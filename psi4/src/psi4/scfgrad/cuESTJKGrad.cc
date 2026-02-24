#include "cuESTJKGrad.h"

#ifdef USING_cuEST

#include <cuda_runtime.h>

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/psi4-dec.h"

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

extern cuestHandle_t cuest_handle;

namespace psi {
namespace scfgrad {

static void check_cuest(cuestStatus_t status, const char* func) {
    if (status != CUEST_STATUS_SUCCESS) {
        std::ostringstream msg;
        msg << "cuEST error in " << func << " (status code " << static_cast<int>(status) << ")";
        throw PSIEXCEPTION(msg.str());
    }
}

#define CHECK_CUEST(call) check_cuest((call), #call)

static void alloc_workspace(cuestWorkspaceDescriptor_t& desc, cuestWorkspace_t& ws) {
    ws = {};
    if (desc.hostBufferSizeInBytes > 0) {
        ws.hostBuffer = reinterpret_cast<uintptr_t>(malloc(desc.hostBufferSizeInBytes));
        ws.hostBufferSizeInBytes = desc.hostBufferSizeInBytes;
    }
    if (desc.deviceBufferSizeInBytes > 0) {
        void* dev_ptr = nullptr;
        cudaMalloc(&dev_ptr, desc.deviceBufferSizeInBytes);
        ws.deviceBuffer = reinterpret_cast<uintptr_t>(dev_ptr);
        ws.deviceBufferSizeInBytes = desc.deviceBufferSizeInBytes;
    }
}

static void free_workspace(cuestWorkspace_t& ws) {
    if (ws.hostBuffer) {
        free(reinterpret_cast<void*>(ws.hostBuffer));
        ws.hostBuffer = 0;
        ws.hostBufferSizeInBytes = 0;
    }
    if (ws.deviceBuffer) {
        cudaFree(reinterpret_cast<void*>(ws.deviceBuffer));
        ws.deviceBuffer = 0;
        ws.deviceBufferSizeInBytes = 0;
    }
}

static cuestAOBasis_t build_cuest_basis(std::shared_ptr<BasisSet> basis,
                                        std::vector<cuestAOShell_t>& shells_out,
                                        cuestWorkspace_t& persistent_ws) {
    auto mol = basis->molecule();
    int natom = mol->natom();

    cuestAOShellParameters_t shell_params;
    CHECK_CUEST(cuestParametersCreate(CUEST_AOSHELL_PARAMETERS, reinterpret_cast<void**>(&shell_params)));

    shells_out.clear();
    std::vector<uint64_t> shells_per_atom(natom);

    for (int A = 0; A < natom; A++) {
        int nshell_on_atom = basis->nshell_on_center(A);
        shells_per_atom[A] = static_cast<uint64_t>(nshell_on_atom);

        for (int Q = 0; Q < nshell_on_atom; Q++) {
            int shell_idx = basis->shell_on_center(A, Q);
            const auto& shell = basis->shell(shell_idx);

            int32_t is_pure = shell.is_pure() ? 1 : 0;
            uint64_t L = static_cast<uint64_t>(shell.am());
            uint64_t nprim = static_cast<uint64_t>(shell.nprimitive());

            cuestAOShell_t cuest_shell;
            CHECK_CUEST(
                cuestAOShellCreate(cuest_handle, is_pure, L, nprim, shell.exps(), shell.coefs(), shell_params, &cuest_shell));

            shells_out.push_back(cuest_shell);
        }
    }

    cuestParametersDestroy(CUEST_AOSHELL_PARAMETERS, shell_params);

    cuestAOBasisParameters_t basis_params;
    CHECK_CUEST(cuestParametersCreate(CUEST_AOBASIS_PARAMETERS, reinterpret_cast<void**>(&basis_params)));

    cuestWorkspaceDescriptor_t persistent_desc = {}, temp_desc = {};
    CHECK_CUEST(cuestAOBasisCreateWorkspaceQuery(cuest_handle, static_cast<uint64_t>(natom), shells_per_atom.data(),
                                                 shells_out.data(), basis_params, &persistent_desc, &temp_desc, nullptr));

    alloc_workspace(persistent_desc, persistent_ws);

    cuestWorkspace_t temp_ws = {};
    alloc_workspace(temp_desc, temp_ws);

    cuestAOBasis_t cuest_basis;
    CHECK_CUEST(cuestAOBasisCreate(cuest_handle, static_cast<uint64_t>(natom), shells_per_atom.data(), shells_out.data(),
                                   basis_params, &persistent_ws, &temp_ws, &cuest_basis));

    free_workspace(temp_ws);
    cuestParametersDestroy(CUEST_AOBASIS_PARAMETERS, basis_params);

    return cuest_basis;
}

cuESTJKGrad::cuESTJKGrad(int deriv, std::shared_ptr<MintsHelper> mints)
    : JKGrad(deriv, mints->get_basisset("ORBITAL")),
      auxiliary_(mints->get_basisset("DF_BASIS_SCF")),
      mints_(mints),
      condition_(1.0E-12) {}

cuESTJKGrad::~cuESTJKGrad() {}

void cuESTJKGrad::print_header() const {
    if (print_) {
        outfile->Printf("  ==> cuESTJKGrad: GPU-Accelerated DF SCF Gradients <==\n\n");
        outfile->Printf("    Gradient:          %11d\n", deriv_);
        outfile->Printf("    J tasked:          %11s\n", (do_J_ ? "Yes" : "No"));
        outfile->Printf("    K tasked:          %11s\n", (do_K_ ? "Yes" : "No"));
        outfile->Printf("    wK tasked:         %11s\n", (do_wK_ ? "Yes" : "No"));
        if (do_wK_) outfile->Printf("    Omega:             %11.3E\n", omega_);
        outfile->Printf("    Fitting Condition: %11.0E\n\n", condition_);
        outfile->Printf("   => Auxiliary Basis Set <=\n\n");
        auxiliary_->print_by_level("outfile", print_);
    }
}

void cuESTJKGrad::compute_gradient() {
    if (!do_J_ && !do_K_) return;

    if (do_wK_) {
        throw PSIEXCEPTION("cuESTJKGrad: Range-separated exchange (wK) gradients not yet supported");
    }

    if (!(Ca_ && Cb_ && Da_ && Db_ && Dt_)) throw PSIEXCEPTION("cuESTJKGrad: Occupation/Density not set");

    int natom = primary_->molecule()->natom();
    int nbf = primary_->nbf();
    size_t nbf2_bytes = static_cast<size_t>(nbf) * nbf * sizeof(double);
    size_t grad_bytes = static_cast<size_t>(natom) * 3 * sizeof(double);

    gradients_.clear();
    if (do_J_) {
        gradients_["Coulomb"] = std::make_shared<Matrix>("Coulomb Gradient", natom, 3);
    }
    if (do_K_) {
        gradients_["Exchange"] = std::make_shared<Matrix>("Exchange Gradient", natom, 3);
    }

    // === Build cuEST infrastructure ===
    std::vector<cuestAOShell_t> primary_shells, auxiliary_shells;
    cuestWorkspace_t primary_ws = {}, auxiliary_ws = {};

    cuestAOBasis_t cuest_primary = build_cuest_basis(primary_, primary_shells, primary_ws);
    cuestAOBasis_t cuest_auxiliary = build_cuest_basis(auxiliary_, auxiliary_shells, auxiliary_ws);

    auto mol = primary_->molecule();
    std::vector<double> xyz(natom * 3);
    for (int A = 0; A < natom; A++) {
        xyz[3 * A + 0] = mol->x(A);
        xyz[3 * A + 1] = mol->y(A);
        xyz[3 * A + 2] = mol->z(A);
    }

    // Build pair list
    cuestAOPairListParameters_t pair_params;
    CHECK_CUEST(cuestParametersCreate(CUEST_AOPAIRLIST_PARAMETERS, reinterpret_cast<void**>(&pair_params)));

    cuestWorkspaceDescriptor_t pair_p_desc = {}, pair_t_desc = {};
    CHECK_CUEST(cuestAOPairListCreateWorkspaceQuery(cuest_handle, cuest_primary, static_cast<uint64_t>(natom), xyz.data(),
                                                    cutoff_, pair_params, &pair_p_desc, &pair_t_desc, nullptr));

    cuestWorkspace_t pair_persistent_ws = {}, pair_temp_ws = {};
    alloc_workspace(pair_p_desc, pair_persistent_ws);
    alloc_workspace(pair_t_desc, pair_temp_ws);

    cuestAOPairList_t pair_list;
    CHECK_CUEST(cuestAOPairListCreate(cuest_handle, cuest_primary, static_cast<uint64_t>(natom), xyz.data(), cutoff_,
                                      pair_params, &pair_persistent_ws, &pair_temp_ws, &pair_list));

    free_workspace(pair_temp_ws);
    cuestParametersDestroy(CUEST_AOPAIRLIST_PARAMETERS, pair_params);

    // Build DF plan (EXCHANGE_FRACTION defaults to 1.0)
    cuestDFIntPlanParameters_t df_params;
    CHECK_CUEST(cuestParametersCreate(CUEST_DFINTPLAN_PARAMETERS, reinterpret_cast<void**>(&df_params)));

    CHECK_CUEST(cuestParametersConfigure(CUEST_DFINTPLAN_PARAMETERS, df_params,
                                         CUEST_DFINTPLAN_PARAMETERS_FITTING_CUTOFF, &condition_, sizeof(double)));

    cuestWorkspaceDescriptor_t df_p_desc = {}, df_t_desc = {};
    CHECK_CUEST(cuestDFIntPlanCreateWorkspaceQuery(cuest_handle, cuest_primary, cuest_auxiliary, pair_list, df_params,
                                                   &df_p_desc, &df_t_desc, nullptr));

    cuestWorkspace_t df_persistent_ws = {}, df_temp_ws = {};
    alloc_workspace(df_p_desc, df_persistent_ws);
    alloc_workspace(df_t_desc, df_temp_ws);

    cuestDFIntPlan_t df_plan;
    CHECK_CUEST(cuestDFIntPlanCreate(cuest_handle, cuest_primary, cuest_auxiliary, pair_list, df_params,
                                     &df_persistent_ws, &df_temp_ws, &df_plan));

    free_workspace(df_temp_ws);
    cuestParametersDestroy(CUEST_DFINTPLAN_PARAMETERS, df_params);

    // === Upload density matrix (Da, alpha density) to GPU ===
    double* d_Da = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_Da), nbf2_bytes);
    cudaMemcpy(d_Da, Da_->get_pointer(), nbf2_bytes, cudaMemcpyHostToDevice);

    // === Prepare MO coefficients for K gradient ===
    double* d_C = nullptr;
    int nocc = Ca_->ncol();
    if (do_K_ && nocc > 0) {
        size_t c_bytes = static_cast<size_t>(nocc) * nbf * sizeof(double);
        std::vector<double> C_row_major(nocc * nbf);
        double** Cp = Ca_->pointer();
        for (int i = 0; i < nocc; i++) {
            for (int mu = 0; mu < nbf; mu++) {
                C_row_major[i * nbf + mu] = Cp[mu][i];
            }
        }
        cudaMalloc(reinterpret_cast<void**>(&d_C), c_bytes);
        cudaMemcpy(d_C, C_row_major.data(), c_bytes, cudaMemcpyHostToDevice);
    }

    double* d_grad = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_grad), grad_bytes);

    cuestWorkspaceDescriptor_t max_ws_desc = {};
    max_ws_desc.hostBufferSizeInBytes = 0;
    max_ws_desc.deviceBufferSizeInBytes = static_cast<size_t>(2) * 1024 * 1024 * 1024;

    // === Compute J gradient ===
    // E_JK = densityScale * E_J(D) + coefficientScale * exchangeFraction * E_K(C)
    // J call: densityScale=2.0 (accounts for RHF Da→Dt doubling), coefficientScale=0.0
    if (do_J_) {
        cuestWorkspaceDescriptor_t j_temp_desc = {};
        CHECK_CUEST(cuestDFSymmetricDerivativeComputeWorkspaceQuery(
            cuest_handle, df_plan, &max_ws_desc, &j_temp_desc, 2.0, nullptr, 0.0, 0, nullptr, nullptr, nullptr));

        cuestWorkspace_t j_temp_ws = {};
        alloc_workspace(j_temp_desc, j_temp_ws);

        cudaMemset(d_grad, 0, grad_bytes);
        CHECK_CUEST(cuestDFSymmetricDerivativeCompute(cuest_handle, df_plan, &max_ws_desc, &j_temp_ws, 2.0, d_Da, 0.0,
                                                      0, nullptr, nullptr, d_grad));
        cudaDeviceSynchronize();

        std::vector<double> grad_host(natom * 3);
        cudaMemcpy(grad_host.data(), d_grad, grad_bytes, cudaMemcpyDeviceToHost);

        double** Jp = gradients_["Coulomb"]->pointer();
        for (int A = 0; A < natom; A++) {
            Jp[A][0] = grad_host[3 * A + 0];
            Jp[A][1] = grad_host[3 * A + 1];
            Jp[A][2] = grad_host[3 * A + 2];
        }

        free_workspace(j_temp_ws);
    }

    // === Compute K gradient ===
    // K call: densityScale=0.0, coefficientScale=1.0 → raw dE_K/dx
    // scf_grad.cc will multiply by -alpha externally
    if (do_K_ && nocc > 0) {
        uint64_t nocc64 = static_cast<uint64_t>(nocc);

        cuestWorkspaceDescriptor_t k_temp_desc = {};
        CHECK_CUEST(cuestDFSymmetricDerivativeComputeWorkspaceQuery(
            cuest_handle, df_plan, &max_ws_desc, &k_temp_desc, 0.0, nullptr, 1.0, 1, &nocc64, nullptr, nullptr));

        cuestWorkspace_t k_temp_ws = {};
        alloc_workspace(k_temp_desc, k_temp_ws);

        cudaMemset(d_grad, 0, grad_bytes);
        CHECK_CUEST(cuestDFSymmetricDerivativeCompute(cuest_handle, df_plan, &max_ws_desc, &k_temp_ws, 0.0, d_Da, 1.0,
                                                      1, &nocc64, d_C, d_grad));
        cudaDeviceSynchronize();

        std::vector<double> grad_host(natom * 3);
        cudaMemcpy(grad_host.data(), d_grad, grad_bytes, cudaMemcpyDeviceToHost);

        double** Kp = gradients_["Exchange"]->pointer();
        for (int A = 0; A < natom; A++) {
            Kp[A][0] = grad_host[3 * A + 0];
            Kp[A][1] = grad_host[3 * A + 1];
            Kp[A][2] = grad_host[3 * A + 2];
        }

        free_workspace(k_temp_ws);
    }

    // === Cleanup ===
    cudaFree(d_grad);
    if (d_C) cudaFree(d_C);
    cudaFree(d_Da);

    cuestDFIntPlanDestroy(df_plan);
    cuestAOPairListDestroy(pair_list);
    cuestAOBasisDestroy(cuest_auxiliary);
    cuestAOBasisDestroy(cuest_primary);

    for (auto& s : primary_shells) cuestAOShellDestroy(s);
    for (auto& s : auxiliary_shells) cuestAOShellDestroy(s);

    free_workspace(df_persistent_ws);
    free_workspace(pair_persistent_ws);
    free_workspace(auxiliary_ws);
    free_workspace(primary_ws);
}

void cuESTJKGrad::compute_hessian() {
    throw PSIEXCEPTION("cuESTJKGrad: Hessian not implemented");
}

}  // namespace scfgrad
}  // namespace psi

#endif  // USING_cuEST
