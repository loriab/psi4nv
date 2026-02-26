#include "cuESTJKGrad.h"

#ifdef USING_cuEST

#include "psi4/libfock/cuESTCommon.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/psi4-dec.h"

#include <vector>

using psi::cuest_common::alloc_workspace;
using psi::cuest_common::free_workspace;
using psi::cuest_common::build_cuest_basis;
using psi::cuest_common::build_cuest_pairlist;

namespace psi {
namespace scfgrad {

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
    cuestWorkspace_t pair_persistent_ws = {};
    cuestAOPairList_t pair_list = build_cuest_pairlist(cuest_primary, natom, xyz.data(), cutoff_, pair_persistent_ws);

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
        cuestDFSymmetricDerivativeComputeParameters_t derivative_params;
        CHECK_CUEST(cuestParametersCreate(CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS, &derivative_params));
        CHECK_CUEST(cuestDFSymmetricDerivativeComputeWorkspaceQuery(
            cuest_handle,
            df_plan,
            derivative_params,
            &max_ws_desc,
            &j_temp_desc,
            2.0,
            nullptr,
            0.0,
            0,
            nullptr,
            nullptr,
            nullptr));

        cuestWorkspace_t j_temp_ws = {};
        alloc_workspace(j_temp_desc, j_temp_ws);

        cudaMemset(d_grad, 0, grad_bytes);
        CHECK_CUEST(cuestDFSymmetricDerivativeCompute(
            cuest_handle,
            df_plan,
            derivative_params,
            &max_ws_desc,
            &j_temp_ws,
            2.0,
            d_Da,
            0.0,
            0,
            nullptr,
            nullptr,
            d_grad));
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
        cuestParametersDestroy(CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS, derivative_params);
    }

    // === Compute K gradient ===
    // K call: densityScale=0.0, coefficientScale=1.0 → raw dE_K/dx
    // scf_grad.cc will multiply by -alpha externally
    if (do_K_ && nocc > 0) {
        uint64_t nocc64 = static_cast<uint64_t>(nocc);
        cuestDFSymmetricExchangeComputeParameters_t exchange_params;
        CHECK_CUEST(cuestParametersCreate(CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS, &exchange_params));
        cuestWorkspaceDescriptor_t k_temp_desc = {};
        CHECK_CUEST(cuestDFSymmetricDerivativeComputeWorkspaceQuery(
            cuest_handle,
            df_plan,
            exchange_params,
            &max_ws_desc,
            &k_temp_desc,
            0.0,
            nullptr,
            1.0,
            1,
            &nocc64,
            nullptr,
            nullptr));

        cuestWorkspace_t k_temp_ws = {};
        alloc_workspace(k_temp_desc, k_temp_ws);

        cudaMemset(d_grad, 0, grad_bytes);
        CHECK_CUEST(cuestDFSymmetricDerivativeCompute(
            cuest_handle,
            df_plan,
            exchange_params,
            &max_ws_desc,
            &k_temp_ws,
            0.0,
            d_Da,
            1.0,
            1,
            &nocc64,
            d_C,
            d_grad));
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
        cuestParametersDestroy(CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS, exchange_params);
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
