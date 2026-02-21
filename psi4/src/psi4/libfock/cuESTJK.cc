#include "cuESTJK.h"

#ifdef USING_cuEST

#include <cuda_runtime.h>

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/integral.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/psi4-dec.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <vector>

extern cuestHandle_t cuest_handle;

namespace psi {

static void check_cuest(cuestStatus_t status, const char* func) {
    if (status != CUEST_STATUS_SUCCESS) {
        std::ostringstream msg;
        msg << "cuEST error in " << func << " (status code " << static_cast<int>(status) << ")";
        throw PSIEXCEPTION(msg.str());
    }
}

#define CHECK_CUEST(call) check_cuest((call), #call)

cuESTJK::cuESTJK(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options)
    : JK(primary),
      options_(options),
      auxiliary_(auxiliary),
      condition_(1.0E-12),
      cuest_primary_basis_(nullptr),
      cuest_auxiliary_basis_(nullptr),
      cuest_pair_list_(nullptr),
      cuest_df_plan_(nullptr),
      plan_built_(false) {
    primary_persistent_ws_ = {};
    auxiliary_persistent_ws_ = {};
    pair_persistent_ws_ = {};
    df_persistent_ws_ = {};
    compute_temp_ws_ = {};
    exchange_max_ws_desc_ = {};
}

cuESTJK::~cuESTJK() {
    destroy_cuest_objects();
}

void cuESTJK::allocate_workspace(cuestWorkspaceDescriptor_t& desc, cuestWorkspace_t& ws) {
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

void cuESTJK::free_workspace(cuestWorkspace_t& ws) {
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

cuestAOBasis_t cuESTJK::build_cuest_basis(std::shared_ptr<BasisSet> basis,
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
            CHECK_CUEST(cuestAOShellCreate(
                cuest_handle, is_pure, L, nprim,
                shell.exps(), shell.coefs(),
                shell_params, &cuest_shell));

            shells_out.push_back(cuest_shell);
        }
    }

    cuestParametersDestroy(CUEST_AOSHELL_PARAMETERS, shell_params);

    cuestAOBasisParameters_t basis_params;
    CHECK_CUEST(cuestParametersCreate(CUEST_AOBASIS_PARAMETERS, reinterpret_cast<void**>(&basis_params)));

    cuestWorkspaceDescriptor_t persistent_desc = {}, temp_desc = {};
    CHECK_CUEST(cuestAOBasisCreateWorkspaceQuery(
        cuest_handle, static_cast<uint64_t>(natom), shells_per_atom.data(),
        shells_out.data(), basis_params,
        &persistent_desc, &temp_desc, nullptr));

    allocate_workspace(persistent_desc, persistent_ws);

    cuestWorkspace_t temp_ws = {};
    allocate_workspace(temp_desc, temp_ws);

    cuestAOBasis_t cuest_basis;
    CHECK_CUEST(cuestAOBasisCreate(
        cuest_handle, static_cast<uint64_t>(natom), shells_per_atom.data(),
        shells_out.data(), basis_params,
        &persistent_ws, &temp_ws, &cuest_basis));

    free_workspace(temp_ws);
    cuestParametersDestroy(CUEST_AOBASIS_PARAMETERS, basis_params);

    return cuest_basis;
}

void cuESTJK::destroy_cuest_objects() {
    if (cuest_df_plan_) {
        cuestDFIntPlanDestroy(cuest_df_plan_);
        cuest_df_plan_ = nullptr;
    }
    if (cuest_pair_list_) {
        cuestAOPairListDestroy(cuest_pair_list_);
        cuest_pair_list_ = nullptr;
    }
    if (cuest_auxiliary_basis_) {
        cuestAOBasisDestroy(cuest_auxiliary_basis_);
        cuest_auxiliary_basis_ = nullptr;
    }
    if (cuest_primary_basis_) {
        cuestAOBasisDestroy(cuest_primary_basis_);
        cuest_primary_basis_ = nullptr;
    }

    for (auto& s : cuest_primary_shells_) cuestAOShellDestroy(s);
    cuest_primary_shells_.clear();
    for (auto& s : cuest_auxiliary_shells_) cuestAOShellDestroy(s);
    cuest_auxiliary_shells_.clear();

    free_workspace(df_persistent_ws_);
    free_workspace(pair_persistent_ws_);
    free_workspace(auxiliary_persistent_ws_);
    free_workspace(primary_persistent_ws_);
    free_workspace(compute_temp_ws_);

    plan_built_ = false;
}

size_t cuESTJK::memory_estimate() {
    return 0;
}

void cuESTJK::print_header() const {
    if (print_) {
        outfile->Printf("  ==> cuESTJK: GPU-Accelerated Density-Fitted J/K Matrices <==\n\n");
        outfile->Printf("    J tasked:          %11s\n", (do_J_ ? "Yes" : "No"));
        outfile->Printf("    K tasked:          %11s\n", (do_K_ ? "Yes" : "No"));
        outfile->Printf("    wK tasked:         %11s\n", (do_wK_ ? "Yes" : "No"));
        if (do_wK_) outfile->Printf("    Omega:             %11.3E\n", omega_);
        outfile->Printf("    Fitting Coverage:  %11.0E\n", condition_);
        outfile->Printf("\n");
    }
}

void cuESTJK::preiterations() {
    if (plan_built_) return;

    cuest_primary_basis_ = build_cuest_basis(primary_, cuest_primary_shells_, primary_persistent_ws_);
    cuest_auxiliary_basis_ = build_cuest_basis(auxiliary_, cuest_auxiliary_shells_, auxiliary_persistent_ws_);

    auto mol = primary_->molecule();
    int natom = mol->natom();

    std::vector<double> xyz(natom * 3);
    for (int A = 0; A < natom; A++) {
        xyz[3 * A + 0] = mol->x(A);
        xyz[3 * A + 1] = mol->y(A);
        xyz[3 * A + 2] = mol->z(A);
    }

    cuestAOPairListParameters_t pair_params;
    CHECK_CUEST(cuestParametersCreate(CUEST_AOPAIRLIST_PARAMETERS, reinterpret_cast<void**>(&pair_params)));

    cuestWorkspaceDescriptor_t pair_persistent_desc = {}, pair_temp_desc = {};
    CHECK_CUEST(cuestAOPairListCreateWorkspaceQuery(
        cuest_handle, cuest_primary_basis_, static_cast<uint64_t>(natom), xyz.data(),
        cutoff_, pair_params, &pair_persistent_desc, &pair_temp_desc, nullptr));

    allocate_workspace(pair_persistent_desc, pair_persistent_ws_);

    cuestWorkspace_t pair_temp_ws = {};
    allocate_workspace(pair_temp_desc, pair_temp_ws);

    CHECK_CUEST(cuestAOPairListCreate(
        cuest_handle, cuest_primary_basis_, static_cast<uint64_t>(natom), xyz.data(),
        cutoff_, pair_params,
        &pair_persistent_ws_, &pair_temp_ws, &cuest_pair_list_));

    free_workspace(pair_temp_ws);
    cuestParametersDestroy(CUEST_AOPAIRLIST_PARAMETERS, pair_params);

    cuestDFIntPlanParameters_t df_params;
    CHECK_CUEST(cuestParametersCreate(CUEST_DFINTPLAN_PARAMETERS, reinterpret_cast<void**>(&df_params)));

    CHECK_CUEST(cuestParametersConfigure(CUEST_DFINTPLAN_PARAMETERS, df_params,
        CUEST_DFINTPLAN_PARAMETERS_FITTING_CUTOFF, &condition_, sizeof(double)));

    cuestWorkspaceDescriptor_t df_persistent_desc = {}, df_temp_desc = {};
    CHECK_CUEST(cuestDFIntPlanCreateWorkspaceQuery(
        cuest_handle, cuest_primary_basis_, cuest_auxiliary_basis_,
        cuest_pair_list_, df_params,
        &df_persistent_desc, &df_temp_desc, nullptr));

    allocate_workspace(df_persistent_desc, df_persistent_ws_);

    cuestWorkspace_t df_temp_ws = {};
    allocate_workspace(df_temp_desc, df_temp_ws);

    CHECK_CUEST(cuestDFIntPlanCreate(
        cuest_handle, cuest_primary_basis_, cuest_auxiliary_basis_,
        cuest_pair_list_, df_params,
        &df_persistent_ws_, &df_temp_ws, &cuest_df_plan_));

    free_workspace(df_temp_ws);
    cuestParametersDestroy(CUEST_DFINTPLAN_PARAMETERS, df_params);

    exchange_max_ws_desc_.hostBufferSizeInBytes = 0;
    exchange_max_ws_desc_.deviceBufferSizeInBytes = static_cast<size_t>(2) * 1024 * 1024 * 1024;

    plan_built_ = true;
}

void cuESTJK::compute_JK() {
    int nbf = primary_->nbf();
    size_t nbf2_bytes = static_cast<size_t>(nbf) * nbf * sizeof(double);

    double* d_D = nullptr;
    double* d_J = nullptr;
    double* d_K = nullptr;
    double* d_C = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_D), nbf2_bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_J), nbf2_bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_K), nbf2_bytes);

    cuestWorkspaceDescriptor_t j_temp_desc = {};
    cuestWorkspaceDescriptor_t k_temp_desc = {};
    size_t max_host = 0, max_device = 0;

    if (do_J_) {
        CHECK_CUEST(cuestDFCoulombComputeWorkspaceQuery(
            cuest_handle, cuest_df_plan_, &j_temp_desc, d_D, d_J));
        max_host = std::max(max_host, j_temp_desc.hostBufferSizeInBytes);
        max_device = std::max(max_device, j_temp_desc.deviceBufferSizeInBytes);
    }

    if (do_K_) {
        for (size_t N = 0; N < D_ao_.size(); N++) {
            int nocc = C_left_ao_[N]->ncol();
            if (nocc == 0) continue;

            CHECK_CUEST(cuestDFSymmetricExchangeComputeWorkspaceQuery(
                cuest_handle, cuest_df_plan_, &exchange_max_ws_desc_,
                &k_temp_desc, static_cast<uint64_t>(nocc), nullptr, d_K));
            max_host = std::max(max_host, k_temp_desc.hostBufferSizeInBytes);
            max_device = std::max(max_device, k_temp_desc.deviceBufferSizeInBytes);
        }
    }

    cuestWorkspaceDescriptor_t total_desc;
    total_desc.hostBufferSizeInBytes = max_host;
    total_desc.deviceBufferSizeInBytes = max_device;

    free_workspace(compute_temp_ws_);
    allocate_workspace(total_desc, compute_temp_ws_);

    for (size_t N = 0; N < D_ao_.size(); N++) {
        if (do_J_) {
            cudaMemcpy(d_D, D_ao_[N]->get_pointer(), nbf2_bytes, cudaMemcpyHostToDevice);
            CHECK_CUEST(cuestDFCoulombCompute(
                cuest_handle, cuest_df_plan_, &compute_temp_ws_, d_D, d_J));
            cudaMemcpy(J_ao_[N]->get_pointer(), d_J, nbf2_bytes, cudaMemcpyDeviceToHost);
        }

        if (do_K_) {
            int nocc = C_left_ao_[N]->ncol();
            if (nocc > 0) {
                size_t c_bytes = static_cast<size_t>(nocc) * nbf * sizeof(double);

                std::vector<double> C_row_major(nocc * nbf);
                double** Cp = C_left_ao_[N]->pointer();
                for (int i = 0; i < nocc; i++) {
                    for (int mu = 0; mu < nbf; mu++) {
                        C_row_major[i * nbf + mu] = Cp[mu][i];
                    }
                }

                cudaFree(d_C);
                cudaMalloc(reinterpret_cast<void**>(&d_C), c_bytes);
                cudaMemcpy(d_C, C_row_major.data(), c_bytes, cudaMemcpyHostToDevice);

                CHECK_CUEST(cuestDFSymmetricExchangeCompute(
                    cuest_handle, cuest_df_plan_, &exchange_max_ws_desc_,
                    &compute_temp_ws_, static_cast<uint64_t>(nocc),
                    d_C, d_K));

                cudaMemcpy(K_ao_[N]->get_pointer(), d_K, nbf2_bytes, cudaMemcpyDeviceToHost);
            }
        }
    }

    cudaFree(d_D);
    cudaFree(d_J);
    cudaFree(d_K);
    if (d_C) cudaFree(d_C);
}

void cuESTJK::postiterations() {
    destroy_cuest_objects();
}

} // namespace psi

#endif // USING_cuEST
