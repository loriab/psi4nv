# cuEST DFT & Gradient Implementation Guide

Detailed implementation notes for extending Psi4's cuEST integration to cover DFT energy, RHF gradients, and full DFT gradients. This supplements the plan at `.cursor/plans/cuest_dft_and_gradients_ce16c569.plan.md`.

## Status

- **Phase 1**: DONE — DFT energy verified with water (B3LYP, PBE) at `tests/cuest-dft-h2o/input.dat`. Energies match DF to ~1e-11 Eh.
- **Phase 3 (J/K Gradient)**: DONE — `cuESTJKGrad` implemented and verified with water (RHF, B3LYP, PBE) at `tests/cuest-grad-h2o/input.dat`. Gradients match DF to ~1e-10 Eh/Bohr. Also verified with cc-pVDZ.
- **Phase 2 (GPU XC)**: DONE — GPU-accelerated XC potential and gradient via cuEST. Verified with water (B3LYP, B3LYP5, PBE, BLYP, PBE0) at `tests/cuest-gpu-xc-h2o/input.dat`. Energies match DF to ~2-7e-8 Eh, gradients to ~2e-6 Eh/Bohr. Opt-in via `PSI4_CUEST_GPU_XC=1` env var. Paxlovid benchmark at `tests/cuest-bench-paxlovid/bench_full_gpu_dft.py`.
- **Phase 4 (Full GPU Gradient)**: DONE — Combined GPU J/K gradient (cuESTJKGrad) + GPU XC gradient (cuEST XC derivative). Verified in Paxlovid benchmark: gradient delta vs DF is ~1.9e-4 Eh/Bohr (grid difference). Full GPU DFT energy speedup: 1.87x, gradient speedup: 1.73x over 8-thread DF.

## Global cuEST Handle

Psi4 has a single global cuEST handle declared in `psi4/src/core.cc`:

```cpp
// psi4/src/core.cc:142
cuestHandle_t cuest_handle = 0;
```

All new cuEST code should `extern` this handle:

```cpp
extern cuestHandle_t cuest_handle;
```

---

## Phase 1: DFT Verification (No Code Changes Needed)

DFT with `SCF_TYPE=CUEST` should already work because:
- `cuESTJK` respects `do_J_` and `do_K_` flags (set by the functional)
- XC potential is computed separately by `VBase` (CPU) — independent of JK
- `CUEST` is already in the valid `SCF_TYPE` list

Test script: `tests/cuest-bench-paxlovid/bench_dft.py` — runs B3LYP and PBE with STO-3G, comparing DF vs CUEST energies.

---

## Phase 2: cuEST XC Potential (GPU-accelerated DFT)

### Overview

Replace Psi4's CPU-based `RV::compute_V()` with cuEST's `cuestXCPotentialRKSCompute` for supported functionals.

### Key API Difference

| | Psi4 `RV::compute_V()` | cuEST `cuestXCPotentialRKSCompute` |
|---|---|---|
| **Input** | Density matrix D (nao×nao) | Occupied MO coefficients C_occ (nocc×nao, row-major) |
| **Output** | Vxc matrix (nao×nao) | Vxc matrix (nao×nao) + XC energy scalar |
| **Grid** | `DFTGrid` (CPU, block-based) | `cuestMolecularGrid_t` (GPU) |
| **Functional** | `SuperFunctional` object | `cuestXCIntPlanParametersFunctional_t` enum |

### cuEST C API Signatures

```c
// XC potential (RKS)
cuestStatus_t cuestXCPotentialRKSCompute(
    cuestHandle_t handle,
    cuestXCIntPlan_t plan,
    const cuestWorkspaceDescriptor_t* maximumWorkspaceDescriptor,
    cuestWorkspace_t* temporaryWorkspace,
    uint64_t numOccupied,
    const double* coefficientMatrix,    // device pointer, nocc×nao row-major
    double* outXCEnergy,                // host pointer to scalar
    double* outXCPotentialMatrix);      // device pointer, nao×nao

// XC potential workspace query
cuestStatus_t cuestXCPotentialRKSComputeWorkspaceQuery(
    cuestHandle_t handle,
    cuestXCIntPlan_t plan,
    const cuestWorkspaceDescriptor_t* maximumWorkspaceDescriptor,
    cuestWorkspaceDescriptor_t* temporaryWorkspaceDescriptor,
    uint64_t numOccupied,
    const double* coefficientMatrix,    // may be NULL
    double* outXCEnergy,                // may be NULL
    double* outXCPotentialMatrix);      // may be NULL
```

### XC Grid Construction

cuEST needs its own molecular grid built from per-atom grids:

```c
// 1. Create atom grids (one per atom)
cuestStatus_t cuestAtomGridCreate(
    cuestHandle_t handle,
    uint64_t numRadialPoints,
    const double* radialNodes,       // Ahlrichs radial quadrature nodes
    const double* radialWeights,     // Ahlrichs radial quadrature weights
    const uint64_t* numAngularPoints, // Lebedev angular points per radial shell
    const cuestAtomGridParameters_t parameters,
    cuestAtomGrid_t* outAtomGrid);

// 2. Create molecular grid from atom grids
cuestStatus_t cuestMolecularGridCreate(
    cuestHandle_t handle,
    uint64_t numAtoms,
    const cuestAtomGrid_t* atomGrid,
    const double* xyz,               // atom coordinates in Bohr
    const cuestMolecularGridParameters_t parameters,
    cuestWorkspace_t* persistentWorkspace,
    cuestWorkspace_t* temporaryWorkspace,
    cuestMolecularGrid_t* outGrid);

// 3. Create XC integral plan
cuestStatus_t cuestXCIntPlanCreate(
    cuestHandle_t handle,
    const cuestAOBasis_t basis,
    const cuestMolecularGrid_t grid,
    cuestXCIntPlanParametersFunctional_t functional,
    const cuestXCIntPlanParameters_t parameters,
    cuestWorkspace_t* persistentWorkspace,
    cuestWorkspace_t* temporaryWorkspace,
    cuestXCIntPlan_t* outPlan);
```

### Ahlrichs Radial Quadrature

The cuEST examples use Ahlrichs radial quadrature (see `cuest/examples/python_api/helpers/grid_utils.py`):

```
alpha = 0.6
n = 1..npoint
z = n * pi / (npoint + 1)
x = cos(z), y = sin(z)
u = ln((1-x)/2)
v = ((1+x)^alpha) / ln(2)
radial_nodes = -R * v * u
radial_weights = pi/(npoint+1) * y * R * v * (-alpha*u/(1+x) + 1/(1-x)) * radial_nodes^2
```

Where R is the Ahlrichs atomic radius (H=0.80, C=1.10, N=0.90, O=0.90, F=0.90, etc.).

### Functional Mapping

```cpp
// cuEST supported functionals (from cuest/include/h/types/cuest_parameter_types.h)
enum cuestXCIntPlanParametersFunctional_t {
    CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_HF     = 0,
    CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP1 = 1,  // B3LYP variant 1
    CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP5 = 2,  // B3LYP variant 5  
    CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B97    = 3,
    CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_BLYP   = 4,
    CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M06L   = 5,
    CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE    = 6,
    CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE0   = 7,
    CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_R2SCAN = 8,
    CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_SVWN5  = 9,
    CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B97MV  = 10,
};
```

Need a mapping function from Psi4's `SuperFunctional::name()` to cuEST enum. Unsupported functionals fall back to CPU `RV`.

### Implementation Approach: BrianQC-style Short-circuit

Rather than creating a full VBase subclass, follow the existing BrianQC pattern in `RV::compute_V()` (psi4/src/psi4/libfock/v.cc line ~1271):

```cpp
void RV::compute_V(std::vector<SharedMatrix> ret) {
    timer_on("RV: Form V");
    
#ifdef USING_BrianQC
    if (brianEnable and brianEnableDFT) {
        // BrianQC short-circuit: compute XC on GPU and return
        brianSCFBuildFockDFT(...);
        return;
    }
#endif

    // ... normal CPU path follows
```

For cuEST, add a similar block:

```cpp
#ifdef USING_cuEST
    if (cuest_xc_enabled_) {
        cuest_compute_V(ret);
        return;
    }
#endif
```

### Getting C_occ into VBase

cuEST XC needs occupied MO coefficients, not density matrix. Options:

1. **Add `set_C()` to VBase** (cleanest):
   - Add `std::vector<SharedMatrix> C_AO_;` member to VBase
   - Add `void set_C(std::vector<SharedMatrix> Cvec)` method
   - Modify `RHF::form_V()` in `psi4/src/psi4/libscf_solver/rhf.cc` to call `potential_->set_C({Ca_})` before `compute_V()`

2. **Reconstruct from D** (hacky, avoid)

### Implementation (DONE)

**Modified** `psi4/src/psi4/libfock/v.h`:
- Added `set_C()` method and `C_AO_` member to VBase
- Added cuEST XC state members (grid, xc plan, workspace, basis) to VBase under `#ifdef USING_cuEST`
- Uses `void*` for cuEST handles and a `CuESTWorkspace` struct to avoid including `cuest.h` in the header

**Modified** `psi4/src/psi4/libfock/v.cc`:
- Added cuEST utility functions before `namespace psi {}`: Ahlrichs quadrature, radii table, functional mapping, workspace helpers, basis builder
- Added `VBase::cuest_xc_initialize()`: builds cuEST AO basis, atom grids (Ahlrichs radial + Lebedev angular), molecular grid, and XC integral plan
- Added `VBase::cuest_xc_cleanup()`: destroys all cuEST objects and frees workspaces
- In `RV::compute_V()`: added cuEST short-circuit before BrianQC block — transposes C_occ, calls `cuestXCPotentialRKSCompute`, downloads Vxc
- In `RV::compute_gradient()`: added cuEST short-circuit — calls `cuestXCDerivativeRKSCompute`, downloads gradient

**Modified** `psi4/src/psi4/libscf_solver/rhf.cc`:
- In `RHF::form_V()`: added `potential_->set_C({Ca_subset("AO", "OCC")})` before `compute_V()`

**Modified** `psi4/src/psi4/scfgrad/scf_grad.cc`:
- In `SCFDeriv::compute_gradient()`: added `potential_->set_C({Ca_occ})` after `set_D()` for RKS

**Modified** `psi4/src/psi4/libscf_solver/CMakeLists.txt`:
- Added `USING_cuEST` compile definition to `scf_solver` target to ensure consistent struct layout

**Opt-in**: Set `PSI4_CUEST_GPU_XC=1` environment variable to enable. Without it, CPU XC is used even with `SCF_TYPE=CUEST`.

### Functional Mapping (Verified)

```cpp
{"B3LYP",  B3LYP1},  // Psi4 B3LYP uses VWN_RPA → cuEST B3LYP1
{"B3LYP5", B3LYP5},  // Psi4 B3LYP5 uses VWN5 → cuEST B3LYP5
{"BLYP",   BLYP},
{"PBE",    PBE},
{"PBE0",   PBE0},
{"M06-L",  M06L},
{"R2SCAN", R2SCAN},
{"B97M-V", B97MV},
```

Note: SVWN mapping had large errors (~196 mEh) and was removed. Needs investigation.

### Verification Results

Water (H2O) / STO-3G with 590 angular, 99 radial points:

| Functional | Energy delta (Eh) |
|------------|-------------------|
| B3LYP      | 2.25e-08          |
| B3LYP5     | 2.25e-08          |
| PBE        | 6.80e-08          |
| BLYP       | 2.28e-08          |
| PBE0       | 6.06e-08          |

Gradient agreement: ~2e-6 Eh/Bohr (from grid quadrature difference).

---

## Phase 3: cuEST J/K Gradient

### Overview

Create `cuESTJKGrad` class inheriting from `JKGrad` using `cuestDFSymmetricDerivativeCompute`.

### cuEST Gradient API

```c
cuestStatus_t cuestDFSymmetricDerivativeCompute(
    cuestHandle_t handle,
    cuestDFIntPlan_t plan,
    const cuestWorkspaceDescriptor_t* maximumWorkspaceDescriptor,
    cuestWorkspace_t* temporaryWorkspace,
    double densityScale,                 // 2.0 for J contribution
    const double* densityMatrix,         // device, nao×nao
    double coefficientScale,             // -alpha for K (alpha = HF exchange fraction)
    uint64_t numCoefficientMatrices,     // 1 for RHF
    const uint64_t* numOccupied,         // [nocc]
    const double* coefficientMatrices,   // device, nocc×nao row-major
    double* outGradient);                // device, natom×3

// Workspace query variant has same signature with descriptors instead of workspaces
```

Key: `densityScale` controls J contribution, `coefficientScale` controls K contribution. The API computes a **combined** J+K gradient in one call.

### Psi4 Gradient Pipeline

```
run_scf_gradient() [proc.py]
  → core.scfgrad(ref_wfn) [wrapper.cc]
    → SCFDeriv::compute_gradient() [scf_grad.cc]
      → Nuclear gradient (molecule)
      → Core gradient: mints->core_hamiltonian_grad(Dt) [kinetic + potential]
      → Overlap gradient: mints->overlap_grad(W)
      → JK gradient: JKGrad::build_JKGrad(1, mints) → compute_gradient()
        → gradients_["Coulomb"], gradients_["Exchange"]
      → XC gradient: potential_->compute_gradient()
      → Sum all terms
```

### Implementation

**Create** `psi4/src/psi4/scfgrad/cuESTJKGrad.h`:

```cpp
#ifndef CUEST_JK_GRAD_H
#define CUEST_JK_GRAD_H

#include "jk_grad.h"

#ifdef USING_cuEST
#include <cuest.h>

namespace psi {
namespace scfgrad {

class cuESTJKGrad : public JKGrad {
protected:
    std::shared_ptr<BasisSet> auxiliary_;
    std::shared_ptr<MintsHelper> mints_;
    double condition_;

public:
    cuESTJKGrad(int deriv, std::shared_ptr<MintsHelper> mints);
    ~cuESTJKGrad() override;

    void compute_gradient() override;
    void compute_hessian() override;
    void print_header() const override;

    void set_condition(double condition) { condition_ = condition; }
};

} // namespace scfgrad
} // namespace psi

#endif // USING_cuEST
#endif // CUEST_JK_GRAD_H
```

**Create** `psi4/src/psi4/scfgrad/cuESTJKGrad.cc`:

Key implementation of `compute_gradient()`:
1. Build cuEST basis, pair list, DF plan (reuse `cuESTJK::build_cuest_basis()` logic or share)
2. Upload D (total density) and C_occ to GPU
3. For pure HF (alpha=1.0): call `cuestDFSymmetricDerivativeCompute` with `densityScale=2.0`, `coefficientScale=-1.0`
4. For hybrid DFT: `coefficientScale = -alpha` (HF exchange fraction)
5. For pure DFT: `coefficientScale = 0.0`, `numCoefficientMatrices = 0`
6. Download gradient (natom×3) from GPU
7. Store in `gradients_["Coulomb"]` (J part) and `gradients_["Exchange"]` (K part)

**Note on J/K separation**: `cuestDFSymmetricDerivativeCompute` computes combined J+K gradient. To separate them for Psi4 (which wants separate "Coulomb" and "Exchange" entries), either:
- Call twice: once with `coefficientScale=0.0` (J only), once with `densityScale=0.0` (K only)
- Or store combined as "Coulomb" and leave "Exchange" as zero, applying the scaling inside

The first approach is cleaner for compatibility with `scf_grad.cc` which scales Exchange by `-alpha` separately.

**Modify** `psi4/src/psi4/scfgrad/jk_grad.cc`:

In `JKGrad::build_JKGrad()` (line 79-119), add before the `else throw` block:

```cpp
} else if (options.get_str("SCF_TYPE") == "CUEST") {
#ifdef USING_cuEST
    cuESTJKGrad* jk = new cuESTJKGrad(deriv, mints);
    if (options["INTS_TOLERANCE"].has_changed())
        jk->set_cutoff(options.get_double("INTS_TOLERANCE"));
    if (options["PRINT"].has_changed())
        jk->set_print(options.get_int("PRINT"));
    if (options["DEBUG"].has_changed())
        jk->set_debug(options.get_int("DEBUG"));
    jk->set_condition(options.get_double("DF_FITTING_CONDITION"));
    return std::shared_ptr<JKGrad>(jk);
#else
    throw PSIEXCEPTION("JKGrad: SCF_TYPE CUEST requires cuEST library");
#endif
}
```

**Modify** `psi4/src/psi4/scfgrad/CMakeLists.txt`:

```cmake
list(APPEND sources
  cuESTJKGrad.cc   # ADD THIS
  jk_grad.cc
  response.cc
  scf_grad.cc
  wrapper.cc
  )

# After target creation, add:
if(TARGET cuEST::cuEST)
  target_compile_definitions(scfgrad
    PUBLIC
      USING_cuEST
    )
  target_link_libraries(scfgrad
    PUBLIC
      cuEST::cuEST
    )
endif()
```

### Reusing cuEST Infrastructure

The `cuESTJK` class already has all the boilerplate for building cuEST basis sets, pair lists, and DF plans. Options:
1. **Extract shared helpers** into a utility header (e.g., `cuESTCommon.h`)
2. **Duplicate the code** in `cuESTJKGrad` (simpler but redundant)
3. **Make cuESTJK's methods static/accessible** (requires refactoring)

Recommendation: Extract `build_cuest_basis()`, `allocate_workspace()`, `free_workspace()` into `cuESTCommon.h` as free functions or a utility class. Both `cuESTJK` and `cuESTJKGrad` can then use them.

---

## Phase 4: Full GPU Gradient

### 4a: One-Electron Gradients via cuEST

Currently in `scf_grad.cc`:

```cpp
// Core (kinetic + potential) gradient
gradients_["Core"] = mints->core_hamiltonian_grad(Dt);

// Overlap gradient  
gradients_["Overlap"] = mints->overlap_grad(W);
```

cuEST provides:

```c
// Overlap gradient
cuestStatus_t cuestOverlapDerivativeCompute(
    cuestHandle_t handle,
    cuestOEIntPlan_t plan,
    cuestWorkspace_t* temporaryWorkspace,
    const double* densityMatrix,     // device, nao×nao
    double* outGradient);            // device, natom×3

// Kinetic energy gradient
cuestStatus_t cuestKineticDerivativeCompute(
    cuestHandle_t handle,
    cuestOEIntPlan_t plan,
    cuestWorkspace_t* temporaryWorkspace,
    const double* densityMatrix,     // device, nao×nao
    double* outGradient);            // device, natom×3

// Nuclear attraction gradient
cuestStatus_t cuestPotentialDerivativeCompute(
    cuestHandle_t handle,
    cuestOEIntPlan_t plan,
    cuestWorkspace_t* temporaryWorkspace,
    uint64_t numCharges,
    const double* xyz,               // charge positions (Bohr)
    const double* q,                 // charge values (nuclear charges)
    const double* densityMatrix,     // device, nao×nao
    double* outBasisGradient,        // device, natom×3 (gradient w.r.t. basis centers)
    double* outChargeGradient);      // device, numCharges×3 (gradient w.r.t. charges)
```

These need an `cuestOEIntPlan_t` built from:

```c
cuestStatus_t cuestOEIntPlanCreate(
    cuestHandle_t handle,
    const cuestAOBasis_t basis,
    const cuestAOPairList_t pairList,
    const cuestOEIntPlanParameters_t parameters,
    cuestWorkspace_t* persistentWorkspace,
    cuestWorkspace_t* temporaryWorkspace,
    cuestOEIntPlan_t* outPlan);
```

**Implementation**: Add a cuEST path in `SCFDeriv::compute_gradient()` (`scf_grad.cc`):

```cpp
#ifdef USING_cuEST
if (options_.get_str("SCF_TYPE") == "CUEST") {
    // Build OE plan (basis + pair list already available from JK grad)
    // Compute kinetic + potential gradients via cuEST
    // Compute overlap gradient via cuEST
    // Store in gradients_["Core"] and gradients_["Overlap"]
} else
#endif
{
    // Existing MintsHelper path
    gradients_["Core"] = mints->core_hamiltonian_grad(Dt);
    gradients_["Overlap"] = mints->overlap_grad(W);
    gradients_["Overlap"]->scale(-1.0);
}
```

**Important**: The overlap gradient uses the energy-weighted density matrix W, not Dt. And the potential gradient needs nuclear charges and positions as the "point charges".

### 4b: XC Gradient via cuEST (DONE)

Implemented as part of Phase 2. The `RV::compute_gradient()` method has a cuEST short-circuit that calls `cuestXCDerivativeRKSCompute`. This is automatically active when GPU XC is enabled (`PSI4_CUEST_GPU_XC=1`).

The XC gradient uses the same cuEST infrastructure (grid, XC plan) built in `VBase::cuest_xc_initialize()`.

---

## Build System Notes

The `fock` library already links cuEST conditionally:

```cmake
# psi4/src/psi4/libfock/CMakeLists.txt (lines 75-84)
if(TARGET cuEST::cuEST)
  target_compile_definitions(fock PUBLIC USING_cuEST)
  target_link_libraries(fock PUBLIC cuEST::cuEST)
endif()
```

The `scfgrad` library needs similar treatment (for Phase 3+4). Since `scfgrad` already links `scf_solver` which depends on `fock`, the cuEST transitive dependency may already be available, but the `USING_cuEST` compile definition needs to be explicitly added to `scfgrad`.

---

## Testing & Benchmarks

### Water tests (correctness)
- `tests/cuest-dft-h2o/input.dat` — DFT energy with GPU J/K, CPU XC (B3LYP, PBE). Matches DF to ~1e-11 Eh.
- `tests/cuest-grad-h2o/input.dat` — J/K gradient (RHF, B3LYP, PBE). Matches DF to ~1e-10 Eh/Bohr.
- `tests/cuest-gpu-xc-h2o/input.dat` — Full GPU XC energy + gradient (B3LYP, PBE). Energy matches to ~2-7e-8 Eh, gradient to ~2e-6 Eh/Bohr.

### Paxlovid benchmarks (performance)
- `tests/cuest-bench-paxlovid/bench_dft.py` — DFT energy (GPU J/K, CPU XC)
- `tests/cuest-bench-paxlovid/bench_gradient.py` — J/K gradient (RHF, B3LYP)
- `tests/cuest-bench-paxlovid/bench_full_gpu_dft.py` — Full GPU DFT (GPU J/K + GPU XC), energy + gradient

### Full GPU DFT Paxlovid Results (B3LYP/STO-3G, 67 atoms, 8 CPU threads)

| Configuration | Energy speedup | Gradient speedup | Energy delta | Grad delta |
|--------------|---------------|-----------------|-------------|-----------|
| GPU J/K + CPU XC | 1.81x | 1.93x | 1.19e-8 Eh | 4.58e-6 |
| GPU J/K + GPU XC | 1.87x | 1.73x | 1.41e-5 Eh | 1.92e-4 |

---

## Remaining Work

- **Phase 4a (OE gradients)**: Not started. One-electron gradients (overlap, kinetic, nuclear attraction) are still on CPU. Could be offloaded to cuEST for further speedup.
- **SVWN functional**: cuEST SVWN5 mapping produces ~196 mEh error. Needs investigation.
- **Larger basis sets**: All benchmarks use STO-3G. Test with cc-pVDZ, cc-pVTZ for more realistic workloads.
- **Make GPU XC default**: Currently opt-in via env var. Consider making it the default when SCF_TYPE=CUEST, or adding a Psi4 option.
- **UKS support**: Only RKS (restricted) is implemented. Add UKS (unrestricted) path for `UV::compute_V()` and `UV::compute_gradient()`.
- **Common utilities**: Extract shared cuEST helpers (basis building, workspace management) into `cuESTCommon.h` used by cuESTJK, cuESTJKGrad, and VBase.
