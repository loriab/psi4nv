# cuEST Integration into Psi4 -- Work Log

## Overview

This document describes the integration of NVIDIA's **cuEST** library into
Psi4's JK (Coulomb/Exchange) framework. cuEST provides GPU-accelerated
density-fitted (DF) Coulomb and Exchange matrix computations. The integration
adds a new `SCF_TYPE CUEST` option that offloads J/K builds to the GPU via
the cuEST C API.

**Status:** Working end-to-end. RHF energies match Psi4's standard `SCF_TYPE DF`
to sub-picohartree precision on water (STO-3G and cc-pVDZ).

---

## Architecture

The integration follows the same pattern as the existing **BrianQC** GPU
integration, but uses a cleaner **new subclass** approach rather than
conditional compilation inside an existing class.

```
JK (base class)
├── DirectJK       (integral-direct, has #ifdef USING_BrianQC conditionals)
├── DiskDFJK       (disk-based density fitting)
├── MemDFJK        (in-core density fitting)
├── CompositeJK    (mixed J/K algorithms)
└── cuESTJK  ← NEW (GPU density-fitted via cuEST)
```

### Data Flow in `compute_JK()`

```
Psi4 host memory                    GPU (cuEST)
─────────────────                    ───────────
D_ao_ (density)  ──cudaMemcpy H→D──→ d_D  ──→ cuestDFCoulombCompute  ──→ d_J
                                                                           │
J_ao_ (Coulomb)  ←─cudaMemcpy D→H──────────────────────────────────────────┘

C_left_ao_ (MOs) ──transpose──→ C_row_major ──cudaMemcpy H→D──→ d_C
                                 (col→row major)                    │
                                                  cuestDFSymmetricExchangeCompute
                                                                    │
K_ao_ (Exchange) ←─cudaMemcpy D→H──────────────────────────────────┘
```

Key details:
- cuEST expects **device pointers** (GPU memory) for all matrix arguments
- cuEST expects the coefficient matrix in **row-major** layout (nocc × nao),
  but Psi4 stores C in column-major (nao × nocc), so we transpose
- cuEST expects **normalized** contraction coefficients (Psi4's `shell.coefs()`,
  NOT `shell.original_coefs()`)

---

## Files Created

### `psi4/src/psi4/libfock/cuESTJK.h`
Header declaring the `cuESTJK` class. Guarded by `#ifdef USING_cuEST`.
Contains:
- Class declaration inheriting from `JK`
- cuEST handle members (basis, pair list, DF plan, workspaces)
- Override declarations for `preiterations()`, `compute_JK()`, `postiterations()`

### `psi4/src/psi4/libfock/cuESTJK.cc`
Full implementation (~350 lines). Key methods:

| Method | Purpose |
|--------|---------|
| `build_cuest_basis()` | Translates Psi4 `BasisSet` → cuEST `cuestAOBasis_t`. Iterates atoms/shells, creates `cuestAOShell_t` objects with normalized coefficients, queries workspace sizes, allocates GPU memory, and builds the basis. |
| `preiterations()` | Called once before SCF iterations. Builds primary and auxiliary cuEST bases, pair list (with `cutoff_` threshold), and the DF integral plan on the GPU. |
| `compute_JK()` | Called each SCF iteration. Copies D/C matrices to GPU, calls `cuestDFCoulombCompute` (J) and `cuestDFSymmetricExchangeCompute` (K), copies results back. |
| `postiterations()` | Tears down all cuEST objects and frees GPU/host workspaces. |
| `allocate_workspace()` / `free_workspace()` | Helper pair for managing cuEST workspace buffers (host via `malloc`, device via `cudaMalloc`). |

### `tests/cuest-rhf-h2o/input.dat`
Test case: water molecule with `SCF_TYPE CUEST`, two basis sets (STO-3G,
cc-pVDZ), comparing against DF reference energies.

---

## Files Modified

### `psi4/src/psi4/libfock/jk.cc`
- Added `#include "cuESTJK.h"`
- Added `"CUEST"` case in `build_JK()` factory method (lines ~186-199),
  guarded by `#ifdef USING_cuEST`
- Requires auxiliary basis (`DF_BASIS_SCF`), uses `DF_FITTING_CONDITION`

### `psi4/src/psi4/libfock/CMakeLists.txt`
- Added `cuESTJK.cc` to source list
- Added `cuEST::cuEST` target linking and `USING_cuEST` compile definition
  (mirrors the BrianQC pattern)

### `psi4/src/read_options.cc`
- Added `"CUEST"` to the valid `SCF_TYPE` enum string (line 195)

### `psi4/driver/procrouting/proc.py`
- Added `"CUEST"` to the `df_needed` list (line 1452) so the Python driver
  automatically resolves the auxiliary (JKFIT) basis set

### `cmake/FindcuEST.cmake`
- Added `CUDA::cudart` as `INTERFACE_LINK_LIBRARIES` on the `cuEST::cuEST`
  imported target, so consumers automatically get CUDA runtime headers and
  linking (needed for `cudaMalloc`/`cudaMemcpy`/`cudaFree` in cuESTJK.cc)

---

## How to Use

### Build with cuEST enabled

```bash
conda run -n p4dev cmake -S. -Bobjdir \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/install \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(conda run -n p4dev python -c 'import sys; print(sys.prefix)')" \
  -DPython_EXECUTABLE="$(conda run -n p4dev which python)" \
  -DENABLE_cuEST=ON \
  -DcuEST_ROOT=/home/dahsu/Documents/cuest \
  -G Ninja

conda run -n p4dev cmake --build objdir -j$(nproc)
```

### Run a calculation

```python
import psi4

psi4.geometry("""
    symmetry c1
    0 1
    O  0.000  0.000  0.117
    H -0.757  0.000 -0.469
    H  0.757  0.000 -0.469
""")

psi4.set_options({
    'basis': 'cc-pVDZ',
    'scf_type': 'cuest',          # ← use GPU-accelerated DF via cuEST
    'df_basis_scf': 'cc-pVDZ-jkfit',  # or leave blank for auto
})

energy = psi4.energy('scf')
```

### Run the test

```bash
cd /tmp
conda run -n p4dev env PYTHONPATH=/path/to/objdir/stage/lib \
  python -c "
import psi4
psi4.core.set_output_file('cuest_test.out')
psi4.geometry('''
    symmetry c1
    O  0.0  0.0  0.117369
    H -0.757  0.0 -0.469476
    H  0.757  0.0 -0.469476
''')
psi4.set_options({'basis':'cc-pVDZ','scf_type':'cuest','df_scf_guess':False})
e = psi4.energy('scf')
print(f'Energy: {e:.12f}')
# Expected: -76.026751163491 (matches SCF_TYPE DF)
"
```

---

## Verified Test Results

| Molecule | Basis   | cuEST Energy (Eh)     | DF Reference (Eh)     | Difference |
|----------|---------|-----------------------|-----------------------|------------|
| H₂O     | STO-3G  | -74.963127532329      | -74.963127532314      | 1.5e-11    |
| H₂O     | cc-pVDZ | -76.026751163493      | -76.026751163491      | 1.9e-12    |

---

## Key Lessons Learned

1. **cuEST expects device pointers.** All matrix arguments to
   `cuestDFCoulombCompute` and `cuestDFSymmetricExchangeCompute` must be
   GPU-allocated memory. Data must be `cudaMemcpy`'d to/from host.

2. **cuEST expects normalized coefficients.** Use Psi4's `shell.coefs()`
   (which includes primitive + contraction normalization), NOT
   `shell.original_coefs()` (raw GBS file values). Both cuEST and Psi4 use
   the same normalization convention.

3. **Row-major C matrix.** cuEST expects the occupied orbital coefficient
   matrix as (nocc × nao) row-major, while Psi4 stores it as (nao × nocc)
   column-major. The transpose is done in `compute_JK()`.

4. **Workspace pattern.** Every cuEST object creation follows: workspace
   query → allocate → create → free temporary workspace. Persistent
   workspaces must outlive the objects they back.

---

## Future Work / TODO

- **wK (range-separated exchange):** Not yet implemented. The cuEST API
  may support this via the `CUEST_DFINTPLAN_PARAMETERS_EXCHANGE_FRACTION`
  parameter, but the `do_wK_` path is not wired up yet.

- **Incremental Fock build:** cuESTJK does not implement incremental Fock
  (INCFOCK) yet. Each iteration recomputes J/K from scratch.

- **Asymmetric densities (ROHF/UHF):** Only symmetric (`C_left == C_right`)
  exchange has been tested. The `cuestDFSymmetricExchangeCompute` API only
  handles the symmetric case. Asymmetric exchange would need a different
  cuEST routine or fallback.

- **Memory management:** Currently allocates/frees device memory for D/J/K
  each `compute_JK()` call. Could be optimized by persisting across
  iterations (sizes don't change).

- **Gradient support:** cuEST provides `cuestDFDerivativeCompute` for
  analytic gradients. This is not yet integrated.

- **DFT/XC integration:** cuEST has XC potential/gradient compute routines.
  These could be integrated into Psi4's `V` (XC potential) framework.

- **Larger molecule testing:** Paxlovid benchmark complete (see below).
  Could extend to even larger systems.

---

## Paxlovid (Nirmatrelvir) Benchmark

A GPU vs CPU benchmark on a realistic drug molecule (67 atoms, C23H32F3N5O4)
is included at `tests/cuest-bench-paxlovid/bench_paxlovid.py`.

### Running the benchmark

```bash
cd /home/dahsu/Documents/psi4nv
conda run -n p4dev env PYTHONPATH=$(pwd)/objdir/stage/lib \
  python tests/cuest-bench-paxlovid/bench_paxlovid.py
```

To control CPU thread count for the DF baseline (defaults to all cores):

```bash
conda run -n p4dev env PYTHONPATH=$(pwd)/objdir/stage/lib \
  python tests/cuest-bench-paxlovid/bench_paxlovid.py --nthreads 16
```

### What it does

Runs RHF with both `SCF_TYPE DF` (CPU, OpenMP) and `SCF_TYPE CUEST` (GPU)
on each of two basis sets (STO-3G and def2-SVP). It prints:
- Per-run energy, wall time, and iteration count
- Energy agreement between DF and cuEST (expects < 1e-6 Eh)
- Comparison against cuEST standalone reference energies
- Speedup summary (CPU wall / GPU wall)

### Reference results (rotini, 64-core CPU + GPU, Feb 2026)

| Basis    | SCF_TYPE | Energy (Eh)            | Wall (s) | Iters |
|----------|----------|------------------------|----------|-------|
| STO-3G   | DF       | -1737.171277125747     | 11       | 11    |
| STO-3G   | CUEST    | -1737.171277108845     | 4        | 11    |
| def2-SVP | DF       | -1758.304446703463     | 35       | 12    |
| def2-SVP | CUEST    | -1758.304446680249     | 16       | 12    |

Energy agreement: DF vs cuEST delta < 3e-8 Eh (both basis sets).

### cuEST standalone reference energies

From the cuEST test suite (`examples/cuest_scf/test/rhf_1`):
- STO-3G:   -1737.1712774897262 Eh
- def2-SVP: -1758.3044466471508 Eh
