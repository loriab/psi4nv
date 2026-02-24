"""
Paxlovid Large-Basis Benchmark: cuEST Full GPU DFT
===================================================
Tests B3LYP on Paxlovid (67 atoms) across multiple basis sets
to characterize GPU performance scaling.

Configurations:
  - DF (CPU)  — density-fitted SCF on CPU
  - CUEST (Full GPU) — GPU J/K + GPU XC + GPU OE gradients

Basis sets: STO-3G, cc-pVDZ, cc-pVTZ (selectable via --basis)

Designed for large-memory GPUs (H200, A100).

Usage:
    conda run -n p4dev python bench_large_basis.py [--basis sto-3g cc-pvdz cc-pvtz] [--nthreads N] [--memory 16GB]
"""

import sys
import os
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Paxlovid large-basis cuEST benchmark")
parser.add_argument("--basis", nargs="+",
                    default=["sto-3g", "cc-pvdz", "cc-pvtz"],
                    help="Basis sets to benchmark (default: sto-3g cc-pvdz cc-pvtz)")
parser.add_argument("--nthreads", type=int, default=os.cpu_count(),
                    help="CPU threads for DF (default: all cores)")
parser.add_argument("--memory", type=str, default="16 GB",
                    help="Psi4 memory allocation (default: '16 GB')")
parser.add_argument("--energy-only", action="store_true",
                    help="Skip gradient computation (useful for very large bases)")
args = parser.parse_args()

stage_lib = os.path.join(os.path.dirname(__file__), "..", "..", "objdir", "stage", "lib")
if os.path.isdir(stage_lib):
    sys.path.insert(0, os.path.abspath(stage_lib))

import psi4

psi4.set_memory(args.memory)
psi4.set_num_threads(args.nthreads)
psi4.core.set_output_file("bench_large_basis.out", False)

mol = psi4.geometry("""
    symmetry c1
    0 1
    F   -5.2651    0.3853    2.1308
    F   -5.8314    2.2135    1.0849
    F   -4.0158    2.1639    2.2891
    O    1.8747   -1.8859   -1.5122
    O    0.0035    0.6408    0.8113
    O   -4.4964    0.3236   -0.6689
    O    3.7449    2.9461    1.9493
    N   -0.6890   -1.0865   -0.5681
    N    2.3460   -1.5920    0.7482
    N   -2.6719    1.4254    0.2686
    N    4.2828    3.4118   -0.2800
    N    5.1671   -3.1201    1.8606
    C   -0.4011   -3.4218   -0.4812
    C   -1.8334   -3.9081   -0.3288
    C   -1.4027   -3.1035   -1.5436
    C    0.1127   -2.1288    0.0777
    C   -1.5341   -1.6173   -1.6420
    C   -2.0815   -5.3814   -0.5422
    C   -2.7379   -3.3010    0.7122
    C    1.5277   -1.8669   -0.3343
    C   -0.7181    0.2182   -0.0939
    C   -1.7133    1.1335   -0.7618
    C   -1.0657    2.3878   -1.3950
    C    3.7515   -1.2850    0.6147
    C   -0.2991    3.2418   -0.3672
    C   -2.1514    3.2635   -2.0527
    C   -0.0789    1.9478   -2.4966
    C    4.0750    0.1111    1.1604
    C    3.6419    1.2203    0.2219
    C   -3.9901    0.9930    0.2272
    C    4.5427   -2.3102    1.3122
    C    4.4285    1.2436   -1.0918
    C    3.8964    2.6052    0.7865
    C   -4.7723    1.4370    1.4280
    C    4.3538    2.6954   -1.5233
    H    0.3660   -4.1612   -0.6712
    H   -1.3276   -3.5945   -2.5039
    H   -0.0235   -2.0721    1.1625
    H   -1.1580   -1.2393   -2.5976
    H   -2.5793   -1.3325   -1.5085
    H   -3.1132   -5.5503   -0.8676
    H   -1.4170   -5.8048   -1.3034
    H   -1.9179   -5.9336    0.3891
    H   -2.5647   -2.2363    0.8852
    H   -3.7849   -3.4159    0.4116
    H   -2.6021   -3.8126    1.6710
    H   -2.2464    0.6413   -1.5739
    H    1.9583   -1.6166    1.6871
    H    4.0266   -1.3394   -0.4442
    H   -2.3548    1.9285    1.0925
    H   -0.9155    3.4957    0.5006
    H    0.0161    4.1902   -0.8199
    H    0.6174    2.7609   -0.0157
    H   -1.7086    4.1207   -2.5729
    H   -2.8522    3.6591   -1.3094
    H   -2.7289    2.6933   -2.7891
    H   -0.5803    1.3412   -3.2591
    H    0.3600    2.8163   -3.0013
    H    0.7496    1.3577   -2.0897
    H    5.1547    0.2036    1.3452
    H    3.5983    0.2466    2.1411
    H    2.5684    1.1566    0.0100
    H    4.0173    0.5709   -1.8503
    H    5.4761    0.9641   -0.9226
    H    5.2292    3.0172   -2.0927
    H    3.4423    2.9087   -2.0903
    H    4.4735    4.4045   -0.1883
""")

common_opts = {
    "reference": "rhf",
    "df_scf_guess": False,
    "e_convergence": 1e-8,
    "d_convergence": 1e-6,
    "dft_spherical_points": 302,
    "dft_radial_points": 75,
}

scf_types = [
    ("DF (CPU)", "DF"),
    ("CUEST (Full GPU)", "CUEST"),
]

print("=" * 90)
print("Paxlovid Large-Basis Benchmark: cuEST Full GPU DFT")
print(f"  Molecule: Paxlovid (C23 H32 F3 N5 O4, 67 atoms)")
print(f"  Functional: B3LYP")
print(f"  Basis sets: {', '.join(args.basis)}")
print(f"  CPU threads: {args.nthreads}")
print(f"  Memory: {args.memory}")
print(f"  Gradient: {'no' if args.energy_only else 'yes'}")
print("=" * 90)

all_results = {}

for basis_name in args.basis:
    print(f"\n{'=' * 90}")
    print(f"  BASIS: {basis_name.upper()}")
    print(f"{'=' * 90}")

    basis_results = {}

    for label, scf_type in scf_types:
        print(f"\n  --- {label} ---")

        # Energy
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.clean_variables()
        psi4.set_options({**common_opts, "basis": basis_name, "scf_type": scf_type})

        t0 = time.perf_counter()
        e = psi4.energy('b3lyp')
        t_energy = time.perf_counter() - t0
        niter = int(psi4.variable("SCF ITERATIONS"))
        nbf = psi4.core.get_variable("NAUX") if scf_type == "DF" else 0

        print(f"    Energy:  {e:20.12f} Eh  ({t_energy:.2f}s, {niter} iters)")

        result = {
            "energy": e, "t_energy": t_energy, "niter": niter,
        }

        # Gradient
        if not args.energy_only:
            psi4.core.clean()
            psi4.core.clean_options()
            psi4.core.clean_variables()
            psi4.set_options({**common_opts, "basis": basis_name, "scf_type": scf_type})

            t0 = time.perf_counter()
            g = psi4.gradient('b3lyp')
            t_grad = time.perf_counter() - t0

            g_np = np.array(g)
            grad_max = np.max(np.abs(g_np))
            grad_rms = np.sqrt(np.mean(g_np ** 2))

            result["grad"] = g_np
            result["t_grad"] = t_grad
            result["grad_max"] = grad_max
            result["grad_rms"] = grad_rms

            print(f"    Gradient: max={grad_max:.2e}, rms={grad_rms:.2e}  ({t_grad:.2f}s)")

        basis_results[label] = result

    all_results[basis_name] = basis_results

# Summary table
print("\n" + "=" * 90)
print("SUMMARY TABLE")
print("=" * 90)

header = f"{'Basis':<12} {'Config':<22} {'Energy (Eh)':>20} {'E time':>8} {'E delta':>10}"
if not args.energy_only:
    header += f" {'G time':>8} {'G delta':>10}"
header += f" {'E speedup':>10}"
if not args.energy_only:
    header += f" {'G speedup':>10}"
print(header)
print("-" * len(header))

for basis_name in args.basis:
    br = all_results[basis_name]
    ref = br["DF (CPU)"]

    for label, _ in scf_types:
        r = br[label]
        e_delta = abs(r["energy"] - ref["energy"])
        e_speedup = ref["t_energy"] / r["t_energy"] if r["t_energy"] > 0 else 0

        row = f"{basis_name.upper():<12} {label:<22} {r['energy']:>20.12f} {r['t_energy']:>7.1f}s {e_delta:>10.2e}"
        if not args.energy_only:
            g_delta = np.max(np.abs(r.get("grad", np.zeros(1)) - ref.get("grad", np.zeros(1)))) if "grad" in r else 0
            g_speedup = ref.get("t_grad", 0) / r.get("t_grad", 1)
            row += f" {r.get('t_grad', 0):>7.1f}s {g_delta:>10.2e}"
        row += f" {e_speedup:>9.2f}x"
        if not args.energy_only:
            row += f" {g_speedup:>9.2f}x"
        print(row)

print("\nDone.")
