"""
Paxlovid Full GPU DFT Benchmark: cuEST GPU J/K + GPU XC
========================================================
Tests B3LYP/STO-3G on Paxlovid (67 atoms) with:
  - DF (CPU) as reference
  - CUEST (GPU J/K + GPU XC + GPU OE gradients)

Compares energies, gradients, and wall times.

Usage:
    conda run -n p4dev python bench_full_gpu_dft.py [--nthreads N]
"""

import sys
import os
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Paxlovid full GPU DFT benchmark")
parser.add_argument("--nthreads", type=int, default=os.cpu_count(),
                    help="CPU threads for DF (default: all cores)")
args = parser.parse_args()

nthreads = args.nthreads

stage_lib = os.path.join(os.path.dirname(__file__), "..", "..", "objdir", "stage", "lib")
if os.path.isdir(stage_lib):
    sys.path.insert(0, os.path.abspath(stage_lib))

import psi4

psi4.set_memory("4 GB")
psi4.set_num_threads(nthreads)
psi4.core.set_output_file("bench_full_gpu_dft.out", False)

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
    "df_scf_guess": False,
    "e_convergence": 1e-8,
    "d_convergence": 1e-6,
    "dft_spherical_points": 302,
    "dft_radial_points": 75,
}

print("=" * 80)
print("Paxlovid Full GPU DFT Benchmark")
print(f"  Molecule: C23 H32 F3 N5 O4  (67 atoms)")
print(f"  Basis: STO-3G")
print(f"  CPU threads: {nthreads}")
print("=" * 80)

# Run 2 configurations for B3LYP
configs = [
    ("DF (CPU)", "DF"),
    ("CUEST (Full GPU)", "CUEST"),
]

results = {}

for label, scf_type in configs:
    print(f"\n--- {label} ---")

    # Energy
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.set_options({**common_opts, "basis": "sto-3g", "scf_type": scf_type})

    t0 = time.perf_counter()
    e = psi4.energy('b3lyp')
    t_energy = time.perf_counter() - t0
    niter = int(psi4.variable("SCF ITERATIONS"))

    # Gradient
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.set_options({**common_opts, "basis": "sto-3g", "scf_type": scf_type})

    t0 = time.perf_counter()
    g = psi4.gradient('b3lyp')
    t_grad = time.perf_counter() - t0

    g_np = np.array(g)
    grad_max = np.max(np.abs(g_np))
    grad_rms = np.sqrt(np.mean(g_np ** 2))

    results[label] = {
        "energy": e, "t_energy": t_energy, "niter": niter,
        "grad": g_np, "t_grad": t_grad,
        "grad_max": grad_max, "grad_rms": grad_rms,
    }

    print(f"  Energy:  {e:20.12f} Eh  ({t_energy:.2f}s, {niter} iters)")
    print(f"  Gradient: max={grad_max:.2e}, rms={grad_rms:.2e}  ({t_grad:.2f}s)")

# Comparison
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

ref_label = "DF (CPU)"
e_ref = results[ref_label]["energy"]
g_ref = results[ref_label]["grad"]

cuest_label = "CUEST (Full GPU)"
e_delta = abs(results[cuest_label]["energy"] - e_ref)
g_delta = np.max(np.abs(results[cuest_label]["grad"] - g_ref))
print(f"\n{cuest_label} vs DF:")
print(f"  Energy delta: {e_delta:.2e} Eh")
print(f"  Gradient max delta: {g_delta:.2e} Eh/Bohr")

# Timing summary
print("\n" + "=" * 80)
print(f"SPEEDUP SUMMARY  (DF baseline using {nthreads} threads)")
print("=" * 80)

t_ref_e = results[ref_label]["t_energy"]
t_ref_g = results[ref_label]["t_grad"]

for label, _ in configs:
    te = results[label]["t_energy"]
    tg = results[label]["t_grad"]
    se = t_ref_e / te if te > 0 else 0
    sg = t_ref_g / tg if tg > 0 else 0
    print(f"  {label}")
    print(f"    Energy:   {te:8.2f}s  (speedup: {se:.2f}x)")
    print(f"    Gradient: {tg:8.2f}s  (speedup: {sg:.2f}x)")

print("\nDone.")
