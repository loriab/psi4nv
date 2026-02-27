"""
Paxlovid Full GPU DFT Benchmark: cuEST GPU J/K + GPU XC
========================================================
Tests B3LYP/STO-3G on Paxlovid (67 atoms) with:
  - DF (CPU) as reference
  - CUEST (GPU J/K + GPU XC + GPU OE gradients)

Compares energies, gradients, and wall times, with detailed per-phase
timing extracted from the Psi4 output file.

Usage:
    conda run -n p4dev python bench_full_gpu_dft.py [--nthreads N]
"""

import sys
import os
import re
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Paxlovid full GPU DFT benchmark")
parser.add_argument("--nthreads", type=int, default=os.cpu_count(),
                    help="CPU threads for DF (default: all cores)")
args = parser.parse_args()

nthreads = args.nthreads

os.environ["PSI4_CUEST_GPU_XC"] = "1"

stage_lib = os.path.join(os.path.dirname(__file__), "..", "..", "objdir", "stage", "lib")
if os.path.isdir(stage_lib):
    sys.path.insert(0, os.path.abspath(stage_lib))

import psi4

psi4.set_memory("4 GB")
psi4.set_num_threads(nthreads)

OUTPUT_FILE = "bench_full_gpu_dft.out"
psi4.core.set_output_file(OUTPUT_FILE, False)


def parse_scf_timing(output_file, start_after_line=0):
    """Parse SCF timing data from a Psi4 output file.

    Returns dict with init breakdown, per-iteration breakdown, and phase timing.
    Only parses content after start_after_line to isolate each computation.
    """
    with open(output_file) as f:
        lines = f.readlines()

    lines = lines[start_after_line:]
    result = {"init": {}, "iterations": [], "phases": {}, "total_lines": start_after_line + len(lines)}

    # Parse SCF Init Timing
    in_init = False
    for line in lines:
        if "==> SCF Init Timing <==" in line:
            in_init = True
            continue
        if in_init:
            m = re.match(r'\s+(\S.*?):\s+([\d.]+)s', line)
            if m:
                result["init"][m.group(1).strip()] = float(m.group(2))
            elif line.strip() == "":
                if result["init"]:
                    in_init = False

    # Parse per-iteration breakdown lines
    iter_re = re.compile(
        r'iter breakdown: G=([\d.]+)ms F\+E=([\d.]+)ms DIIS=([\d.]+)ms '
        r'C=([\d.]+)ms D=([\d.]+)ms other=([\d.]+)ms'
    )
    # Parse form_G sub-breakdown: V(XC) vs JK
    formg_re = re.compile(
        r'form_G breakdown: V\(XC\)=([\d.]+)ms\s+JK=([\d.]+)ms\s+combine=([\d.]+)ms'
    )
    formg_entries = []
    for line in lines:
        mg = formg_re.search(line)
        if mg:
            formg_entries.append({
                "V_XC": float(mg.group(1)),
                "JK": float(mg.group(2)),
                "combine": float(mg.group(3)),
            })

    # formg_entries includes the SAD iteration (no corresponding iter breakdown),
    # so formg_entries[i+1] aligns with the i-th iter breakdown line.
    formg_offset = max(0, len(formg_entries) - sum(1 for l in lines if iter_re.search(l)))

    iter_idx = 0
    for line in lines:
        m = iter_re.search(line)
        if m:
            entry = {
                "G": float(m.group(1)),
                "F+E": float(m.group(2)),
                "DIIS": float(m.group(3)),
                "C": float(m.group(4)),
                "D": float(m.group(5)),
                "other": float(m.group(6)),
            }
            fg_idx = iter_idx + formg_offset
            if fg_idx < len(formg_entries):
                entry["V_XC"] = formg_entries[fg_idx]["V_XC"]
                entry["JK"] = formg_entries[fg_idx]["JK"]
            iter_idx += 1
            result["iterations"].append(entry)

    # Parse SCF Phase Timing
    in_phase = False
    for line in lines:
        if "==> SCF Phase Timing <==" in line:
            in_phase = True
            continue
        if in_phase:
            m = re.match(r'\s+(\S.*?):\s+([\d.]+)s', line)
            if m:
                result["phases"][m.group(1).strip()] = float(m.group(2))
            elif line.strip() == "":
                if result["phases"]:
                    in_phase = False

    return result


def summarize_iter_timing(timing_data):
    """Aggregate per-iteration timing into totals and averages."""
    iters = timing_data["iterations"]
    if not iters:
        return None

    phases = ["G", "F+E", "DIIS", "C", "D", "other"]
    totals = {p: sum(it[p] for it in iters) for p in phases}
    total_all = sum(totals.values())
    n = len(iters)

    has_vxc = "V_XC" in iters[0]
    if has_vxc:
        totals["V_XC"] = sum(it.get("V_XC", 0) for it in iters)
        totals["JK"] = sum(it.get("JK", 0) for it in iters)

    result = {
        "n_iters": n,
        "totals_ms": totals,
        "total_iter_ms": total_all,
        "avg_ms": {p: totals[p] / n for p in totals},
        "pct": {p: 100 * totals[p] / total_all if total_all > 0 else 0 for p in phases},
        "has_vxc": has_vxc,
    }
    if has_vxc:
        g_total = totals["G"]
        result["pct_of_G"] = {
            "V_XC": 100 * totals["V_XC"] / g_total if g_total > 0 else 0,
            "JK": 100 * totals["JK"] / g_total if g_total > 0 else 0,
        }
    return result


def print_timing_detail(label, timing_data):
    """Print a detailed timing breakdown for one configuration."""
    summary = summarize_iter_timing(timing_data)
    if not summary:
        print(f"  (no per-iteration timing data found)")
        return summary

    init = timing_data["init"]
    phases = timing_data["phases"]

    print(f"\n  Initialization:")
    for k, v in init.items():
        print(f"    {k:<20s} {v:8.3f}s")

    print(f"\n  SCF Iterations ({summary['n_iters']} iters, {summary['total_iter_ms']:.0f}ms total):")
    print(f"    {'Phase':<10s} {'Total (ms)':>12s} {'Avg (ms)':>10s} {'% of iter':>10s}")
    print(f"    {'-'*44}")
    for p in ["G", "F+E", "DIIS", "C", "D", "other"]:
        print(f"    {p:<10s} {summary['totals_ms'][p]:12.1f} {summary['avg_ms'][p]:10.1f} {summary['pct'][p]:9.1f}%")

    if summary.get("has_vxc"):
        print(f"\n    form_G breakdown (V(XC) vs J/K):")
        print(f"      V(XC):   {summary['totals_ms']['V_XC']:8.1f}ms avg={summary['avg_ms']['V_XC']:7.1f}ms  ({summary['pct_of_G']['V_XC']:5.1f}% of G)")
        print(f"      JK:      {summary['totals_ms']['JK']:8.1f}ms avg={summary['avg_ms']['JK']:7.1f}ms  ({summary['pct_of_G']['JK']:5.1f}% of G)")

    if phases:
        print(f"\n  SCF Phases:")
        for k, v in phases.items():
            print(f"    {k:<20s} {v:8.3f}s")

    return summary


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

configs = [
    ("DF (CPU)", "DF"),
    ("CUEST (Full GPU)", "CUEST"),
]

results = {}
output_line_cursor = 0

for label, scf_type in configs:
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    # --- Energy ---
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.set_options({**common_opts, "basis": "sto-3g", "scf_type": scf_type})

    t0 = time.perf_counter()
    e = psi4.energy('b3lyp')
    t_energy = time.perf_counter() - t0
    niter = int(psi4.variable("SCF ITERATIONS"))

    energy_timing = parse_scf_timing(OUTPUT_FILE, start_after_line=output_line_cursor)
    output_line_cursor = energy_timing["total_lines"]

    print(f"\n  Energy:  {e:20.12f} Eh  ({t_energy:.2f}s, {niter} iters)")
    energy_summary = print_timing_detail(label + " Energy", energy_timing)

    # --- Gradient ---
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.set_options({**common_opts, "basis": "sto-3g", "scf_type": scf_type})

    t0 = time.perf_counter()
    g = psi4.gradient('b3lyp')
    t_grad = time.perf_counter() - t0

    grad_timing = parse_scf_timing(OUTPUT_FILE, start_after_line=output_line_cursor)
    output_line_cursor = grad_timing["total_lines"]

    g_np = np.array(g)
    grad_max = np.max(np.abs(g_np))
    grad_rms = np.sqrt(np.mean(g_np ** 2))

    print(f"\n  Gradient: max={grad_max:.2e}, rms={grad_rms:.2e}  ({t_grad:.2f}s)")
    grad_summary = print_timing_detail(label + " Gradient", grad_timing)

    results[label] = {
        "energy": e, "t_energy": t_energy, "niter": niter,
        "grad": g_np, "t_grad": t_grad,
        "grad_max": grad_max, "grad_rms": grad_rms,
        "energy_timing": energy_timing,
        "grad_timing": grad_timing,
        "energy_summary": energy_summary,
        "grad_summary": grad_summary,
    }

# === Accuracy comparison ===
print(f"\n{'='*80}")
print("ACCURACY COMPARISON")
print(f"{'='*80}")

ref_label = "DF (CPU)"
e_ref = results[ref_label]["energy"]
g_ref = results[ref_label]["grad"]

cuest_label = "CUEST (Full GPU)"
e_delta = abs(results[cuest_label]["energy"] - e_ref)
g_delta = np.max(np.abs(results[cuest_label]["grad"] - g_ref))
print(f"\n  {cuest_label} vs DF:")
print(f"    Energy delta:       {e_delta:.2e} Eh")
print(f"    Gradient max delta: {g_delta:.2e} Eh/Bohr")

# === Overall speedup ===
print(f"\n{'='*80}")
print(f"OVERALL SPEEDUP  (DF baseline: {nthreads} CPU threads)")
print(f"{'='*80}")

t_ref_e = results[ref_label]["t_energy"]
t_ref_g = results[ref_label]["t_grad"]

for label, _ in configs:
    te = results[label]["t_energy"]
    tg = results[label]["t_grad"]
    se = t_ref_e / te if te > 0 else 0
    sg = t_ref_g / tg if tg > 0 else 0
    print(f"\n  {label}")
    print(f"    Energy:   {te:8.2f}s  (speedup: {se:.2f}x)")
    print(f"    Gradient: {tg:8.2f}s  (speedup: {sg:.2f}x)")

# === Detailed phase-by-phase comparison ===
print(f"\n{'='*80}")
print("PHASE-BY-PHASE COMPARISON  (Energy SCF iterations)")
print(f"{'='*80}")

ref_s = results[ref_label].get("energy_summary")
cu_s = results[cuest_label].get("energy_summary")

if ref_s and cu_s:
    print(f"\n  {'Phase':<10s} {'DF total(ms)':>14s} {'DF avg(ms)':>12s}"
          f" {'cuEST total(ms)':>16s} {'cuEST avg(ms)':>14s} {'Speedup':>8s}")
    print(f"  {'-'*76}")
    for p in ["G", "F+E", "DIIS", "C", "D", "other"]:
        df_tot = ref_s["totals_ms"][p]
        df_avg = ref_s["avg_ms"][p]
        cu_tot = cu_s["totals_ms"][p]
        cu_avg = cu_s["avg_ms"][p]
        sp = df_tot / cu_tot if cu_tot > 0 else float("inf")
        print(f"  {p:<10s} {df_tot:14.1f} {df_avg:12.1f}"
              f" {cu_tot:16.1f} {cu_avg:14.1f} {sp:7.2f}x")

    df_total = ref_s["total_iter_ms"]
    cu_total = cu_s["total_iter_ms"]
    sp_total = df_total / cu_total if cu_total > 0 else float("inf")
    print(f"  {'TOTAL':<10s} {df_total:14.1f} {df_total/ref_s['n_iters']:12.1f}"
          f" {cu_total:16.1f} {cu_total/cu_s['n_iters']:14.1f} {sp_total:7.2f}x")

    # V(XC) vs JK sub-breakdown of G
    if ref_s.get("has_vxc") and cu_s.get("has_vxc"):
        print(f"\n  form_G sub-breakdown (V(XC) vs J/K):")
        print(f"  {'Component':<10s} {'DF total(ms)':>14s} {'DF avg(ms)':>12s}"
              f" {'cuEST total(ms)':>16s} {'cuEST avg(ms)':>14s} {'Speedup':>8s}")
        print(f"  {'-'*76}")
        for p in ["V_XC", "JK"]:
            df_tot = ref_s["totals_ms"][p]
            df_avg = ref_s["avg_ms"][p]
            cu_tot = cu_s["totals_ms"][p]
            cu_avg = cu_s["avg_ms"][p]
            sp = df_tot / cu_tot if cu_tot > 0 else float("inf")
            lbl = "V(XC)" if p == "V_XC" else "J/K"
            print(f"  {lbl:<10s} {df_tot:14.1f} {df_avg:12.1f}"
                  f" {cu_tot:16.1f} {cu_avg:14.1f} {sp:7.2f}x")

    # Init time comparison
    df_init = results[ref_label]["energy_timing"]["init"]
    cu_init = results[cuest_label]["energy_timing"]["init"]
    if df_init and cu_init:
        print(f"\n  Initialization:")
        for k in df_init:
            df_v = df_init.get(k, 0)
            cu_v = cu_init.get(k, 0)
            sp = df_v / cu_v if cu_v > 0 else float("inf")
            print(f"    {k:<20s}  DF: {df_v:7.3f}s   cuEST: {cu_v:7.3f}s   ({sp:.2f}x)")

    # Where the time goes
    print(f"\n  Time budget (Energy, wall clock):")
    for lbl in [ref_label, cuest_label]:
        r = results[lbl]
        init_total = r["energy_timing"]["init"].get("Total init", 0)
        iter_total = r["energy_summary"]["total_iter_ms"] / 1000 if r["energy_summary"] else 0
        finalize = r["energy_timing"]["phases"].get("Finalize", 0)
        accounted = init_total + iter_total + finalize
        unaccounted = r["t_energy"] - accounted
        print(f"    {lbl}:")
        print(f"      Init:        {init_total:7.2f}s  ({100*init_total/r['t_energy']:5.1f}%)")
        print(f"      Iterations:  {iter_total:7.2f}s  ({100*iter_total/r['t_energy']:5.1f}%)")
        s = r['energy_summary']
        print(f"        form_G:    {s['totals_ms']['G']/1000:7.2f}s  ({s['pct']['G']:5.1f}% of iter)")
        if s.get("has_vxc"):
            print(f"          V(XC):   {s['totals_ms']['V_XC']/1000:7.2f}s  ({s['pct_of_G']['V_XC']:5.1f}% of G)")
            print(f"          J/K:     {s['totals_ms']['JK']/1000:7.2f}s  ({s['pct_of_G']['JK']:5.1f}% of G)")
        print(f"        DIIS:      {s['totals_ms']['DIIS']/1000:7.2f}s  ({s['pct']['DIIS']:5.1f}% of iter)")
        print(f"        other:     {(iter_total - s['totals_ms']['G']/1000 - s['totals_ms']['DIIS']/1000):7.2f}s")
        print(f"      Finalize:    {finalize:7.2f}s  ({100*finalize/r['t_energy']:5.1f}%)")
        if unaccounted > 0.01:
            print(f"      Unaccounted: {unaccounted:7.2f}s  ({100*unaccounted/r['t_energy']:5.1f}%)")

print(f"\n{'='*80}")
print("NOTES")
print(f"{'='*80}")
print("  - 'G' = form_G() = J/K build + XC integration (the GPU-accelerated part)")
print("  - 'F+E' = Fock matrix assembly + energy computation")
print("  - 'DIIS' = convergence acceleration")
print("  - 'C' = orbital diagonalization")
print("  - 'D' = density matrix formation")
print("  - form_G dominates iteration time; further V(XC) vs J/K split")
print("    is visible in the Psi4 output file timers")
print()
print("Done.")
