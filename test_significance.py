"""Significance test: 20 runs at Large scale, Welch t-test for all pairs."""
import numpy as np
from scipy import stats
import copy

from config import *
from model import *
from data_loader import generate_suppliers, generate_orders, generate_distances
from racf_stages import run_racf
from benchmarks import BENCHMARKS, run_rep_nostable, run_rep_notopsis
from experiments_multicycle import run_multicycle

N, M = 50, 150  # Large scale
N_CYCLES = 10
N_RUNS = 20

rng = np.random.default_rng(BASE_SEED)
suppliers = generate_suppliers(M, rng)
orders = generate_orders(N, rng)
distances = generate_distances(M, N, rng)

methods = ["REP-OBLIV", "REP-STATIC", "REP-NOTOPSIS", "REP-NOSTABLE", "RACF"]
per_run = {m: {'reliability': [], 'utility': []} for m in methods}

print(f"Running {N_RUNS} independent runs at Large scale ({N_CYCLES} cycles each)...")
for run in range(N_RUNS):
    run_rng = np.random.default_rng(BASE_SEED + run)
    for method in methods:
        metrics, _ = run_multicycle(suppliers, orders, distances, method,
                                    n_cycles=N_CYCLES, rng=run_rng)
        per_run[method]['reliability'].append(metrics['reliability'])
        max_w = sum(o.p * o.Q for o in orders)
        per_run[method]['utility'].append(metrics['utility'] / max_w)
    if (run + 1) % 5 == 0:
        print(f"  {run+1}/{N_RUNS} runs completed")

print("\n=== MEANS ===")
for m in methods:
    rels = per_run[m]['reliability']
    utils = per_run[m]['utility']
    print(f"{m:15s}  Reliability: {np.mean(rels):.4f} ± {np.std(rels, ddof=1):.4f}  "
          f"Utility: {np.mean(utils):.4f} ± {np.std(utils, ddof=1):.4f}")

print("\n=== WELCH T-TEST (Reliability) ===")
print(f"{'':15s}", end="")
for m in methods:
    print(f"{m:>12s}", end="")
print()

for m1 in methods:
    print(f"{m1:15s}", end="")
    for m2 in methods:
        if m1 == m2:
            print(f"{'--':>12s}", end="")
        else:
            t, p = stats.ttest_ind(per_run[m1]['reliability'],
                                   per_run[m2]['reliability'], equal_var=False)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            print(f"{p:.4f} {sig:>4s}", end="")
    print()

print("\n=== WELCH T-TEST (Utility) ===")
print(f"{'':15s}", end="")
for m in methods:
    print(f"{m:>12s}", end="")
print()

for m1 in methods:
    print(f"{m1:15s}", end="")
    for m2 in methods:
        if m1 == m2:
            print(f"{'--':>12s}", end="")
        else:
            t, p = stats.ttest_ind(per_run[m1]['utility'],
                                   per_run[m2]['utility'], equal_var=False)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            print(f"{p:.4f} {sig:>4s}", end="")
    print()

print("\nDone.")
