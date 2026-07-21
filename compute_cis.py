"""Compute bootstrap CIs for all experimental tables."""
import numpy as np
import copy
from config import *
from model import *
from data_loader import generate_suppliers, generate_orders, generate_distances
from racf_stages import run_racf
from benchmarks import BENCHMARKS, run_rep_nostable, run_rep_notopsis
from experiments_multicycle import run_multicycle

N_RUNS = 20
N_CYCLES = 10
N_BOOTSTRAP = 1000

def bootstrap_ci(data, n_bootstrap=N_BOOTSTRAP, alpha=0.01):
    data = np.array(data)
    rng = np.random.default_rng(BASE_SEED)
    means = np.array([np.mean(rng.choice(data, size=len(data), replace=True))
                      for _ in range(n_bootstrap)])
    m = np.mean(data)
    lo = np.percentile(means, 0.5)
    hi = np.percentile(means, 99.5)
    return m, (lo, hi), (m - lo + hi - m) / 2  # mean, (lo,hi), half-width

print("Computing bootstrap CIs for all scales and methods...")
all_results = {}

for scale_name in SCALES:
    N, M = SCALES[scale_name]['N'], SCALES[scale_name]['M']
    print(f"\n  Scale: {scale_name} (N={N}, M={M})")

    rng = np.random.default_rng(BASE_SEED)
    suppliers = generate_suppliers(M, rng)
    orders = generate_orders(N, rng)
    distances = generate_distances(M, N, rng)
    max_w = sum(o.p * o.Q for o in orders)

    methods = list(BENCHMARKS.keys())
    per_run = {m: {'reliability': [], 'utility': []} for m in methods}

    for run in range(N_RUNS):
        run_rng = np.random.default_rng(BASE_SEED + run)
        for method in methods:
            metrics, _ = run_multicycle(suppliers, orders, distances, method,
                                        n_cycles=N_CYCLES, rng=run_rng)
            per_run[method]['reliability'].append(metrics['reliability'])
            per_run[method]['utility'].append(metrics['utility'] / max_w)

    scale_result = {}
    for m in methods:
        rel_mean, rel_ci, rel_hw = bootstrap_ci(per_run[m]['reliability'])
        util_mean, util_ci, util_hw = bootstrap_ci(per_run[m]['utility'])
        scale_result[m] = {
            'reliability_mean': round(rel_mean, 4),
            'reliability_ci_hw': round(rel_hw, 4),
            'utility_mean': round(util_mean, 4),
            'utility_ci_hw': round(util_hw, 4),
        }
        print(f"    {m:15s} rel={rel_mean:.4f}±{rel_hw:.4f}  util={util_mean:.4f}±{util_hw:.4f}")
    all_results[scale_name] = scale_result

# Print formatted for LaTeX
print("\n\n=== FOR LaTeX: RELIABILITY TABLE ===")
for scale_name in SCALES:
    row = f"{scale_name:<10}"
    for m in BENCHMARKS:
        r = all_results[scale_name][m]
        row += f" & {r['reliability_mean']:.2f}\\,({r['reliability_ci_hw']:.3f})"
    row += " \\\\"
    print(row)

print("\n=== FOR LaTeX: UTILITY TABLE ===")
for scale_name in SCALES:
    row = f"{scale_name:<10}"
    for m in BENCHMARKS:
        r = all_results[scale_name][m]
        row += f" & {r['utility_mean']:.2f}\\,({r['utility_ci_hw']:.3f})"
    row += " \\\\"
    print(row)

print("\nDone.")
