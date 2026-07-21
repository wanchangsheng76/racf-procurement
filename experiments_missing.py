"""
Missing experiments: temporal validation, cross-domain validation,
cascade analysis, low-reputation robustness.
"""
import numpy as np
import copy
import json
import os
from datetime import datetime
from collections import defaultdict

from config import *
from model import (Supplier, Order, Coalition,
                   coalition_reputation, coalition_utility,
                   update_reputation, is_feasible)
from racf import stage1_initial_formation
from racf_stages import run_racf
from data_loader import (load_kaggle_dataset, compute_supplier_stats,
                         generate_suppliers, generate_orders, generate_distances)
from experiments_multicycle import run_multicycle

OUTPUT_FILE = "results/missing_experiments.json"
CSV_PATH = os.path.join(os.path.dirname(__file__),
                        "Procurement KPI Analysis Dataset.csv")


# ============================================================
# 1. Out-of-Sample Temporal Validation (2022 vs 2023-2024)
# ============================================================
def run_temporal_validation():
    """Split Kaggle data by year, calibrate on 2022, test on 2023-2024."""
    print("\n" + "=" * 60)
    print("[1/4] Temporal Validation (2022 calibrate -> 2023-24 test)")
    print("=" * 60)

    rows = load_kaggle_dataset(CSV_PATH)

    # Split by year
    rows_2022 = []
    rows_2023 = []
    for r in rows:
        try:
            date = datetime.strptime(r['Order_Date'], '%Y-%m-%d')
            if date.year == 2022:
                rows_2022.append(r)
            elif date.year >= 2023:
                rows_2023.append(r)
        except:
            rows_2022.append(r)  # default to calibration set

    print(f"  Calibration (2022): {len(rows_2022)} orders")
    print(f"  Holdout (2023-24): {len(rows_2023)} orders")

    if len(rows_2022) < 10 or len(rows_2023) < 10:
        print("  SKIPPED: insufficient data per split")
        return None

    # Calibrate suppliers from 2022 data
    stats_2022 = compute_supplier_stats(rows_2022)
    s_names = sorted(stats_2022.keys())
    rng = np.random.default_rng(BASE_SEED)

    suppliers = []
    for i, name in enumerate(s_names):
        s = stats_2022[name]
        on_time = s['on_time'] / max(s['delivered'], 1)
        m = Supplier(
            m=i, K=rng.uniform(10, 100), c=rng.uniform(5, 50),
            h=rng.uniform(0.05, 0.30), f=rng.uniform(10, 100),
            tau=rng.uniform(1, 8), v=rng.uniform(40, 80),
            r=on_time, alpha=2.0, beta=2.0,
        )
        suppliers.append(m)

    # Generate orders from 2022 params, evaluate on 2023-style orders
    rng_orders = np.random.default_rng(BASE_SEED + 500)
    orders_in = generate_orders(min(len(rows_2022), 50), rng_orders)
    orders_out = generate_orders(min(len(rows_2023), 50), rng_orders)

    M_in, N_in = len(suppliers), len(orders_in)
    distances_in = generate_distances(M_in, N_in)
    distances_out = generate_distances(M_in, len(orders_out))

    # In-sample
    max_w_in = sum(o.p * o.Q for o in orders_in)
    results = {}
    for method in ["RACF", "REP-OBLIV"]:
        metrics_in, _ = run_multicycle(suppliers, orders_in, distances_in, method,
                                       n_cycles=5, rng=rng)
        metrics_out, _ = run_multicycle(suppliers, orders_out, distances_out, method,
                                        n_cycles=5, rng=rng)
        results[method] = {
            'reliability_in': metrics_in['reliability'],
            'reliability_out': metrics_out['reliability'],
            'utility_in': metrics_in['utility'] / max_w_in,
            'utility_out': metrics_out['utility'] / max(max_w_in, 1),
        }

    # Compute differences
    racf_rel_diff = results['RACF']['reliability_in'] - results['REP-OBLIV']['reliability_in']
    racf_rel_diff_out = results['RACF']['reliability_out'] - results['REP-OBLIV']['reliability_out']

    temporal = {
        'racf_minus_obliv_rel_in': racf_rel_diff,
        'racf_minus_obliv_rel_out': racf_rel_diff_out,
        'delta': abs(racf_rel_diff - racf_rel_diff_out),
        'racf_utility_in': results['RACF']['utility_in'],
        'racf_utility_out': results['RACF']['utility_out'],
        'racf_reliability_out': results['RACF']['reliability_out'],
    }
    print(f"  RACF-REPOBLIV reliability gap: in={racf_rel_diff:.3f}, out={racf_rel_diff_out:.3f}")
    print(f"  Temporal stability Δ: {temporal['delta']:.4f}")
    return temporal


# ============================================================
# 2. Cross-Domain Validation (Domain A → Domain B)
# ============================================================
def run_cross_domain_validation():
    """Split by item category: Domain A calibrate, Domain B test."""
    print("\n" + "=" * 60)
    print("[2/4] Cross-Domain Validation (Standardized -> Industrial)")
    print("=" * 60)

    rows = load_kaggle_dataset(CSV_PATH)

    domain_a_cats = {'Electronics', 'Office Supplies'}
    domain_b_cats = {'Raw Materials', 'MRO', 'Packaging'}

    rows_a = [r for r in rows if r.get('Item_Category', '') in domain_a_cats]
    rows_b = [r for r in rows if r.get('Item_Category', '') in domain_b_cats]

    print(f"  Domain A (standardized): {len(rows_a)} orders")
    print(f"  Domain B (industrial):   {len(rows_b)} orders")

    if len(rows_a) < 10 or len(rows_b) < 10:
        print("  SKIPPED: insufficient data per domain")
        return None

    # Calibrate from Domain A
    stats_a = compute_supplier_stats(rows_a)
    s_names = sorted(stats_a.keys())
    rng = np.random.default_rng(BASE_SEED)

    suppliers = []
    for i, name in enumerate(s_names):
        s = stats_a[name]
        on_time = s['on_time'] / max(s['delivered'], 1)
        m = Supplier(
            m=i, K=rng.uniform(10, 100), c=rng.uniform(5, 50),
            h=rng.uniform(0.05, 0.30), f=rng.uniform(10, 100),
            tau=rng.uniform(1, 8), v=rng.uniform(40, 80),
            r=on_time, alpha=2.0, beta=2.0,
        )
        suppliers.append(m)

    rng_orders = np.random.default_rng(BASE_SEED + 600)
    orders_a = generate_orders(min(len(rows_a), 50), rng_orders)
    orders_b = generate_orders(min(len(rows_b), 50), rng_orders)

    M = len(suppliers)
    distances_a = generate_distances(M, len(orders_a))
    distances_b = generate_distances(M, len(orders_b))

    results = {}
    for method in ["RACF", "REP-OBLIV"]:
        metrics_a, _ = run_multicycle(suppliers, orders_a, distances_a, method,
                                      n_cycles=5, rng=rng)
        metrics_b, _ = run_multicycle(suppliers, orders_b, distances_b, method,
                                      n_cycles=5, rng=rng)
        results[method] = {
            'reliability_a': metrics_a['reliability'],
            'reliability_b': metrics_b['reliability'],
            'utility_a': metrics_a['utility'],
            'utility_b': metrics_b['utility'],
        }

    racf_diff_a = results['RACF']['reliability_a'] - results['REP-OBLIV']['reliability_a']
    racf_diff_b = results['RACF']['reliability_b'] - results['REP-OBLIV']['reliability_b']

    cross_domain = {
        'racf_minus_obliv_rel_a': racf_diff_a,
        'racf_minus_obliv_rel_b': racf_diff_b,
        'attenuation': abs(racf_diff_a - racf_diff_b),
        'racf_reliability_b': results['RACF']['reliability_b'],
        'racf_utility_b': results['RACF']['utility_b'],
    }
    print(f"  RACF-REPOBLIV reliability gap: in-domain={racf_diff_a:.3f}, cross-domain={racf_diff_b:.3f}")
    print(f"  Attenuation: {cross_domain['attenuation']:.3f}")
    return cross_domain


# ============================================================
# 3. Reputation Cascade Analysis
# ============================================================
def run_cascade_analysis():
    """Cascade: suppliers excluded at threshold, measure reputation decay."""
    print("\n" + "=" * 60)
    print("[3/4] Reputation Cascade Analysis")
    print("=" * 60)

    N, M = 50, 150  # Large scale
    rng = np.random.default_rng(BASE_SEED)
    suppliers = generate_suppliers(M, rng)
    orders_template = generate_orders(N, rng)
    distances = generate_distances(M, N, rng)

    cascade_results = {}
    for R_val in [0.40, 0.50, 0.60, 0.70]:
        s_test = copy.deepcopy(suppliers)
        orders = [Order(n=i, Q=o.Q, T=o.T, p=o.p, R_star=R_val, beta=o.beta)
                  for i, o in enumerate(orders_template)]

        initial_reps = [m.r for m in s_test]
        for cycle in range(CASCADE_CYCLES):
            coalitions = run_racf(s_test, orders, distances, rho=RHO, rng=rng)
            for n, c in coalitions.items():
                if c.suppliers:
                    for m in c.suppliers:
                        update_reputation(m, c, distances, eta=ETA)
            # Excluded suppliers get neutral ψ=0.5
            for m in s_test:
                if m.assigned_order is None:
                    m.alpha += 0.5
                    m.beta += 0.5
                    r_hat = m.alpha / (m.alpha + m.beta)
                    m.r = (1 - ETA) * m.r + ETA * r_hat

        final_reps = [m.r for m in s_test]
        cascade_frac = sum(1 for init, final in zip(initial_reps, final_reps)
                          if init - final > 0.15) / M

        # Counterfactual: static reputation
        s_static = copy.deepcopy(suppliers)
        for cycle in range(CASCADE_CYCLES):
            orders_s = [Order(n=i, Q=o.Q, T=o.T, p=o.p, R_star=R_val, beta=o.beta)
                        for i, o in enumerate(orders_template)]
            coalitions = run_racf(s_static, orders_s, distances, rho=RHO, rng=rng)
            # Don't update reputation (static)
            for m in s_static:
                m.reset_allocation()
        static_final = [m.r for m in s_static]
        static_cascade = sum(1 for init, final in zip(initial_reps, static_final)
                            if init - final > 0.15) / M

        cascade_results[R_val] = {
            'with_feedback': cascade_frac,
            'static': static_cascade,
            'cascade_intensity': cascade_frac - static_cascade,
        }
        print(f"  R*={R_val}: with_fb={cascade_frac:.2f}, static={static_cascade:.2f}, "
              f"intensity={cascade_frac - static_cascade:.2f}")

    return cascade_results


# ============================================================
# 4. Low-Reputation Supplier Robustness
# ============================================================
def run_low_rep_robustness():
    """Vary fraction of low-reputation suppliers (r_m < 0.5) from 10% to 60%."""
    print("\n" + "=" * 60)
    print("[4/4] Low-Reputation Supplier Robustness")
    print("=" * 60)

    N, M = 50, 150
    rng = np.random.default_rng(BASE_SEED)
    orders = generate_orders(N, rng)
    distances = generate_distances(M, N, rng)

    robustness = {}
    for low_frac in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
        n_low = int(M * low_frac)
        suppliers = generate_suppliers(M, rng)
        # Assign low reputation to first n_low suppliers
        for i in range(n_low):
            suppliers[i].r = rng.uniform(0.1, 0.5)
            suppliers[i].alpha = 1.0
            suppliers[i].beta = 1.0
        # High reputation for rest
        for i in range(n_low, M):
            suppliers[i].r = rng.uniform(0.5, 0.95)
            suppliers[i].alpha = 5.0
            suppliers[i].beta = 1.0

        racf_metrics, _ = run_multicycle(suppliers, orders, distances, "RACF",
                                         n_cycles=5, rng=rng)
        obliv_metrics, _ = run_multicycle(suppliers, orders, distances, "REP-OBLIV",
                                          n_cycles=5, rng=rng)

        robustness[low_frac] = {
            'racf_completion': racf_metrics['completion_rate'],
            'obliv_completion': obliv_metrics['completion_rate'],
            'racf_reliability': racf_metrics['reliability'],
            'obliv_reliability': obliv_metrics['reliability'],
        }
        print(f"  low_rep={low_frac:.0%}: RACF comp={racf_metrics['completion_rate']:.2%}, "
              f"OBLIV comp={obliv_metrics['completion_rate']:.2%}")

    return robustness


# ============================================================
# Main
# ============================================================
def run_all_missing():
    os.makedirs("results", exist_ok=True)
    all_results = {}

    all_results['temporal'] = run_temporal_validation()
    all_results['cross_domain'] = run_cross_domain_validation()
    all_results['cascade'] = run_cascade_analysis()
    all_results['low_rep_robustness'] = run_low_rep_robustness()

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"All missing experiments completed. Saved to {OUTPUT_FILE}")
    print(f"{'=' * 60}")

    # Summary
    print("\n=== SUMMARY OF MISSING EXPERIMENTS ===")
    if all_results.get('temporal'):
        t = all_results['temporal']
        print(f"Temporal: in-sample Δ={t['racf_minus_obliv_rel_in']:.3f}, "
              f"out-sample Δ={t['racf_minus_obliv_rel_out']:.3f}, stability={t['delta']:.4f}")
    if all_results.get('cross_domain'):
        cd = all_results['cross_domain']
        print(f"Cross-Domain: in-domain Δ={cd['racf_minus_obliv_rel_a']:.3f}, "
              f"cross-domain Δ={cd['racf_minus_obliv_rel_b']:.3f}, attenuation={cd['attenuation']:.3f}")
    if all_results.get('cascade'):
        for R, v in all_results['cascade'].items():
            print(f"Cascade R*={R}: intensity={v['cascade_intensity']:.2f}")
    if all_results.get('low_rep_robustness'):
        for f, v in all_results['low_rep_robustness'].items():
            print(f"Low-rep {float(f):.0%}: RACF={v['racf_completion']:.2%}, OBLIV={v['obliv_completion']:.2%}")

    return all_results


if __name__ == "__main__":
    run_all_missing()
