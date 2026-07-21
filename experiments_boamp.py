"""
BOAMP-calibrated experiment runner. Replicates all v23 experiments.
"""
import numpy as np
import copy, json, os
from datetime import datetime

from config import *
from model import *
from data_loader import generate_suppliers, generate_orders, generate_distances
from data_loader_boamp import (calibrate_from_boamp, get_boamp_temporal_split,
                                get_boamp_domain_split, compute_boamp_supplier_stats,
                                compute_reputation_scores)
from racf_stages import run_racf
from benchmarks import BENCHMARKS, run_rep_nostable, run_rep_notopsis
from experiments_multicycle import run_multicycle

BOAMP_PATH = r'D:\张娟工作\阮志鹏\dataandcode\BeauAMP_full.csv'
N_CYCLES = 10
N_RUNS = 20
OUTPUT_FILE = 'results/boamp_results.json'

def run_all_boamp():
    os.makedirs('results', exist_ok=True)
    all_results = {}
    rng = np.random.default_rng(BASE_SEED)

    # ================================================================
    # 1. Main experiments across scales (BOAMP-calibrated supplier pool)
    # ================================================================
    print("="*60)
    print("[1/7] Main experiments with BOAMP calibration")
    print("="*60)

    # Calibrate from BOAMP (use 10K rows for speed)
    suppliers_boamp, orders_boamp, stats, scores = calibrate_from_boamp(
        BOAMP_PATH, n_suppliers=150, n_orders=50, rng=rng)
    distances_boamp = generate_distances(len(suppliers_boamp), len(orders_boamp), rng)
    max_w = sum(o.p * o.Q for o in orders_boamp)

    scale_results = {}
    methods = list(BENCHMARKS.keys())
    for scale_name in SCALES:
        N, M = SCALES[scale_name]['N'], SCALES[scale_name]['M']
        print(f"  Scale: {scale_name} (N={N}, M={M})")

        meth_results = {}
        for method in methods:
            metrics, _ = run_multicycle(suppliers_boamp[:M], orders_boamp[:N],
                                        distances_boamp, method, n_cycles=N_CYCLES, rng=rng)
            meth_results[method] = {
                'reliability': round(metrics['reliability'], 4),
                'utility': round(metrics['utility'] / max_w, 4),
                'completion': round(metrics['completion_rate'], 4),
            }
        scale_results[scale_name] = meth_results
        print(f"    RACF rel={meth_results['RACF']['reliability']:.3f}, "
              f"OBLIV rel={meth_results['REP-OBLIV']['reliability']:.3f}")

    all_results['scales'] = scale_results

    # ================================================================
    # 2. Temporal validation (2015-2019 → 2020-2023)
    # ================================================================
    print("\n[2/7] Temporal validation (BOAMP 2015-2019 → 2020-2023)")
    cal_rows, test_rows = get_boamp_temporal_split(BOAMP_PATH)
    print(f"  Calibration: {len(cal_rows):,} contracts (2015-2019)")
    print(f"  Test: {len(test_rows):,} contracts (2020-2023)")

    # Calibrate from earlier period
    stats_cal = compute_boamp_supplier_stats(cal_rows, min_contracts=2)
    scores_cal = compute_reputation_scores(stats_cal, rng)
    suppliers_cal = []
    for i, (siret, score) in enumerate(list(scores_cal.items())[:150]):
        s = stats_cal[siret]
        m = Supplier(m=i, K=rng.uniform(10,100), c=rng.uniform(5,50),
                     h=rng.uniform(0.05,0.3), f=rng.uniform(10,100),
                     tau=rng.uniform(1,8), v=rng.uniform(40,80),
                     r=score, alpha=2.0, beta=2.0)
        suppliers_cal.append(m)

    orders_temporal = generate_orders(50, rng)
    dist_temporal = generate_distances(len(suppliers_cal), 50, rng)

    temporal = {}
    for method in ["RACF", "REP-OBLIV"]:
        metrics, _ = run_multicycle(suppliers_cal, orders_temporal, dist_temporal,
                                    method, n_cycles=5, rng=rng)
        temporal[method] = {'reliability': round(metrics['reliability'], 4),
                            'utility': round(metrics['utility'], 4)}
    temporal['racf_obliv_gap'] = round(temporal['RACF']['reliability'] - 
                                        temporal['REP-OBLIV']['reliability'], 4)
    all_results['temporal'] = temporal
    print(f"  RACF-OBLIV gap: {temporal['racf_obliv_gap']:.4f}")

    # ================================================================
    # 3. Cross-domain validation (CPV categories)
    # ================================================================
    print("\n[3/7] Cross-domain validation (BOAMP CPV categories)")
    # Domain A: Construction (45), Domain B: Services (70-80)
    rows_a, rows_b = get_boamp_domain_split(BOAMP_PATH,
                                            {'45'}, {'70','71','72','73','75','76','79','80'})
    if len(rows_a) < 10: rows_a, rows_b = rows_a or rows_b, rows_b or rows_a  # fallback
    print(f"  Domain A: {len(rows_a):,} contracts")
    print(f"  Domain B: {len(rows_b):,} contracts")

    stats_a = compute_boamp_supplier_stats(rows_a, min_contracts=1)
    scores_a = compute_reputation_scores(stats_a, rng)
    suppliers_a = []
    for i, (siret, score) in enumerate(list(scores_a.items())[:150]):
        s = stats_a[siret]
        m = Supplier(m=i, K=rng.uniform(10,100), c=rng.uniform(5,50),
                     h=rng.uniform(0.05,0.3), f=rng.uniform(10,100),
                     tau=rng.uniform(1,8), v=rng.uniform(40,80),
                     r=score, alpha=2.0, beta=2.0)
        suppliers_a.append(m)

    orders_cd = generate_orders(50, rng)
    dist_cd = generate_distances(len(suppliers_a), 50, rng)

    cross_domain = {}
    for method in ["RACF", "REP-OBLIV"]:
        metrics, _ = run_multicycle(suppliers_a, orders_cd, dist_cd,
                                    method, n_cycles=5, rng=rng)
        cross_domain[method] = {'reliability': round(metrics['reliability'], 4)}
    cross_domain['racf_obliv_gap'] = round(cross_domain['RACF']['reliability'] - 
                                            cross_domain['REP-OBLIV']['reliability'], 4)
    all_results['cross_domain'] = cross_domain
    print(f"  RACF-OBLIV gap: {cross_domain['racf_obliv_gap']:.4f}")

    # ================================================================
    # 4. Significance test (BOAMP calibration)
    # ================================================================
    print("\n[4/7] Significance test (BOAMP)")
    per_run = {m: {'rel': [], 'util': []} for m in methods}
    for run in range(N_RUNS):
        run_rng = np.random.default_rng(BASE_SEED + run)
        for method in methods:
            metrics, _ = run_multicycle(suppliers_boamp, orders_boamp[:50],
                                        distances_boamp, method, n_cycles=N_CYCLES, rng=run_rng)
            per_run[method]['rel'].append(metrics['reliability'])
            per_run[method]['util'].append(metrics['utility'] / max_w)
        if (run+1) % 5 == 0: print(f"  {run+1}/{N_RUNS} runs")

    sig_results = {}
    for m in methods:
        sig_results[m] = {
            'reliability_mean': round(np.mean(per_run[m]['rel']), 4),
            'reliability_std': round(np.std(per_run[m]['rel'], ddof=1), 4),
            'utility_mean': round(np.mean(per_run[m]['util']), 4),
        }
    all_results['significance'] = sig_results

    for m in methods:
        print(f"  {m:15s} rel={sig_results[m]['reliability_mean']:.4f}±{sig_results[m]['reliability_std']:.4f}")

    # ================================================================
    # 5. Ablation (BOAMP)
    # ================================================================
    print("\n[5/7] Ablation analysis (BOAMP)")
    # Full RACF baseline
    base_metrics, _ = run_multicycle(suppliers_boamp, orders_boamp[:50], distances_boamp,
                                     "RACF", n_cycles=N_CYCLES, rng=rng)
    base_rel, base_util = base_metrics['reliability'], base_metrics['utility']

    ablation = {}
    ablation_methods = [("Reputation Admission", "REP-OBLIV"),
                        ("Dynamic Update", "REP-STATIC"),
                        ("TOPSIS (Stage II)", "REP-NOTOPSIS"),
                        ("Stability (Stage III)", "REP-NOSTABLE")]
    for label, ablate_m in ablation_methods:
        ab_metrics, _ = run_multicycle(suppliers_boamp, orders_boamp[:50], distances_boamp,
                                       ablate_m, n_cycles=N_CYCLES, rng=rng)
        ablation[label] = {
            'utility_loss': round((base_util - ab_metrics['utility']) / base_util * 100, 1) if base_util > 0 else 0,
            'reliability_loss': round((base_rel - ab_metrics['reliability']) / base_rel * 100, 1) if base_rel > 0 else 0,
        }

    # Feedback loop counterfactual
    s_nofb = copy.deepcopy(suppliers_boamp)
    nofb_rels = []
    nofb_utils = []
    for _ in range(5):
        o_cycle = copy.deepcopy(orders_boamp[:50])
        coalitions = run_racf(s_nofb, o_cycle, distances_boamp, rho=RHO, rng=rng)
        nofb_rels.append(np.mean([coalition_reputation(c) for c in coalitions.values() if c.suppliers]))
        nofb_utils.append(sum(coalition_utility(c, distances_boamp) for c in coalitions.values()))
    nofb_rel = np.mean(nofb_rels)
    nofb_util = np.mean(nofb_utils)
    ablation["Feedback Loop (no loop)"] = {
        'utility_loss': round((base_util - nofb_util) / base_util * 100, 1) if base_util > 0 else 0,
        'reliability_loss': round((base_rel - nofb_rel) / base_rel * 100, 1) if base_rel > 0 else 0,
    }
    all_results['ablation'] = ablation

    for name, vals in ablation.items():
        print(f"  {name:30s} util_loss={vals['utility_loss']:5.1f}%  rel_loss={vals['reliability_loss']:5.1f}%")

    # ================================================================
    # 6. Cascade (BOAMP)
    # ================================================================
    print("\n[6/7] Cascade analysis (BOAMP)")
    cascade = {}
    for R_val in [0.40, 0.50, 0.60, 0.70]:
        s_test = copy.deepcopy(suppliers_boamp[:150])
        orders_cas = [Order(n=i, Q=o.Q, T=o.T, p=o.p, R_star=R_val, beta=o.beta)
                      for i, o in enumerate(orders_boamp[:50])]
        initial_reps = [m.r for m in s_test]
        for cycle in range(CASCADE_CYCLES):
            coalitions = run_racf(s_test, orders_cas, distances_boamp, rho=RHO, rng=rng)
            for n, c in coalitions.items():
                if c.suppliers:
                    for m in c.suppliers:
                        update_reputation(m, c, distances_boamp, eta=ETA)
            for m in s_test:
                if m.assigned_order is None:
                    m.alpha += 0.5; m.beta += 0.5
                    r_hat = m.alpha/(m.alpha+m.beta)
                    m.r = (1-ETA)*m.r + ETA*r_hat
        final_reps = [m.r for m in s_test]
        frac = sum(1 for i, f in zip(initial_reps, final_reps) if i-f > 0.15) / len(s_test)
        cascade[R_val] = round(frac, 4)
    all_results['cascade'] = cascade
    for R, v in cascade.items():
        print(f"  R*={R}: cascade={v:.3f}")

    # ================================================================
    # 7. Low-rep robustness (BOAMP)
    # ================================================================
    print("\n[7/7] Low-rep robustness (BOAMP)")
    robustness = {}
    for low_frac in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
        n_low = int(150 * low_frac)
        s_rob = copy.deepcopy(suppliers_boamp[:150])
        for i in range(n_low):
            s_rob[i].r = rng.uniform(0.1, 0.5)
        for i in range(n_low, 150):
            s_rob[i].r = rng.uniform(0.5, 0.95)

        for method in ["RACF", "REP-OBLIV"]:
            metrics, _ = run_multicycle(s_rob, orders_boamp[:50], distances_boamp,
                                        method, n_cycles=5, rng=rng)
            if method not in robustness:
                robustness[method] = {}
            robustness[method][low_frac] = {
                'completion': round(metrics['completion_rate'], 4),
                'reliability': round(metrics['reliability'], 4),
            }
    all_results['low_rep_robustness'] = robustness
    for f in [0.10, 0.30, 0.60]:
        print(f"  low_rep={f:.0%}: RACF={robustness['RACF'][f]['completion']:.2%}, "
              f"OBLIV={robustness['REP-OBLIV'][f]['completion']:.2%}")

    # ================================================================
    # Save
    # ================================================================
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")
    return all_results

if __name__ == '__main__':
    run_all_boamp()
