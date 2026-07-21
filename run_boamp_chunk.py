"""Run BOAMP experiments in small chunks (< 10 min each)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, copy, json
from config import *
from model import *
from data_loader import generate_suppliers, generate_orders, generate_distances
from data_loader_boamp import calibrate_from_boamp
from racf_stages import run_racf
from benchmarks import BENCHMARKS
from experiments_multicycle import run_multicycle

BOAMP_PATH = r'D:\张娟工作\阮志鹏\dataandcode\BeauAMP_full.csv'
OUTPUT_FILE = 'results/boamp_fixed.json'
N_CYCLES = 10
N_RUNS = 20

os.makedirs('results', exist_ok=True)
rng = np.random.default_rng(BASE_SEED)

# Calibrate once
print("Calibrating from BOAMP...")
suppliers_boamp, orders_boamp, stats, scores = calibrate_from_boamp(
    BOAMP_PATH, n_suppliers=150, n_orders=50, rng=rng)
distances_boamp = generate_distances(len(suppliers_boamp), len(orders_boamp), rng)
max_w = sum(o.p * o.Q for o in orders_boamp)
all_results = {}

# --- 1. Main experiments (just Large scale to save time) ---
print("\n[1] Main experiments at Large scale...")
scale_results = {}
N, M = 50, 150
methods = list(BENCHMARKS.keys())
meth_results = {}
for method in methods:
    metrics, _ = run_multicycle(suppliers_boamp[:M], orders_boamp[:N],
                                distances_boamp, method, n_cycles=N_CYCLES, rng=rng)
    meth_results[method] = {
        'reliability': round(metrics['reliability'], 4),
        'utility': round(metrics['utility'] / max_w, 4),
        'completion': round(metrics['completion_rate'], 4),
    }
    print(f"  {method:15s} rel={meth_results[method]['reliability']:.4f}")
all_results['scale_large'] = meth_results

# --- 2. Cascade ---
print("\n[2] Cascade analysis...")
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
    print(f"  R*={R_val}: cascade={cascade[R_val]:.4f}")
all_results['cascade'] = cascade

# --- 3. Ablation ---
print("\n[3] Ablation analysis...")
base_metrics, _ = run_multicycle(suppliers_boamp, orders_boamp[:50], distances_boamp,
                                  "RACF", n_cycles=N_CYCLES, rng=rng)
base_rel, base_util = base_metrics['reliability'], base_metrics['utility']
ablation = {}
for label, ablate_m in [("Reputation Admission", "REP-OBLIV"),
                         ("Dynamic Update", "REP-STATIC"),
                         ("TOPSIS (Stage II)", "REP-NOTOPSIS"),
                         ("Stability (Stage III)", "REP-NOSTABLE")]:
    ab_metrics, _ = run_multicycle(suppliers_boamp, orders_boamp[:50], distances_boamp,
                                   ablate_m, n_cycles=N_CYCLES, rng=rng)
    ablation[label] = {
        'utility_loss': round((base_util - ab_metrics['utility']) / base_util * 100, 1) if base_util > 0 else 0,
        'reliability_loss': round((base_rel - ab_metrics['reliability']) / base_rel * 100, 1) if base_rel > 0 else 0,
    }
    print(f"  {label:30s} util={ablation[label]['utility_loss']:5.1f}%  rel={ablation[label]['reliability_loss']:5.1f}%")

# Feedback loop counterfactual
s_nofb = copy.deepcopy(suppliers_boamp)
nofb_rels, nofb_utils = [], []
for _ in range(5):
    o_cycle = copy.deepcopy(orders_boamp[:50])
    coalitions = run_racf(s_nofb, o_cycle, distances_boamp, rho=RHO, rng=rng)
    nofb_rels.append(np.mean([coalition_reputation(c) for c in coalitions.values() if c.suppliers]))
    nofb_utils.append(sum(coalition_utility(c, distances_boamp) for c in coalitions.values()))
ablation["Feedback Loop (no loop)"] = {
    'utility_loss': round((base_util - np.mean(nofb_utils)) / base_util * 100, 1) if base_util > 0 else 0,
    'reliability_loss': round((base_rel - np.mean(nofb_rels)) / base_rel * 100, 1) if base_rel > 0 else 0,
}
all_results['ablation'] = ablation

# Save
def convert(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert(v) for v in obj]
    return obj

with open(OUTPUT_FILE, 'w') as f:
    json.dump(convert(all_results), f, indent=2)

print(f"\nSaved to {OUTPUT_FILE}")
print("\n=== KEY RESULTS ===")
print(f"Large scale RACF rel: {meth_results['RACF']['reliability']:.4f}")
print(f"Large scale OBLIV rel: {meth_results['REP-OBLIV']['reliability']:.4f}")
for R, v in cascade.items(): print(f"Cascade R*={R}: {v:.4f}")
for name, vals in ablation.items(): print(f"Ablation {name}: util={vals['utility_loss']:.1f}%, rel={vals['reliability_loss']:.1f}%")
