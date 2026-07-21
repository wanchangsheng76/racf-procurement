"""
Multi-cycle experiment runner with endogenous reputation feedback.
This is the key innovation of the paper: reputation evolves across cycles.
"""
import numpy as np
import copy
import os
import json
from datetime import datetime

from config import *
from model import (Supplier, Order, Coalition,
                   coalition_reputation, coalition_utility,
                   update_reputation, is_feasible)
from racf import stage1_initial_formation
from racf_stages import run_racf, stage2_reallocation, stage3_refinement
from benchmarks import BENCHMARKS, run_rep_nostable, run_rep_notopsis
from data_loader import (generate_suppliers, generate_orders,
                         generate_distances, load_kaggle_dataset,
                         calibrate_from_real_data)


def run_multicycle(suppliers, orders, distances, method, n_cycles=10, rng=None,
                    eta=ETA, rho=RHO):
    """Run multiple procurement cycles with endogenous reputation feedback."""
    if rng is None:
        rng = np.random.default_rng()

    suppliers_evolving = copy.deepcopy(suppliers)
    all_metrics = []

    for cycle in range(n_cycles):
        orders_cycle = copy.deepcopy(orders)

        # Run allocation for this cycle
        if method == "RACF":
            coalitions = run_racf(suppliers_evolving, orders_cycle, distances, rho=rho, rng=rng)
        elif method == "REP-OBLIV":
            saved_R = [o.R_star for o in orders_cycle]
            for o in orders_cycle:
                o.R_star = 0.0
            coalitions = stage1_initial_formation(suppliers_evolving, orders_cycle, distances, rho=0.0, rng=rng)
            coalitions = stage3_refinement(coalitions, suppliers_evolving, distances, rho=0.0)
            for o, r in zip(orders_cycle, saved_R):
                o.R_star = r
        elif method == "REP-STATIC":
            # Static reputation: use fixed r_m from initial state
            frozen_reps = [m.r for m in suppliers_evolving]
            coalitions = stage1_initial_formation(suppliers_evolving, orders_cycle, distances, rho=rho, rng=rng)
        elif method == "REP-NOTOPSIS":
            coalitions = run_rep_notopsis(suppliers_evolving, orders_cycle, distances, rho=rho, rng=rng)
        elif method == "REP-NOSTABLE":
            coalitions = run_rep_nostable(suppliers_evolving, orders_cycle, distances, rho=rho, rng=rng)
        else:
            coalitions = run_racf(suppliers_evolving, orders_cycle, distances, rho=rho, rng=rng)

        # Metrics for this cycle
        rep = np.mean([coalition_reputation(c) for c in coalitions.values() if c.suppliers])
        wel = sum(coalition_utility(c, distances) for c in coalitions.values())
        task_comp = sum(1 for c in coalitions.values() if c.chi == 1) / len(coalitions)

        all_metrics.append({'reliability': rep, 'utility': wel, 'completion': task_comp})

        # Reputation feedback: update reputation based on coalition outcomes
        for n, c in coalitions.items():
            if c.suppliers:
                for m in c.suppliers:
                    update_reputation(m, c, distances, eta=eta)

        # Also give neutral evaluation to EXCLUDED suppliers (they lose reputation)
        for m in suppliers_evolving:
            if m.assigned_order is None:
                # Supplier got no order → neutral ψ=0.5 → Bayesian update
                m.alpha += 0.5
                m.beta += 0.5
                r_hat = m.alpha / (m.alpha + m.beta)
                m.r = (1 - eta) * m.r + eta * r_hat

    # Return average across cycles
    avg_metrics = {
        'reliability': np.mean([m['reliability'] for m in all_metrics]),
        'utility': np.mean([m['utility'] for m in all_metrics]),
        'completion_rate': np.mean([m['completion'] for m in all_metrics]),
    }
    return avg_metrics, all_metrics


def run_full_experiments():
    """Run the complete experimental suite with multi-cycle feedback."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}
    N_CYCLES = 10

    print("=" * 60)
    print("RACF Experiment Reproduction (Multi-Cycle Feedback)")
    print("=" * 60)

    # 1. Main experiments across scales
    print("\n[1/5] Multi-cycle experiments across 5 scales...")
    scale_results = {}
    for scale_name in SCALES:
        print(f"  Scale: {scale_name} ({SCALES[scale_name]['N']} orders, {SCALES[scale_name]['M']} suppliers)...")
        N, M = SCALES[scale_name]['N'], SCALES[scale_name]['M']
        rng = np.random.default_rng(BASE_SEED)
        suppliers = generate_suppliers(M, rng)
        orders = generate_orders(N, rng)
        distances = generate_distances(M, N, rng)
        max_w = sum(o.p * o.Q for o in orders)

        method_results = {}
        for method in BENCHMARKS:
            metrics, _ = run_multicycle(suppliers, orders, distances, method, n_cycles=N_CYCLES, rng=rng)
            method_results[method] = {
                'reliability': metrics['reliability'],
                'utility': metrics['utility'] / max_w,
                'completion': metrics['completion_rate'],
            }

        scale_results[scale_name] = method_results
        print(f"    RACF rel: {method_results['RACF']['reliability']:.3f}, "
              f"REP-OBLIV rel: {method_results['REP-OBLIV']['reliability']:.3f}")

    all_results['scales'] = scale_results

    # 2. Ablation analysis
    print("\n[2/5] Ablation analysis...")
    scale = SCALES["Large"]
    N, M = scale['N'], scale['M']
    rng = np.random.default_rng(BASE_SEED)
    suppliers = generate_suppliers(M, rng)
    orders = generate_orders(N, rng)
    distances = generate_distances(M, N, rng)

    # Full RACF baseline
    base_metrics, _ = run_multicycle(suppliers, orders, distances, "RACF", n_cycles=N_CYCLES, rng=rng)
    base_rel = base_metrics['reliability']
    base_util = base_metrics['utility']

    ablation_results = {}
    # Each ablation: remove one component
    for label, ablate_method in [
        ("Reputation Admission", "REP-OBLIV"),
        ("Dynamic Update", "REP-STATIC"),
        ("TOPSIS (Stage II)", "REP-NOTOPSIS"),
        ("Stability (Stage III)", "REP-NOSTABLE"),
    ]:
        ab_metrics, _ = run_multicycle(suppliers, orders, distances, ablate_method, n_cycles=N_CYCLES, rng=rng)
        ablation_results[label] = {
            'utility_loss': (base_util - ab_metrics['utility']) / base_util * 100 if base_util > 0 else 0,
            'reliability_loss': (base_rel - ab_metrics['reliability']) / base_rel * 100 if base_rel > 0 else 0,
        }
    # Feedback loop counterfactual
    rng2 = np.random.default_rng(BASE_SEED + 100)
    s_nofb = copy.deepcopy(suppliers)
    # Run without feedback: don't update reputation (freeze after initial)
    no_fb_metrics_list = []
    for cycle in range(N_CYCLES):
        o_cycle = copy.deepcopy(orders)
        coalitions = run_racf(s_nofb, o_cycle, distances, rho=RHO, rng=rng2)
        no_fb_metrics_list.append({
            'reliability': np.mean([coalition_reputation(c) for c in coalitions.values() if c.suppliers]),
            'utility': sum(coalition_utility(c, distances) for c in coalitions.values()),
        })
    no_fb_rel = np.mean([m['reliability'] for m in no_fb_metrics_list])
    no_fb_util = np.mean([m['utility'] for m in no_fb_metrics_list])
    ablation_results["Endogenous Feedback (no loop)"] = {
        'utility_loss': (base_util - no_fb_util) / base_util * 100 if base_util > 0 else 0,
        'reliability_loss': (base_rel - no_fb_rel) / base_rel * 100 if base_rel > 0 else 0,
    }

    all_results['ablation'] = ablation_results
    print("  done")

    # 3. Sensitivity analysis
    print("\n[3/5] Parameter sensitivity (η-ρ heatmap)...")
    grid = np.zeros((len(ETA_RANGE), len(RHO_RANGE)))
    for i, eta in enumerate(ETA_RANGE):
        for j, rho in enumerate(RHO_RANGE):
            s_test = copy.deepcopy(suppliers)
            o_test = copy.deepcopy(orders)
            metrics, _ = run_multicycle(s_test, o_test, distances, "RACF",
                                        n_cycles=5, rng=rng, eta=eta, rho=rho)
            grid[i, j] = metrics['reliability']

    all_results['sensitivity'] = {'grid': grid.tolist(), 'etas': ETA_RANGE.tolist(), 'rhos': RHO_RANGE.tolist()}
    print("  done")

    # 4. Real-data validation
    print("\n[4/5] Real-data validation...")
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "Procurement KPI Analysis Dataset.csv")
        rows = load_kaggle_dataset(csv_path)
        suppliers_real, orders_real = calibrate_from_real_data(rows)
        distances_real = generate_distances(len(suppliers_real), len(orders_real))

        real_results = {}
        for method in ["RACF", "REP-OBLIV", "REP-STATIC", "REP-NOTOPSIS", "REP-NOSTABLE"]:
            metrics, _ = run_multicycle(suppliers_real, orders_real, distances_real, method, n_cycles=5, rng=rng)
            real_results[method] = {
                'reliability': metrics['reliability'],
                'utility': metrics['utility'],
                'completion': metrics['completion_rate'],
            }
        all_results['real_data'] = real_results
        print("  done")
    except Exception as e:
        print(f"  Skipped: {e}")

    # 5. Save
    print("\n[5/5] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"results_multicycle_{timestamp}.json")

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"  Saved to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Coalition Reliability (Multi-Cycle)")
    print("=" * 80)
    header = f"{'Scale':<10} " + " ".join(f"{m:<12}" for m in BENCHMARKS)
    print(header)
    print("-" * len(header))
    for scale_name in SCALES:
        row = f"{scale_name:<10} "
        for method in BENCHMARKS:
            val = scale_results[scale_name][method]['reliability']
            row += f"{val:<12.3f}"
        print(row)

    print("\nAblation Results (Large Scale):")
    for name, vals in ablation_results.items():
        print(f"  {name:<35} Util loss: {vals['utility_loss']:5.1f}%  Rel loss: {vals['reliability_loss']:5.1f}%")

    print("\nDone!")
    return all_results


if __name__ == "__main__":
    run_full_experiments()
