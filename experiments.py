"""
Main experiment runner — reproduces all experiments from Section 6.
"""
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List

from config import *
from model import (Supplier, Order, Coalition,
                   coalition_reputation, coalition_utility,
                   update_reputation, is_feasible)
from racf import stage1_initial_formation
from racf_stages import run_racf, stage2_reallocation, stage3_refinement
from benchmarks import BENCHMARKS, run_rep_notopsis, run_rep_nostable
from data_loader import (generate_suppliers, generate_orders,
                         generate_distances, load_kaggle_dataset,
                         calibrate_from_real_data)


def total_welfare(coalitions, distances):
    """Compute Φ(Π) = Σ W(C_n) — the potential function."""
    return sum(coalition_utility(c, distances) for c in coalitions.values())


def avg_coalition_reputation(coalitions):
    """Average coalition reputation across all orders."""
    reps = []
    for n, c in coalitions.items():
        if c.suppliers:
            reps.append(coalition_reputation(c))
    return np.mean(reps) if reps else 0.0


def task_completion_rate(coalitions):
    """Fraction of orders with feasible coalitions."""
    if not coalitions:
        return 0.0
    feasible = sum(1 for c in coalitions.values() if c.chi == 1 and c.suppliers)
    return feasible / len(coalitions)


def compute_metrics(coalitions, distances):
    """Compute all evaluation metrics."""
    return {
        'reliability': avg_coalition_reputation(coalitions),
        'utility': total_welfare(coalitions, distances),
        'completion_rate': task_completion_rate(coalitions),
    }


def run_simulation(suppliers, orders, distances, method, rng, **kwargs):
    """Run one simulation with a given method."""
    # Deep copy suppliers to avoid mutation across runs
    import copy
    suppliers_copy = copy.deepcopy(suppliers)
    orders_copy = copy.deepcopy(orders)

    if method == "RACF":
        coalitions = run_racf(suppliers_copy, orders_copy, distances, **kwargs)
    else:
        coalitions = BENCHMARKS[method](suppliers_copy, orders_copy, distances, **kwargs)

    # Simulate reputation feedback cycle
    for _ in range(3):
        for n, c in coalitions.items():
            if c.suppliers:
                for m in c.suppliers:
                    update_reputation(m, c, distances, eta=ETA)
        # Excluded suppliers get neutral evaluation
        for m in suppliers_copy:
            if m.assigned_order is None:
                m.alpha += 0.5
                m.beta += 0.5
                r_hat = m.alpha / (m.alpha + m.beta)
                m.r = (1 - ETA) * m.r + ETA * r_hat

    metrics = compute_metrics(coalitions, distances)
    return metrics


def run_experiment_scale(scale_name, n_runs=N_RUNS, seed=BASE_SEED):
    """Run experiments at a given scale across all methods."""
    scale = SCALES[scale_name]
    N, M = scale['N'], scale['M']

    rng = np.random.default_rng(seed)

    # Generate problem instance
    suppliers = generate_suppliers(M, rng)
    orders = generate_orders(N, rng)
    distances = generate_distances(M, N, rng)
    # Normalize utility
    max_possible_welfare = sum(o.p * o.Q for o in orders)

    methods = list(BENCHMARKS.keys())  # REP-OBLIV, REP-STATIC, REP-NOTOPSIS, REP-NOSTABLE, RACF
    results = {m: {'reliability': [], 'utility': [], 'completion_rate': []} for m in methods}

    for run in range(n_runs):
        run_rng = np.random.default_rng(seed + run)
        for method in methods:
            metrics = run_simulation(suppliers, orders, distances, method, run_rng, rho=RHO)
            results[method]['reliability'].append(metrics['reliability'])
            results[method]['utility'].append(metrics['utility'] / max_possible_welfare)
            results[method]['completion_rate'].append(metrics['completion_rate'])

    # Summarize
    summary = {}
    for m in methods:
        summary[m] = {
            'reliability_mean': np.mean(results[m]['reliability']),
            'reliability_ci': _bootstrap_ci(results[m]['reliability']),
            'utility_mean': np.mean(results[m]['utility']),
            'utility_ci': _bootstrap_ci(results[m]['utility']),
            'completion_mean': np.mean(results[m]['completion_rate']),
        }

    return summary, results


def run_ablation(scale_name="Large", n_runs=N_RUNS, seed=BASE_SEED):
    """Ablation analysis: remove one component at a time."""
    scale = SCALES[scale_name]
    N, M = scale['N'], scale['M']
    rng = np.random.default_rng(seed)
    suppliers = generate_suppliers(M, rng)
    orders = generate_orders(N, rng)
    distances = generate_distances(M, N, rng)

    import copy
    results = {}
    # Full RACF
    metrics_list = []
    for run in range(n_runs):
        s = copy.deepcopy(suppliers)
        o = copy.deepcopy(orders)
        run_rng = np.random.default_rng(seed + run)
        coalitions = run_racf(s, o, distances, rho=RHO, rng=run_rng)
        metrics_list.append(compute_metrics(coalitions, distances))
    base_rel = np.mean([m['reliability'] for m in metrics_list])
    base_util = np.mean([m['utility'] for m in metrics_list])

    # Components to ablate
    ablations = {
        'Reputation Admission': lambda s, o, d, rng: stage1_initial_formation(
            s, [Order(n=i, Q=o[i].Q, T=o[i].T, p=o[i].p, R_star=0.0, beta=o[i].beta)
                for i in range(len(o))], d, rho=RHO, rng=rng),
        'Dynamic Update': lambda s, o, d, rng: stage1_initial_formation(s, o, d, rho=RHO, rng=rng),
        'TOPSIS (Stage II)': lambda s, o, d, rng: run_rep_notopsis(s, o, d, rho=RHO, rng=rng),
        'Stability (Stage III)': lambda s, o, d, rng: run_rep_nostable(s, o, d, rho=RHO, rng=rng),
    }

    for name, func in ablations.items():
        metrics_list = []
        for run in range(n_runs):
            s = copy.deepcopy(suppliers)
            o = copy.deepcopy(orders)
            run_rng = np.random.default_rng(seed + run)
            coalitions = func(s, o, distances, run_rng)
            metrics_list.append(compute_metrics(coalitions, distances))
        mean_rel = np.mean([m['reliability'] for m in metrics_list])
        mean_util = np.mean([m['utility'] for m in metrics_list])
        results[name] = {
            'utility_loss': (base_util - mean_util) / base_util * 100 if base_util > 0 else 0,
            'reliability_loss': (base_rel - mean_rel) / base_rel * 100 if base_rel > 0 else 0,
        }

    return results


def run_sensitivity(scale_name="Large", seed=BASE_SEED):
    """Parameter sensitivity: η-ρ heatmap."""
    scale = SCALES[scale_name]
    N, M = scale['N'], scale['M']
    rng = np.random.default_rng(seed)
    suppliers = generate_suppliers(M, rng)
    orders = generate_orders(N, rng)
    distances = generate_distances(M, N, rng)

    import copy
    grid = np.zeros((len(ETA_RANGE), len(RHO_RANGE)))

    for i, eta in enumerate(ETA_RANGE):
        for j, rho in enumerate(RHO_RANGE):
            s = copy.deepcopy(suppliers)
            o = copy.deepcopy(orders)
            coalitions = run_racf(s, o, distances, rho=rho, rng=rng)
            grid[i, j] = avg_coalition_reputation(coalitions)

    return grid, ETA_RANGE, RHO_RANGE


def run_cascade_analysis(scale_name="Large", seed=BASE_SEED):
    """Reputation cascade effect under varying thresholds."""
    scale = SCALES[scale_name]
    N, M = scale['N'], scale['M']
    rng = np.random.default_rng(seed)
    suppliers = generate_suppliers(M, rng)
    orders = generate_orders(N, rng)
    distances = generate_distances(M, N, rng)

    import copy
    results = {}
    for R_val in CASCADE_R_STAR_RANGE:
        s = copy.deepcopy(suppliers)
        o = [Order(n=i, Q=ord.Q, T=ord.T, p=ord.p, R_star=R_val, beta=ord.beta)
             for i, ord in enumerate(orders)]
        initial_reps = [m.r for m in s]
        for cycle in range(CASCADE_CYCLES):
            coalitions = run_racf(s, o, distances, rho=RHO, rng=rng)
            # Reputation update
            for n, c in coalitions.items():
                if c.suppliers:
                    for m in c.suppliers:
                        update_reputation(m, c, distances, eta=ETA)
            # Excluded suppliers get neutral evaluation
            for m in s:
                if m.assigned_order is None:
                    m.alpha += 0.5
                    m.beta += 0.5
                    r_hat = m.alpha / (m.alpha + m.beta)
                    m.r = (1 - ETA) * m.r + ETA * r_hat
        final_reps = [m.r for m in s]
        cascade_frac = sum(1 for init, final in zip(initial_reps, final_reps)
                          if init - final > 0.15) / M
        results[R_val] = cascade_frac

    return results


def _bootstrap_ci(data, n_bootstrap=N_BOOTSTRAP, alpha=SIGNIFICANCE_LEVEL):
    """Bootstrap confidence interval."""
    data = np.array(data)
    means = []
    rng = np.random.default_rng(BASE_SEED)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lo = np.percentile(means, alpha / 2 * 100)
    hi = np.percentile(means, (1 - alpha / 2) * 100)
    return (lo, hi)


def run_all_experiments():
    """Run the complete experimental suite."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    print("=" * 60)
    print("RACF Experiment Reproduction")
    print("=" * 60)

    # 1. Main experiments across scales
    print("\n[1/6] Running main experiments across 5 scales...")
    scale_results = {}
    for scale_name in SCALES:
        print(f"  Scale: {scale_name}...", end=" ")
        summary, raw = run_experiment_scale(scale_name, n_runs=N_RUNS)
        scale_results[scale_name] = summary
        print("done")
    all_results['scales'] = scale_results

    # 2. Ablation analysis
    print("\n[2/6] Running ablation analysis...")
    ablation = run_ablation("Large", n_runs=N_RUNS)
    all_results['ablation'] = ablation
    print("  done")

    # 3. Sensitivity analysis
    print("\n[3/6] Running parameter sensitivity (η-ρ heatmap)...")
    grid, etas, rhos = run_sensitivity("Large")
    all_results['sensitivity'] = {'grid': grid.tolist(), 'etas': etas.tolist(), 'rhos': rhos.tolist()}
    print("  done")

    # 4. Cascade analysis
    print("\n[4/6] Running reputation cascade analysis...")
    cascade = run_cascade_analysis("Large")
    all_results['cascade'] = cascade
    print("  done")

    # 5. Real-data validation
    print("\n[5/6] Running real-data validation...")
    try:
        csv_path = os.path.join(os.path.dirname(__file__),
                                "Procurement KPI Analysis Dataset.csv")
        rows = load_kaggle_dataset(csv_path)
        suppliers_real, orders_real = calibrate_from_real_data(rows)
        distances_real = generate_distances(len(suppliers_real), len(orders_real))

        real_results = {}
        for method in BENCHMARKS:
            metrics_list = []
            for run in range(min(N_RUNS, 5)):
                import copy
                s = copy.deepcopy(suppliers_real)
                o = copy.deepcopy(orders_real)
                run_rng = np.random.default_rng(BASE_SEED + run)
                metrics = run_simulation(s, o, distances_real, method, run_rng, rho=RHO)
                metrics_list.append(metrics)
            real_results[method] = {
                'reliability_mean': np.mean([m['reliability'] for m in metrics_list]),
                'utility_mean': np.mean([m['utility'] for m in metrics_list]),
                'completion_mean': np.mean([m['completion_rate'] for m in metrics_list]),
            }
        all_results['real_data'] = real_results
        print("  done")
    except Exception as e:
        print(f"  Skipped (data issue): {e}")

    # 6. Save results
    print("\n[6/6] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"results_{timestamp}.json")

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"  Saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Coalition Reliability")
    print("=" * 60)
    header = f"{'Scale':<10} " + " ".join(f"{m:<12}" for m in BENCHMARKS)
    print(header)
    print("-" * len(header))
    for scale_name in SCALES:
        row = f"{scale_name:<10} "
        for method in BENCHMARKS:
            val = scale_results[scale_name][method]['reliability_mean']
            row += f"{val:<12.2f}"
        print(row)

    print("\nDone! All experiments completed.")
    return all_results


if __name__ == "__main__":
    run_all_experiments()
