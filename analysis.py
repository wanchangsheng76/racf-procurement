"""
Analysis and visualization module.
Statistical tests, bootstrap CIs, and plots.
"""
import numpy as np
import json
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def bootstrap_ci(data, n_bootstrap=1000, alpha=0.01, seed=42):
    """Compute bootstrap confidence interval."""
    data = np.array(data)
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(data, size=len(data), replace=True))
                      for _ in range(n_bootstrap)])
    lo = np.percentile(means, alpha / 2 * 100)
    hi = np.percentile(means, (1 - alpha / 2) * 100)
    return np.mean(data), (lo, hi)


def welch_ttest(data1, data2):
    """Welch's two-sample t-test."""
    n1, n2 = len(data1), len(data2)
    m1, m2 = np.mean(data1), np.mean(data2)
    v1, v2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    se = np.sqrt(v1/n1 + v2/n2)
    if se == 0:
        return 0, 1.0
    t = (m1 - m2) / se
    # Welch-Satterthwaite degrees of freedom
    df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
    from scipy import stats
    p = 2 * stats.t.sf(abs(t), df)
    return t, p


def plot_reliability_comparison(results, output_path="fig_reliability.png"):
    """Plot coalition reliability comparison across scales."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plot")
        return

    scales = list(results['scales'].keys())
    methods = list(results['scales'][scales[0]].keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(scales))
    width = 0.15

    for i, method in enumerate(methods):
        vals = [results['scales'][s][method]['reliability'] for s in scales]
        ax.bar(x + i * width, vals, width, label=method)

    ax.set_xlabel('Scale')
    ax.set_ylabel('Coalition Reliability')
    ax.set_title('Coalition Reliability Across Scales')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(scales)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_sensitivity_heatmap(results, output_path="fig_sensitivity.png"):
    """Plot η-ρ sensitivity heatmap."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plot")
        return

    data = results.get('sensitivity', {})
    if not data:
        return
    grid = np.array(data['grid'])
    etas = np.array(data['etas'])
    rhos = np.array(data['rhos'])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(rhos, etas, grid, shading='auto', cmap='RdYlGn')
    ax.set_xlabel('ρ (Incentive Coefficient)')
    ax.set_ylabel('η (Smoothing Factor)')
    ax.set_title('RACF Relative Improvement (Coalition Reliability)')
    plt.colorbar(im, ax=ax, label='Reliability')
    # Mark optimal
    opt_idx = np.unravel_index(np.argmax(grid), grid.shape)
    ax.plot(rhos[opt_idx[1]], etas[opt_idx[0]], 'k*', markersize=15,
            label=f'Optimal (η={etas[opt_idx[0]]:.2f}, ρ={rhos[opt_idx[1]]:.2f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_ablation(results, output_path="fig_ablation.png"):
    """Plot ablation analysis."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plot")
        return

    ablation = results.get('ablation', {})
    if not ablation:
        return

    names = list(ablation.keys())
    util_loss = [ablation[n]['utility_loss'] for n in names]
    rel_loss = [ablation[n]['reliability_loss'] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, util_loss, width, label='Utility Loss (%)', color='steelblue')
    ax.bar(x + width/2, rel_loss, width, label='Reliability Loss (%)', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Loss (%)')
    ax.set_title('Ablation Analysis (Large Scale)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def generate_report(results_path, output_dir="results"):
    """Generate a report from saved results."""
    with open(results_path, 'r') as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    if HAS_MPL:
        plot_reliability_comparison(results, os.path.join(output_dir, "fig_reliability.png"))
        plot_ablation(results, os.path.join(output_dir, "fig_ablation.png"))
        plot_sensitivity_heatmap(results, os.path.join(output_dir, "fig_sensitivity.png"))

    # Text report
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, 'w') as f:
        f.write("RACF Experiment Results\n")
        f.write("=" * 60 + "\n\n")

        if 'scales' in results:
            f.write("Coalition Reliability by Scale:\n")
            scales = list(results['scales'].keys())
            methods = list(results['scales'][scales[0]].keys())
            f.write(f"{'Scale':<12}" + "".join(f"{m:<14}" for m in methods) + "\n")
            f.write("-" * (12 + 14 * len(methods)) + "\n")
            for s in scales:
                row = f"{s:<12}"
                for m in methods:
                    row += f"{results['scales'][s][m]['reliability']:<14.4f}"
                f.write(row + "\n")

        if 'ablation' in results:
            f.write("\nAblation Analysis:\n")
            for name, vals in results['ablation'].items():
                f.write(f"  {name}: Util={vals['utility_loss']:.1f}%, Rel={vals['reliability_loss']:.1f}%\n")

    print(f"Report saved to {report_path}")
    return report_path


if __name__ == "__main__":
    import glob
    results_files = glob.glob("results/results_multicycle_*.json")
    if results_files:
        latest = max(results_files, key=os.path.getmtime)
        print(f"Generating report from {latest}")
        generate_report(latest)
    else:
        print("No results files found")
