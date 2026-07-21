"""Generate Fig 6 sensitivity heatmap with reduced grid (11×7, 3 cycles)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import BASE_SEED, ETA, RHO
from experiments_multicycle import run_multicycle
from data_loader import generate_suppliers, generate_orders, generate_distances

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..')

print("Generating Fig 6 sensitivity heatmap...")
N, M = 50, 150
rng = np.random.default_rng(BASE_SEED)
suppliers = generate_suppliers(M, rng)
orders = generate_orders(N, rng)
distances = generate_distances(M, N, rng)

etas = np.linspace(0.05, 0.95, 11)   # 11 points
rhos = np.linspace(0.25, 3.0, 7)     # 7 points (11×7 = 77 grid)
grid = np.zeros((len(etas), len(rhos)))

for i, eta in enumerate(etas):
    for j, rho_iter in enumerate(rhos):
        s_test = copy.deepcopy(suppliers)
        o_test = copy.deepcopy(orders)
        metrics, _ = run_multicycle(s_test, o_test, distances, "RACF",
                                    n_cycles=3, rng=rng, eta=eta, rho=rho_iter)
        grid[i, j] = metrics['reliability']
        print(f"  eta={eta:.2f}, rho={rho_iter:.2f}: rel={metrics['reliability']:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(7.5, 5.5))
im = ax.pcolormesh(rhos, etas, grid, shading='auto', cmap='RdYlGn', vmin=0.45, vmax=0.70)
cbar = plt.colorbar(im, ax=ax, label='Coalition Reliability', shrink=0.85)
cbar.ax.tick_params(labelsize=8)

opt_idx = np.unravel_index(np.argmax(grid), grid.shape)
ax.plot(rhos[opt_idx[1]], etas[opt_idx[0]], 'k*', markersize=14,
        markeredgewidth=0.5, markeredgecolor='white',
        label=f'Best ($\\eta$={etas[opt_idx[0]]:.2f}, $\\rho$={rhos[opt_idx[1]]:.2f})')

ax.axhline(y=0.30, color='blue', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(0.35, 0.32, '$\\eta{=}0.3$ (paper calibration)', fontsize=8, color='blue', va='bottom')
ax.set_xlabel('Incentive coefficient $\\rho$', fontsize=11)
ax.set_ylabel('Smoothing factor $\\eta$', fontsize=11)
ax.set_title('RACF Coalition Reliability: $\\eta$-$\\rho$ Sensitivity\n(Large scale, N=50, M=150, 3-cycle feedback)', fontsize=11)
ax.legend(fontsize=8, loc='lower right')
ax.tick_params(labelsize=9)

plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, 'fig6_sensitivity_heatmap.pdf')
fig.savefig(outpath, dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"\nSaved {outpath}")
print(f"Best: eta={etas[opt_idx[0]]:.2f}, rho={rhos[opt_idx[1]]:.2f}, rel={grid[opt_idx]:.4f}")
