"""Quick test: verify eta/rho parameters now propagate correctly."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, copy
from config import BASE_SEED
from experiments_multicycle import run_multicycle
from data_loader import generate_suppliers, generate_orders, generate_distances

N, M = 50, 150
etas = [0.05, 0.30, 0.50, 0.70, 0.95]
rhos = [0.25, 1.0, 1.5, 2.0, 3.0]
rng = np.random.default_rng(BASE_SEED)
suppliers = generate_suppliers(M, rng)
orders = generate_orders(N, rng)
distances = generate_distances(M, N, rng)

print("eta/rho sensitivity test (5x5, 3 cycles each):")
header = "eta\\rho".rjust(8)
for r in rhos:
    header += f"{r:>9}"
print(header)
print("-" * (8 + 9*5))

for eta in etas:
    row = f"{eta:>8.2f}"
    for rho in rhos:
        s = copy.deepcopy(suppliers)
        o = copy.deepcopy(orders)
        metrics, _ = run_multicycle(s, o, distances, "RACF",
                                    n_cycles=3, rng=rng, eta=eta, rho=rho)
        row += f"{metrics['reliability']:>9.4f}"
    print(row)
print("\nDone.")
