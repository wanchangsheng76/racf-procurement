"""Two-basin trajectory experiment and cascade counterfactual."""
import numpy as np
import copy
from config import *
from model import *
from data_loader import generate_suppliers, generate_orders, generate_distances
from racf_stages import run_racf

N, M = 50, 150
N_CYCLES = 20
RNG = np.random.default_rng(BASE_SEED)

suppliers = generate_suppliers(M, RNG)
orders_template = generate_orders(N, RNG)
distances = generate_distances(M, N, RNG)

print("=== EXPERIMENT 1: Two-Basin Trajectory (R*=0.6) ===")
R_val = 0.6
s_test = copy.deepcopy(suppliers)
orders = [Order(n=i, Q=o.Q, T=o.T, p=o.p, R_star=R_val, beta=o.beta)
          for i, o in enumerate(orders_template)]

# Track specific suppliers
track_ids = []
track_trajectories = {}
for i, m in enumerate(s_test):
    if 0.55 < m.r < 0.65:  # near threshold
        track_ids.append(i)
        track_trajectories[i] = []
        if len(track_ids) >= 5:
            break

for cycle in range(N_CYCLES):
    coalitions = run_racf(s_test, orders, distances, rho=RHO, rng=RNG)
    for n, c in coalitions.items():
        if c.suppliers:
            for m in c.suppliers:
                update_reputation(m, c, distances, eta=ETA)
    for m in s_test:
        if m.assigned_order is None:
            m.alpha += 0.5; m.beta += 0.5
            r_hat = m.alpha / (m.alpha + m.beta)
            m.r = (1 - ETA) * m.r + ETA * r_hat
    for tid in track_ids:
        track_trajectories[tid].append(round(s_test[tid].r, 4))

print("Trajectories of suppliers near R*=0.6:")
for tid in track_ids:
    traj = track_trajectories[tid]
    status = "EXCLUDED" if traj[-1] < R_val else "INCLUDED"
    print(f"  Supplier {tid}: {traj[0]:.4f} -> {traj[-1]:.4f} [{status}]")
    print(f"    Full: {[f'{x:.3f}' for x in traj]}")

# Count converging to 0.5
converged = sum(1 for tid in track_ids if abs(track_trajectories[tid][-1] - 0.5) < 0.05)
print(f"  {converged}/{len(track_ids)} converge to ~0.5")

print("\n=== EXPERIMENT 2: Cascade Counterfactual (R*: 0.6 -> 0.4) ===")
s_test2 = copy.deepcopy(suppliers)
orders2_high = [Order(n=i, Q=o.Q, T=o.T, p=o.p, R_star=0.6, beta=o.beta)
                for i, o in enumerate(orders_template)]
orders2_low = [Order(n=i, Q=o.Q, T=o.T, p=o.p, R_star=0.4, beta=o.beta)
               for i, o in enumerate(orders_template)]

# Find suppliers excluded at R*=0.6
initial_reps = [m.r for m in s_test2]
for cycle in range(10):
    coalitions = run_racf(s_test2, orders2_high, distances, rho=RHO, rng=RNG)
    for n, c in coalitions.items():
        if c.suppliers:
            for m in c.suppliers:
                update_reputation(m, c, distances, eta=ETA)
    for m in s_test2:
        if m.assigned_order is None:
            m.alpha += 0.5; m.beta += 0.5
            r_hat = m.alpha / (m.alpha + m.beta)
            m.r = (1 - ETA) * m.r + ETA * r_hat

after_high = [m.r for m in s_test2]
excluded_ids = [i for i in range(M) if after_high[i] < 0.6 and initial_reps[i] >= 0.6]
print(f"  After 10 cycles at R*=0.6: {len(excluded_ids)} suppliers excluded")

# Track recovery of excluded suppliers
recovery_traj = {i: [after_high[i]] for i in excluded_ids[:5]}
for cycle in range(10):
    coalitions = run_racf(s_test2, orders2_low, distances, rho=RHO, rng=RNG)
    for n, c in coalitions.items():
        if c.suppliers:
            for m in c.suppliers:
                update_reputation(m, c, distances, eta=ETA)
    for m in s_test2:
        if m.assigned_order is None:
            m.alpha += 0.5; m.beta += 0.5
            r_hat = m.alpha / (m.alpha + m.beta)
            m.r = (1 - ETA) * m.r + ETA * r_hat
    for i in excluded_ids[:5]:
        recovery_traj[i].append(round(s_test2[i].r, 4))

recovered = sum(1 for i in excluded_ids[:5] if recovery_traj[i][-1] >= 0.4)
print(f"  After 10 cycles at R*=0.4 (lowered): {recovered}/{min(5, len(excluded_ids))} excluded suppliers recovered")
for i in excluded_ids[:5]:
    t = recovery_traj[i]
    print(f"    Supplier {i}: {t[0]:.4f} -> ... -> {t[-1]:.4f} [{'RECOVERED' if t[-1]>=0.4 else 'STILL EXCLUDED'}]")

# Format for LaTeX table
print("\n=== FOR LaTeX ===")
print("Two-basin trajectories (R*=0.6):")
for tid in track_ids[:5]:
    traj = track_trajectories[tid]
    r0 = traj[0]; r5 = traj[4] if len(traj)>4 else traj[-1]; r10 = traj[9] if len(traj)>9 else traj[-1]; r20 = traj[-1]
    print(f"  {r0:.3f} & {r5:.3f} & {r10:.3f} & {r20:.3f} \\\\")

print("\nCascade counterfactual (R* 0.6->0.4):")
for i in excluded_ids[:5]:
    t = recovery_traj[i]
    print(f"  {t[0]:.3f} & {t[-1]:.3f} \\\\")

print("\nDone.")
