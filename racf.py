"""
RACF Three-Stage Coalition Formation Mechanism.
Implements Algorithm 1 and Stages I-III.
"""
import numpy as np
from typing import List, Dict
from model import Supplier, Order, Coalition
from model import individual_utility, coalition_utility, is_feasible


def stage1_initial_formation(suppliers, orders, distances, rho=1.5, rng=None):
    """Algorithm 1: Reputation-filtered greedy matching."""
    if rng is None:
        rng = np.random.default_rng()

    coalitions = {n: Coalition(order=ord) for n, ord in enumerate(orders)}
    free = list(suppliers)
    rng.shuffle(free)

    for m in free:
        m.reset_allocation()
        A_m = []
        for n, coalition in coalitions.items():
            order = orders[n]
            if m.r < order.R_star:
                continue
            if len(coalition.suppliers) >= order.S_max:
                continue
            q_each = order.Q / max(len(coalition.suppliers) + 1, 1)
            u = individual_utility(m, order, q_each, rho)
            if u >= 0:
                A_m.append((n, u))

        if A_m:
            n_best = max(A_m, key=lambda x: x[1])[0]
            order = orders[n_best]
            existing = len(coalitions[n_best].suppliers) + 1
            q = order.Q / existing
            # Redistribute quantities equally
            for s in coalitions[n_best].suppliers:
                s.q_allocated = q
                coalitions[n_best].quantities[s.m] = q
            coalitions[n_best].add_supplier(m, q)

    # Mark feasibility
    for n, c in coalitions.items():
        c.chi = 1 if is_feasible(c, distances) else 0

    return coalitions
