"""
Benchmark implementations for comparison.
REP-OBLIV, REP-STATIC, REP-NOTOPSIS, REP-NOSTABLE.
"""
import numpy as np
from typing import List, Dict
from model import Supplier, Order, Coalition, individual_utility
from racf import stage1_initial_formation
from racf_stages import stage2_reallocation, stage3_refinement


def run_rep_obliv(suppliers, orders, distances, rho=0.0, rng=None):
    """REP-OBLIV: Greedy matching WITHOUT any reputation influence (R*=0, ρ=0)."""
    if rng is None:
        rng = np.random.default_rng()

    # Remove all reputation influence: admission threshold + utility bonus
    saved_R = [o.R_star for o in orders]
    for o in orders:
        o.R_star = 0.0

    coalitions = stage1_initial_formation(suppliers, orders, distances, rho=0.0, rng=rng)

    # Restore
    for o, r in zip(orders, saved_R):
        o.R_star = r

    return coalitions


def run_rep_static(suppliers, orders, distances, rho=1.5, rng=None):
    """REP-STATIC: Fixed reputation thresholds, no dynamic update."""
    # This is essentially Stage I with frozen reputation
    coalitions = stage1_initial_formation(suppliers, orders, distances, rho, rng)
    return coalitions


def run_rep_notopsis(suppliers, orders, distances, rho=1.5, rng=None):
    """REP-NOTOPSIS: RACF without Stage II."""
    coalitions = stage1_initial_formation(suppliers, orders, distances, rho, rng)
    coalitions = stage3_refinement(coalitions, suppliers, distances, rho)
    return coalitions


def run_rep_nostable(suppliers, orders, distances, rho=1.5, rng=None):
    """REP-NOSTABLE: RACF without Stage III."""
    coalitions = stage1_initial_formation(suppliers, orders, distances, rho, rng)
    coalitions = stage2_reallocation(coalitions, suppliers, distances, rho, rng)
    return coalitions


BENCHMARKS = {
    "REP-OBLIV": run_rep_obliv,
    "REP-STATIC": run_rep_static,
    "REP-NOTOPSIS": run_rep_notopsis,
    "REP-NOSTABLE": run_rep_nostable,
    "RACF": None,  # handled separately
}
