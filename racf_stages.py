"""
Stage II: TOPSIS-Based Priority Reallocation.
Stage III: Stability-Driven Refinement.
"""
import numpy as np
from typing import List, Dict
from model import Supplier, Order, Coalition
from model import individual_utility, coalition_utility, is_feasible
from model import coalition_completion_time
from config import TOPSIS_WEIGHTS
from racf import stage1_initial_formation


# ============================================================
# Stage II: TOPSIS-Based Priority Reallocation
# ============================================================

def _zeta(coalition: Coalition) -> float:
    """Capacity gap (Eq. 15)."""
    if not coalition.suppliers:
        return 1.0
    total_cap = sum(m.K * coalition.order.T / coalition.order.Q for m in coalition.suppliers)
    return max(1 - total_cap, 0)


def topsis_priority(coalitions: Dict[int, Coalition], eps=1e-6):
    """TOPSIS ranking for infeasible coalitions (Eqs. 16-18)."""
    infeasible = {n: c for n, c in coalitions.items() if c.chi == 0 and c.suppliers}
    if not infeasible:
        return {}

    n_ids = list(infeasible.keys())
    # Feature vectors
    A = np.zeros((len(n_ids), 3))
    for i, n in enumerate(n_ids):
        c = infeasible[n]
        A[i, 0] = c.order.R_star                     # reputation baseline
        A[i, 1] = 1.0 / (c.order.T + eps)            # deadline urgency
        A[i, 2] = 1.0 - _zeta(c)                     # reparability

    # TOPSIS
    norm = np.sqrt((A ** 2).sum(axis=0)) + eps
    B = A / norm
    C = B * np.array(TOPSIS_WEIGHTS)
    c_plus = C.max(axis=0)
    c_minus = C.min(axis=0)
    d_plus = np.sqrt(((C - c_plus) ** 2).sum(axis=1))
    d_minus = np.sqrt(((C - c_minus) ** 2).sum(axis=1))
    pi = d_minus / (d_plus + d_minus + eps)

    return {n_ids[i]: pi[i] for i in range(len(n_ids))}


def stage2_reallocation(coalitions, suppliers, distances, rho=1.5, rng=None):
    """Stage II: TOPSIS-based priority reallocation."""
    if rng is None:
        rng = np.random.default_rng()

    # Collect all unassigned suppliers
    unassigned = [m for m in suppliers if m.assigned_order is None]
    if not unassigned:
        return coalitions

    for _ in range(len(unassigned)):
        priorities = topsis_priority(coalitions)
        if not priorities:
            break

        # Try ALL infeasible orders in TOPSIS priority order
        ordered = sorted(priorities, key=priorities.get, reverse=True)
        repaired = False

        for n_target in ordered:
            c_target = coalitions[n_target]

            # Find best feasible supplier for this coalition
            best_s = None
            best_u = -float('inf')
            for s in unassigned:
                if s.r < c_target.order.R_star:
                    continue
                if len(c_target.suppliers) >= c_target.order.S_max:
                    continue
                # Pre-check: would coalition be feasible after adding?
                trial_q = c_target.order.Q / (len(c_target.suppliers) + 1)
                trial_suppliers = list(c_target.suppliers) + [s]
                # Quick feasibility estimate: check individual rationality
                u = individual_utility(s, c_target.order, trial_q, rho)
                if u < 0:
                    continue
                if u > best_u:
                    best_u = u
                    best_s = s

            if best_s is not None:
                q = c_target.order.Q / (len(c_target.suppliers) + 1)
                for s_existing in c_target.suppliers:
                    s_existing.q_allocated = q
                    c_target.quantities[s_existing.m] = q
                c_target.add_supplier(best_s, q)
                unassigned.remove(best_s)
                c_target.chi = 1 if is_feasible(c_target, distances) else 0
                repaired = True
                break  # one repair per outer iteration, then reprioritize

        if not repaired:
            break  # no infeasible order can be repaired with remaining unassigned

    return coalitions


# ============================================================
# Stage III: Stability-Driven Refinement
# ============================================================

def _redistribute(coalition):
    """Evenly redistribute order quantity among current suppliers."""
    if not coalition.suppliers:
        coalition.quantities = {}
        return
    q = coalition.order.Q / len(coalition.suppliers)
    coalition.quantities = {}
    for s in coalition.suppliers:
        coalition.quantities[s.m] = q
        s.q_allocated = q
        s.assigned_order = coalition.order.n


def _try_switch(m, from_n, to_n, coalitions, distances, rho=1.5):
    """Try to switch supplier m from coalition from_n to to_n.
    Checks all three Definition~3 conditions: feasibility, utility, welfare."""
    c_from = coalitions[from_n]
    c_to = coalitions[to_n]

    # Pre-checks (cheap filters)
    if m.r < c_to.order.R_star:
        return False
    if len(c_to.suppliers) >= c_to.order.S_max:
        return False

    # Compute utility change
    q_to = c_to.order.Q / (len(c_to.suppliers) + 1)
    u_new = individual_utility(m, c_to.order, q_to, rho)
    q_from = c_from.quantities.get(m.m, 0)
    u_old = individual_utility(m, c_from.order, q_from, rho) if from_n >= 0 else 0
    if u_new <= u_old:
        return False

    # Save old state
    old_from_sup = list(c_from.suppliers)
    old_to_sup = list(c_to.suppliers)
    w_old = coalition_utility(c_from, distances) + coalition_utility(c_to, distances)

    # Apply tentative switch with unified redistribution
    c_from.remove_supplier(m)
    c_to.add_supplier(m, q_to)
    _redistribute(c_from)
    _redistribute(c_to)

    # Check feasibility (Definition~3 condition i)
    feasible = True
    if c_from.suppliers and not is_feasible(c_from, distances):
        feasible = False
    if not is_feasible(c_to, distances):
        feasible = False

    if feasible:
        w_new = (coalition_utility(c_from, distances) if c_from.suppliers else 0) + \
                (coalition_utility(c_to, distances) if c_to.suppliers else 0)
        accepted = (w_new > w_old)
    else:
        accepted = False

    # Restore old state
    c_from.suppliers = old_from_sup
    c_to.suppliers = old_to_sup
    _redistribute(c_from)
    _redistribute(c_to)

    return accepted


def _apply_switch(m, from_n, to_n, coalitions):
    """Apply the switch using unified redistribution."""
    c_from = coalitions[from_n]
    c_to = coalitions[to_n]
    q_to = c_to.order.Q / (len(c_to.suppliers) + 1)
    c_from.remove_supplier(m)
    c_to.add_supplier(m, q_to)
    _redistribute(c_from)
    _redistribute(c_to)


def stage3_refinement(coalitions, suppliers, distances, rho=1.5, max_iter=500):
    """Stage III: Stability-driven refinement via switching."""
    for iteration in range(max_iter):
        improved = False
        assigned = [m for m in suppliers if m.assigned_order is not None]

        for m in assigned:
            from_n = m.assigned_order
            for to_n in coalitions:
                if to_n == from_n:
                    continue
                if _try_switch(m, from_n, to_n, coalitions, distances, rho):
                    _apply_switch(m, from_n, to_n, coalitions)
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    # Update feasibility flags
    for n, c in coalitions.items():
        c.chi = 1 if is_feasible(c, distances) else 0

    return coalitions


def run_racf(suppliers, orders, distances, rho=1.5, rng=None):
    """Run the full RACF three-stage mechanism."""
    if rng is None:
        rng = np.random.default_rng()

    coalitions = stage1_initial_formation(suppliers, orders, distances, rho, rng)
    coalitions = stage2_reallocation(coalitions, suppliers, distances, rho, rng)
    coalitions = stage3_refinement(coalitions, suppliers, distances, rho)

    return coalitions
