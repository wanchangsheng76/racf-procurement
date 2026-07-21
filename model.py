"""
Core model: Supplier, Order, Coalition, and Reputation Dynamics.
Implements Eqs. (1)-(9) from the paper.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Supplier:
    """A procurement supplier (provider)."""
    m: int                          # Supplier ID
    K: float                        # Production capacity (units/period)
    c: float                        # Unit production cost
    h: float                        # Overhead cost factor
    f: float                        # Fixed participation cost
    tau: float                      # Fixed setup time (hours)
    v: float                        # Logistics speed (km/h)
    # Reputation state
    r: float                        # Current reputation score r_m ∈ [0,1]
    alpha: float                    # Beta-Bayesian prior α
    beta: float                     # Beta-Bayesian prior β
    # Allocation
    assigned_order: Optional[int] = None   # Order ID currently assigned to
    q_allocated: float = 0.0               # Quantity allocated to this supplier

    @property
    def cost_per_unit(self):
        return self.c * (1 + self.h)

    def reset_allocation(self):
        self.assigned_order = None
        self.q_allocated = 0.0


@dataclass
class Order:
    """A procurement order (demander)."""
    n: int                          # Order ID
    Q: float                        # Total procurement quantity
    T: float                        # Delivery deadline (hours)
    p: float                        # Unit price offered
    R_star: float                   # Minimum required coalition reliability
    beta: float                     # Delivery penalty sensitivity
    # Feasibility
    S_max: int = 15                 # Maximum coalition size

    def copy(self):
        return Order(self.n, self.Q, self.T, self.p, self.R_star, self.beta, self.S_max)


@dataclass
class Coalition:
    """A procurement coalition for an order."""
    order: Order
    suppliers: List[Supplier] = field(default_factory=list)
    chi: int = 1                    # Feasibility flag: 1=feasible, 0=infeasible
    quantities: dict = field(default_factory=dict)  # m -> q_nm

    @property
    def size(self):
        return len(self.suppliers)

    @property
    def total_q(self):
        return sum(self.quantities.values())

    def add_supplier(self, m: Supplier, q: float):
        self.suppliers.append(m)
        self.quantities[m.m] = q
        m.assigned_order = self.order.n
        m.q_allocated = q

    def remove_supplier(self, m: Supplier):
        if m in self.suppliers:
            self.suppliers.remove(m)
        self.quantities.pop(m.m, None)
        m.reset_allocation()

    def copy(self):
        return Coalition(
            order=self.order.copy(),
            suppliers=list(self.suppliers),
            chi=self.chi,
            quantities=dict(self.quantities),
        )


# ============================================================
# Production & Delivery Model (Eqs. 2-5)
# ============================================================

def production_time(supplier: Supplier, q: float) -> float:
    """Eq. (2): t_nm^prod = τ_m + q_nm / K_m"""
    return supplier.tau + q / supplier.K


def transportation_time(supplier: Supplier, d_nm: float) -> float:
    """Eq. (3): t_nm^trans = d_nm / v_m"""
    return d_nm / supplier.v


def coalition_completion_time(coalition: Coalition, distances: dict) -> float:
    """Eq. (4): T(C_n) = max_{m∈S_n} {t_nm}"""
    if not coalition.suppliers:
        return float('inf')
    order_dist = distances.get(coalition.order.n, {})
    times = []
    for m in coalition.suppliers:
        t_prod = production_time(m, coalition.quantities[m.m])
        d_nm = order_dist.get(m.m, 100.0) if isinstance(order_dist, dict) else 100.0
        t_trans = transportation_time(m, d_nm)
        times.append(t_prod + t_trans)
    return max(times)


def delivery_discount(coalition: Coalition, distances: dict) -> float:
    """Eq. (5): γ_n(C_n) = exp(-β_n·[T̃(C_n) - T_n]^+)"""
    T_actual = coalition_completion_time(coalition, distances)
    delay = max(T_actual - coalition.order.T, 0)
    return np.exp(-coalition.order.beta * delay)


# ============================================================
# Reputation Dynamics (Eqs. 6-9)
# ============================================================

def evaluation_score(supplier: Supplier, coalition: Coalition,
                     distances: dict, lambdas=(0.5, 0.3, 0.2),
                     subjective_rating: float = None) -> float:
    """Eq. (6-7): Collaborative evaluation score ψ_nm."""
    l1, l2, l3 = lambdas
    # On-time component
    t_prod = production_time(supplier, coalition.quantities[supplier.m])
    order_dist = distances.get(coalition.order.n, {})
    d_nm = order_dist.get(supplier.m, 100.0) if isinstance(order_dist, dict) else 100.0
    t_trans = transportation_time(supplier, d_nm)
    t_total = t_prod + t_trans
    delay = max(t_total - coalition.order.T, 0)
    on_time = np.exp(-coalition.order.beta * delay)

    # Contribution share
    x_nm = coalition.quantities[supplier.m] / coalition.order.Q

    # Subjective rating
    if subjective_rating is None:
        subjective_rating = np.random.uniform(0.4, 1.0)

    return l1 * on_time + l2 * x_nm + l3 * subjective_rating


def beta_bayesian_update(supplier: Supplier, psi: float) -> float:
    """Eq. (8): α_m ← α_m + ψ, β_m ← β_m + (1-ψ), r̂_m = α_m/(α_m+β_m)"""
    supplier.alpha += psi
    supplier.beta += (1 - psi)
    return supplier.alpha / (supplier.alpha + supplier.beta)


def exponential_smoothing(supplier: Supplier, r_hat: float, eta: float = 0.3) -> float:
    """Eq. (9): r_m^new = (1-η)·r_m^old + η·r̂_m"""
    supplier.r = (1 - eta) * supplier.r + eta * r_hat
    return supplier.r


def update_reputation(supplier: Supplier, coalition: Coalition,
                      distances: dict, eta: float = 0.3,
                      lambdas=(0.5, 0.3, 0.2)) -> float:
    """Full reputation update cycle: ψ → Beta-Bayesian → Exponential Smoothing."""
    psi = evaluation_score(supplier, coalition, distances, lambdas)
    r_hat = beta_bayesian_update(supplier, psi)
    return exponential_smoothing(supplier, r_hat, eta)


def coalition_reputation(coalition: Coalition) -> float:
    """Eq. (10): R_n = Σ x_nm · r_m"""
    if not coalition.suppliers:
        return 0.0
    Q_total = coalition.order.Q
    return sum((coalition.quantities[m.m] / Q_total) * m.r
               for m in coalition.suppliers)


# ============================================================
# Utility Functions (Eqs. 11-13)
# ============================================================

def individual_utility(supplier: Supplier, order: Order,
                       q: float, rho: float = 1.5) -> float:
    """Eq. (12): u_m(C_n) = [p_n·(1+ρ·r_m) - c_m·(1+h_m)]·q_nm - f_m"""
    revenue = order.p * (1 + rho * supplier.r) * q
    cost = supplier.cost_per_unit * q + supplier.f
    return revenue - cost


def coalition_utility(coalition: Coalition, distances: dict) -> float:
    """Eq. (13): W(C_n) = p_n·Q_n·γ_n(C_n) - Σ C_mn"""
    if not coalition.suppliers:
        return 0.0
    gamma = delivery_discount(coalition, distances)
    revenue = coalition.order.p * coalition.order.Q * gamma
    total_cost = sum(m.cost_per_unit * coalition.quantities[m.m] + m.f
                     for m in coalition.suppliers)
    return revenue - total_cost


def is_feasible(coalition: Coalition, distances: dict) -> bool:
    """Definition 4: Coalition feasibility check."""
    if not coalition.suppliers:
        return False
    # (i) T(C_n) ≤ T_n
    T = coalition_completion_time(coalition, distances)
    if T > coalition.order.T:
        return False
    # (ii) r_m ≥ R_n* for all m
    for m in coalition.suppliers:
        if m.r < coalition.order.R_star:
            return False
    # (iii) |S_n| ≤ S_n^max
    if len(coalition.suppliers) > coalition.order.S_max:
        return False
    # (iv) u_m(C_n) ≥ 0 for all m
    for m in coalition.suppliers:
        if individual_utility(m, coalition.order, coalition.quantities[m.m]) < 0:
            return False
    return True
