"""
Configuration parameters for RACF experiments.
All values match the paper's Section 6 (Computational Experiments).
"""

import numpy as np

# ============================================================
# Problem Scales (Table 1)
# ============================================================
SCALES = {
    "Small":    {"N": 10,  "M": 30,  "coalition_size": 5},
    "Medium":   {"N": 25,  "M": 75,  "coalition_size": 7},
    "Large":    {"N": 50,  "M": 150, "coalition_size": 10},
    "X-Large":  {"N": 100, "M": 300, "coalition_size": 12},
    "Massive":  {"N": 200, "M": 600, "coalition_size": 15},
}

# ============================================================
# Supplier Parameters (calibrated to industry benchmarks)
# ============================================================
SUPPLIER_PARAMS = {
    "capacity_range": (10, 100),       # K_m: units/period
    "cost_range": (5, 50),             # c_m: unit production cost
    "overhead_range": (0.05, 0.30),    # h_m: overhead factor
    "fixed_cost_range": (10, 100),     # f_m: fixed participation cost
    "setup_time_range": (1, 8),        # τ_m: hours
    "logistics_speed_range": (40, 80), # v_m: km/h
    "initial_reputation_alpha": 2.0,   # Beta(2,2) prior for new suppliers
    "initial_reputation_beta": 2.0,
}

# ============================================================
# Order Parameters
# ============================================================
ORDER_PARAMS = {
    "quantity_range": (50, 500),       # Q_n: units
    "deadline_range": (48, 168),       # T_n: hours
    "unit_price_range": (8, 60),       # p_n: offered unit price
    "beta_range": (0.01, 0.05),        # β_n: delivery penalty sensitivity
    "distance_range": (50, 500),       # d_nm: km
    "subjective_rating_range": (0.4, 1.0),  # v_nm
}

# ============================================================
# Key Mechanism Parameters (Section 6.1)
# ============================================================
ETA = 0.3           # Reputation smoothing factor (empirically calibrated)
RHO = 1.5           # Reputation incentive coefficient
LAMBDA = (0.5, 0.3, 0.2)  # λ₁,λ₂,λ₃ for evaluation score
TOPSIS_WEIGHTS = (0.40, 0.35, 0.25)  # ω for Stage II

# ============================================================
# Reputation threshold defaults
# ============================================================
DEFAULT_R_STAR = 0.5  # Default admission threshold when not specified

# ============================================================
# Simulation & Statistical Parameters
# ============================================================
N_RUNS = 20              # Independent simulation runs per config
N_BOOTSTRAP = 1000       # Bootstrap resamples
SIGNIFICANCE_LEVEL = 0.01

# ============================================================
# Sensitivity Analysis Ranges (Section 6.3)
# ============================================================
ETA_RANGE = np.linspace(0.05, 0.95, 19)   # 0.05 to 0.95
RHO_RANGE = np.linspace(0.25, 3.0, 12)    # 0.25 to 3.0
LOW_REP_FRAC_RANGE = np.linspace(0.10, 0.60, 6)  # 10% to 60%

# ============================================================
# Cascade Analysis Parameters
# ============================================================
CASCADE_CYCLES = 10
CASCADE_R_STAR_RANGE = [0.40, 0.50, 0.60, 0.70]

# ============================================================
# Random Seed
# ============================================================
BASE_SEED = 42

# ============================================================
# Output
# ============================================================
OUTPUT_DIR = "results"
