"""
BOAMP (French Public Procurement) data loader and supplier calibration.
Replaces Kaggle dataset. Uses 1.16M contracts across 294K suppliers.
"""
import csv
import numpy as np
from collections import defaultdict
from model import Supplier, Order
from config import SUPPLIER_PARAMS, ORDER_PARAMS


def load_boamp_dataset(filepath: str, max_rows: int = None):
    """Load the BeauAMP full dataset."""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append(row)
    return rows


def compute_boamp_supplier_stats(rows, min_contracts=3):
    """Compute per-supplier statistics from BOAMP data.
    
    Reputation proxy: supplier's win frequency relative to peers.
    A supplier who wins many contracts consistently = higher reputation.
    """
    suppliers = defaultdict(lambda: {
        'contracts': 0, 'total_value': 0, 'years': set(),
        'categories': set(), 'buyers': set(), 'first_year': 9999, 'last_year': 0
    })
    
    for row in rows:
        siret = row.get('WIN_SIRET', '').strip()
        name = row.get('WIN_STATED_NAME', '').strip()
        if not siret or not name:
            continue
        
        s = suppliers[siret]
        s['name'] = name
        s['contracts'] += 1
        
        # Price
        try:
            price = float(row.get('AWARD_PRICE', '0') or 0)
            s['total_value'] += price
        except:
            pass
        
        # Year
        date_str = row.get('AWARD_DATE', '')
        year = date_str[:4] if date_str else ''
        if year and year.isdigit():
            y = int(year)
            if y >= 2000:  # Filter out bad dates
                s['years'].add(y)
                s['first_year'] = min(s['first_year'], y)
                s['last_year'] = max(s['last_year'], y)
        
        # Category
        cpv = row.get('CPV', '').strip()[:2]
        if cpv:
            s['categories'].add(cpv)
        
        # Buyer (using SIREN as proxy if BUYER_STATED_NAME is missing)
        buyer = row.get('CAE_SIREN', '').strip()
        if buyer:
            s['buyers'].add(buyer)
    
    # Filter suppliers with enough data
    qualified = {k: v for k, v in suppliers.items() if v['contracts'] >= min_contracts}
    return qualified


def compute_reputation_scores(supplier_stats, rng=None):
    """Compute reputation scores from supplier win statistics.
    
    Reputation r_m is computed as a normalized score based on:
    - Win frequency (contracts per active year)
    - Total contract value (log scale, normalized)
    - Buyer diversity (number of unique buyers)
    - Category breadth (number of CPV categories)
    
    Final score is a weighted combination, mapped to [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Compute raw metrics
    metrics = {}
    for siret, s in supplier_stats.items():
        active_years = max(s['last_year'] - s['first_year'] + 1, 1)
        freq = s['contracts'] / active_years
        avg_value = s['total_value'] / max(s['contracts'], 1)
        buyer_count = len(s['buyers'])
        cat_count = len(s['categories'])
        
        metrics[siret] = {
            'freq': freq,
            'avg_value': np.log1p(avg_value),
            'buyer_count': buyer_count,
            'cat_count': cat_count,
            'contracts': s['contracts'],
        }
    
    # Normalize to [0, 1] using percentiles
    all_freq = [m['freq'] for m in metrics.values()]
    all_val = [m['avg_value'] for m in metrics.values()]
    all_buy = [m['buyer_count'] for m in metrics.values()]
    all_cat = [m['cat_count'] for m in metrics.values()]
    
    def percentile_score(vals, x):
        """Map x to its percentile in vals, scaled to [0.1, 0.95]."""
        if len(vals) <= 1:
            return 0.5
        rank = sum(1 for v in vals if v < x) / len(vals)
        return 0.1 + 0.85 * rank  # Scale to [0.1, 0.95]
    
    # Compute scores
    scores = {}
    for siret, m in metrics.items():
        s_freq = percentile_score(all_freq, m['freq'])
        s_val = percentile_score(all_val, m['avg_value'])
        s_buy = percentile_score(all_buy, m['buyer_count'])
        s_cat = percentile_score(all_cat, m['cat_count'])
        
        # Weighted combination: emphasize frequency and value
        score = 0.35 * s_freq + 0.30 * s_val + 0.20 * s_buy + 0.15 * s_cat
        scores[siret] = np.clip(score, 0.05, 0.95)
    
    return scores


def calibrate_from_boamp(filepath, n_suppliers=150, n_orders=50, rng=None):
    """Calibrate supplier pool and orders from BOAMP data."""
    if rng is None:
        rng = np.random.default_rng()
    
    print(f"  Loading BOAMP data...")
    rows = load_boamp_dataset(filepath)
    print(f"  {len(rows):,} contracts loaded")
    
    print(f"  Computing supplier statistics...")
    stats = compute_boamp_supplier_stats(rows, min_contracts=3)
    print(f"  {len(stats):,} suppliers with >=3 contracts")
    
    print(f"  Computing reputation scores...")
    scores = compute_reputation_scores(stats, rng)
    
    # Select top N suppliers by score for diversity
    sorted_suppliers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select suppliers across score distribution (top, middle, bottom)
    n_top = n_suppliers // 2
    n_mid = n_suppliers // 4
    n_low = n_suppliers - n_top - n_mid
    
    selected = []
    selected.extend(sorted_suppliers[:n_top])
    mid_start = len(sorted_suppliers) // 2 - n_mid // 2
    selected.extend(sorted_suppliers[mid_start:mid_start + n_mid])
    low_start = max(0, len(sorted_suppliers) - n_low - 10)
    selected.extend(sorted_suppliers[low_start:low_start + n_low])
    
    # Build Supplier objects
    suppliers = []
    siret_to_idx = {}
    for i, (siret, score) in enumerate(selected):
        s = stats[siret]
        m = Supplier(
            m=i,
            K=rng.uniform(*SUPPLIER_PARAMS['capacity_range']),
            c=rng.uniform(*SUPPLIER_PARAMS['cost_range']),
            h=rng.uniform(*SUPPLIER_PARAMS['overhead_range']),
            f=rng.uniform(*SUPPLIER_PARAMS['fixed_cost_range']),
            tau=rng.uniform(*SUPPLIER_PARAMS['setup_time_range']),
            v=rng.uniform(*SUPPLIER_PARAMS['logistics_speed_range']),
            r=score,
            alpha=SUPPLIER_PARAMS['initial_reputation_alpha'],
            beta=SUPPLIER_PARAMS['initial_reputation_beta'],
        )
        suppliers.append(m)
        siret_to_idx[siret] = i
    
    # Calibrate orders from price distribution
    all_prices = []
    for row in rows:
        try:
            p = float(row.get('AWARD_PRICE', '0') or 0)
            if 1000 < p < 1e9:
                all_prices.append(p)
        except:
            pass
    
    all_prices = np.array(all_prices)
    log_prices = np.log10(all_prices)
    p_mean, p_std = np.mean(log_prices), np.std(log_prices)
    
    orders = []
    for i in range(n_orders):
        log_p = rng.normal(p_mean, p_std)
        price = min(max(10 ** log_p / 100, 1), 200)  # Scale to unit price, cap at 200
        order = Order(
            n=i,
            Q=rng.uniform(*ORDER_PARAMS['quantity_range']),
            T=rng.uniform(*ORDER_PARAMS['deadline_range']),
            p=max(price, 1),
            R_star=rng.uniform(0.3, 0.7),
            beta=rng.uniform(*ORDER_PARAMS['beta_range']),
        )
        orders.append(order)
    
    print(f"  Calibrated {len(suppliers)} suppliers, {len(orders)} orders")
    print(f"  Reputation range: [{min(scores.values()):.3f}, {max(scores.values()):.3f}]")
    print(f"  Reputation mean: {np.mean(list(scores.values())):.3f}")
    
    return suppliers, orders, stats, scores


def get_boamp_temporal_split(filepath, calibrate_years=(2015, 2019), test_years=(2020, 2023)):
    """Split BOAMP data by year for temporal validation."""
    rows = load_boamp_dataset(filepath)
    
    cal_rows = []
    test_rows = []
    for row in rows:
        date_str = row.get('AWARD_DATE', '')
        year_str = date_str[:4]
        if year_str.isdigit():
            y = int(year_str)
            if calibrate_years[0] <= y <= calibrate_years[1]:
                cal_rows.append(row)
            elif test_years[0] <= y <= test_years[1]:
                test_rows.append(row)
    
    return cal_rows, test_rows


def get_boamp_domain_split(filepath, domain_a_cpv_prefixes, domain_b_cpv_prefixes):
    """Split BOAMP data by CPV category prefix for cross-domain validation."""
    rows = load_boamp_dataset(filepath)
    
    rows_a = []
    rows_b = []
    for row in rows:
        cpv = row.get('CPV', '').strip()[:2]
        if cpv in domain_a_cpv_prefixes:
            rows_a.append(row)
        elif cpv in domain_b_cpv_prefixes:
            rows_b.append(row)
    
    return rows_a, rows_b


def analyze_reputation_distribution(scores):
    """Analyze the computed reputation distribution."""
    vals = np.array(list(scores.values()))
    return {
        'mean': np.mean(vals),
        'std': np.std(vals),
        'min': np.min(vals),
        'max': np.max(vals),
        'p25': np.percentile(vals, 25),
        'p50': np.percentile(vals, 50),
        'p75': np.percentile(vals, 75),
    }
