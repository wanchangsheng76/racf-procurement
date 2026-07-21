"""
Kaggle Procurement KPI dataset loader and calibration.
Parses CSV, computes supplier on-time rates, cost profiles, etc.
"""
import csv
import numpy as np
from datetime import datetime
from typing import List, Dict
from model import Supplier, Order
from config import SUPPLIER_PARAMS, ORDER_PARAMS


def load_kaggle_dataset(filepath: str) -> List[Dict]:
    """Load the Procurement KPI Analysis Dataset."""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_supplier_stats(rows: List[Dict]) -> Dict:
    """Compute per-supplier statistics from real data."""
    suppliers = {}
    for row in rows:
        sname = row['Supplier']
        if sname not in suppliers:
            suppliers[sname] = {'total': 0, 'on_time': 0, 'delivered': 0,
                                'total_qty': 0, 'defective': 0,
                                'prices': [], 'categories': set()}

        stats = suppliers[sname]
        try:
            qty = float(row['Quantity']) if row['Quantity'] else 0
            defective = float(row['Defective_Units']) if row['Defective_Units'] else 0
        except ValueError:
            qty = 0
            defective = 0

        stats['total'] += 1
        stats['total_qty'] += qty
        stats['defective'] += defective
        stats['categories'].add(row.get('Item_Category', ''))

        status = row.get('Order_Status', '')
        if status == 'Delivered':
            stats['delivered'] += 1
            # Check if on-time
            try:
                order_date = datetime.strptime(row['Order_Date'], '%Y-%m-%d')
                delivery_date = datetime.strptime(row['Delivery_Date'], '%Y-%m-%d')
                # Simple heuristic: delivered within 14 days = on-time
                if (delivery_date - order_date).days <= 14:
                    stats['on_time'] += 1
            except:
                pass

        # Price data
        try:
            price = float(row['Unit_Price']) if row['Unit_Price'] else None
            if price:
                stats['prices'].append(price)
        except ValueError:
            pass

    return suppliers


def calibrate_from_real_data(rows, rng=None):
    """Calibrate supplier and order parameters from Kaggle data."""
    if rng is None:
        rng = np.random.default_rng()

    stats = compute_supplier_stats(rows)
    s_names = sorted(stats.keys())

    suppliers = []
    for i, name in enumerate(s_names):
        s = stats[name]
        on_time_rate = s['on_time'] / max(s['delivered'], 1)
        avg_price = np.mean(s['prices']) if s['prices'] else 30.0

        m = Supplier(
            m=i,
            K=rng.uniform(*SUPPLIER_PARAMS['capacity_range']),
            c=max(5, avg_price * rng.uniform(0.3, 0.7)),
            h=rng.uniform(*SUPPLIER_PARAMS['overhead_range']),
            f=rng.uniform(*SUPPLIER_PARAMS['fixed_cost_range']),
            tau=rng.uniform(*SUPPLIER_PARAMS['setup_time_range']),
            v=rng.uniform(*SUPPLIER_PARAMS['logistics_speed_range']),
            r=on_time_rate,
            alpha=SUPPLIER_PARAMS['initial_reputation_alpha'],
            beta=SUPPLIER_PARAMS['initial_reputation_beta'],
        )
        suppliers.append(m)

    # Generate orders based on data patterns
    all_categories = set()
    for row in rows:
        all_categories.add(row.get('Item_Category', ''))

    orders = []
    for i in range(min(len(rows), 200)):
        row = rows[i]
        try:
            qty = float(row['Quantity']) if row['Quantity'] else rng.uniform(*ORDER_PARAMS['quantity_range'])
        except ValueError:
            qty = rng.uniform(*ORDER_PARAMS['quantity_range'])
        try:
            price = float(row['Unit_Price']) if row['Unit_Price'] else rng.uniform(*ORDER_PARAMS['unit_price_range'])
        except ValueError:
            price = rng.uniform(*ORDER_PARAMS['unit_price_range'])

        order = Order(
            n=i,
            Q=max(10, qty),
            T=rng.uniform(*ORDER_PARAMS['deadline_range']),
            p=max(1, price),
            R_star=rng.uniform(0.3, 0.7),
            beta=rng.uniform(*ORDER_PARAMS['beta_range']),
        )
        orders.append(order)

    return suppliers, orders


def generate_suppliers(M, rng=None, real_data_calibration=False):
    """Generate M synthetic suppliers."""
    if rng is None:
        rng = np.random.default_rng()
    p = SUPPLIER_PARAMS
    suppliers = []
    for i in range(M):
        if real_data_calibration:
            r_init = rng.beta(p['initial_reputation_alpha'], p['initial_reputation_beta'])
        else:
            r_init = rng.beta(p['initial_reputation_alpha'], p['initial_reputation_beta'])
        m = Supplier(
            m=i,
            K=rng.uniform(*p['capacity_range']),
            c=rng.uniform(*p['cost_range']),
            h=rng.uniform(*p['overhead_range']),
            f=rng.uniform(*p['fixed_cost_range']),
            tau=rng.uniform(*p['setup_time_range']),
            v=rng.uniform(*p['logistics_speed_range']),
            r=r_init,
            alpha=p['initial_reputation_alpha'],
            beta=p['initial_reputation_beta'],
        )
        suppliers.append(m)
    return suppliers


def generate_orders(N, rng=None):
    """Generate N synthetic orders."""
    if rng is None:
        rng = np.random.default_rng()
    p = ORDER_PARAMS
    orders = []
    for i in range(N):
        order = Order(
            n=i,
            Q=rng.uniform(*p['quantity_range']),
            T=rng.uniform(*p['deadline_range']),
            p=rng.uniform(*p['unit_price_range']),
            R_star=rng.uniform(0.3, 0.7),
            beta=rng.uniform(*p['beta_range']),
        )
        orders.append(order)
    return orders


def generate_distances(M, N, rng=None):
    """Generate random distances between suppliers and orders."""
    if rng is None:
        rng = np.random.default_rng()
    distances = {}
    for n in range(N):
        distances[n] = {m: rng.uniform(*ORDER_PARAMS['distance_range']) for m in range(M)}
    return distances
