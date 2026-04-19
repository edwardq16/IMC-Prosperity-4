"""
IMC Prosperity 4 - Round 1 Backtester
Supports parameter sweeps over max_skew and spread.

Usage:
    py backtester.py                        # run with default params
    py backtester.py --sweep                # sweep over max_skew values
    py backtester.py --max-skew 2 --spread 2  # run with specific params

CSV files must be in the same folder as this script, named:
    prices_round_1_day_0.csv
    prices_round_1_day_1.csv  (etc.)
    trades_round_1_day_0.csv
    trades_round_1_day_1.csv  (etc.)
"""

import csv
import os
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ── Data structures (mirrors the real datamodel) ─────────────────────────────

@dataclass
class Order:
    symbol: str
    price: int
    quantity: int  # positive = buy, negative = sell


@dataclass
class OrderDepth:
    buy_orders: Dict[int, int] = field(default_factory=dict)   # price -> quantity
    sell_orders: Dict[int, int] = field(default_factory=dict)  # price -> quantity (negative)


@dataclass
class TradingState:
    timestamp: int
    order_depths: Dict[str, OrderDepth]
    position: Dict[str, int]
    traderData: str = ""


# ── Trader (your strategy lives here) ─────────────────────────────────────────

class Trader:
    LIMIT = 80

    def __init__(self, max_skew: float = 1.0, spread: int = 2):
        self.max_skew = max_skew
        self.spread = spread

    def trade(self, state: TradingState, product: str, fair_value: float) -> List[Order]:
        orders = []
        position = state.position.get(product, 0)
        depth = state.order_depths[product]

        # ── 1. Arbitrage: take all mispriced orders immediately ───────────────
        for price in sorted(depth.sell_orders.keys()):
            if price < fair_value and position < self.LIMIT:
                qty = min(self.LIMIT - position, -depth.sell_orders[price])
                orders.append(Order(product, price, qty))
                position += qty

        for price in sorted(depth.buy_orders.keys(), reverse=True):
            if price > fair_value and position > -self.LIMIT:
                qty = min(self.LIMIT + position, depth.buy_orders[price])
                orders.append(Order(product, price, -qty))
                position -= qty

        # ── 2. Inventory skew ────────────────────────────────────────────────
        skew = (position / self.LIMIT) * self.max_skew

        # ── 3. Market make with skewed quotes ────────────────────────────────
        bid_price = round(fair_value - self.spread - skew)
        ask_price = round(fair_value + self.spread - skew)

        buy_capacity  = self.LIMIT - position
        sell_capacity = self.LIMIT + position

        if buy_capacity > 0 and bid_price < fair_value:
            orders.append(Order(product, bid_price, buy_capacity))

        if sell_capacity > 0 and ask_price > fair_value:
            orders.append(Order(product, ask_price, -sell_capacity))

        return orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {}
        for product in state.order_depths:
            position = state.position.get(product, 0)

            if product == "ASH_COATED_OSMIUM":
                fair_value = 10000.0

            elif product == "INTARIAN_PEPPER_ROOT":
                fair_value = 0.001 * state.timestamp + 13000.0

            else:
                continue

            result[product] = self.trade(state, product, fair_value)

        return result, 0, ""


# ── CSV parsing ───────────────────────────────────────────────────────────────

def parse_prices(filepath: str) -> Dict[int, Dict[str, OrderDepth]]:
    """Returns {timestamp: {product: OrderDepth}}"""
    data = defaultdict(lambda: defaultdict(OrderDepth))

    with open(filepath, newline='') as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                day       = int(row[0])
                timestamp = int(row[1])
                product   = row[2].strip()
            except (ValueError, IndexError):
                continue

            depth = data[timestamp][product]

            # Columns 3-8: bid prices/volumes (up to 3 levels)
            # Columns 9-14: ask prices/volumes (up to 3 levels)
            bid_cols = [(3,4), (5,6), (7,8)]
            ask_cols = [(9,10), (11,12), (13,14)]

            for pc, qc in bid_cols:
                try:
                    p = row[pc].strip()
                    q = row[qc].strip()
                    if p and q:
                        depth.buy_orders[int(float(p))] = int(float(q))
                except (ValueError, IndexError):
                    pass

            for pc, qc in ask_cols:
                try:
                    p = row[pc].strip()
                    q = row[qc].strip()
                    if p and q:
                        depth.sell_orders[int(float(p))] = -int(float(q))
                except (ValueError, IndexError):
                    pass

    return data


def parse_trades(filepath: str) -> Dict[int, List[Tuple[str, int, int]]]:
    """Returns {timestamp: [(product, price, quantity), ...]}"""
    data = defaultdict(list)

    with open(filepath, newline='') as f:
        for row in csv.reader(f):
            try:
                timestamp = int(row[0])
                product   = row[4].strip()
                price     = int(float(row[5]))
                quantity  = int(float(row[6]))
                data[timestamp].append((product, price, quantity))
            except (ValueError, IndexError):
                continue

    return data


# ── Order matching ─────────────────────────────────────────────────────────────

def match_orders(
    orders: List[Order],
    depth: OrderDepth,
    position: int,
    limit: int
) -> Tuple[int, float]:
    """
    Match orders against the order book.
    Returns (new_position, pnl_change).
    """
    pnl = 0.0
    pos = position

    for order in orders:
        if order.quantity > 0:  # buy order
            # match against sell side of book
            for ask_price in sorted(depth.sell_orders.keys()):
                if ask_price > order.price:
                    break
                if pos >= limit:
                    break
                available = -depth.sell_orders[ask_price]
                fill = min(order.quantity, available, limit - pos)
                if fill <= 0:
                    continue
                pnl -= fill * ask_price
                pos += fill
                order.quantity -= fill

        else:  # sell order
            qty = -order.quantity
            for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_price < order.price:
                    break
                if pos <= -limit:
                    break
                available = depth.buy_orders[bid_price]
                fill = min(qty, available, limit + pos)
                if fill <= 0:
                    continue
                pnl += fill * bid_price
                pos -= fill
                qty -= fill

    return pos, pnl


# ── Main simulation ────────────────────────────────────────────────────────────

PRODUCTS = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]
DAYS = [0, 1, 2]
LIMIT = 80


def run_backtest(max_skew: float, spread: int, data_dir: str, verbose: bool = False) -> float:
    """Run a full backtest and return total PnL across all days."""
    trader = Trader(max_skew=max_skew, spread=spread)
    total_pnl = 0.0

    for day in DAYS:
        prices_file = os.path.join(data_dir, f"prices_round_1_day_{day}.csv")
        trades_file = os.path.join(data_dir, f"trades_round_1_day_{day}.csv")

        if not os.path.exists(prices_file):
            continue

        prices = parse_prices(prices_file)
        trades = parse_trades(trades_file) if os.path.exists(trades_file) else {}

        positions = {p: 0 for p in PRODUCTS}
        pnl       = {p: 0.0 for p in PRODUCTS}
        timestamps = sorted(prices.keys())

        for ts in timestamps:
            depth_map = prices[ts]

            # Build TradingState
            state = TradingState(
                timestamp=ts,
                order_depths=dict(depth_map),
                position=dict(positions),
            )

            orders_by_product, _, _ = trader.run(state)

            for product, orders in orders_by_product.items():
                if product not in depth_map:
                    continue
                new_pos, delta_pnl = match_orders(
                    orders, depth_map[product], positions[product], LIMIT
                )
                positions[product] = new_pos
                pnl[product] += delta_pnl

        # Mark to market at end of day
        last_ts = timestamps[-1] if timestamps else 0
        for product in PRODUCTS:
            if product in prices.get(last_ts, {}):
                depth = prices[last_ts][product]
                all_prices = list(depth.buy_orders.keys()) + list(depth.sell_orders.keys())
                if all_prices:
                    mid = sum(all_prices) / len(all_prices)
                    pnl[product] += positions[product] * mid

        day_pnl = sum(pnl.values())
        total_pnl += day_pnl

        if verbose:
            print(f"  Day {day}: PnL = {day_pnl:,.0f}  "
                  f"(ACO: {pnl['ASH_COATED_OSMIUM']:,.0f}, "
                  f"IPR: {pnl['INTARIAN_PEPPER_ROOT']:,.0f})  "
                  f"Final positions: ACO={positions['ASH_COATED_OSMIUM']}, "
                  f"IPR={positions['INTARIAN_PEPPER_ROOT']}")

    return total_pnl


# ── Parameter sweep ────────────────────────────────────────────────────────────

def run_sweep(data_dir: str):
    skew_values   = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    spread_values = [1, 2, 3]

    print(f"\n{'Spread':<10} {'Max skew':<12} {'Total PnL':>12}")
    print("-" * 36)

    best_pnl    = float('-inf')
    best_params = None

    for spread in spread_values:
        for skew in skew_values:
            pnl = run_backtest(skew, spread, data_dir, verbose=False)
            marker = " <-- best" if pnl > best_pnl else ""
            if pnl > best_pnl:
                best_pnl    = pnl
                best_params = (skew, spread)
            print(f"{spread:<10} {skew:<12} {pnl:>12,.0f}{marker}")

    print(f"\nBest params: max_skew={best_params[0]}, spread={best_params[1]}, PnL={best_pnl:,.0f}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMC Prosperity 4 Round 1 Backtester")
    parser.add_argument("--sweep",     action="store_true",  help="Sweep over parameter combinations")
    parser.add_argument("--max-skew",  type=float, default=1.0, help="Inventory skew parameter")
    parser.add_argument("--spread",    type=int,   default=2,   help="Half-spread for market making")
    parser.add_argument("--data-dir",  type=str,   default=".", help="Directory containing CSV files")
    args = parser.parse_args()

    if args.sweep:
        print("Running parameter sweep...")
        run_sweep(args.data_dir)
    else:
        print(f"Running backtest: max_skew={args.max_skew}, spread={args.spread}")
        pnl = run_backtest(args.max_skew, args.spread, args.data_dir, verbose=True)
        print(f"\nTotal PnL: {pnl:,.0f}")
