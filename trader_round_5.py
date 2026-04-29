import numpy as np
import math
from datamodel import TradingState, Order
from typing import Dict, List

class Trader:

    def trade_pebbles_etf(self, state: TradingState) -> Dict[str, List[Order]]:
        pebbles_basket = ["PEBBLES_M", "PEBBLES_S", "PEBBLES_L", "PEBBLES_XS", "PEBBLES_XL"]
        max_pos = 10
        basket_price = 50000
        quote_threshold = 3
        skew_per_unit = 0.5
        edge = 1
        orders = {}
        best_bids = {}
        best_asks = {}
        mid_price = {}

        for i in pebbles_basket:
            best_bids[i] = max(state.order_depths.buy_orders.keys())
            best_asks[i] = min(state.order_depths.sell_orders.keys())
            mid_price[i] = (best_bids[i] + best_asks[i]) / 2

        basket_sum = sum(mid_price.values())
        error = basket_price - basket_sum

        if abs(error) < quote_threshold:
            return {}

        leg_adj = error / 5

        for i in pebbles_basket:
            product_position = state.position.get(i, 0)
            leg_fair = mid_price[i] + leg_adj
            skew = product_position * skew_per_unit
            bid_price = int(round(leg_fair - edge - skew))
            ask_price = int(round(leg_fair + edge - skew))
            bid_price = min(bid_price, best_asks[i] - 1)
            ask_price = max(ask_price, best_bids[i] + 1)
            buy_cap = max_pos - product_position
            sell_cap = max_pos + product_position
            leg_orders = []

            if error > 0 and buy_cap > 0:
                leg_orders.append(Order(i, bid_price, buy_cap))
            if error < 0 and sell_cap > 0:
                leg_orders.append(Order(i, ask_price, -sell_cap))

            if leg_orders:
                orders[i] = leg_orders

        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        pebbles_orders = self.trade_pebbles_etf(state)
        for product, order_list in pebbles_orders.items():
            result.setdefault(product, []).extend(order_list)

        traderData = ""
        conversions = 0
        return result, conversions, traderData