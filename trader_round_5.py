import numpy as np
import math
from datamodel import TradingState, Order
from typing import Dict, List

class Trader:

    def trade_pebbles_basket_arb(self, state: TradingState) -> Dict[str, List[Order]]:
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
            best_bids[i] = max(state.order_depths[i].buy_orders.keys())
            best_asks[i] = min(state.order_depths[i].sell_orders.keys())
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

    def trade_mean_reverting(self, state: TradingState, product: str) -> List[Order]:
        orders = []
        product_position = state.position.get(product, 0)
        order_depth = state.order_depths[product]
        max_position = 10
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        best_bid_vol = order_depth.buy_orders[best_bid]
        best_ask_vol = -order_depth.sell_orders[best_ask]
        imbalance = (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol)
        imbalance = max(-0.5, min(0.5, imbalance))

        spread = max(2, best_ask - best_bid)
        mid = (best_bid + best_ask) / 2

        k = spread * 0.15
        fair_price = mid + k * imbalance

        ## ARBITRAGE ##
        arb_limit = 10
        arb_threshold = spread * 0.45

        for price, quantity in sorted(order_depth.sell_orders.items()):
            edge = fair_price - price
            if edge > arb_threshold and product_position < arb_limit:
                buy_quantity = min(max_position - product_position, -quantity)
                orders.append(Order(product, price, buy_quantity))
                product_position += buy_quantity

        for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True):
            edge = price - fair_price
            if edge > arb_threshold and product_position > -arb_limit:
                sell_quantity = min(max_position + product_position, quantity)
                orders.append(Order(product, price, -sell_quantity))
                product_position -= sell_quantity

        ## MARKET MAKE ##
        if order_depth.buy_orders and order_depth.sell_orders:
            inv = product_position / max_position
            kappa = 0.5
            base_size = 20
            bid_size = max(0, int(base_size * (1 - inv)))
            ask_size = max(0, int(base_size * (1 + inv)))

            spread *= 0.9
            reservation_price = fair_price - kappa * inv

            if bid_size > 0:
                bid_price = round(reservation_price - spread / 2)
                orders.append(Order(product, bid_price, bid_size))

            if ask_size > 0:
                ask_price = round(reservation_price + spread / 2)
                orders.append(Order(product, ask_price, -ask_size))

        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        mean_reverting = [
            "UV_VISOR_ORANGE",
            "ROBOT_VACUUMING",
            "ROBOT_DISHES",
            "PANEL_2X2",
            "OXYGEN_SHAKE_MINT",
            "MICROCHIP_TRIANGLE"
        ]

        for product in mean_reverting:
            order_book = self.trade_mean_reverting(state, product)
            result[product] = order_book

        traderData = ""
        conversions = 0
        return result, conversions, traderData