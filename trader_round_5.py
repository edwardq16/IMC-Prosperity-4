import numpy as np
import math
from datamodel import TradingState, Order
from typing import Dict, List

class Trader:

    def trade_pebbles_etf(self, state: TradingState) -> Dict[str, List[Order]]:
        pebbles_basket = ["PEBBLES_M", "PEBBLES_S", "PEBBLES_L", "PEBBLES_XS", "PEBBLES_XL"]
        orders = []
        max_pos = 10
        basket_price = 50000
        order_depths = {}

        for i in pebbles_basket:
            order_depths = {state.order_depths.get(i)}
            pos = state.position.get(i, 0)
            best_bid = max(order_depths[i].buy_orders.keys())
            best_ask = min(order_depths[i].sell_orders.keys())
            mid_price = {i: (best_bid + best_ask)/2}

        basket_sum = sum(mid_price.items())

        if basket_sum <




    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        traderData = ""
        conversions = 0
        return result, conversions, traderData