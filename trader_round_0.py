from datamodel import TradingState, Order, OrderDepth
from typing import Dict, List

class Trader:

    def trade_emeralds(self, state: TradingState, product: str, product_position: int) -> List[Order]:
        orders = []

        ## ARBITRAGE ##
        for price, quantity in sorted(state.order_depths[product].sell_orders.items()):
            if price < 10000 and product_position < 80:
                buy_quantity = min(80 - product_position, -quantity)
                orders.append(Order(product, price, buy_quantity))
                product_position += buy_quantity

        for price, quantity in sorted(state.order_depths[product].buy_orders.items(), reverse=True):
            if price > 10000 and product_position > -80:
                sell_quantity = min(80 + product_position, quantity)
                orders.append(Order(product, price, -sell_quantity))
                product_position -= sell_quantity

        ## MARKET MAKE ##
        best_bid = max(state.order_depths[product].buy_orders)
        best_ask = min(state.order_depths[product].sell_orders)

        if best_bid + 1 < 10000:
            orders.append(Order(product, best_bid + 1, 80 - product_position))
        if best_ask - 1 > 10000:
            orders.append(Order(product, best_ask - 1, -(80 + product_position)))

        return orders

    def trade_tomatoes(self, state: TradingState, product: str, product_position: int) -> List[Order]:
        orders = []
        best_bid = max(state.order_depths[product].buy_orders)
        best_ask = min(state.order_depths[product].sell_orders)
        mid_price = (best_bid + best_ask) / 2

        ## ARBITRAGE ##
        for price, quantity in sorted(state.order_depths[product].sell_orders.items()):
            if price < mid_price and product_position < 80:
                buy_quantity = min(80 - product_position, -quantity)
                orders.append(Order(product, price, buy_quantity))
                product_position += buy_quantity

        for price, quantity in sorted(state.order_depths[product].buy_orders.items(), reverse=True):
            if price > mid_price and product_position > -80:
                sell_quantity = min(80 + product_position, quantity)
                orders.append(Order(product, price, -sell_quantity))
                product_position -= sell_quantity

        ## MARKET MAKE ##
        if best_bid + 1 < mid_price:
            orders.append(Order(product, best_bid + 1, 80 - product_position))
        if best_ask - 1 > mid_price:
            orders.append(Order(product, best_ask - 1, -(80 + product_position)))

        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for product in state.order_depths:
            product_position = state.position.get(product, 0)

            if product == "EMERALDS":
                order_book = self.trade_emeralds(state, product, product_position)

            elif product == "TOMATOES":
                order_book = self.trade_tomatoes(state, product, product_position)

            else:
                order_book = []

            result[product] = order_book

        traderData = ""
        conversions = 0
        return result, conversions, traderData