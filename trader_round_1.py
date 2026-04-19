from datamodel import TradingState, Order
from typing import Dict, List

class Trader:
    SPREAD = 2
    MAX_SKEW = 1.0

    def trade_ipr(self, state: TradingState, product: str, product_position: int) -> List[Order]:
        orders = []
        fair_price = 0.001 * state.timestamp + 13000

        ## ARBITRAGE ##
        for price, quantity in sorted(state.order_depths[product].sell_orders.items()):
            if price < fair_price and product_position < 80:
                buy_quantity = min(80 - product_position, -quantity)
                orders.append(Order(product, price, buy_quantity))
                product_position += buy_quantity

        for price, quantity in sorted(state.order_depths[product].buy_orders.items(), reverse=True):
            if price > fair_price and product_position > -80:
                sell_quantity = min(80 + product_position, quantity)
                orders.append(Order(product, price, -sell_quantity))
                product_position -= sell_quantity

        ## MARKET MAKE ##
        skew = product_position / 80 * self.MAX_SKEW
        bid_price = round(fair_price - self.SPREAD - skew)
        ask_price = round(fair_price + self.SPREAD + skew)
        buy_capacity = 80 - product_position
        sell_capacity = 80 + product_position

        if state.order_depths[product].buy_orders and state.order_depths[product].sell_orders:
            if buy_capacity > 0:
                orders.append(Order(product, bid_price, buy_capacity))
            if sell_capacity > 0:
                orders.append(Order(product, ask_price, -sell_capacity))

        return orders

    def trade_aco(self, state: TradingState, product: str, product_position: int) -> List[Order]:
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
        fair_price = 10000
        skew = product_position / 80 * self.MAX_SKEW
        bid_price = round(fair_price - self.SPREAD - skew)
        ask_price = round(fair_price + self.SPREAD + skew)
        buy_capacity = 80 - product_position
        sell_capacity = 80 + product_position

        if state.order_depths[product].buy_orders and state.order_depths[product].sell_orders:
            if buy_capacity > 0:
                orders.append(Order(product, bid_price, buy_capacity))
            if sell_capacity > 0:
                orders.append(Order(product, ask_price, -sell_capacity))

        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for product in state.order_depths:
            product_position = state.position.get(product, 0)

            if product == "INTARIAN_PEPPER_ROOT":
                order_book = self.trade_ipr(state, product, product_position)

            elif product == "ASH_COATED_OSMIUM":
                order_book = self.trade_aco(state, product, product_position)

            else:
                order_book = []

            result[product] = order_book

        traderData = ""
        conversions = 0
        return result, conversions, traderData