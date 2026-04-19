from datamodel import TradingState, Order
from typing import Dict, List

class Trader:

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
        if state.order_depths[product].buy_orders and state.order_depths[product].sell_orders:
            best_bid = max(state.order_depths[product].buy_orders)
            best_ask = min(state.order_depths[product].sell_orders)

            if best_bid + 1 < fair_price:
                orders.append(Order(product, best_bid + 1, 80 - product_position))
            if best_ask - 1 > fair_price:
                orders.append(Order(product, best_ask - 1, -(80 + product_position)))

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
        if state.order_depths[product].buy_orders and state.order_depths[product].sell_orders:
            best_bid = max(state.order_depths[product].buy_orders)
            best_ask = min(state.order_depths[product].sell_orders)

            if best_bid + 1 < 10000:
                orders.append(Order(product, best_bid + 1, 80 - product_position))
            if best_ask - 1 > 10000:
                orders.append(Order(product, best_ask - 1, -(80 + product_position)))

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


    pass