from datamodel import TradingState, Order
from typing import Dict, List

class Trader:
    def trade_hp(self, state: TradingState, product: str, product_position: int) -> List[Order]:
        orders = []

        ## CODE HERE ##

        return orders

    def trade_vev(self, state: TradingState, product: str, product_position: int) -> List[Order]:
        orders = []

        order_depth = state.order_depths[product]

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        ## ARBITRAGE ##
        for price, quantity in sorted(state.order_depths[product].sell_orders.items()):
            if price < mid_price and product_position < 200:
                buy_quantity = min(200 - product_position, -quantity)
                orders.append(Order(product, price, buy_quantity))
                product_position += buy_quantity

        for price, quantity in sorted(state.order_depths[product].buy_orders.items(), reverse=True):
            if price > mid_price and product_position > -200:
                sell_quantity = min(200 + product_position, quantity)
                orders.append(Order(product, price, -sell_quantity))
                product_position -= sell_quantity

        ## CODE HERE ##

        return orders
    def trade_vev_vouchers(self, state: TradingState, product: str, product_position: int) -> List[Order]:
        orders = []

        ## CODE HERE ##

        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for product in state.order_depths:
            product_position = state.position.get(product, 0)

            if product == "HYDROGEL_PACK":
                order_book = self.trade_hp(state, product, product_position)

            elif product == "VELVETFRUIT_EXTRACT":
                order_book = self.trade_vev(state, product, product_position)

            elif product == "VELVETFRUIT_EXTRACT_VOUCHER":
                order_book = self.trade_vev_vouchers(state, product, product_position)

            else:
                order_book = []

            result[product] = order_book

        traderData = ""
        conversions = 0
        return result, conversions, traderData