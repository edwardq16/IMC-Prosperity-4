from datamodel import TradingState, Order
from typing import Dict, List

class Trader:
    def trade_hp(self, state: TradingState, product: str, product_position: int) -> List[Order]:
        orders = []

        ## CODE HERE ##

        return orders

    def trade_vev(self, state: TradingState, product: str, product_position: int) -> List[Order]:
        orders = []

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