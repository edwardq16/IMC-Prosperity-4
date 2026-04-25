import numpy as np
import math
from datamodel import TradingState, Order
from typing import Dict, List

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + 0.5 * (sigma ** 2) * T)/(sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))
    N_d1 = norm_cdf(d1)
    N_d2 = norm_cdf(d2)
    C = (S * N_d1) - (K * N_d2)
    return C

def bs_delta(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        if S > K:
            return 1.0
        else:
            return 0.0
    d1 = (np.log(S / K) + 0.5 * (sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm_cdf(d1)

def implied_vol(C_mkt, S, K, T, tolerance=1e-5, max_iterations=100):
    if T <= 0 or C_mkt < max(S - K, 0) or C_mkt > S:
        return None
    low = 1e-4
    high = 5.0
    mid = 0.5 * (low + high)
    for i in range(max_iterations):
        mid = 0.5 * (low + high)
        price = bs_call(S, K, T, mid)
        if abs(price - C_mkt) < tolerance:
            return mid
        if price < C_mkt:
            low = mid
        if price > C_mkt:
            high = mid
    return mid

class Trader:
    voucher_strikes = {"VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000, "VEV_6500": 6500}

    def trade_hp(self, state: TradingState) -> List[Order]:
        orders = []

        ## CODE HERE ##

        return orders

    def trade_vev(self, state: TradingState) -> List[Order]:
        orders = []
        product = "VELVETFRUIT_EXTRACT"
        product_position = state.position.get(product, 0)
        order_depth = state.order_depths[product]
        max_position = 200

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        ## ARBITRAGE ##
        for price, quantity in sorted(order_depth.sell_orders.items()):
            if price < mid_price and product_position < max_position:
                buy_quantity = min(max_position - product_position, -quantity)
                orders.append(Order(product, price, buy_quantity))
                product_position += buy_quantity

        for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True):
            if price > mid_price and product_position > -max_position:
                sell_quantity = min(max_position + product_position, quantity)
                orders.append(Order(product, price, -sell_quantity))
                product_position -= sell_quantity

        ## MARKET MAKE ##

        if order_depth.buy_orders and order_depth.sell_orders:
            gamma = 1
            std = 0.000215
            bid_size = max_position - product_position
            ask_size = max_position + product_position
            reservation_price = mid_price - product_position * gamma * (std ** 2) * (1_000_000 - state.timestamp)
            spread = 2

            if bid_size > 0:
                orders.append(Order(product, round(reservation_price - spread / 2), bid_size))

            if ask_size > 0:
                orders.append(Order(product, round(reservation_price + spread / 2), -ask_size))

        return orders

    def trade_vev_vouchers(self, state: TradingState) -> Dict[str, List[Order]]:
        voucher_orders = {}
        voucher_data = {}
        max_position = 300
        deviation_threshold = 5e-3
        underlying = "VELVETFRUIT_EXTRACT"
        order_depth = state.order_depths[underlying]
        T = 5 - state.timestamp / 1_000_000

        if order_depth.buy_orders and order_depth.sell_orders and T >0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            S = (best_bid + best_ask) / 2

            ## IMPLIED VOLATILITY FOR ALL VOUCHERS ##
            for voucher_name, K in self.voucher_strikes.items():
                if state.order_depths[voucher_name].buy_orders and state.order_depths[voucher_name].sell_orders:
                    best_bid = max(state.order_depths[voucher_name].buy_orders)
                    best_ask = min(state.order_depths[voucher_name].sell_orders)
                    mid = (best_bid + best_ask) / 2
                    m = (np.log(K/S))/np.sqrt(T)
                    iv = implied_vol(mid, S, K, T)
                    if iv is not None:
                        voucher_data[voucher_name] = {"iv": iv, "mid": mid, "moneyness": m}

            moneynesses = [data["moneyness"] for data in voucher_data.values()]
            ivs = [data["iv"] for data in voucher_data.values()]

            if len(ivs) >= 5:

                ## FIT SMILE AND FIND DEVIATION ##
                coeffs = np.polyfit(moneynesses, ivs, deg=2)
                for voucher_name, data in voucher_data.items():
                    voucher_orders[voucher_name] = []
                    product_position = state.position.get(voucher_name, 0)
                    m = data["moneyness"]
                    market_iv = data["iv"]
                    fitted_iv = np.polyval(coeffs, m)
                    deviation = market_iv - fitted_iv
                    if abs(deviation) >= deviation_threshold:

                        ## TRADE ##
                        if deviation > 0:
                            for price, quantity in sorted(state.order_depths[voucher_name].buy_orders.items(), reverse=True):
                                if product_position <= -max_position:
                                    break
                                sell_quantity = min(max_position + product_position, quantity)
                                voucher_orders[voucher_name].append(Order(voucher_name, price, -sell_quantity))
                                product_position -= sell_quantity

                        elif deviation < 0:
                            for price, quantity in sorted(state.order_depths[voucher_name].sell_orders.items()):
                                if product_position >= max_position:
                                    break
                                buy_quantity = min(max_position - product_position, -quantity)
                                voucher_orders[voucher_name].append(Order(voucher_name, price, buy_quantity))
                                product_position += buy_quantity

        return voucher_orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        result["HYDROGEL_PACK"] = self.trade_hp(state)
        result["VELVETFRUIT_EXTRACT"] = self.trade_vev(state)
        voucher_orders = self.trade_vev_vouchers(state)
        for voucher_name, orders in voucher_orders.items():
            result[voucher_name] = orders

        traderData = ""
        conversions = 0
        return result, conversions, traderData