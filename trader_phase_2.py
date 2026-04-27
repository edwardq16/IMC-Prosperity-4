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
        product = "HYDROGEL_PACK"
        product_position = state.position.get(product, 0)
        order_depth = state.order_depths[product]
        max_position = 200
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        best_bid_vol = order_depth.buy_orders[best_bid]
        best_ask_vol = -order_depth.sell_orders[best_ask]
        imbalance = (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol)

        spread = max(2, best_ask - best_bid)
        mid = (best_bid + best_ask) / 2
        k = 3
        fair_price = mid + k * imbalance

        ## ARBITRAGE ##
        arb_limit = 80
        for price, quantity in sorted(order_depth.sell_orders.items()):
            edge = fair_price - price

            if edge > spread / 2 and product_position < arb_limit:
                buy_quantity = min(max_position - product_position, -quantity)
                orders.append(Order(product, price, buy_quantity))
                product_position += buy_quantity

        for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True):
            edge = price - fair_price

            if edge > spread / 2 and product_position > -arb_limit:
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
                orders.append(Order(product, round(reservation_price - spread / 2), bid_size))
                product_position += bid_size

            if ask_size > 0:
                orders.append(Order(product, round(reservation_price + spread / 2), -ask_size))
                product_position -= ask_size

        return orders

    def trade_ve(self, state: TradingState) -> List[Order]:
        orders = []
        product = "VELVETFRUIT_EXTRACT"
        product_position = state.position.get(product, 0)
        order_depth = state.order_depths[product]
        max_position = 200
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())


        best_bid_vol = order_depth.buy_orders[best_bid]
        best_ask_vol = -order_depth.sell_orders[best_ask]
        imbalance = (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol)

        spread = max(2, best_ask - best_bid)
        mid = (best_bid + best_ask) / 2
        k = 3
        fair_price = mid + k * imbalance

        ## ARBITRAGE ##
        arb_limit = 80
        for price, quantity in sorted(order_depth.sell_orders.items()):
            edge = fair_price - price

            if edge > spread / 2 and product_position < arb_limit:
                buy_quantity = min(max_position - product_position, -quantity)
                orders.append(Order(product, price, buy_quantity))
                product_position += buy_quantity

        for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True):
            edge = price - fair_price

            if edge > spread / 2 and product_position > -arb_limit:
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
                orders.append(Order(product, round(reservation_price - spread / 2), bid_size))
                product_position += bid_size

            if ask_size > 0:
                orders.append(Order(product, round(reservation_price + spread / 2), -ask_size))
                product_position -= ask_size

        return orders

   
    def trade_vev(self, state: TradingState) -> Dict[str, List[Order]]:
        voucher_orders = {}
        voucher_data = {}
        max_position = 300
        underlying = "VELVETFRUIT_EXTRACT"
        order_depth = state.order_depths[underlying]
        T = 5 - state.timestamp / 100_000

        if order_depth.buy_orders and order_depth.sell_orders and T > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())

        AGGRESSIVE_THRESHOLD = 0.0010
        PASSIVE_THRESHOLD = 0.0001

        if order_depth.buy_orders and order_depth.sell_orders and T > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            S = (best_bid + best_ask) / 2

            for voucher_name, K in self.voucher_strikes.items():
                if state.order_depths[voucher_name].buy_orders and state.order_depths[voucher_name].sell_orders:
                    v_bid = max(state.order_depths[voucher_name].buy_orders)
                    v_ask = min(state.order_depths[voucher_name].sell_orders)
                    mid = (v_bid + v_ask) / 2
                    m = (np.log(S / K)) / np.sqrt(T)
                    if abs(m) > 0.075:
                        continue
                    iv = implied_vol(mid, S, K, T)
                    if iv is not None:
                        voucher_data[voucher_name] = {"iv": iv, "mid": mid, "moneyness": m, "v_bid": v_bid, "v_ask": v_ask}

            moneynesses = [data["moneyness"] for data in voucher_data.values()]
            ivs = [data["iv"] for data in voucher_data.values()]

            if len(ivs) >= 5:
                coeffs = np.polyfit(moneynesses, ivs, deg=3)
                for voucher_name, data in voucher_data.items():
                    voucher_orders[voucher_name] = []
                    product_position = state.position.get(voucher_name, 0)
                    m = data["moneyness"]
                    market_iv = data["iv"]
                    fitted_iv = np.polyval(coeffs, m)
                    deviation = market_iv - fitted_iv
                    v_bid = data["v_bid"]
                    v_ask = data["v_ask"]

                    if abs(deviation) >= AGGRESSIVE_THRESHOLD:
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

                    elif abs(deviation) >= PASSIVE_THRESHOLD:
                        if deviation > 0:
                            post_price = v_ask - 1
                            if post_price >= v_bid and product_position > -max_position:
                                sell_quantity = max_position + product_position
                                if sell_quantity > 0:
                                    voucher_orders[voucher_name].append(Order(voucher_name, post_price, -sell_quantity))

                        elif deviation < 0:
                            post_price = v_bid + 1
                            if post_price <= v_ask and product_position < max_position:
                                buy_quantity = max_position - product_position
                                if buy_quantity > 0:
                                    voucher_orders[voucher_name].append(Order(voucher_name, post_price, buy_quantity))



        return voucher_orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        result["HYDROGEL_PACK"] = self.trade_hp(state)
        result["VELVETFRUIT_EXTRACT"] = self.trade_ve(state)
        voucher_orders = self.trade_vev(state)
        for voucher_name, orders in voucher_orders.items():
            result[voucher_name] = orders

        traderData = ""
        conversions = 0
        return result, conversions, traderData