def trade_vev(self, state: TradingState):
    voucher_orders = {}
    voucher_data = {}
    hedge_data = {}
    max_position = 300
    underlying = "VELVETFRUIT_EXTRACT"
    order_depth = state.order_depths[underlying]
    T = 5 - state.timestamp / 1_000_000

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
                m = (np.log(K / S)) / np.sqrt(T)
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
                K = self.voucher_strikes[voucher_name]
                product_position = state.position.get(voucher_name, 0)
                m = data["moneyness"]
                market_iv = data["iv"]
                fitted_iv = np.polyval(coeffs, m)
                deviation = market_iv - fitted_iv
                v_bid = data["v_bid"]
                v_ask = data["v_ask"]

                ## ALWAYS COMPUTE HEDGE DATA regardless of whether we trade ##
                delta = bs_delta(S, K, T, market_iv)
                hedge_data[voucher_name] = {"delta": delta, "position": product_position}

                if abs(deviation) >= AGGRESSIVE_THRESHOLD:
                    if deviation > 0:
                        for price, quantity in sorted(state.order_depths[voucher_name].buy_orders.items(),
                                                      reverse=True):
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

    return {}, {}