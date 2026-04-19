from trader_round_0 import Trader
from datamodel import TradingState, OrderDepth
import random

if __name__ == "__main__":
    emerald_depth = OrderDepth()

    for i in range(20):
        emerald_depth.sell_orders[random.randint(9990, 10010)] = random.randint(0, 80)
        emerald_depth.buy_orders[random.randint(9990, 10010)] = random.randint(0, 80)

    order_depths = {"EMERALDS": emerald_depth}

    state = TradingState("", 0, {}, order_depths, {},{},{}, None)

    trader = Trader()
    print(trader.run(state))