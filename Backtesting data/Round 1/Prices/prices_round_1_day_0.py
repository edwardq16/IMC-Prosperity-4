import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("prices_round_1_day_0.csv", sep=";")

ipr_prices = df[df["product"] == "INTARIAN_PEPPER_ROOT"]
ipr_clean = ipr_prices[ipr_prices["mid_price"] > 5000]
ipr_clean["best_bid"] = ipr_clean[["bid_price_1", "bid_price_2", "bid_price_3"]].max(axis=1)
ipr_clean["best_ask"] = ipr_clean[["ask_price_1", "ask_price_2", "ask_price_3"]].min(axis=1)

aco_prices = df[df["product"] == "ASH_COATED_OSMIUM"]
aco_clean = aco_prices[aco_prices["mid_price"] > 5000]

plt.plot(ipr_clean["timestamp"], ipr_clean["mid_price"], label="mid_price")
plt.plot(ipr_clean["timestamp"], ipr_clean["best_bid"], label="best bid")
plt.plot(ipr_clean["timestamp"], ipr_clean["best_ask"], label="best ask")
plt.legend()
plt.show()

plt.plot(aco_clean["timestamp"], aco_clean["mid_price"])
plt.ylim(9950, 10050)
plt.show()