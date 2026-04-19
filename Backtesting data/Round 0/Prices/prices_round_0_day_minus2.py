import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("prices_round_0_day_-2.csv", sep=";")

emerald_prices = df[df["product"] == "EMERALDS"]
tomato_prices = df[df["product"] == "TOMATOES"]

plt.plot(emerald_prices["timestamp"], emerald_prices["mid_price"])
plt.ylim(9990, 10010)
plt.show()

plt.plot(tomato_prices["timestamp"], tomato_prices["mid_price"])
plt.ylim(4984, 5040)
plt.show()
