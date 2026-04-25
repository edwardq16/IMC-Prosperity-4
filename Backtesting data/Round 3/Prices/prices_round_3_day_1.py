import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("prices_round_3_day_1.csv", sep=";")

hp_prices = df[df["product"] == "HYDROGEL_PACK"]
hp_clean = hp_prices[hp_prices["mid_price"] > 5000]
hp_clean["best_bid"] = hp_clean[["bid_price_1", "bid_price_2", "bid_price_3"]].max(axis=1)
hp_clean["best_ask"] = hp_clean[["ask_price_1", "ask_price_2", "ask_price_3"]].min(axis=1)

vev_prices = df[df["product"] == "VELVETFRUIT_EXTRACT"]
vev_clean = vev_prices[vev_prices["mid_price"] > 5000]

plt.plot(hp_clean["timestamp"], hp_clean["mid_price"], label="mid_price")
plt.plot(hp_clean["timestamp"], hp_clean["best_bid"], label="best bid")
plt.plot(hp_clean["timestamp"], hp_clean["best_ask"], label="best ask")
plt.legend()
plt.show()
print(f"HP mean mid: {hp_clean['mid_price'].mean():.2f}")

plt.plot(vev_clean["timestamp"], vev_clean["mid_price"])
plt.show()
print(f"VE mean mid: {vev_clean['mid_price'].mean():.2f}")

log_returns = np.log(vev_prices["mid_price"] / vev_prices["mid_price"].shift(1)).dropna()
S_prev = vev_prices["mid_price"].shift(1).dropna()
plt.scatter(log_returns, S_prev, s=0.1)
plt.xlabel('S_{i-1} (previous mid price)')
plt.ylabel('Log return r_i')
plt.title('Log return vs previous price')
plt.show()

plt.hist(log_returns, bins=200, density=True)
x = np.linspace(log_returns.min(), log_returns.max(), 200)
mu, sigma = log_returns.mean(), log_returns.std()
plt.plot(x, (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2), 'r-', label='Normal fit')

plt.xlabel('Log return')
plt.ylabel('Density')
plt.legend()
plt.show()

log_returns2 = np.log(hp_prices["mid_price"] / hp_prices["mid_price"].shift(1)).dropna()
print(f"Std:  {log_returns2.std():.6f}")
print(f"Mean: {log_returns.mean():.6f}")
print(f"Std:  {log_returns.std():.6f}")
print(f"Min:  {log_returns.min():.6f}")
print(f"Max:  {log_returns.max():.6f}")