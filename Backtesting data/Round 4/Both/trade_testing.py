import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv("trades_round_4_day_1.csv", sep=";")
df2 = pd.read_csv("trades_round_4_day_2.csv", sep=";")
df3 = pd.read_csv("trades_round_4_day_3.csv", sep=";")

df1["day"] = 1
df2["day"] = 2
df3["day"] = 3

trades = pd.concat([df1, df2, df3], ignore_index=True)

df4 = pd.read_csv("prices_round_4_day_1.csv", sep=";")
df5 = pd.read_csv("prices_round_4_day_2.csv", sep=";")
df6 = pd.read_csv("prices_round_4_day_3.csv", sep=";")

df4["day"] = 1
df5["day"] = 2
df6["day"] = 3

prices = pd.concat([df4, df5, df6], ignore_index=True)

price_lookup = {}
for (day, product), group in prices.groupby(['day', 'product']):
    g = group.sort_values('timestamp')
    price_lookup[(day, product)] = (g['timestamp'].values, g['mid_price'].values)

def get_mid_at(day, product, timestamp):
    key = (day, product)
    times, mids = price_lookup[key]
    position = np.searchsorted(times, timestamp, side='left')
    if position >= len(times):
        return np.nan
    return mids[position]

horizons = [100, 500, 2000, 10000]

for h in horizons:
    trades[f'future_mid_{h}'] = trades.apply(lambda row: get_mid_at(row['day'], row['symbol'], row['timestamp'] + h), axis=1)
    trades[f'buyer_mo_{h}']  = trades[f'future_mid_{h}'] - trades['price']
    trades[f'seller_mo_{h}'] = trades['price'] - trades[f'future_mid_{h}']

print("\n=== Per-day buyer markout (horizon=500) ===")
per_day_buy = trades.groupby(['day', 'symbol', 'buyer'])['buyer_mo_500'].agg(['mean', 'count'])
print(per_day_buy)

print("\n=== Per-day seller markout (horizon=500) ===")
per_day_sell = trades.groupby(['day', 'symbol', 'seller'])['seller_mo_500'].agg(['mean', 'count'])
print(per_day_sell)

prices['spread'] = prices['ask_price_1'] - prices['bid_price_1']
print("\n=== Bid-ask spread by product ===")
print(prices.groupby('product')['spread'].agg(['mean', 'median']))

print("\n=== Volume by buyer (sum and mean trade size) ===")
print(trades.groupby(['symbol', 'buyer'])['quantity'].agg(['sum', 'mean', 'count']))

print("\n=== Volume by seller (sum and mean trade size) ===")
print(trades.groupby(['symbol', 'seller'])['quantity'].agg(['sum', 'mean', 'count']))

print("\n=== Buyer markout across horizons (mean) ===")
agg_cols = [f'buyer_mo_{h}' for h in horizons]
print(trades.groupby(['symbol', 'buyer'])[agg_cols].mean())

print("\n=== Seller markout across horizons (mean) ===")
agg_cols = [f'seller_mo_{h}' for h in horizons]
print(trades.groupby(['symbol', 'seller'])[agg_cols].mean())

trades['mid_at_trade'] = trades.apply(lambda row: get_mid_at(row['day'], row['symbol'], row['timestamp']), axis=1)

trades['price_vs_mid'] = trades['price'] - trades['mid_at_trade']

print("\n=== Buyer: trade price vs mid (positive = bought ABOVE mid, i.e. lifted offer) ===")
print(trades.groupby(['symbol', 'buyer'])['price_vs_mid'].agg(['mean', 'std', 'count']))

print("\n=== Seller: trade price vs mid (positive = sold ABOVE mid, i.e. hit by aggressor or sat on offer) ===")
print(trades.groupby(['symbol', 'seller'])['price_vs_mid'].agg(['mean', 'std', 'count']))

mark38_hp = trades[(trades['symbol'] == 'HYDROGEL_PACK') & ((trades['buyer'] == 'Mark 38') | (trades['seller'] == 'Mark 38'))].sort_values(['day', 'timestamp']).copy()

mark38_hp['gap'] = mark38_hp.groupby('day')['timestamp'].diff()

print("Mark 38 in HYDROGEL — gaps between consecutive trades:")
print(mark38_hp['gap'].describe())
print(f"\nFraction of gaps ≤ 100:  {(mark38_hp['gap'] <= 100).mean():.2%}")
print(f"Fraction of gaps ≤ 300:  {(mark38_hp['gap'] <= 300).mean():.2%}")
print(f"Fraction of gaps ≤ 1000: {(mark38_hp['gap'] <= 1000).mean():.2%}")