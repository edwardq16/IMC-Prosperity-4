import pandas as pd
import numpy as np

dfs = []
for day in [0, 1, 2]:
    df = pd.read_csv(f'prices_round_3_day_{day}.csv', sep=';')
    dfs.append(df)
df = pd.concat(dfs)

for product in ['VELVETFRUIT_EXTRACT', 'HYDROGEL_PACK']:
    p = df[df['product'] == product]

    natural_spread = p['ask_price_1'] - p['bid_price_1']
    mid_prices = p['mid_price']
    log_returns = np.log(mid_prices / mid_prices.shift(1)).dropna()

    print(f"\n{product}")
    print(
        f"  Natural spread - mean: {natural_spread.mean():.2f}, median: {natural_spread.median():.2f}, min: {natural_spread.min():.2f}, max: {natural_spread.max():.2f}")
    print(f"  Sigma timestep: {log_returns.std():.6f}")
    print(f"  Mid price range: {mid_prices.min():.1f} to {mid_prices.max():.1f}")