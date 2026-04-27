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

horizon = 50
price_lookup = {}

for (day, product), group in prices.groupby(['day', 'product']):
    g = group.sort_values('timestamp')
    price_lookup[(day, product)] = (g['timestamp'].values, g['mid_price'].values)

def get_mid_at(day, product, timestamp):
    time = price_lookup[(day, product)][0]
    mid_price = price_lookup[(day, product)][1]

    position = np.searchsorted(time, timestamp, side='left')

    return mid_price[position]

horizon = 500


