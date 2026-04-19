import pandas as pd

df = pd.read_csv("trades_round_0_day_-2.csv", sep=";")

emerald_trades = df[df["symbol"] == "EMERALDS"]
tomato_trades = df[df["symbol"] == "TOMATOES"]

print(emerald_trades.describe())
print(tomato_trades.describe())
