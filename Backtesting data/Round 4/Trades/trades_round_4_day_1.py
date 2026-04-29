import pandas as pd
df = pd.read_csv("trades_round_4_day_1.csv", sep=";")
print(df['buyer'].value_counts())
print(df['seller'].value_counts())
print(df.groupby(['symbol', 'buyer']).size().unstack(fill_value=0))
print(df.groupby(['symbol', 'seller']).size().unstack(fill_value=0))