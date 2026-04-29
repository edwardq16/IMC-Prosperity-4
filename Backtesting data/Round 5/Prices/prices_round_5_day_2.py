import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("prices_round_5_day_2.csv", sep=";")

group = df["product"].str.split("_").str[0]

galaxy = df[group == "GALAXY"]
sleep = df[group == "SLEEP"]
microchip = df[group == "MICROCHIP"]
pebbles = df[group == "PEBBLES"]
robot = df[group == "ROBOT"]
uv = df[group == "UV"]
translator = df[group == "TRANSLATOR"]
panel = df[group == "PANEL"]
oxygen = df[group == "OXYGEN"]
snackpack = df[group == "SNACKPACK"]

plt.plot(galaxy["timestamp"], galaxy["mid_price"], label="GALAXY")
plt.legend()
plt.show()