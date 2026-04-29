import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("prices_round_5_day_2.csv", sep=";")

df[["group", "item"]] = df["product"].str.split("_", n=1, expand=True)

figures = []

for group_name, group_df in df.groupby("group"):
    fig, ax = plt.subplots()

    # Plot each item in the group
    for item_name, item_df in group_df.groupby("item"):
        item_df = item_df.sort_values("timestamp")
        ax.plot(item_df["timestamp"], item_df["mid_price"], label=item_name)

    # Special case: add the berry sum to the SNACKPACK plot
    if group_name == "SNACKPACK":
        snack_pivot = group_df.pivot(index="timestamp", columns="item", values="mid_price")
        if "STRAWBERRY" in snack_pivot.columns and "RASPBERRY" in snack_pivot.columns:
            berry_sum = snack_pivot["STRAWBERRY"] + snack_pivot["RASPBERRY"]
            ax.plot(berry_sum.index, berry_sum,
                    label="STRAWBERRY + RASPBERRY",
                    linestyle="--", linewidth=2)
            print(f"Berry sum: mean={berry_sum.mean():.2f}, std={berry_sum.std():.2f}")

    ax.set_title(group_name)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mid Price")
    ax.legend()

    figures.append(fig)

plt.show()