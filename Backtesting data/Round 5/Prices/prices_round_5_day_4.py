import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("prices_round_5_day_4.csv", sep=";")

parts = df["product"].str.split("_")

df[["group", "item"]] = df["product"].str.split("_", n=1, expand=True)

figures = []

for group_name, group_df in df.groupby("group"):
    fig, ax = plt.subplots()

    for item_name, item_df in group_df.groupby("item"):
        item_df = item_df.sort_values("timestamp")
        ax.plot(item_df["timestamp"], item_df["mid_price"], label=item_name)

        if group_name == "SNACKPACK":
            snack_pivot = group_df.pivot(index="timestamp", columns="item", values="mid_price")

            if "STRAWBERRY" in snack_pivot.columns and "RASPBERRY" in snack_pivot.columns:
                snack_pivot["SUM_BERRIES"] = (
                        snack_pivot["STRAWBERRY"] + snack_pivot["RASPBERRY"]
                )

                ax.plot(
                    snack_pivot.index,
                    snack_pivot["SUM_BERRIES"],
                    label="SUM (STRAWBERRY + RASPBERRY)",
                    linestyle="--",
                    linewidth=2
                )

    ax.set_title(group_name)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mid Price")
    ax.legend()
    fig.autofmt_xdate()

    figures.append(fig)

plt.show()