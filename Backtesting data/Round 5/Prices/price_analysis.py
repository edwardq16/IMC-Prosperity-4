import pandas as pd
import numpy as np
from itertools import combinations

# ---------- Load all 3 days ----------
days = []
for d in [2, 3, 4]:
    day_df = pd.read_csv(f"prices_round_5_day_{d}.csv", sep=";")
    day_df["day"] = d
    days.append(day_df)
df = pd.concat(days, ignore_index=True)

print(f"Loaded {len(df)} rows, {df['product'].nunique()} products, days {sorted(df['day'].unique())}")

# ---------- Per-product summary ----------
summary = df.groupby("product")["mid_price"].agg(["mean", "std", "min", "max"])
summary["range"] = summary["max"] - summary["min"]
summary["cv"] = summary["std"] / summary["mean"]
print("\n=== Per-product summary (sorted by cv) ===")
print(summary.sort_values("cv"))

# ---------- Group definitions ----------
GROUPS = {
    "SNACKPACK":     ["SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "SNACKPACK_VANILLA", "SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY"],
    "TRANSLATOR":    ["TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_GRAPHITE_MIST", "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_VOID_BLUE", "TRANSLATOR_ECLIPSE_CHARCOAL"],
    "PANEL":         ["PANEL_2X2", "PANEL_4X4", "PANEL_2X4", "PANEL_1X4", "PANEL_1X2"],
    "OXYGEN_SHAKE":  ["OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_GARLIC", "OXYGEN_SHAKE_EVENING_BREATH"],
    "UV_VISOR":      ["UV_VISOR_ORANGE", "UV_VISOR_RED", "UV_VISOR_MAGENTA", "UV_VISOR_AMBER", "UV_VISOR_YELLOW"],
    "ROBOT":         ["ROBOT_VACUUMING", "ROBOT_LAUNDRY", "ROBOT_MOPPING", "ROBOT_DISHES", "ROBOT_IRONING"],
    "MICROCHIP":     ["MICROCHIP_TRIANGLE", "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_RECTANGLE", "MICROCHIP_SQUARE"],
    "GALAXY_SOUNDS": ["GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_SOLAR_WINDS", "GALAXY_SOUNDS_SOLAR_FLAMES", "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_BLACK_HOLES"],
    "SLEEP_POD":     ["SLEEP_POD_NYLON", "SLEEP_POD_COTTON", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_SUEDE", "SLEEP_POD_POLYESTER"],
    "PEBBLES":       ["PEBBLES_M", "PEBBLES_S", "PEBBLES_L", "PEBBLES_XS", "PEBBLES_XL"],
}

# ---------- Build wide format: rows = (day, timestamp), cols = product ----------
wide = df.pivot_table(index=["day", "timestamp"], columns="product", values="mid_price").sort_index()
returns = wide.pct_change().dropna()

# ---------- Pair-level scan ----------
findings = []  # we'll collect everything and sort at the end

for group_name, products in GROUPS.items():
    for a, b in combinations(products, 2):
        if a not in wide.columns or b not in wide.columns:
            continue
        pa = wide[a].dropna()
        pb = wide[b].dropna()
        common = pa.index.intersection(pb.index)
        pa, pb = pa.loc[common], pb.loc[common]

        # Correlation of returns (for the pair)
        ra = returns[a].loc[returns.index.isin(common)]
        rb = returns[b].loc[returns.index.isin(common)]
        if len(ra) < 100:
            continue
        corr = ra.corr(rb)

        # Constant-sum check (within each day)
        s = pa + pb
        sum_per_day = s.groupby(level="day").agg(["mean", "std"])
        # we care about how flat the sum is *within* day, normalized by mean
        sum_cv_per_day = sum_per_day["std"] / sum_per_day["mean"]
        sum_score = sum_cv_per_day.max()  # worst-case day's relative noise in the sum
        sum_means = sum_per_day["mean"].values

        # Constant-difference check
        d_ = pa - pb
        diff_per_day = d_.groupby(level="day").agg(["mean", "std"])
        # for differences, normalise by typical price level (use mean of pa)
        diff_score = (diff_per_day["std"] / pa.groupby(level="day").mean()).max()

        # Constant-ratio check
        ratio = pa / pb
        ratio_per_day = ratio.groupby(level="day").agg(["mean", "std"])
        ratio_score = (ratio_per_day["std"] / ratio_per_day["mean"]).max()

        findings.append({
            "group": group_name,
            "a": a, "b": b,
            "corr": corr,
            "sum_cv_max_day": sum_score,    # smaller = flatter sum
            "diff_cv_max_day": diff_score,  # smaller = flatter difference
            "ratio_cv_max_day": ratio_score,# smaller = flatter ratio
            "sum_means": sum_means,         # to spot day-to-day drift
        })

results = pd.DataFrame(findings)

# ---------- Report: top constant-sum pairs ----------
print("\n=== Top 10 most constant-sum pairs (smallest within-day cv of sum) ===")
print(results.nsmallest(10, "sum_cv_max_day")[
    ["group", "a", "b", "corr", "sum_cv_max_day", "sum_means"]
].to_string(index=False))

print("\n=== Top 10 most constant-difference pairs ===")
print(results.nsmallest(10, "diff_cv_max_day")[
    ["group", "a", "b", "corr", "diff_cv_max_day"]
].to_string(index=False))

print("\n=== Top 10 most constant-ratio pairs ===")
print(results.nsmallest(10, "ratio_cv_max_day")[
    ["group", "a", "b", "corr", "ratio_cv_max_day"]
].to_string(index=False))

print("\n=== Top 10 most positively correlated pairs ===")
print(results.nlargest(10, "corr")[["group", "a", "b", "corr"]].to_string(index=False))

print("\n=== Top 10 most negatively correlated pairs ===")
print(results.nsmallest(10, "corr")[["group", "a", "b", "corr"]].to_string(index=False))

# ---------- Group-level structure: average within-group correlation ----------
print("\n=== Within-group correlation summary ===")
for group_name, products in GROUPS.items():
    available = [p for p in products if p in returns.columns]
    if len(available) < 2:
        continue
    sub = returns[available].corr()
    # Strip diagonal, take mean of off-diagonal
    off_diag = sub.where(~np.eye(len(sub), dtype=bool))
    avg_corr = off_diag.stack().mean()
    max_corr = off_diag.stack().max()
    min_corr = off_diag.stack().min()
    print(f"{group_name:15s}  avg={avg_corr:+.2f}  min={min_corr:+.2f}  max={max_corr:+.2f}")

# ---------- Basket detection within each group ----------
# For each group, regress each product against the other 4 and report R^2
from numpy.linalg import lstsq
print("\n=== Basket detection (R² of each product regressed on its 4 group-mates) ===")
for group_name, products in GROUPS.items():
    available = [p for p in products if p in wide.columns]
    if len(available) < 5:
        continue
    sub = wide[available].dropna()
    print(f"\n{group_name}:")
    for target in available:
        others = [p for p in available if p != target]
        X = np.column_stack([sub[others].values, np.ones(len(sub))])
        y = sub[target].values
        coef, residuals, rank, sv = lstsq(X, y, rcond=None)
        y_pred = X @ coef
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        if r2 > 0.95:
            print(f"  {target:30s} R²={r2:.4f}  coefs={dict(zip(others, coef[:-1].round(3)))}")
        else:
            print(f"  {target:30s} R²={r2:.4f}")

# Verify the 5-way sum is constant for PEBBLES and SNACKPACK

PEBBLES = ["PEBBLES_M", "PEBBLES_S", "PEBBLES_L", "PEBBLES_XS", "PEBBLES_XL"]
SNACKPACK = ["SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "SNACKPACK_VANILLA",
             "SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY"]

for group_name, products in [("PEBBLES", PEBBLES), ("SNACKPACK", SNACKPACK)]:
    # Sum the 5 columns at each (day, timestamp)
    basket = wide[products].sum(axis=1)

    print(f"\n=== {group_name} 5-way sum ===")
    print(f"Overall:   mean={basket.mean():.4f}  std={basket.std():.4f}  "
          f"min={basket.min():.2f}  max={basket.max():.2f}  "
          f"range={basket.max() - basket.min():.2f}")

    # Per-day breakdown
    per_day = basket.groupby(level="day").agg(["mean", "std", "min", "max"])
    per_day["range"] = per_day["max"] - per_day["min"]
    print("Per day:")
    print(per_day)