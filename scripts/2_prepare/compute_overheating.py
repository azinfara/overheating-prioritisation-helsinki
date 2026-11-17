"""
Overheating analysis — outdoor classification, apartment & building metrics, plots.

Usage:
    python compute_overheating.py

Inputs:
    final/hourly_indoor_temperature.parquet
    final/hourly_weather.csv
    final/buildings_metadata.csv

Outputs:
    final/outdoor_daily.csv
    final/apartment_metrics.csv
    final/building_metrics.csv
    final/buildings_metadata.csv  (updated)
    plots/A1_outdoor_temperature_timeline.png
    plots/A2_heatwave_calendar.png
    plots/A3_day_classification_summary.png
    plots/B1_degree_hours_distribution.png
    plots/B2_temp_mean_vs_dh27.png
    plots/B3_peak_hour_distribution.png
    plots/B4_peak_lag_distribution.png
    plots/B5_delta_distribution.png
    plots/C1_spread_mean_vs_peak.png
    plots/C2_rank_correlation.png
    plots/C3_pct_above_thresholds.png
    plots/C4_heatwave_spread_worsening.png
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

OUT_DATA = "final"
OUT_PLOT = "plots"
os.makedirs(OUT_PLOT, exist_ok=True)

PAL = {"blue": "#3266ad", "teal": "#1D9E75", "coral": "#D85A30",
       "pink": "#D4537E", "gray": "#73726c", "green": "#639922",
       "amber": "#BA7517", "red": "#E24B4A", "purple": "#534AB7",
       "dkgreen": "#2d6a2e"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#ddd", "axes.grid": False, "font.family": "sans-serif",
    "font.size": 10, "savefig.dpi": 180, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#eee", linewidth=0.5)
    ax.set_axisbelow(True)

def save_plot(fig, name):
    p = os.path.join(OUT_PLOT, name)
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    print(f"    -> {p}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LOAD DATA                                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("=" * 65)
print("OVERHEATING ANALYSIS")
print("=" * 65)
print("\n[LOAD]")

temp = pd.read_parquet(os.path.join(OUT_DATA, "hourly_indoor_temperature.parquet"))
temp["timestamp"] = pd.to_datetime(temp["timestamp"])
temp["date"] = temp["timestamp"].dt.normalize()
temp["hour"] = temp["timestamp"].dt.hour
print(f"  Indoor:   {len(temp):>12,} rows | {temp['device'].nunique():,} devices | "
      f"{temp['building_id'].nunique()} buildings")

weather = pd.read_csv(os.path.join(OUT_DATA, "hourly_weather.csv"), parse_dates=["timestamp"])
weather["date"] = weather["timestamp"].dt.normalize()
weather["hour"] = weather["timestamp"].dt.hour
print(f"  Weather:  {len(weather):>12,} rows | {weather['timestamp'].min().date()} to {weather['timestamp'].max().date()}")

buildings = pd.read_csv(os.path.join(OUT_DATA, "buildings_metadata.csv"))
print(f"  Buildings:{len(buildings):>12,} rows")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART A — OUTDOOR DAILY CLASSIFICATION                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("\n[PART A] Outdoor daily classification")

od = weather.groupby("date").agg(
    out_max=("outdoor_temp", "max"),
    out_min=("outdoor_temp", "min"),
    out_mean=("outdoor_temp", "mean"),
    n_hours=("outdoor_temp", "count"),
).reset_index()
od["date"] = pd.to_datetime(od["date"])
od = od.sort_values("date").reset_index(drop=True)

# Classifications
od["hot_day"]    = od["out_max"] > 25  # published definition: daily max strictly above 25 C
od["fmi_yellow"] = (od["out_max"] >= 27) & (od["out_mean"] >= 20)
od["fmi_orange"] = (od["out_max"] >= 30) & (od["out_mean"] >= 24)
od["fmi_red"]    = (od["out_max"] >= 35) & (od["out_mean"] >= 28)

# Heatwave: more than 4 consecutive hot days (published definition)
streak = 0
streaks = []
for h in od["hot_day"]:
    streak = streak + 1 if h else 0
    streaks.append(streak)
od["hot_streak"] = streaks

od["heatwave_day"] = False
for i in range(len(od) - 1, -1, -1):
    if od.loc[i, "hot_streak"] >= 5:
        start = i - od.loc[i, "hot_streak"] + 1
        od.loc[max(0, start):i, "heatwave_day"] = True

# Find heatwave periods
hw_periods = []
in_hw = False
for _, row in od.iterrows():
    if row["heatwave_day"] and not in_hw:
        hw_start = row["date"]; in_hw = True
    elif not row["heatwave_day"] and in_hw:
        hw_periods.append((hw_start, prev)); in_hw = False
    prev = row["date"]
if in_hw:
    hw_periods.append((hw_start, prev))

# Print summary
n = {k: od[k].sum() for k in ["hot_day", "heatwave_day", "fmi_yellow", "fmi_orange", "fmi_red"]}
print(f"  Days total:     {len(od)}")
for k, v in n.items():
    print(f"  {k:18s} {v}")
print(f"  Heatwave periods: {len(hw_periods)}")
for s, e in hw_periods:
    print(f"    {s.strftime('%Y-%m-%d')} to {e.strftime('%Y-%m-%d')} ({(e-s).days+1} days)")

od.to_csv(os.path.join(OUT_DATA, "outdoor_daily.csv"), index=False, encoding="utf-8-sig")
print(f"  Saved outdoor_daily.csv")

# --- Plot A1: Outdoor temperature timeline (2021 only) ---
print("  Plotting...")
od_plot = od[od["date"].dt.year == 2021].copy()
hw_periods_plot = [(s, e) for s, e in hw_periods if s.year == 2021]

fig, ax = plt.subplots(figsize=(16, 5))

# Heatwave bands
for s, e in hw_periods_plot:
    ax.axvspan(s, e + pd.Timedelta(days=1), alpha=0.15, color=PAL["coral"], zorder=0)

ax.fill_between(od_plot["date"], od_plot["out_min"], od_plot["out_max"], alpha=0.2, color=PAL["blue"], label="Min–Max range")
ax.plot(od_plot["date"], od_plot["out_max"], color=PAL["red"], linewidth=1.2, label="Daily max")
ax.plot(od_plot["date"], od_plot["out_mean"], color=PAL["blue"], linewidth=1.2, label="Daily mean")
ax.plot(od_plot["date"], od_plot["out_min"], color=PAL["teal"], linewidth=1, alpha=0.7, label="Daily min")

ax.axhline(25, color="#f5c542", linestyle="--", linewidth=0.8, alpha=0.7, label="Hot day (25°C)")
ax.axhline(27, color="#e85d2a", linestyle="--", linewidth=0.8, alpha=0.7, label="FMI yellow (27°C)")
ax.axhline(30, color="#b80020", linestyle="--", linewidth=0.8, alpha=0.7, label="FMI orange (30°C)")

ax.set_ylabel("Temperature (°C)")
ax.set_title("Outdoor daily temperature — summer 2021", fontsize=14, fontweight="bold", loc="left")
ax.legend(loc="upper left", fontsize=8, ncol=4)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45)
style_ax(ax)
# Add heatwave label
for s, e in hw_periods_plot:
    mid = s + (e - s) / 2
    ax.text(mid, ax.get_ylim()[1] - 1, "HEATWAVE", ha="center", fontsize=8,
            color=PAL["coral"], fontweight="bold", alpha=0.8)
save_plot(fig, "A1_outdoor_temperature_timeline.png")

# --- Plot A2: Heatwave calendar heatmap ---
od_2021 = od[od["date"].dt.year == 2021].copy()
od_2021["month"] = od_2021["date"].dt.month
od_2021["day"] = od_2021["date"].dt.day

fig, ax = plt.subplots(figsize=(14, 3.5))
months = od_2021["month"].unique()
for _, row in od_2021.iterrows():
    x = (row["date"] - od_2021["date"].min()).days
    color = "#b80020" if row["fmi_orange"] else (
            "#e85d2a" if row["fmi_yellow"] else (
            "#f5c542" if row["hot_day"] else "#dde4ed"))
    ax.barh(0, 1, left=x, color=color, edgecolor="white", linewidth=0.3, height=0.8)
    if row["out_max"] > 25:
        ax.text(x + 0.5, 0, f"{row['out_max']:.0f}", ha="center", va="center",
                fontsize=6, color="white" if row["fmi_yellow"] else "#555")

ax.set_xlim(0, len(od_2021))
ax.set_yticks([])
# x-axis as dates
tick_positions = []
tick_labels = []
for _, row in od_2021.iterrows():
    x = (row["date"] - od_2021["date"].min()).days
    if row["day"] in [1, 8, 15, 22]:
        tick_positions.append(x)
        tick_labels.append(row["date"].strftime("%b %d"))
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)

patches = [mpatches.Patch(color=c, label=l) for c, l in [
    ("#dde4ed", "Normal"), ("#f5c542", "Hot (≥25)"),
    ("#e85d2a", "FMI Yellow (≥27)"), ("#b80020", "FMI Orange (≥30)")]]
ax.legend(handles=patches, loc="upper right", fontsize=8, ncol=4)
ax.set_title("Day-by-day outdoor heat classification — summer 2021", fontsize=13, fontweight="bold", loc="left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
save_plot(fig, "A2_heatwave_calendar.png")

# --- Plot A3: Summary bar ---
fig, ax = plt.subplots(figsize=(8, 4))
cats = ["Total days", "Hot (≥25)", "Heatwave\n(>4 consec)", "FMI Yellow\n(≥27+mean≥20)",
        "FMI Orange\n(≥30+mean≥24)", "FMI Red\n(≥35+mean≥28)"]
vals = [len(od), n["hot_day"], n["heatwave_day"], n["fmi_yellow"], n["fmi_orange"], n["fmi_red"]]
colors = [PAL["blue"], PAL["amber"], PAL["coral"], "#e8c840", PAL["coral"], PAL["red"]]
bars = ax.bar(range(len(cats)), vals, color=colors, width=0.6)
ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=9)
ax.set_title("Outdoor day classification summary", fontsize=13, fontweight="bold", loc="left")
style_ax(ax)
for b in bars:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h + 0.5, str(int(h)), ha="center", fontsize=10, color="#555")
save_plot(fig, "A3_day_classification_summary.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART B — APARTMENT-LEVEL METRICS                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("\n[PART B] Apartment-level metrics")

# Merge outdoor daily to indoor
print("  Merging outdoor classification to indoor data...")
temp = temp.merge(
    od[["date", "out_max", "out_mean", "hot_day", "heatwave_day", "fmi_yellow", "fmi_orange"]],
    on="date", how="left"
)
# Merge hourly outdoor temp
temp = temp.merge(weather[["timestamp", "outdoor_temp"]], on="timestamp", how="left")
print(f"  Rows with outdoor hourly: {temp['outdoor_temp'].notna().sum():,} / {len(temp):,}")

# --- Vectorized degree-hour helper ---
def dh_col(series, threshold):
    excess = series - threshold
    return excess.clip(lower=0)

temp["excess_27"] = dh_col(temp["temperature"], 27)
temp["excess_30"] = dh_col(temp["temperature"], 30)
temp["excess_32"] = dh_col(temp["temperature"], 32)
temp["delta"] = temp["temperature"] - temp["outdoor_temp"]

# --- Vectorized peak hour / lag per device per day ---
print("  Computing peak hours (vectorized)...")
# Indoor peak hour per device per day
daily_indoor = temp.groupby(["device", "date"]).agg(
    indoor_peak_val=("temperature", "max"),
    indoor_peak_hour=("hour", lambda x: x.iloc[temp.loc[x.index, "temperature"].values.argmax()]),
    daily_mean=("temperature", "mean"),
).reset_index()

# Outdoor peak hour per day
daily_outdoor = temp.dropna(subset=["outdoor_temp"]).groupby("date").agg(
    outdoor_peak_hour=("hour", lambda x: x.iloc[temp.loc[x.index, "outdoor_temp"].values.argmax()]),
).reset_index()

daily_indoor = daily_indoor.merge(daily_outdoor, on="date", how="left")
daily_indoor["peak_lag"] = daily_indoor["indoor_peak_hour"] - daily_indoor["outdoor_peak_hour"]
daily_indoor.loc[daily_indoor["peak_lag"] < 0, "peak_lag"] += 24

# Add day classification
daily_indoor = daily_indoor.merge(
    od[["date", "hot_day", "heatwave_day", "fmi_yellow", "fmi_orange"]],
    on="date", how="left"
)

# --- Aggregate per apartment ---
print("  Aggregating per apartment...")

# Full summer stats
grp = temp.groupby("device")
apt = grp.agg(
    building_id=("building_id", "first"),
    address=("address", "first"),
    city=("city", "first"),
    n_hours=("temperature", "count"),
    n_days=("date", "nunique"),
    date_start=("timestamp", "min"),
    date_end=("timestamp", "max"),
    temp_mean=("temperature", "mean"),
    temp_max=("temperature", "max"),
    temp_min=("temperature", "min"),
    temp_std=("temperature", "std"),
    temp_p95=("temperature", lambda x: x.quantile(0.95)),
    dh_27=("excess_27", "sum"),
    dh_30=("excess_30", "sum"),
    dh_32=("excess_32", "sum"),
    hours_above_25=("temperature", lambda x: (x > 25).sum()),
    hours_above_27=("temperature", lambda x: (x > 27).sum()),
    hours_above_30=("temperature", lambda x: (x > 30).sum()),
    hours_above_32=("temperature", lambda x: (x > 32).sum()),
    delta_mean=("delta", "mean"),
    delta_max=("delta", "max"),
).reset_index()

# Hot day stats
temp_hot = temp[temp["hot_day"] == True]
grp_hot = temp_hot.groupby("device")
hot_stats = grp_hot.agg(
    hot_temp_mean=("temperature", "mean"),
    hot_temp_max=("temperature", "max"),
    hot_dh_27=("excess_27", "sum"),
    hot_dh_30=("excess_30", "sum"),
    hot_hours=("temperature", "count"),
    hot_delta_mean=("delta", "mean"),
).reset_index()

# Heatwave stats
temp_hw = temp[temp["heatwave_day"] == True]
grp_hw = temp_hw.groupby("device")
hw_stats = grp_hw.agg(
    hw_temp_mean=("temperature", "mean"),
    hw_temp_max=("temperature", "max"),
    hw_dh_27=("excess_27", "sum"),
    hw_dh_30=("excess_30", "sum"),
    hw_hours=("temperature", "count"),
    hw_delta_mean=("delta", "mean"),
).reset_index()

# FMI yellow stats
temp_yel = temp[temp["fmi_yellow"] == True]
if len(temp_yel) > 0:
    yel_stats = temp_yel.groupby("device").agg(
        fmi_yellow_temp_mean=("temperature", "mean"),
        fmi_yellow_temp_max=("temperature", "max"),
        fmi_yellow_dh_27=("excess_27", "sum"),
    ).reset_index()
else:
    yel_stats = pd.DataFrame({"device": apt["device"]})

# FMI orange stats
temp_ora = temp[temp["fmi_orange"] == True]
if len(temp_ora) > 0:
    ora_stats = temp_ora.groupby("device").agg(
        fmi_orange_temp_mean=("temperature", "mean"),
        fmi_orange_temp_max=("temperature", "max"),
        fmi_orange_dh_30=("excess_30", "sum"),
    ).reset_index()
else:
    ora_stats = pd.DataFrame({"device": apt["device"]})

# Peak hour / lag — hot days only
di_hot = daily_indoor[daily_indoor["hot_day"] == True]
peak_hot = di_hot.groupby("device").agg(
    peak_hour_hot=("indoor_peak_hour", "mean"),
    peak_lag_hot=("peak_lag", "mean"),
).reset_index()

# Peak hour / lag — heatwave days
di_hw = daily_indoor[daily_indoor["heatwave_day"] == True]
peak_hw = di_hw.groupby("device").agg(
    peak_hour_hw=("indoor_peak_hour", "mean"),
    peak_lag_hw=("peak_lag", "mean"),
).reset_index()

# Peak hour / lag — FMI orange days
di_ora = daily_indoor[daily_indoor["fmi_orange"] == True]
if len(di_ora) > 0:
    peak_ora = di_ora.groupby("device").agg(
        peak_hour_orange=("indoor_peak_hour", "mean"),
        peak_lag_orange=("peak_lag", "mean"),
    ).reset_index()
else:
    peak_ora = pd.DataFrame({"device": apt["device"]})

# Merge all
for df in [hot_stats, hw_stats, yel_stats, ora_stats, peak_hot, peak_hw, peak_ora]:
    apt = apt.merge(df, on="device", how="left")

# Flag: exceeds 150 dh above 27 (Finnish building code)
apt["exceeds_code_150dh"] = apt["dh_27"] > 150

# Round
for col in apt.select_dtypes(include=[np.number]).columns:
    apt[col] = apt[col].round(2)

apt.to_csv(os.path.join(OUT_DATA, "apartment_metrics.csv"), index=False, encoding="utf-8-sig")
print(f"  Saved apartment_metrics.csv: {len(apt)} apartments x {len(apt.columns)} cols")
print(f"  Apartments exceeding 150 dh@27: {apt['exceeds_code_150dh'].sum()} / {len(apt)}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART C — BUILDING-LEVEL METRICS                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("\n[PART C] Building-level metrics")

def safe_spread(x):
    return x.max() - x.min() if len(x) > 1 else 0

bg = apt.groupby("building_id")

bld_m = bg.agg(
    bld_n_apartments=("device", "count"),

    # Summer-wide
    bld_temp_mean=("temp_mean", "mean"),
    bld_temp_max=("temp_max", "max"),
    bld_temp_std_across=("temp_mean", "std"),

    # Mean-based spread
    bld_warmest_apt_mean=("temp_mean", "max"),
    bld_coolest_apt_mean=("temp_mean", "min"),
    bld_spread_mean=("temp_mean", safe_spread),

    # Peak-based spread
    bld_warmest_apt_peak=("temp_max", "max"),
    bld_coolest_apt_peak=("temp_max", "min"),
    bld_spread_peak=("temp_max", safe_spread),

    # Degree hours
    bld_dh_27_mean=("dh_27", "mean"),
    bld_dh_27_max=("dh_27", "max"),
    bld_dh_27_min=("dh_27", "min"),
    bld_dh_30_mean=("dh_30", "mean"),
    bld_dh_30_max=("dh_30", "max"),
    bld_dh_32_mean=("dh_32", "mean"),

    # % exceeding thresholds
    bld_pct_above_27=("hours_above_27", lambda x: (x > 0).mean() * 100),
    bld_pct_above_30=("hours_above_30", lambda x: (x > 0).mean() * 100),
    bld_pct_exceed_code=("exceeds_code_150dh", lambda x: x.mean() * 100),

    # Peak hour / lag on hot days — building averages
    bld_peak_hour_hot_mean=("peak_hour_hot", "mean"),
    bld_peak_hour_hot_peak=("peak_hour_hot", "max"),
    bld_peak_lag_hot_mean=("peak_lag_hot", "mean"),
    bld_peak_hour_hot_spread=("peak_hour_hot", safe_spread),
    bld_peak_hour_hot_std=("peak_hour_hot", "std"),
    bld_peak_lag_hot_spread=("peak_lag_hot", safe_spread),

    # Peak hour / lag on heatwave days
    bld_peak_hour_hw_mean=("peak_hour_hw", "mean"),
    bld_peak_hour_hw_peak=("peak_hour_hw", "max"),
    bld_peak_lag_hw_mean=("peak_lag_hw", "mean"),

    # Heatwave temps
    bld_hw_temp_mean=("hw_temp_mean", "mean"),
    bld_hw_temp_max=("hw_temp_max", "max"),
    bld_hw_dh_27_mean=("hw_dh_27", "mean"),
    bld_hw_dh_27_max=("hw_dh_27", "max"),
    bld_hw_spread_mean=("hw_temp_mean", safe_spread),

    # Delta
    bld_delta_mean=("delta_mean", "mean"),
    bld_delta_max=("delta_max", "max"),
    bld_hw_delta_mean=("hw_delta_mean", "mean"),
).reset_index()

# Same-worst analysis: is warmest-by-mean also warmest-by-peak?
print("  Computing same-worst apartment analysis...")
same_worst = []
rank_corrs = []
for bid, group in apt.groupby("building_id"):
    if len(group) < 3:
        same_worst.append(np.nan)
        rank_corrs.append(np.nan)
        continue
    worst_mean = group.loc[group["temp_mean"].idxmax(), "device"]
    worst_peak = group.loc[group["temp_max"].idxmax(), "device"]
    same_worst.append(worst_mean == worst_peak)

    # Spearman rank correlation between mean rank and peak rank
    if len(group) >= 4:
        rho, _ = sp_stats.spearmanr(group["temp_mean"].rank(), group["temp_max"].rank())
        rank_corrs.append(rho)
    else:
        rank_corrs.append(np.nan)

bld_m["bld_same_worst"] = same_worst
bld_m["bld_rank_corr_mean_peak"] = rank_corrs

# Heatwave worsening: does spread increase during heatwave vs normal?
bld_m["bld_spread_worsens"] = bld_m["bld_hw_spread_mean"] > bld_m["bld_spread_mean"]

# Peak hour on orange days if available
if "peak_hour_orange" in apt.columns and apt["peak_hour_orange"].notna().sum() > 0:
    ora_bld = apt.groupby("building_id").agg(
        bld_peak_hour_orange_mean=("peak_hour_orange", "mean"),
        bld_peak_hour_orange_peak=("peak_hour_orange", "max"),
        bld_peak_lag_orange_mean=("peak_lag_orange", "mean"),
    ).reset_index()
    bld_m = bld_m.merge(ora_bld, on="building_id", how="left")

# Round
for col in bld_m.select_dtypes(include=[np.number]).columns:
    bld_m[col] = bld_m[col].round(2)

bld_m.to_csv(os.path.join(OUT_DATA, "building_metrics.csv"), index=False, encoding="utf-8-sig")
print(f"  Saved building_metrics.csv: {len(bld_m)} buildings x {len(bld_m.columns)} cols")

# Update buildings_metadata
print("  Updating buildings_metadata.csv...")
old_cols = [c for c in buildings.columns if c.startswith("bld_") or c.startswith("temp_") or
            c in ["has_temp_data", "n_temp_apartments"]]
buildings = buildings.drop(columns=old_cols, errors="ignore")
buildings = buildings.merge(bld_m, on="building_id", how="left")
buildings["has_temp_data"] = buildings["bld_n_apartments"].notna()
buildings.to_csv(os.path.join(OUT_DATA, "buildings_metadata.csv"), index=False, encoding="utf-8-sig")
print(f"  Saved buildings_metadata.csv: {len(buildings)} rows x {len(buildings.columns)} cols")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART D — VISUALIZATIONS                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("\n[PART D] Visualizations")

# --- B1: Degree hours distribution ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (col, thr, color) in zip(axes, [
    ("dh_27", "27°C", PAL["amber"]),
    ("dh_30", "30°C", PAL["coral"]),
    ("dh_32", "32°C", PAL["red"])]):
    data = apt[col].dropna()
    ax.hist(data[data > 0], bins=40, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(f"Degree-hours above {thr}", fontsize=11, fontweight="medium", loc="left")
    ax.set_xlabel("Degree-hours"); ax.set_ylabel("Apartments")
    if col == "dh_27":
        ax.axvline(150, color="black", linestyle="--", linewidth=1, label="Code limit (150 dh)")
        ax.legend(fontsize=8)
    ax.text(0.95, 0.95, f"n>0: {(data>0).sum()}/{len(data)}", transform=ax.transAxes,
            fontsize=8, ha="right", va="top", color="#888")
    style_ax(ax)
fig.suptitle("Apartment degree-hours distribution — summer 2021", fontsize=13, fontweight="bold", y=1.03)
save_plot(fig, "B1_degree_hours_distribution.png")

# --- B2: temp_mean vs dh_27 ---
fig, ax = plt.subplots(figsize=(8, 6))
mask = apt["dh_27"].notna() & apt["temp_mean"].notna()
ax.scatter(apt.loc[mask, "temp_mean"], apt.loc[mask, "dh_27"], s=10, alpha=0.3,
           color=PAL["coral"], edgecolors="none")
ax.set_xlabel("Summer mean temperature (°C)")
ax.set_ylabel("Degree-hours above 27°C")
ax.set_title("Mean temperature vs overheating severity", fontsize=13, fontweight="bold", loc="left")
ax.axhline(150, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Code limit 150 dh")
r, p = sp_stats.pearsonr(apt.loc[mask, "temp_mean"], apt.loc[mask, "dh_27"])
ax.text(0.05, 0.95, f"r={r:.2f}, n={mask.sum()}", transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(facecolor="white", edgecolor="#ddd", alpha=0.8))
ax.legend(fontsize=8)
style_ax(ax)
save_plot(fig, "B2_temp_mean_vs_dh27.png")

# --- B3: Peak hour distribution (hot days vs heatwave) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, (col, label, color) in zip(axes, [
    ("peak_hour_hot", "Hot days (max≥25)", PAL["amber"]),
    ("peak_hour_hw", "Heatwave days", PAL["coral"])]):
    data = apt[col].dropna()
    if len(data) > 0:
        ax.hist(data, bins=np.arange(-0.5, 24.5, 1), color=color, edgecolor="white", alpha=0.85)
        ax.axvline(data.median(), color=PAL["red"], linestyle="--", linewidth=1.5,
                   label=f"median = {data.median():.1f}h")
        ax.legend(fontsize=8)
    ax.set_title(f"Indoor peak hour — {label}", fontsize=11, fontweight="medium", loc="left")
    ax.set_xlabel("Hour of day (0–23)"); ax.set_ylabel("Apartments")
    ax.set_xlim(-0.5, 23.5)
    style_ax(ax)
fig.suptitle("When do apartments get hottest?", fontsize=13, fontweight="bold", y=1.03)
save_plot(fig, "B3_peak_hour_distribution.png")

# --- B4: Peak lag distribution ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, (col, label, color) in zip(axes, [
    ("peak_lag_hot", "Hot days", PAL["blue"]),
    ("peak_lag_hw", "Heatwave days", PAL["purple"])]):
    data = apt[col].dropna()
    if len(data) > 0:
        ax.hist(data, bins=30, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(data.median(), color=PAL["red"], linestyle="--", linewidth=1.5,
                   label=f"median = {data.median():.1f}h")
        ax.legend(fontsize=8)
    ax.set_title(f"Thermal lag — {label}", fontsize=11, fontweight="medium", loc="left")
    ax.set_xlabel("Hours after outdoor peak"); ax.set_ylabel("Apartments")
    style_ax(ax)
fig.suptitle("How long after outdoor peak does indoor peak occur?", fontsize=13, fontweight="bold", y=1.03)
save_plot(fig, "B4_peak_lag_distribution.png")

# --- B5: Indoor-outdoor delta ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, (col, label, color) in zip(axes, [
    ("delta_mean", "Full summer", PAL["blue"]),
    ("hw_delta_mean", "Heatwave days only", PAL["coral"])]):
    data = apt[col].dropna()
    if len(data) > 0:
        ax.hist(data, bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.3)
        ax.axvline(data.median(), color=PAL["red"], linestyle="--", linewidth=1.5,
                   label=f"median = {data.median():+.1f}°C")
        ax.legend(fontsize=8)
    ax.set_title(f"Indoor − Outdoor delta — {label}", fontsize=11, fontweight="medium", loc="left")
    ax.set_xlabel("Temperature difference (°C)"); ax.set_ylabel("Apartments")
    style_ax(ax)
fig.suptitle("Does the building amplify (+) or buffer (−) outdoor heat?", fontsize=13, fontweight="bold", y=1.03)
save_plot(fig, "B5_delta_distribution.png")

# --- C1: Spread mean vs peak ---
fig, ax = plt.subplots(figsize=(8, 6))
mask = (bld_m["bld_spread_mean"].notna() & bld_m["bld_spread_peak"].notna() &
        (bld_m["bld_n_apartments"] >= 3))
ax.scatter(bld_m.loc[mask, "bld_spread_mean"], bld_m.loc[mask, "bld_spread_peak"],
           s=20, alpha=0.4, color=PAL["purple"], edgecolors="none")
ax.set_xlabel("Spread in apartment means (°C)")
ax.set_ylabel("Spread in apartment peaks (°C)")
ax.set_title("Intra-building temperature inequality: mean vs peak", fontsize=13, fontweight="bold", loc="left")
if mask.sum() > 10:
    r, _ = sp_stats.pearsonr(bld_m.loc[mask, "bld_spread_mean"], bld_m.loc[mask, "bld_spread_peak"])
    ax.text(0.05, 0.95, f"r={r:.2f}, n={mask.sum()}", transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="#ddd", alpha=0.8))
lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, lim], [0, lim], color="#ccc", linestyle="--", linewidth=0.8, label="1:1 line")
ax.legend(fontsize=8)
style_ax(ax)
save_plot(fig, "C1_spread_mean_vs_peak.png")

# --- C2: Rank correlation distribution ---
fig, ax = plt.subplots(figsize=(8, 4.5))
data = bld_m["bld_rank_corr_mean_peak"].dropna()
if len(data) > 10:
    ax.hist(data, bins=30, color=PAL["teal"], edgecolor="white", alpha=0.85)
    ax.axvline(data.median(), color=PAL["red"], linestyle="--", linewidth=1.5,
               label=f"median = {data.median():.2f}")
    ax.legend(fontsize=9)
    pct_high = (data > 0.8).mean() * 100
    ax.text(0.95, 0.95, f"{pct_high:.0f}% of buildings have r > 0.8\n(same apartment worst by both measures)",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="#ddd", alpha=0.8))
ax.set_xlabel("Spearman rank correlation (mean-rank vs peak-rank)")
ax.set_ylabel("Buildings")
ax.set_title("Is the warmest-on-average apartment also the hottest-peaking?",
             fontsize=13, fontweight="bold", loc="left")
style_ax(ax)
save_plot(fig, "C2_rank_correlation.png")

# --- C3: % of apartments above thresholds ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, (col, label, color) in zip(axes, [
    ("bld_pct_above_27", "% apartments ever above 27°C", PAL["amber"]),
    ("bld_pct_above_30", "% apartments ever above 30°C", PAL["red"])]):
    data = bld_m[col].dropna()
    ax.hist(data, bins=20, color=color, edgecolor="white", alpha=0.85)
    ax.set_xlabel("% of apartments"); ax.set_ylabel("Buildings")
    ax.set_title(label, fontsize=11, fontweight="medium", loc="left")
    pct_100 = (data == 100).mean() * 100
    ax.text(0.95, 0.95, f"{pct_100:.0f}% of buildings: ALL apartments exceed",
            transform=ax.transAxes, fontsize=8, ha="right", va="top", color="#888")
    style_ax(ax)
fig.suptitle("Building-level overheating prevalence", fontsize=13, fontweight="bold", y=1.03)
save_plot(fig, "C3_pct_above_thresholds.png")

# --- C4: Heatwave spread worsening ---
fig, ax = plt.subplots(figsize=(8, 6))
mask = (bld_m["bld_spread_mean"].notna() & bld_m["bld_hw_spread_mean"].notna() &
        (bld_m["bld_n_apartments"] >= 3))
ax.scatter(bld_m.loc[mask, "bld_spread_mean"], bld_m.loc[mask, "bld_hw_spread_mean"],
           s=20, alpha=0.4, color=PAL["coral"], edgecolors="none")
ax.set_xlabel("Normal spread between apartments (°C)")
ax.set_ylabel("Heatwave spread between apartments (°C)")
ax.set_title("Does apartment inequality worsen during heatwaves?",
             fontsize=13, fontweight="bold", loc="left")
lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, lim], [0, lim], color="#ccc", linestyle="--", linewidth=0.8, label="No change (1:1)")
pct_worse = bld_m.loc[mask, "bld_spread_worsens"].mean() * 100
ax.text(0.05, 0.95, f"{pct_worse:.0f}% of buildings: spread WORSENS during heatwaves",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(facecolor="white", edgecolor="#ddd", alpha=0.8))
ax.legend(fontsize=8)
style_ax(ax)
save_plot(fig, "C4_heatwave_spread_worsening.png")


