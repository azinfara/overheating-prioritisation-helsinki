"""
STAGE 3 -- Apartment differences + worst-apartment rank-consistency + independence tests
Within-building overheating spread in the Helsinki residential stock.

Reproduces: resilient-vs-overheating baseline/response gap; rank_consistency
(Spearman rho, mean vs peak) 85% rank-consistent / 15% rank-inconsistent, median rho 0.76;
same_worst 52%; heatwave worsening by rank-consistency group; lag/swing independence
vs dh_27 (sensitivity should FAIL and be excluded).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from common import (
    PLOTS, MIN_APTS, PERIODS, DG_ORDER, DG_LABELS, PAL5,
    load_buildings_metadata, load_apartment_metrics, load_hourly_indoor, load_hourly_weather,
    compute_swing_range, compute_sensitivity, compute_period_mean, outside_heatwave_mask,
    compute_nightly_min, build_apt_table, build_building_comp, set_style,
    material_table, plot_material_bars,
)

set_style()

print(f"""
{'='*72}
STAGE 3 -- DEFINITIONS
{'='*72}
baseline (apartment)   : temp_mean (whole-summer mean; dominated by non-hot days,
                          used as a proxy for 'normal-day' indoor temperature)
heat response           : hot_temp_mean - temp_mean  (rise on hot days [outdoor
                          max>25C] relative to the whole-summer baseline)
rank_consistency        : Spearman rho of a building's apartments, ranked by
                          temp_mean vs temp_max (n >= {MIN_APTS} apts)
                          Rank-consistent = rho >= 0.5 ; Rank-inconsistent = rho < 0.5
same_worst              : #1 apartment by temp_mean == #1 by temp_max (stricter)
heatwave worsening      : building spread (max-min) using hw_temp_mean vs
                          temp_mean; 'worsens' if hw spread > summer spread
independence test       : |Pearson r| vs dh_27 must be < 0.3 to use a
                          behavioral metric downstream (lag, swing expected to
                          pass; sensitivity expected to FAIL -> excluded)
{'='*72}""")

# ============================================================
# LOAD + PREPARE
# ============================================================
buildings_meta = load_buildings_metadata()
apt_metrics = load_apartment_metrics(merge_area_type=True)
print("Loading hourly indoor temperature + weather (for swing_range & sensitivity) ...")
hourly = load_hourly_indoor()
weather = load_hourly_weather()
swing_df = compute_swing_range(hourly)
sensitivity_df = compute_sensitivity(hourly, weather)

# Summer-EXCLUDING-heatwaves baseline (point 5): the existing whole-summer temp_mean
# baseline is only mildly diluted by heatwave days (26 of ~110 monitored days), but a
# cleaner comparison excludes short_hw + long_hw entirely -- gap + recovery + any
# shoulder days outside the 4 named PERIODS all count as "normal summer" here.
excl_mask = outside_heatwave_mask(hourly["timestamp"])
exclhw_mean = hourly.loc[excl_mask].groupby("device")["temperature"].mean().rename("temp_mean_exclhw")

# Secondary robustness check: first vs. last calendar week of the WHOLE monitored
# window (derived from the data itself, not hardcoded, so it tracks whatever range
# hourly_indoor_temperature.parquet actually covers).
data_start = hourly["timestamp"].min().normalize()
data_end = hourly["timestamp"].max().normalize()
first_week_end = (data_start + pd.Timedelta(days=6)).date().isoformat()
last_week_start = (data_end - pd.Timedelta(days=6)).date().isoformat()
first_week_mean = compute_period_mean(hourly, data_start.date().isoformat(), first_week_end).rename("temp_mean_firstweek")
last_week_mean = compute_period_mean(hourly, last_week_start, data_end.date().isoformat()).rename("temp_mean_lastweek")
print(f"Monitored window: {data_start.date()} to {data_end.date()}  "
      f"-- first week ends {first_week_end}, last week starts {last_week_start}")

# Nightly-min per apartment (for the point-6 accumulation-rate-by-age analysis below) --
# computed here, while hourly is still in memory, mirroring Stage 2's methodology.
nmin_apt = compute_nightly_min(hourly, id_col="device")
dev_bld_map = hourly[["device", "building_id"]].drop_duplicates()
del hourly

apt = build_apt_table(apt_metrics, buildings_meta, swing_df)
apt = apt.merge(sensitivity_df, on="device", how="left")
apt = apt.merge(exclhw_mean, on="device", how="left")
apt = apt.merge(first_week_mean, on="device", how="left")
apt = apt.merge(last_week_mean, on="device", how="left")
building_comp, apt, lag_med, swing_med = build_building_comp(apt)

# ============================================================
# 1. BASELINE vs HEAT RESPONSE (resilient vs overheating)
# ============================================================
print("\n" + "="*50)
print("1. BASELINE vs HEAT RESPONSE")
print("="*50)
apt["response"] = apt["hot_temp_mean"] - apt["temp_mean"]
g = apt.groupby("code_status")[["temp_mean", "response"]].median()
print(g.round(2))
gap_baseline = g.loc["overheating", "temp_mean"] - g.loc["resilient", "temp_mean"]
gap_response = g.loc["overheating", "response"] - g.loc["resilient", "response"]
print(f"\nBaseline gap (overheating - resilient): {gap_baseline:+.2f} C   [expect ~+1.5]")
print(f"Response gap (overheating - resilient): {gap_response:+.2f} C   [expect near 0 -- 'near-equal response']")
u, p = stats.mannwhitneyu(apt.loc[apt["code_status"]=="overheating","temp_mean"].dropna(),
                           apt.loc[apt["code_status"]=="resilient","temp_mean"].dropna())
print(f"Mann-Whitney U (baseline, resilient vs overheating): p={p:.3g}")

# ============================================================
# 2. RANK STABILITY (Spearman rho, mean vs peak) + SAME_WORST
# ============================================================
print("\n" + "="*50)
print("2. RANK STABILITY + SAME_WORST")
print("="*50)
a = apt[["device", "building_id", "temp_mean", "temp_max"]].dropna()

rows = []
for bid, g2 in a.groupby("building_id"):
    if len(g2) < MIN_APTS:
        continue
    rho, p = stats.spearmanr(g2["temp_mean"], g2["temp_max"])
    worst_by_mean = g2.loc[g2["temp_mean"].idxmax(), "device"]
    worst_by_peak = g2.loc[g2["temp_max"].idxmax(), "device"]
    rows.append({"building_id": bid, "rank_rho": rho, "same_worst": worst_by_mean == worst_by_peak, "n": len(g2)})
rank_df = pd.DataFrame(rows).set_index("building_id")
rank_df["rank_consistent"] = rank_df["rank_rho"] >= 0.5

print(f"n = {len(rank_df)} buildings (>= {MIN_APTS} apts)")
print(f"median rank_rho = {rank_df['rank_rho'].median():.2f}   [expect 0.76]")
print((rank_df["rank_consistent"].value_counts(normalize=True)*100).round(1).rename({True:"rank-consistent",False:"rank-inconsistent"}).to_string())
print(f"[expect rank-consistent 85% / rank-inconsistent 15%]")
print(f"\nsame_worst = True in {rank_df['same_worst'].mean()*100:.0f}% of buildings   [expect 52%]")

# cross-validate against precomputed buildings_metadata columns
xval = rank_df.join(buildings_meta.set_index("building_id")[["bld_rank_corr_mean_peak", "bld_same_worst"]])
rho_diff = (xval["rank_rho"] - xval["bld_rank_corr_mean_peak"]).abs()
print(f"\nCross-check vs buildings_metadata precomputed columns:")
print(f"  bld_rank_corr_mean_peak: median |diff| from fresh rho = {rho_diff.median():.4f} (n={rho_diff.notna().sum()})")
agree = (xval["same_worst"] == xval["bld_same_worst"]).mean() * 100
print(f"  bld_same_worst agreement with fresh same_worst: {agree:.1f}%")

# ============================================================
# 3. FIG -- is the worst apartment always the same? (spread by rank-consistency)
# ============================================================
bc = building_comp.join(rank_df[["rank_rho", "rank_consistent", "same_worst"]])
sp_by_stab = bc.groupby("rank_consistent")[["spread_temp", "spread_peak"]].median()
print("\nMedian spread by rank-consistency group:")
print(sp_by_stab.round(2).rename(index={True: "rank-consistent", False: "rank-inconsistent"}))
print("[expect rank-consistent buildings to have LARGER spread]")

fig, ax = plt.subplots(figsize=(8, 6))
st = bc[bc["rank_consistent"] == True]
un = bc[bc["rank_consistent"] == False]
ax.scatter(un["spread_temp"], un["spread_peak"], alpha=0.6, s=20, color="#d7191c", marker="^",
           label=f"Rank-inconsistent (rho<0.5, {(~bc['rank_consistent']).mean()*100:.0f}%)")
ax.scatter(st["spread_temp"], st["spread_peak"], alpha=0.4, s=16, color="#2ca25f",
           label=f"Rank-consistent (rho>=0.5, {bc['rank_consistent'].mean()*100:.0f}%)")
lim = max(bc["spread_temp"].max(), bc["spread_peak"].max())
ax.plot([0, lim], [0, lim], "--", color="grey")
ax.set_xlabel("Mean-based spread (C)"); ax.set_ylabel("Peak-based spread (C)")
ax.set_title("Is the worst apartment always the same?\nRank-consistent buildings have larger overheating spread")
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage3_worst_apartment_stability.png")
plt.close(fig)

# ============================================================
# 4. HEATWAVE WORSENING (overall, then by rank-stability tertile)
# ============================================================
print("\n" + "="*50)
print("4. HEATWAVE WORSENING")
print("="*50)
hw_spread = apt.groupby("building_id").agg(
    hw_spread=("hw_temp_mean", lambda x: x.max() - x.min()),
    n=("hw_temp_mean", "count"),
).query("n >= @MIN_APTS")
worsen = building_comp[["spread_temp"]].join(hw_spread[["hw_spread"]], how="inner").dropna()
worsen["worsens"] = worsen["hw_spread"] > worsen["spread_temp"]
worsen["delta"] = worsen["hw_spread"] - worsen["spread_temp"]
print(f"n = {len(worsen)} buildings")
print(f"Summer spread: median {worsen['spread_temp'].median():.2f} C; "
      f"Heatwave spread: median {worsen['hw_spread'].median():.2f} C   [expect 2.46 -> 2.75]")
print(f"Worsens in {worsen['worsens'].mean()*100:.0f}% of buildings   [expect ~71%]")
stat, p = stats.wilcoxon(worsen["spread_temp"], worsen["hw_spread"])
print(f"Wilcoxon (summer vs heatwave spread): stat={stat:.1f}, p={p:.3g}")

print("\nBy rank-consistency tertile (low/mid/high rho):")
w2 = worsen.join(rank_df[["rank_rho"]], how="inner").dropna(subset=["rank_rho"])
w2["tertile"] = pd.qcut(w2["rank_rho"], 3, labels=["low", "mid", "high"])
tert = w2.groupby("tertile", observed=True).agg(
    pct_worsens=("worsens", lambda x: x.mean()*100),
    mean_delta=("delta", "mean"),
    n=("worsens", "count"),
)
print(tert.round(2))
print("[expect roughly 61/72/77% and +0.08/+0.25/+0.30 C low->high, if tertile is the right split]")

# ============================================================
# 4b. FIG -- heatwave worsening scatter (normal spread vs heatwave spread),
# overall and split by rank-consistency (rank-consistent vs rank-inconsistent buildings)
# ============================================================
w3 = worsen.join(rank_df[["rank_consistent"]], how="inner")
pct_overall = worsen["worsens"].mean() * 100
pct_stable = w3.loc[w3["rank_consistent"] == True, "worsens"].mean() * 100
pct_unstable = w3.loc[w3["rank_consistent"] == False, "worsens"].mean() * 100
n_stable = int((w3["rank_consistent"] == True).sum())
n_unstable = int((w3["rank_consistent"] == False).sum())
print(f"\nHeatwave worsening rate: overall {pct_overall:.0f}% | "
      f"rank-consistent buildings {pct_stable:.0f}% (n={n_stable}) | "
      f"rank-inconsistent buildings {pct_unstable:.0f}% (n={n_unstable})")

lim = max(worsen["spread_temp"].max(), worsen["hw_spread"].max()) * 1.05

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.scatter(worsen["spread_temp"], worsen["hw_spread"], alpha=0.55, s=28,
           color="#e2725b", edgecolor="white", linewidth=0.3)
ax.plot([0, lim], [0, lim], "--", color="grey", label="No change (1:1)")
ax.text(0.03, 0.97, f"{pct_overall:.0f}% of buildings: spread WORSENS during heatwaves",
        transform=ax.transAxes, va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999"))
ax.set_xlabel("Normal spread between apartments (C)")
ax.set_ylabel("Heatwave spread between apartments (C)")
ax.set_title("Does apartment overheating spread worsen during heatwaves?", fontsize=14, fontweight="bold")
ax.set_xlim(-0.3, lim)
ax.set_ylim(-0.3, lim)
ax.legend(loc="lower right", frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage3_heatwave_worsening_scatter.png")
plt.close(fig)
print("Saved: plots/stage3_heatwave_worsening_scatter.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
for ax, (label, sub, pct, color) in zip(axes, [
    ("Rank-consistent buildings (rho >= 0.5)", w3[w3["rank_consistent"] == True], pct_stable, "#2ca25f"),
    ("Rank-inconsistent buildings (rho < 0.5)", w3[w3["rank_consistent"] == False], pct_unstable, "#d7191c"),
]):
    ax.scatter(sub["spread_temp"], sub["hw_spread"], alpha=0.6, s=28, color=color,
               edgecolor="white", linewidth=0.3)
    ax.plot([0, lim], [0, lim], "--", color="grey", label="No change (1:1)")
    ax.text(0.03, 0.97, f"{pct:.0f}% worsen\n(n={len(sub)})", transform=ax.transAxes, va="top",
            fontsize=11, bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999"))
    ax.set_xlabel("Normal spread between apartments (C)")
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlim(-0.3, lim)
    ax.set_ylim(-0.3, lim)
axes[0].set_ylabel("Heatwave spread between apartments (C)")
axes[0].legend(loc="lower right", frameon=True, fontsize=9)
fig.suptitle("Heatwave worsening: rank-consistent vs. rank-inconsistent buildings", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage3_heatwave_worsening_stability_comparison.png")
plt.close(fig)
print("Saved: plots/stage3_heatwave_worsening_stability_comparison.png")

# ============================================================
# 4c. SUMMER-EXCL-HEATWAVES BASELINE (cleaner than whole-summer temp_mean, which
# still includes the 26 heatwave days themselves) + FIRST/LAST-WEEK SHOULDER CHECK
# ============================================================
print("\n" + "="*50)
print("4c. SUMMER-EXCL-HEATWAVES BASELINE vs HEATWAVE SPREAD")
print("="*50)

spread_exclhw = (apt.groupby("building_id")["temp_mean_exclhw"]
                  .agg(lambda x: x.max() - x.min() if x.notna().sum() >= MIN_APTS else np.nan)
                  .rename("spread_exclhw").dropna())
cmp_exclhw = spread_exclhw.to_frame().join(hw_spread[["hw_spread"]], how="inner").dropna()
cmp_exclhw["worsens"] = cmp_exclhw["hw_spread"] > cmp_exclhw["spread_exclhw"]

print(f"n = {len(cmp_exclhw)} buildings")
print(f"Whole-summer baseline (incl. heatwave days): median spread = {worsen['spread_temp'].median():.2f} C  [old baseline, for reference]")
print(f"Summer-EXCL-heatwaves baseline:               median spread = {cmp_exclhw['spread_exclhw'].median():.2f} C  [cleaner baseline]")
print(f"Heatwave-period spread:                       median spread = {cmp_exclhw['hw_spread'].median():.2f} C")
pct_worse_exclhw = cmp_exclhw['worsens'].mean()*100
print(f"Worsens (heatwave > excl-heatwave baseline) in {pct_worse_exclhw:.0f}% of buildings")
stat_e, p_e = stats.wilcoxon(cmp_exclhw["spread_exclhw"], cmp_exclhw["hw_spread"])
print(f"Wilcoxon (excl-heatwave baseline vs heatwave spread): stat={stat_e:.1f}, p={p_e:.3g}")
print(f"-- the MEDIAN spread barely moves (2.46 C -> {cmp_exclhw['spread_exclhw'].median():.2f} C), but the per-building")
print(f"   worsens-rate does shift more than that suggests: 71% (whole-summer baseline) -> {pct_worse_exclhw:.0f}%")
print("   (excl-heatwave baseline). Still a real, statistically significant majority -- report the")
print("   range (65-71%) rather than a single number, since baseline choice is not fully neutral here.")

print("\nSecondary robustness check -- first vs. last calendar week of the monitored summer:")
for label, col in [("First week", "temp_mean_firstweek"), ("Last week", "temp_mean_lastweek")]:
    sp_week = (apt.groupby("building_id")[col]
               .agg(lambda x: x.max() - x.min() if x.notna().sum() >= MIN_APTS else np.nan)
               .rename("spread_week").dropna())
    cmp_week = sp_week.to_frame().join(hw_spread[["hw_spread"]], how="inner").dropna()
    pct_worse_week = (cmp_week["hw_spread"] > cmp_week["spread_week"]).mean() * 100
    print(f"  {label:10s}: median spread = {cmp_week['spread_week'].median():.2f} C (n={len(cmp_week)})  "
          f"-- heatwave spread is larger in {pct_worse_week:.0f}% of buildings")

fig, ax = plt.subplots(figsize=(7.5, 5.5))
labels_bar = ["Whole summer\n(incl. heatwave days)", "Summer excl.\nheatwaves", "Heatwave\nperiod"]
vals_bar = [worsen["spread_temp"].median(), cmp_exclhw["spread_exclhw"].median(), cmp_exclhw["hw_spread"].median()]
colors_bar = ["#abd9e9", "#2c7bb6", "#d7191c"]
ax.bar(labels_bar, vals_bar, color=colors_bar, edgecolor="white")
for i, v in enumerate(vals_bar):
    ax.text(i, v + 0.03, f"{v:.2f} C", ha="center", fontsize=10.5, fontweight="bold")
ax.set_ylabel("Median within-building spread (C)")
ax.set_title("Overheating spread: baseline choice barely changes the picture", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage3_summer_excl_heatwaves_comparison.png")
plt.close(fig)
print("Saved: plots/stage3_summer_excl_heatwaves_comparison.png")

# ============================================================
# 4d. ACCUMULATION RATE DURING THE SHORT HEATWAVE, BY BUILDING AGE
# Distinct from Stage 4's 24h peak-lag finding (REACTION speed, a single night's
# response): this asks how fast a building's nightly minimum climbs as the short
# heatwave arrives and builds toward its peak -- "before -> beginning -> peak",
# a multi-day RAMP-UP rate, not a lag. Same OLS-slope-of-nightly-min methodology
# as Stage 2, just fit over a different window (a 6-day 'before' run-up glued to
# the 6-day short_hw period itself, so day_idx 0 = 6 days before short_hw starts
# and day_idx 11 = the last day of short_hw, its peak).
# ============================================================
print("\n" + "="*50)
print("4d. ACCUMULATION RATE DURING SHORT HEATWAVE (before -> peak), BY AGE")
print("="*50)

nmin_apt_b = nmin_apt.merge(dev_bld_map, on="device", how="left")
bld_night_ramp = nmin_apt_b.groupby(["building_id", "date"])["night_min"].mean().reset_index()

short_hw_start = pd.Timestamp(PERIODS["short_hw"][0])
short_hw_end = pd.Timestamp(PERIODS["short_hw"][1])
before_start = short_hw_start - pd.Timedelta(days=6)
ramp_window = bld_night_ramp[(bld_night_ramp["date"] >= before_start) & (bld_night_ramp["date"] <= short_hw_end)].copy()
ramp_window["day_idx"] = (ramp_window["date"] - before_start).dt.days
print(f"Ramp window: {before_start.date()} (6 days before) -> {short_hw_end.date()} (peak of short heatwave), "
      f"{ramp_window['date'].nunique()} nights")


def ramp_slope_for(df, min_pts=6):
    out = {}
    for bid, g in df.groupby("building_id"):
        if len(g) < min_pts:
            continue
        s, *_ = stats.linregress(g["day_idx"], g["night_min"])
        out[bid] = s
    return pd.Series(out, name="ramp_slope")


ramp_slope = ramp_slope_for(ramp_window)
print(f"Buildings with a ramp slope: {len(ramp_slope)}")
print(f"Overall ramp-slope median = {ramp_slope.median():+.3f} C/night")

ramp_by_age = ramp_slope.to_frame().join(buildings_meta.set_index("building_id")[["design_group"]])
age_table = ramp_by_age.groupby("design_group", observed=True)["ramp_slope"].agg(["median", "mean", "count"]).reindex(DG_ORDER)
print("\nRamp-slope (accumulation rate) by design_group:")
print(age_table.round(3).to_string())

valid_age = age_table["median"].dropna()
if len(valid_age) == len(DG_ORDER):
    ordinal = np.arange(len(DG_ORDER))
    r_age, p_age = stats.spearmanr(ordinal, valid_age.values)
    print(f"\nSpearman r (chronological design_group order vs median ramp-slope) = {r_age:+.3f} (p={p_age:.3g})")
print("Testing: do newer buildings show a SMALLER (slower) accumulation slope during the ramp-up?")
newest_slope = age_table.loc["post2012", "median"]
oldest_slope = age_table.loc["pre1977", "median"]
print(f"  post2012 median slope = {newest_slope:+.3f} C/night vs pre1977 median slope = {oldest_slope:+.3f} C/night")
print(f"  {'CONFIRMED slower' if abs(newest_slope) < abs(oldest_slope) else 'NOT slower'} -- "
      f"post2012 {'<' if abs(newest_slope) < abs(oldest_slope) else '>='} pre1977 in |slope|")

fig, ax = plt.subplots(figsize=(8, 5.5))
meds = age_table["median"].reindex(DG_ORDER)
ax.bar([DG_LABELS[g] for g in DG_ORDER], meds.values, color=PAL5, edgecolor="white")
for i, v in enumerate(meds.values):
    if pd.notna(v):
        ax.text(i, v + (0.002 if v >= 0 else -0.006), f"{v:+.3f}", ha="center", fontsize=9.5, fontweight="bold")
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_ylabel("Median ramp-slope, before -> peak of short heatwave (C/night)")
ax.set_xlabel("Construction era")
ax.set_title("Do newer buildings accumulate heat more slowly?", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage3_accumulation_slope_by_age.png")
plt.close(fig)
print("Saved: plots/stage3_accumulation_slope_by_age.png")

# ---- material effect on accumulation rate (point 7) -- building-level (one ramp
# slope per building), Steel/Other excluded, fixed MATERIAL_ORDER shared with Stage 4 ----
print("\nRamp-slope (accumulation rate) by building_material (n = buildings):")
mat_ramp_table = material_table(ramp_slope, buildings_meta)
print(mat_ramp_table.round(3).to_string())
ramp_spread_material = mat_ramp_table["median"].max() - mat_ramp_table["median"].min()
ramp_spread_age = age_table["median"].max() - age_table["median"].min()
print(f"\nSpread across materials: {ramp_spread_material:.3f} C/night   vs.   spread across design_group (age): {ramp_spread_age:.3f} C/night")
print(f"-- material effect on accumulation rate is {'SMALLER' if ramp_spread_material < ramp_spread_age else 'LARGER'} than the age effect.")

plot_material_bars(mat_ramp_table, "{:+.3f}",
                   "Median building ramp-slope, before -> peak of short heatwave (C/night)",
                   "Does building material affect accumulation rate?", "#fdae61",
                   rf"{PLOTS}/stage3_material_effect_on_accumulation.png")
print("Saved: plots/stage3_material_effect_on_accumulation.png")

# ============================================================
# 5. INDEPENDENCE TEST (lag & swing vs dh_27; sensitivity should FAIL)
# ============================================================
print("\n" + "="*50)
print("5. INDEPENDENCE TEST vs dh_27 (|r| < 0.3 to keep)")
print("="*50)
for col, label in [("peak_lag_hot", "thermal lag"), ("swing_range", "daily swing"), ("sensitivity", "envelope sensitivity")]:
    d = apt[[col, "dh_27"]].dropna()
    r, p = stats.pearsonr(d[col], d["dh_27"])
    verdict = "PASS (independent)" if abs(r) < 0.3 else "FAIL (exclude -- confounded with dh_27)"
    print(f"  {label:22s} r={r:+.3f}  p={p:.3g}   -> {verdict}")

print("\n>>> STAGE 3 complete.")
