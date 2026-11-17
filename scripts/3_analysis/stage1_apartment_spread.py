"""
STAGE 1 -- Within-building overheating spread + composition
Within-building overheating spread in the Helsinki residential stock.

Reproduces: mean_spread/peak_spread histograms (medians ~2.5/3.0 C, peak>mean
in ~72% of buildings), 4 composition classifications per building
(overheating, lag, swing, behavior 2x2), and within-building spread medians.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import (
    PLOTS, MIN_APTS, PAL5,
    load_buildings_metadata, load_apartment_metrics, load_hourly_indoor,
    compute_swing_range, build_apt_table, build_building_comp, set_style,
)

set_style()

print(f"""
{'='*72}
STAGE 1 -- DEFINITIONS
{'='*72}
mean_spread (building)  : max - min of apartment temp_mean          -> expect median ~2.3-2.5 C
peak_spread (building)  : max - min of apartment temp_max           -> expect median ~2.9-3.0 C
coolest apartment       : per building, the apartment with min temp_mean / min temp_max / min
                           dh_27 (three different ways to define 'coolest') -- is IT still over
                           criteria (dh_27>150)? The dh_27-based version is the strict best case and
                           equals the 'all_overheating' composition share (consistency check).
is_fast                 : peak_lag_hot < population median          (short lag = fast)
is_volatile             : swing_range >= population median          (big swing = volatile)
behavior_type           : (fast/slow lag) x (volatile/stable swing), 2x2
composition class       : f = fraction of a building's apartments flagged positive
                           f=0 all_neg | 0<f<0.4 mostly_neg | 0.4<=f<=0.6 mixed
                           0.6<f<1 mostly_pos | f=1 all_pos
unit of analysis        : building, >= {MIN_APTS} apartments
{'='*72}""")

# ============================================================
# LOAD + PREPARE (standalone -- recomputes from raw sources)
# ============================================================
buildings_meta = load_buildings_metadata()
apt_metrics = load_apartment_metrics(merge_area_type=True)
print("Loading hourly indoor temperature to compute swing_range ...")
hourly = load_hourly_indoor()
swing_df = compute_swing_range(hourly)
del hourly

apt = build_apt_table(apt_metrics, buildings_meta, swing_df)
building_comp, apt, lag_med, swing_med = build_building_comp(apt)
print(f"lag_med={lag_med:.2f} h, swing_med={swing_med:.2f} C")
print(f"building_comp built: {building_comp.shape}")

# ============================================================
# MEAN vs PEAK SPREAD
# ============================================================
print("\n" + "="*50)
print("MEAN vs PEAK SPREAD")
print("="*50)
med_temp = building_comp["spread_temp"].median()
med_peak = building_comp["spread_peak"].median()
pct_peak_gt_mean = (building_comp["spread_peak"] > building_comp["spread_temp"]).mean() * 100
print(f"median mean_spread (spread_temp) = {med_temp:.2f} C   [expect ~2.3-2.5]")
print(f"median peak_spread               = {med_peak:.2f} C   [expect ~2.9-3.0]")
print(f"peak_spread > mean_spread in {pct_peak_gt_mean:.0f}% of buildings   [expect ~72%]")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(building_comp["spread_temp"], bins=25, alpha=0.6, color="#4575b4",
        label=f"Mean spread (median {med_temp:.1f} C)")
ax.hist(building_comp["spread_peak"], bins=25, alpha=0.6, color="#d7191c",
        label=f"Peak spread (median {med_peak:.1f} C)")
ax.axvline(med_temp, color="#4575b4", ls="--")
ax.axvline(med_peak, color="#d7191c", ls="--")
ax.set_xlabel("Temperature difference: warmest - coolest apartment (C)")
ax.set_ylabel("Number of buildings")
ax.set_title("How different are apartments in the same building?")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage1_mean_vs_peak_spread.png")
plt.close(fig)

# ============================================================
# IS THE COOLEST APARTMENT STILL OVERHEATED?
# The spread numbers above (~2.5 C mean, ~3.0 C peak) say buildings are
# unequal -- they don't say whether being "the coolest" is actually safe.
# This asks that directly: pick the coolest apartment in each building
# (three different ways to define "coolest"), then check whether THAT
# apartment still exceeds the criteria threshold (dh_27>150).
# ============================================================
print("\n" + "="*50)
print("IS THE COOLEST APARTMENT STILL OVERHEATED?")
print("="*50)

apt_valid = apt.dropna(subset=["temp_mean", "temp_max", "dh_27", "dh_30", "building_id", "device"])
bld_counts = apt_valid.groupby("building_id").size()
eligible = bld_counts[bld_counts >= MIN_APTS].index
apt_valid = apt_valid[apt_valid["building_id"].isin(eligible)]


def coolest_by(df, col):
    idx = df.groupby("building_id")[col].idxmin()
    return df.loc[idx].set_index("building_id")


coolest_mean = coolest_by(apt_valid, "temp_mean")   # coolest by average indoor temp
coolest_peak = coolest_by(apt_valid, "temp_max")    # coolest by peak indoor temp
coolest_dh27 = coolest_by(apt_valid, "dh_27")       # least-overheated apartment, strict best case

n_bld = len(coolest_mean)
pct_mean_over = (coolest_mean["dh_27"] > 150).mean() * 100
pct_peak_over = (coolest_peak["dh_27"] > 150).mean() * 100
pct_dh27_over = (coolest_dh27["dh_27"] > 150).mean() * 100

# dh_30 is a STRICTER criterion than dh_27>150 (degree-hours above the higher 30 C
# reference, not 27 C). Its distribution is heavily zero-inflated (68% of apartment-
# summers have dh_30 EXACTLY 0 -- see Stage 7's severity-cluster docstring), so a
# percentile-matched threshold degenerates to 0 anyway (verified separately). The
# natural, non-arbitrary, data-driven cut is therefore dh_30 > 0: "any accumulated
# exposure at all above the stricter 30 C threshold" -- not a percentile pick.
pct_mean_over30 = (coolest_mean["dh_30"] > 0).mean() * 100
pct_peak_over30 = (coolest_peak["dh_30"] > 0).mean() * 100
pct_dh27_over30 = (coolest_dh27["dh_30"] > 0).mean() * 100

print(f"n = {n_bld} buildings (>= {MIN_APTS} apartments)")
print(f"\nCoolest apartment BY MEAN TEMP:")
print(f"  still exceeds criteria (dh_27>150)         in {pct_mean_over:.1f}% of buildings  (median dh_27={coolest_mean['dh_27'].median():.0f})")
print(f"  still exceeds STRICTER criteria (dh_30>0)  in {pct_mean_over30:.1f}% of buildings")
print(f"Coolest apartment BY PEAK TEMP:")
print(f"  still exceeds criteria (dh_27>150)         in {pct_peak_over:.1f}% of buildings  (median dh_27={coolest_peak['dh_27'].median():.0f})")
print(f"  still exceeds STRICTER criteria (dh_30>0)  in {pct_peak_over30:.1f}% of buildings")
print(f"Least-overheated apartment BY DH_27 (strict best case):")
print(f"  still exceeds criteria (dh_27>150)         in {pct_dh27_over:.1f}% of buildings")
print(f"  [this equals the 'all_overheating' composition share printed below -- consistency check]")
print(f"  still exceeds STRICTER criteria (dh_30>0)  in {pct_dh27_over30:.1f}% of buildings")
print(f"\n-- dh_30>0 is a materially stricter bar: even the coolest apartment clears dh_27's 150-hour")
print(f"   line more easily than it clears 'zero accumulated exposure above 30 C at all'.")

same_best = (coolest_mean["device"].values == coolest_dh27["device"].reindex(coolest_mean.index).values).mean() * 100
print(f"\nCoolest-by-mean apartment IS the least-overheated (by dh_27) apartment in {same_best:.0f}% of buildings")
print(f"  (mirrors the Stage 3 same_worst check, applied to the BEST apartment instead of the worst)")

print("\nDistribution of dh_27 among the COOLEST apartment, by definition:")
for label, series in [("Coolest by mean", coolest_mean["dh_27"]), ("Coolest by peak", coolest_peak["dh_27"]),
                       ("Least-overheated (strict)", coolest_dh27["dh_27"])]:
    d = series.describe(percentiles=[0.25, 0.5, 0.75])
    print(f"  {label:26s} median={d['50%']:.0f}  mean={d['mean']:.0f}  "
          f"IQR=[{d['25%']:.0f}, {d['75%']:.0f}]  max={d['max']:.0f}")

xmax = max(coolest_mean["dh_27"].max(), coolest_peak["dh_27"].max(), coolest_dh27["dh_27"].max())
bins = np.linspace(0, xmax, 31)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, (label, series, pct) in zip(axes, [
    ("Coolest by\naverage temp", coolest_mean["dh_27"], pct_mean_over),
    ("Coolest by\npeak temp", coolest_peak["dh_27"], pct_peak_over),
    ("Least-overheated\napartment (strict)", coolest_dh27["dh_27"], pct_dh27_over),
]):
    ax.hist(series, bins=bins, color="#2c7bb6", edgecolor="white")
    ax.axvline(150, color="#d7191c", ls="--", linewidth=1.5)
    ax.set_xlabel("dh_27 (degree-hours > 27 C)")
    ax.set_title(f"{label}\n{pct:.0f}% still over criteria", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Number of buildings")
axes[0].text(150, axes[0].get_ylim()[1] * 0.95, "  criteria\n  limit (150)", color="#d7191c", fontsize=8.5, va="top")
fig.suptitle("Distribution of dh_27 among the COOLEST apartment in each building", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage1_coolest_apartment_still_overheated.png")
plt.close(fig)

# ---- simple, readable summary: grouped bars, dh_27 criterion vs stricter dh_30 criterion ----
rows = [
    ("Coolest by\naverage temp", pct_mean_over, pct_mean_over30),
    ("Coolest by\npeak temp", pct_peak_over, pct_peak_over30),
    ("Least-overheated\napartment (strict)", pct_dh27_over, pct_dh27_over30),
]
fig, ax = plt.subplots(figsize=(9, 4.8))
y = np.arange(len(rows))
bar_h = 0.32
for i, (label, pct27, pct30) in enumerate(rows):
    ax.barh(i - bar_h/2 - 0.02, pct27, color="#d7191c", edgecolor="white", height=bar_h, label="dh_27 > 150 (criteria)" if i == 0 else None)
    ax.barh(i + bar_h/2 + 0.02, pct30, color="#8c1515", edgecolor="white", height=bar_h, label="dh_30 > 0 (stricter)" if i == 0 else None)
    ax.text(pct27 + 1.5, i - bar_h/2 - 0.02, f"{pct27:.0f}%", va="center", fontsize=9.5, color="#d7191c", fontweight="bold")
    ax.text(pct30 + 1.5, i + bar_h/2 + 0.02, f"{pct30:.0f}%", va="center", fontsize=9.5, color="#8c1515", fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=11)
ax.set_xlim(0, 100)
ax.set_xlabel(f"% of buildings still over the threshold (n={n_bld})")
ax.set_xticks([0, 25, 50, 75, 100])
ax.invert_yaxis()
ax.legend(loc="lower right", frameon=False, fontsize=9.5)
ax.set_title("Is a building's coolest apartment actually safe? (dh_27 vs. stricter dh_30)", fontsize=13.5, fontweight="bold")
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage1_coolest_apartment_summary.png")
plt.close(fig)

# ============================================================
# COMPOSITION SHARES (4 dimensions)
# ============================================================
ORDER_OVER = ["all_resilient", "mostly_resilient", "mixed", "mostly_overheating", "all_overheating"]
ORDER_FAST = ["all_slow", "mostly_slow", "mixed", "mostly_fast", "all_fast"]
ORDER_VOL = ["all_stable", "mostly_stable", "mixed", "mostly_volatile", "all_volatile"]
BT_ORDER = ["buffered_stable", "buffered_volatile", "fast_stable", "fast_volatile"]

print("\n" + "="*50)
print("COMPOSITION SHARES (% of buildings)")
print("="*50)
for nm, col_frac, col_cls, order in [
    ("OVERHEATING (dh_27>150)", "frac_over", "class_over", ORDER_OVER),
    ("LAG (fast = short lag)", "frac_fast", "class_fast", ORDER_FAST),
    ("SWING (volatile = big swing)", "frac_vol", "class_vol", ORDER_VOL),
]:
    print(f"\n{nm}  (n={building_comp[col_cls].notna().sum()}):")
    print((building_comp[col_cls].value_counts(normalize=True) * 100).reindex(order).round(1).to_string())

dom_pct = (building_comp["dominant_behavior"].value_counts(normalize=True) * 100).reindex(BT_ORDER)
print(f"\nDOMINANT BEHAVIORAL TYPE (lag x swing)  (n={building_comp['dominant_behavior'].notna().sum()}):")
print(dom_pct.round(1).to_string())

print("\nSPREAD MEDIANS (within-building, max-min):")
sp = building_comp[["spread_temp", "spread_peak", "spread_dh", "spread_lag", "spread_swing", "spread_area"]].median()
for k, v in sp.items():
    print(f"  {k:14s} = {v:.2f}")

# ============================================================
# FIG -- 4-panel composition breakdown
# ============================================================
def comp_pct(col, order):
    return (building_comp[col].value_counts(normalize=True) * 100).reindex(order).fillna(0)

panels = [
    ("Overheating", comp_pct("class_over", ORDER_OVER),
     ["all resilient", "mostly resilient", "mixed", "mostly overheating", "all overheating"]),
    ("Thermal lag", comp_pct("class_fast", ORDER_FAST),
     ["all slow", "mostly slow", "mixed", "mostly fast", "all fast"]),
    ("Daily swing", comp_pct("class_vol", ORDER_VOL),
     ["all stable", "mostly stable", "mixed", "mostly volatile", "all volatile"]),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (title, vals, labels) in zip(axes.flat, panels):
    ax.barh(range(5), vals.values, color=PAL5, edgecolor="white")
    ax.set_yticks(range(5)); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    for i, v in enumerate(vals.values):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_title(title); ax.set_xlabel("% of buildings"); ax.set_xlim(0, max(60, vals.max()*1.15))

ax = axes.flat[3]
btp = dom_pct.fillna(0)
ax.barh(range(4), btp.values, color=["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"], edgecolor="white")
ax.set_yticks(range(4)); ax.set_yticklabels([b.replace("_", " ") for b in BT_ORDER], fontsize=9)
ax.invert_yaxis()
for i, v in enumerate(btp.values):
    ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=9)
ax.set_title("Behavioral type (lag x swing)"); ax.set_xlabel("% of buildings"); ax.set_xlim(0, max(45, btp.max()*1.15))
fig.suptitle("Building composition across four thermal dimensions", fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage1_four_compositions.png")
plt.close(fig)

print("\nSaved: plots/stage1_mean_vs_peak_spread.png, plots/stage1_coolest_apartment_still_overheated.png,")
print("       plots/stage1_coolest_apartment_summary.png, plots/stage1_four_compositions.png")
print(">>> STAGE 1 complete.")
