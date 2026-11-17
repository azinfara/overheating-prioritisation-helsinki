"""
STAGE 4 -- Age structure (do once, correctly)
Within-building overheating spread in the Helsinki residential stock.

design_group x outcomes (dh_27, temp_mean, temp_max, peak_lag_hot, swing_range):
median table + automatic shape diagnosis. Expect: overheating/mean/peak/swing
are HUMP-shaped (worst at 2004-2009); lag is MONOTONIC (newer=shorter, r~-0.39).
Age is non-monotonic -- design_group categories are used throughout, never a
linear year term (guardrail #1).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from common import (
    PLOTS, MIN_APTS, DG_ORDER, DG_LABELS,
    load_buildings_metadata, load_apartment_metrics, load_hourly_indoor,
    compute_swing_range, build_apt_table, set_style,
    material_table, plot_material_bars,
)

set_style()

print(f"""
{'='*72}
STAGE 4 -- DEFINITIONS
{'='*72}
design_group   : pre1977 | 1977-2003 | 2004-2009 | 2010-2011 | post2012
                 (non-linear age bins tied to Finnish code revisions -- NEVER
                 treat as a linear year term; guardrail #1)
shape diagnosis: HUMP  = worst (max) outcome occurs at the middle group
                 (y2004_2009), i.e. buildings from just before the 2010/2012
                 code tightening are the worst, not simply 'older = worse'
                 MONOTONIC = outcome steadily rises or falls across the
                 5 ordered groups (checked via Spearman rho vs group order)
small-n caveat : y2004_2009 (n=26 buildings) and y2010_2011 (n=20 buildings)
                 are DESCRIPTIVE ONLY -- too few buildings for inference
{'='*72}""")

# ============================================================
# LOAD + PREPARE
# ============================================================
buildings_meta = load_buildings_metadata()
apt_metrics = load_apartment_metrics(merge_area_type=True)
print("Loading hourly indoor temperature to compute swing_range ...")
hourly = load_hourly_indoor()
swing_df = compute_swing_range(hourly)
del hourly

apt = build_apt_table(apt_metrics, buildings_meta, swing_df)

n_bld_by_group = buildings_meta.dropna(subset=["design_group"])["design_group"].value_counts().reindex(DG_ORDER)
print("Buildings per design_group (n):")
print(n_bld_by_group.to_string())

# ============================================================
# APARTMENT-LEVEL MEDIANS BY DESIGN GROUP
# ============================================================
OUTCOMES = ["dh_27", "temp_mean", "temp_max", "peak_lag_hot", "swing_range"]
print("\n" + "="*50)
print("APARTMENT-LEVEL MEDIANS BY DESIGN GROUP  (descriptive; clustering-inflated if tested)")
print("="*50)
med_table = apt.groupby("design_group", observed=True)[OUTCOMES].median().reindex(DG_ORDER)
print(med_table.round(2))

# ============================================================
# BUILDING-LEVEL CROSS-CHECK (precomputed bld_* columns)
# ============================================================
bld_cols = {"dh_27": "bld_dh_27_mean", "temp_mean": "bld_temp_mean", "temp_max": "bld_temp_max",
            "peak_lag_hot": "bld_peak_lag_hot_mean"}
print("\nBUILDING-LEVEL cross-check (precomputed bld_* means, by design_group):")
bm = buildings_meta.dropna(subset=["design_group"])
bld_med_table = bm.groupby("design_group", observed=True)[list(bld_cols.values())].median().reindex(DG_ORDER)
print(bld_med_table.round(2))

# ============================================================
# SHAPE DIAGNOSIS
# ============================================================
print("\n" + "="*50)
print("SHAPE DIAGNOSIS")
print("="*50)
ordinal = np.arange(len(DG_ORDER))


def diagnose(vals):
    vals = vals.reindex(DG_ORDER)
    peak_group = vals.idxmax()
    trough_group = vals.idxmin()
    r, p = stats.spearmanr(ordinal, vals.values)
    is_hump = peak_group == "y2004_2009"
    shape = "HUMP (peak at y2004_2009)" if is_hump else f"NOT hump (peak at {peak_group})"
    return shape, r, p, peak_group, trough_group


print("(spearman r here is over the 5 design_group MEDIANS -- n=5 points, a shape check, not an inferential test)")
for col in OUTCOMES:
    shape, r, p, peak_group, trough_group = diagnose(med_table[col])
    print(f"  {col:14s} shape={shape:28s}  spearman r(5-group medians)={r:+.2f}  p={p:.3g}  "
          f"[worst={peak_group}, best={trough_group}]")

# proper row-level check for the LAG monotonic claim (continuous year, not the 5-point group check above)
apt_year = apt.dropna(subset=["year_of_construction", "peak_lag_hot"])
lag_r_row, lag_p_row = stats.spearmanr(apt_year["year_of_construction"], apt_year["peak_lag_hot"])
print(f"\npeak_lag_hot vs year_of_construction, ROW-LEVEL (apartments, n={len(apt_year)}):")
print(f"  spearman r={lag_r_row:+.2f}, p={lag_p_row:.3g}   [expect r~-0.39 -- this is the real reportable stat,")
print(f"  the 5-point group-median check above is just a coarse shape sanity check, not the headline number]")

# ============================================================
# MATERIAL EFFECT ON THERMAL LAG (point 7) -- does building_material affect
# reaction speed the way design_group/age does? BUILDING-LEVEL: each building's
# lag = mean peak_lag_hot of its apartments (>= MIN_APTS), so large buildings
# don't dominate. Steel/Other are thin-sample and excluded; materials are plotted
# in the fixed MATERIAL_ORDER shared with Stage 3.
# ============================================================
print("\n" + "="*50)
print("MATERIAL EFFECT ON THERMAL LAG (peak_lag_hot, building-level)")
print("="*50)
bld_lag = apt.groupby("building_id")["peak_lag_hot"].agg(["mean", "count"])
bld_lag = bld_lag.loc[bld_lag["count"] >= MIN_APTS, "mean"]
mat_lag = material_table(bld_lag, buildings_meta)
print(mat_lag.round(2).to_string())
print("(count = number of buildings)")
lag_spread_material = mat_lag["median"].max() - mat_lag["median"].min()
# age spread on the same building-level basis, so the two are comparable
bld_lag_dg = bld_lag.to_frame().join(buildings_meta.set_index("building_id")["design_group"])
lag_by_dg = bld_lag_dg.groupby("design_group", observed=True)["mean"].median().reindex(DG_ORDER)
lag_spread_age = lag_by_dg.max() - lag_by_dg.min()
print(f"\nSpread across materials: {lag_spread_material:.2f} h   vs.   spread across design_group (age): {lag_spread_age:.2f} h")
print(f"-- material effect on lag is {'SMALLER' if lag_spread_material < lag_spread_age else 'LARGER'} than the age effect "
      f"({'age/design_group is the dominant driver of thermal lag' if lag_spread_material < lag_spread_age else 'material effect is at least as large as age'}).")

plot_material_bars(mat_lag, "{:.1f}h",
                   "Median building thermal lag, peak_lag_hot (hours)",
                   "Does building material affect reaction speed?", "#2c7bb6",
                   rf"{PLOTS}/stage4_material_effect_on_lag.png")
print("Saved: plots/stage4_material_effect_on_lag.png")

# ============================================================
# PASSIVE CONSISTENCY CHECK (Farahani 2023 BRI)
# ============================================================
print("\n" + "="*50)
print("PASSIVE CONSISTENCY CHECK")
print("="*50)
post2012_max = med_table.loc["post2012", "temp_max"]
peak_era_max = med_table.loc["y2004_2009", "temp_max"]
print(f"median temp_max: post2012={post2012_max:.2f} C vs y2004_2009 (peak era)={peak_era_max:.2f} C")
if post2012_max < peak_era_max:
    print("post2012 overheats LESS than the 2004-2009 peak -- consistent with Farahani 2023 (BRI).")
else:
    print("WARNING: post2012 does NOT overheat less than the 2004-2009 peak -- inconsistent with Farahani 2023 (BRI).")

# ============================================================
# FIG -- hump (dh_27) vs monotonic (lag)
# ============================================================
labels = [DG_LABELS[g] for g in DG_ORDER]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
vals = med_table["dh_27"].reindex(DG_ORDER)
colors = ["#2c7bb6" if g != "y2004_2009" else "#d7191c" for g in DG_ORDER]
ax.bar(labels, vals.values, color=colors, edgecolor="white")
ax.set_ylabel("Median dh_27 (degree-hours > 27C)")
ax.set_title("Overheating: HUMP shape (worst at 2004-2009)")
ax.set_xlabel("Construction era")

ax = axes[1]
vals = med_table["peak_lag_hot"].reindex(DG_ORDER)
ax.plot(labels, vals.values, marker="o", color="#2c7bb6", linewidth=2)
ax.set_ylabel("Median thermal lag (hours)")
ax.set_title(f"Thermal lag: MONOTONIC (newer=shorter, r={lag_r_row:+.2f})")
ax.set_xlabel("Construction era")

for ax in axes:
    ax.tick_params(axis="x", rotation=20)
fig.suptitle("Age structure is non-monotonic for overheating but monotonic for lag", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage4_hump_vs_monotonic.png")
plt.close(fig)

print("\nSaved: plots/stage4_hump_vs_monotonic.png")
print(">>> STAGE 4 complete.")
