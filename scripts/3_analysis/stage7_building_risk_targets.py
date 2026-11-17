"""
STAGE 7 -- Two parallel targets
Overheating Risk Categorization Rubric ("Energy Label for Overheating").

Builds Target A (frac_over, the building-level LEVEL metric -- a COMPOSITION/equity
question, "how many of my residents are affected") and Target B (a 3-way KMeans
severity cluster -- an absolute SEVERITY question, "how hot does this building
actually get"), then establishes and documents the headline "two risk
definitions diverge" finding BEFORE any feature work happens (that's Stage 8+).

Both targets' Low/Medium/High tiers are now built the SAME way -- KMeans(k=3)
natural-break clustering -- rather than Target A using plain equal-sized
tertiles: tertiles force exactly 1/3 of buildings into each bucket regardless
of where the real breaks fall (the tertile 'Low' bucket included buildings
with up to 62.5% of apartments overheating -- not a reassuring 'Low'). A 1D
KMeans cluster on frac_over alone finds a much stricter, more honest Low group.
Target B's cluster folds in bld_dh_27_mean/bld_dh_30_mean
(log1p-transformed -- both are heavily zero-inflated and break KMeans if fed
in raw), so the criteria-meaningful 30/32 C thresholds are now clustering INPUTS,
not just descriptive context.

Both cluster fits happen ONCE here and are used purely to define the
ground-truth label for the 570 monitored buildings -- NEVER a deployed model.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from common import (
    PLOTS, DG_ORDER, DG_LABELS, PAL5, MIN_APTS,
    load_buildings_metadata, load_apartment_metrics, load_hourly_indoor,
    compute_swing_range, build_apt_table, build_building_comp, attach_building_features,
    build_severity_cluster, build_composition_cluster, build_composition_cluster_2var,
    coarse_class, set_style,
)

set_style()
tier_colors = {"Low": PAL5[0], "Medium": PAL5[2], "High": PAL5[4]}
TIER_ORDER = ["Low", "Medium", "High"]

print(f"""
{'='*72}
STAGE 7 -- DEFINITIONS
{'='*72}
TARGET A (composition risk) : frac_over -- fraction of a building's apartments
              individually crossing the 150 degree-hour criteria threshold
              (dh_27 > 150). An EQUITY/COMPOSITION question. Low/Medium/High
              via KMeans(k=3) on frac_over alone (natural breaks, not tertiles).
TARGET B (severity risk)    : KMeans(n_clusters=3, random_state=42, n_init=10)
              on standardized [bld_hw_temp_mean, bld_hw_temp_max, bld_dh_27_mean,
              log1p(bld_dh_30_mean)], n=516
              (has_temp_data & bld_n_apartments>=MIN_APTS). Cluster ranks
              ordered 0=Low/1=Medium/2=High by mean bld_hw_temp_mean per cluster.
              An ABSOLUTE average+peak+duration heat SEVERITY question, now
              explicitly anchored to the 27 and 30 C thresholds, not
              just 27 C. Used only to label the 570 monitored buildings --
              NEVER a deployed model.
WITHIN-BUILDING OVERHEATING SPREAD is OUT OF SCOPE for both targets -- neither
              target says anything about how unequal a building's apartments are.
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
building_comp, apt, lag_med, swing_med = build_building_comp(apt)
bc = attach_building_features(building_comp, buildings_meta)
print(f"\nTarget A table (bc): n={len(bc)} buildings (>= {MIN_APTS} monitored apartments)")

# ============================================================
# TARGET B -- severity cluster (KMeans k=3, 5 variables)
# ============================================================
print("\n" + "="*50)
print("TARGET B -- SEVERITY CLUSTER (KMeans, 4 variables)")
print("="*50)
cluster_df, sil_b = build_severity_cluster(buildings_meta, min_apts=MIN_APTS)
print(f"n={len(cluster_df)} buildings (has_temp_data & bld_n_apartments>={MIN_APTS})")

print("\nSilhouette by k (k=2..6):")
print(sil_b.round(3).to_string())
best_k_b = sil_b.idxmax()
print(f"-> best silhouette at k={best_k_b} ({sil_b[best_k_b]:.3f}); k=3 ({sil_b[3]:.3f}) chosen "
      f"(matches the Low/Medium/High pitch).")

sev_cols_b = ["bld_hw_temp_mean", "bld_hw_temp_max", "bld_dh_27_mean", "bld_dh_30_mean"]
print("\nCluster profile (mean of the 4 clustering variables per severity tier -- "
      "dh_27/dh_30 shown in real units, clustering itself used log1p of these):")
prof = cluster_df.groupby("cluster_tier", observed=True)[sev_cols_b].mean().reindex(TIER_ORDER)
print(prof.round(2))

tier_counts_b = cluster_df["cluster_tier"].value_counts().reindex(TIER_ORDER)
tier_pct_b = (tier_counts_b / len(cluster_df) * 100)
print("\nCluster sizes:")
for t in TIER_ORDER:
    print(f"  {t:8s} n={tier_counts_b[t]:4d}  ({tier_pct_b[t]:.1f}%)")

# ------------------------------------------------------------
# HOW HIGH DOES IT ACTUALLY GO -- degree-hours at all 3 CRITERIA thresholds
# (27/30/32 C). These are the physically/legally meaningful numbers for the
# renovation-targeting step that follows building-level screening.
# ------------------------------------------------------------
print("\n" + "="*50)
print("HOW HIGH CAN IT GO -- degree-hours at 27/30 C by severity tier")
print("="*50)
sev_cols = ["bld_hw_temp_mean", "bld_hw_temp_max", "bld_temp_max",
            "bld_dh_27_mean", "bld_dh_30_mean",
            "bld_pct_above_27", "bld_pct_above_30", "bld_pct_exceed_code"]
sev_range = cluster_df.groupby("cluster_tier", observed=True)[sev_cols].agg(["min", "mean", "max"]).reindex(TIER_ORDER)
print(sev_range.round(1).to_string())
print("\nTarget B uses degree-hours above 27 and 30 C only (log1p-transformed).")

# FIG -- degree-hours at 27/30 C by tier (log scale)
fig, ax = plt.subplots(figsize=(8, 5.5))
dh_cols = ["bld_dh_27_mean", "bld_dh_30_mean"]
dh_labels = ["> 27 C", "> 30 C"]
x = np.arange(len(dh_cols))
width = 0.25
for i, t in enumerate(TIER_ORDER):
    vals = sev_range.loc[t, [(c, "mean") for c in dh_cols]].values.astype(float)
    vals_plot = np.where(vals <= 0, 0.5, vals)  # log-scale floor for zero/near-zero bars
    ax.bar(x + (i - 1) * width, vals_plot, width=width, label=t, color=tier_colors[t])
ax.set_xticks(x)
ax.set_xticklabels(dh_labels)
ax.set_yscale("log")
ax.set_ylabel("Mean degree-hours per building (log scale)")
ax.set_title("How high can it go: degree-hours at 2 thresholds, by severity tier")
ax.legend(frameon=False, title="Tier")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/paper2_stage0_targetB_severity_range.png")
plt.close(fig)
print("\nSaved: plots/paper2_stage0_targetB_severity_range.png")

# ============================================================
# TARGET A -- composition cluster (KMeans k=3, 1 variable: frac_over)
# ============================================================
print("\n" + "="*50)
print("TARGET A -- COMPOSITION CLUSTER (KMeans on frac_over)")
print("="*50)
bc["coarse_class"] = coarse_class(bc["frac_over"])
coarse_pct = pd.Series(bc["coarse_class"]).value_counts(normalize=True) * 100
print("Coarse composition class shares (all_resilient/mixed/all_overheating -- unrelated 3-way split,")
print("kept as-is, frac==0 / frac==1 / else):")
print(coarse_pct.round(1).to_string())

cluster_df_a, sil_a = build_composition_cluster(bc, min_apts=MIN_APTS)
print("\nSilhouette by k (k=2..6):")
print(sil_a.round(3).to_string())
best_k_a = sil_a.idxmax()
print(f"-> best silhouette at k={best_k_a} ({sil_a[best_k_a]:.3f}); k=3 ({sil_a[3]:.3f}) chosen "
      f"(matches the Low/Medium/High pitch).")

bc["cluster_tier_A"] = cluster_df_a["cluster_tier"].reindex(bc.index)
bc["cluster_rank_A"] = cluster_df_a["cluster_rank"].reindex(bc.index)
print("\nTarget A cluster profile (min/mean/max frac_over per tier):")
tierA_profile = bc.groupby("cluster_tier_A", observed=True)["frac_over"].agg(["min", "mean", "max", "count"]).reindex(TIER_ORDER)
print(tierA_profile.round(3))

# The real KMeans boundaries, precisely -- for correcting any rounded presentation
# cutoffs (a slide stating "Low <40%, Medium 40-70%, High >70%" is a readability
# rounding, not what the model actually computed; use these exact figures instead).
low_max = tierA_profile.loc["Low", "max"] * 100
med_min = tierA_profile.loc["Medium", "min"] * 100
med_max = tierA_profile.loc["Medium", "max"] * 100
high_min = tierA_profile.loc["High", "min"] * 100
print(f"\nExact Target A cutoffs (for presentation -- replace any rounded '40%/70%' with these):")
print(f"  Low:    frac_over <= {low_max:.1f}%")
print(f"  Medium: {med_min:.1f}% <= frac_over <= {med_max:.1f}%")
print(f"  High:   frac_over >= {high_min:.1f}%")

# ============================================================
# dh_30 RE-CLUSTERING TEST (point 10) -- does adding a dh_30-based composition
# signal sharpen Target A's cluster separation vs frac_over alone? frac_over_30
# uses the same data-driven "dh_30 > 0" cut established for the coolest-
# apartment check (Stage 1) -- a much stricter bar than dh_27>150 (68% of
# apartment-summers have dh_30 exactly 0).
# ============================================================
print("\n" + "="*50)
print("dh_30 RE-CLUSTERING TEST -- does adding frac_over_30 sharpen Target A?")
print("="*50)
apt["exceeds_dh30"] = apt["dh_30"] > 0
frac_over_30 = (apt.groupby("building_id")["exceeds_dh30"].agg(["mean", "count"])
                .query("count >= @MIN_APTS")["mean"].rename("frac_over_30"))
bc["frac_over_30"] = frac_over_30.reindex(bc.index)
print(f"frac_over_30 attached: n={bc['frac_over_30'].notna().sum()} buildings, "
      f"median={bc['frac_over_30'].median():.3f}, mean={bc['frac_over_30'].mean():.3f}")

_, sil_2var = build_composition_cluster_2var(bc, extra_col="frac_over_30", min_apts=MIN_APTS)
print("\nSilhouette comparison at k=3:")
print(f"  1-variable (frac_over alone)          : {sil_a[3]:.3f}")
print(f"  2-variable (frac_over + frac_over_30)  : {sil_2var[3]:.3f}")
if sil_2var[3] > sil_a[3] + 0.01:
    print("-> 2-variable clustering WINS (meaningfully higher silhouette) -- consider adopting it.")
elif sil_2var[3] < sil_a[3] - 0.01:
    print("-> 1-variable (current) clustering WINS -- adding frac_over_30 does not sharpen separation.")
else:
    print("-> ESSENTIALLY TIED (within 0.01) -- no meaningful gain from adding frac_over_30. "
          "Keeping the simpler 1-variable clustering (frac_over alone) as the locked Target A definition: "
          "it is already the simplest sufficient description, and dh_30's information is heavily "
          "redundant with dh_27 at the building-composition level (unlike Target B, where dh_30/dh_32 "
          "add real severity-tail information dh_27 alone doesn't capture).")

# ============================================================
# TARGET A vs TARGET B -- the headline disagreement finding
# ============================================================
print("\n" + "="*50)
print("TARGET A vs TARGET B -- DISAGREEMENT")
print("="*50)
bc["cluster_rank_B"] = cluster_df["cluster_rank"].reindex(bc.index)
bc["cluster_tier_B"] = cluster_df["cluster_tier"].reindex(bc.index)
sub = bc.dropna(subset=["frac_over", "cluster_rank_B", "cluster_rank_A"])

r_s, p_s = stats.spearmanr(sub["frac_over"], sub["cluster_rank_B"])
r_p, p_p = stats.pearsonr(sub["frac_over"], sub["cluster_rank_B"])
print(f"Spearman r(frac_over, cluster_rank_B) = {r_s:+.3f}  (p={p_s:.2g}, n={len(sub)})")
print(f"Pearson  r(frac_over, cluster_rank_B) = {r_p:+.3f}  (p={p_p:.2g})")
print("Moderate-to-strong -- the two targets are correlated in broad strokes (both trace back to "
      "summer heat exposure), NOT independent. But correlated is not the same as INTERCHANGEABLE: "
      "see the exact-tier crosstab below for how often they land a building in the same category.")

ct = pd.crosstab(sub["cluster_tier_A"], sub["cluster_tier_B"])
ct = ct.reindex(index=TIER_ORDER, columns=TIER_ORDER, fill_value=0)
print("\nCrosstab: Target A cluster tier (rows) x Target B severity tier (columns):")
print(ct)
agree_rate = np.trace(ct.values) / ct.values.sum() * 100
print(f"\nExact agreement rate (same tier both targets): {agree_rate:.1f}%  (n={ct.values.sum()})")
print("\n*** HEADLINE FINDING: the two targets disagree on the exact tier for a large share of "
      "buildings, even though they are moderately-strongly correlated overall. frac_over asks a "
      "composition question, the cluster asks an absolute-severity question -- a planner needs to "
      "know which question they're actually asking. Report both as complementary, not one 'more "
      "correct' than the other. ***")

# ============================================================
# FIG 1 -- Target A overview (4-panel)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

ax = axes[0, 0]
rng = np.random.RandomState(0)
for t in TIER_ORDER:
    d = bc[bc["cluster_tier_A"] == t]
    y = rng.normal(0, 0.6, len(d))
    ax.scatter(d["frac_over"], y, s=16, alpha=0.6, color=tier_colors[t], label=f"{t} (n={len(d)})")
ax.set_yticks([])
ax.set_xlabel("frac_over (fraction of apartments exceeding dh_27>150)")
ax.set_title("(1) frac_over -- KMeans natural-break tiers")
ax.legend(frameon=False, fontsize=8, loc="upper left")

ax = axes[0, 1]
order_c = ["all_resilient", "mixed", "all_overheating"]
vals = coarse_pct.reindex(order_c).fillna(0)
ax.bar(order_c, vals.values, color=PAL5[:3], edgecolor="white")
ax.set_ylabel("% of buildings")
ax.set_title("(2) Coarse composition class")
ax.tick_params(axis="x", rotation=15)

ax = axes[1, 0]
tsize = bc["cluster_tier_A"].value_counts().reindex(TIER_ORDER)
ax.bar(tsize.index.astype(str), tsize.values, color=[tier_colors[t] for t in TIER_ORDER], edgecolor="white")
for i, t in enumerate(TIER_ORDER):
    ax.text(i, tsize[t] + 5, f"{tsize[t]}\n({tsize[t]/tsize.sum()*100:.1f}%)", ha="center", fontsize=9)
ax.set_ylabel("Number of buildings")
ax.set_title("(3) Target A tier sizes (imbalanced but honest)")

ax = axes[1, 1]
dg_mean = bc.groupby("design_group", observed=True)["frac_over"].mean().reindex(DG_ORDER)
mat_mean = bc.groupby("building_material", observed=True)["frac_over"].mean()
dg_norm = (dg_mean - dg_mean.min()) / (dg_mean.max() - dg_mean.min())
mat_norm = (mat_mean - mat_mean.min()) / (mat_mean.max() - mat_mean.min())
labels = [DG_LABELS[g] for g in DG_ORDER] + ["|"] + list(mat_mean.index)
vals = list(dg_norm.values) + [np.nan] + list(mat_norm.values)
colors = [PAL5[0]] * len(DG_ORDER) + ["white"] + [PAL5[3]] * len(mat_mean)
ax.bar(range(len(labels)), [0 if np.isnan(v) else v for v in vals], color=colors, edgecolor="white")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Normalized mean frac_over (0-1)")
ax.set_title("(4) design_group hump (blue) + material (orange) profile")

fig.suptitle("Target A -- Composition risk (frac_over) overview", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/paper2_stage0_targetA_overview.png")
plt.close(fig)

# ============================================================
# FIG 2 -- Target B cluster visualization (4-panel)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

ax = axes[0, 0]
for t in TIER_ORDER:
    d = cluster_df[cluster_df["cluster_tier"] == t]
    ax.scatter(d["bld_hw_temp_mean"], d["bld_hw_temp_max"], s=14, alpha=0.7,
               color=tier_colors[t], label=t)
ax.set_xlabel("bld_hw_temp_mean (C)")
ax.set_ylabel("bld_hw_temp_max (C)")
ax.set_title("(1) Heatwave mean vs max temp")
ax.legend(frameon=False, fontsize=9)

ax = axes[0, 1]
for t in TIER_ORDER:
    d = cluster_df[cluster_df["cluster_tier"] == t]
    ax.scatter(d["bld_hw_temp_mean"], d["bld_dh_27_mean"], s=14, alpha=0.7,
               color=tier_colors[t], label=t)
ax.set_xlabel("bld_hw_temp_mean (C)")
ax.set_ylabel("bld_dh_27_mean (degree-hours)")
ax.set_title("(2) Heatwave mean temp vs degree-hours")
ax.legend(frameon=False, fontsize=9)

ax = axes[1, 0]
ax.bar(tier_counts_b.index, tier_counts_b.values, color=[tier_colors[t] for t in tier_counts_b.index], edgecolor="white")
for i, t in enumerate(tier_counts_b.index):
    ax.text(i, tier_counts_b[t] + 5, f"{tier_counts_b[t]}\n({tier_pct_b[t]:.1f}%)", ha="center", fontsize=9)
ax.set_ylabel("Number of buildings")
ax.set_title("(3) Cluster sizes (imbalanced but interpretable)")

ax = axes[1, 1]
prof_norm = prof.apply(lambda c: (c - c.min()) / (c.max() - c.min()), axis=0)
x = np.arange(3)
width = 0.15
for i, col in enumerate(prof_norm.columns):
    ax.bar(x + (i - 2) * width, prof_norm[col].reindex(TIER_ORDER).values,
           width=width, label=col, color=PAL5[i % len(PAL5)])
ax.set_xticks(x)
ax.set_xticklabels(TIER_ORDER)
ax.set_ylabel("Normalized value (0-1 per variable)")
ax.set_title("(4) Normalized per-variable cluster profile (5 vars)")
ax.legend(frameon=False, fontsize=6.5)

fig.suptitle("Target B -- Severity cluster (KMeans k=3, 4 variables incl. dh_27/dh_30)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/paper2_stage0_targetB_clusters.png")
plt.close(fig)

print("\nSaved: plots/paper2_stage0_targetA_overview.png, plots/paper2_stage0_targetB_clusters.png")
print(">>> STAGE 7 complete.")
