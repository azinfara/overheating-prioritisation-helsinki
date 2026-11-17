"""
STAGE 8 -- Feature scoring
Overheating Risk Categorization Rubric ("Energy Label for Overheating").

Constructs greenery_score and attaches n_apartments (raw registry count) onto
the Target A table; joins Target B's cluster_rank; prints the univariate
Spearman r of all 9 locked features against BOTH targets (the "Feature set"
table from the plan); computes the 9-feature CV R2 benchmark for both targets
on the SAME design matrix (apples-to-apples); renders the R2 comparison figure.
This is the honesty anchor restated next to every rubric card in Stage 11 --
what a 9-feature hand score is approximating.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
from common import (
    PLOTS, PAL5, MIN_APTS, ALL_9_FEATURES, NUMERIC_7, DG_ORDER, DG_LABELS,
    load_buildings_metadata, load_apartment_metrics, load_hourly_indoor,
    compute_swing_range, build_apt_table, build_building_comp, attach_building_features,
    attach_risk_features, build_severity_cluster, set_style,
)

set_style()

print(f"""
{'='*72}
STAGE 8 -- FEATURE SCORING
{'='*72}
9 locked feature roles (not data-driven selection), tested against BOTH targets:
  {ALL_9_FEATURES}
Design matrix (identical for both targets): design_group + building_material
  dummies (drop_first) + {NUMERIC_7}
Model: sklearn LinearRegression, 5-fold CV (KFold(5, shuffle=True, random_state=42))
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
bc = attach_risk_features(bc, buildings_meta)

cluster_df, sil = build_severity_cluster(buildings_meta, min_apts=MIN_APTS)
bc["cluster_rank"] = cluster_df["cluster_rank"].reindex(bc.index)
bc["cluster_tier"] = cluster_df["cluster_tier"].reindex(bc.index)

print(f"\nbc table ready: n={len(bc)}; greenery_score/n_apartments/cluster_rank attached.")

# ============================================================
# UNIVARIATE EFFECT SIZE -- all 9 LOCKED features vs both targets.
# 7 numeric features -> signed Spearman r. The 2 categorical features
# (design_group, building_material) have no natural order for a numeric
# variable to correlate against, so they use eta (correlation ratio,
# sqrt(between-group SS / total SS)) -- the standard UNSIGNED effect-size
# for a nominal category vs a continuous/ordinal outcome. design_group is
# also shown via a naive CHRONOLOGICAL Spearman r for comparison -- it comes
# out much weaker than eta (confirms the age-structure hump-shape finding: a linear
# year-order badly undersells the real effect, which is why the point-card
# uses a lookup table instead of a linear year score).
# ============================================================
print("\n" + "="*50)
print("UNIVARIATE EFFECT SIZE -- ALL 9 LOCKED FEATURES vs BOTH TARGETS")
print("="*50)
rows = []
for feat in NUMERIC_7:
    dA = bc[[feat, "frac_over"]].dropna()
    rA, pA = stats.spearmanr(dA[feat], dA["frac_over"])
    dB = bc[[feat, "cluster_rank"]].dropna()
    rB, pB = stats.spearmanr(dB[feat], dB["cluster_rank"])
    rows.append({"feature": feat, "stat": "Spearman r", "n_A": len(dA), "r_vs_A": round(rA, 3),
                 "p_A<.05": pA < 0.05, "n_B": len(dB), "r_vs_B": round(rB, 3), "p_B<.05": pB < 0.05})


def eta(cats, y):
    """Correlation ratio: unsigned association strength, nominal category vs continuous/ordinal y."""
    d = pd.DataFrame({"c": cats, "y": y}).dropna()
    grand_mean = d["y"].mean()
    ss_between = d.groupby("c")["y"].apply(lambda g: len(g) * (g.mean() - grand_mean) ** 2).sum()
    ss_total = ((d["y"] - grand_mean) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)), len(d)


for feat in ["design_group", "building_material"]:
    eA, nA = eta(bc[feat].astype(str), bc["frac_over"])
    eB, nB = eta(bc[feat].astype(str), bc["cluster_rank"])
    rows.append({"feature": feat, "stat": "eta (unsigned)", "n_A": nA, "r_vs_A": round(eA, 3),
                 "p_A<.05": None, "n_B": nB, "r_vs_B": round(eB, 3), "p_B<.05": None})

uni = pd.DataFrame(rows)
print(uni.sort_values("r_vs_A", key=abs, ascending=False).to_string(index=False))

dg_ord = {g: i for i, g in enumerate(DG_ORDER)}
bc["dg_chrono"] = bc["design_group"].astype(str).map(dg_ord)
dA = bc[["dg_chrono", "frac_over"]].dropna()
dB = bc[["dg_chrono", "cluster_rank"]].dropna()
r_chrono_A, _ = stats.spearmanr(dA["dg_chrono"], dA["frac_over"])
r_chrono_B, _ = stats.spearmanr(dB["dg_chrono"], dB["cluster_rank"])
print(f"\ndesign_group naive CHRONOLOGICAL Spearman r (for comparison, NOT used in the card): "
      f"vs A={r_chrono_A:+.3f}, vs B={r_chrono_B:+.3f}")
print(f"  -- much weaker than eta above ({uni.loc[uni['feature']=='design_group','r_vs_A'].iloc[0]:.3f}/"
      f"{uni.loc[uni['feature']=='design_group','r_vs_B'].iloc[0]:.3f}) because the true relationship is a")
print(f"  HUMP (worst at 2004-2009), not monotonic with age -- confirms why the card uses a lookup")
print(f"  table for design_group instead of a linear year-based score.")

# ============================================================
# FIG A -- univariate effect size, the 7 NUMERIC features only, Target A vs B
# (categoricals moved to their own dedicated figure below -- an eta number
# squeezed onto the same signed-r axis obscured more than it showed)
# ============================================================
uni_numeric = uni[uni["stat"] == "Spearman r"].sort_values("r_vs_A", key=abs, ascending=True).set_index("feature")
fig, ax = plt.subplots(figsize=(9, 5.5))
ypos = np.arange(len(uni_numeric))
ax.barh(ypos - 0.2, uni_numeric["r_vs_A"], height=0.4, color="#2c7bb6", label="Target A (frac_over)")
ax.barh(ypos + 0.2, uni_numeric["r_vs_B"], height=0.4, color="#d7191c", label="Target B (cluster_rank)")
ax.set_yticks(ypos)
ax.set_yticklabels(uni_numeric.index, fontsize=10)
ax.axvline(0, color="grey", linewidth=0.8)
ax.set_xlabel("Spearman r")
ax.set_title("Why these 7 numeric features: real effect sizes vs both targets", fontsize=14, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage8_feature_selection_numeric.png")
plt.close(fig)
print("\nSaved: plots/stage8_feature_selection_numeric.png")

# ============================================================
# FIG B -- the 2 CATEGORICAL features, dedicated figure: mean target value per
# category (design_group / building_material), for both targets -- shows the
# actual hump/material pattern directly instead of an abstract unsigned eta
# number that can't be compared to the numeric features' signed r on one axis.
# ============================================================
mat_order_by_frac = bc.groupby("building_material", observed=True)["frac_over"].mean().sort_values().index.tolist()
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
panels = [
    ("design_group", "frac_over", DG_ORDER, [DG_LABELS[g] for g in DG_ORDER], "Target A: frac_over", "#2c7bb6"),
    ("design_group", "cluster_rank", DG_ORDER, [DG_LABELS[g] for g in DG_ORDER], "Target B: cluster_rank", "#d7191c"),
    ("building_material", "frac_over", mat_order_by_frac, mat_order_by_frac, "Target A: frac_over", "#2c7bb6"),
    ("building_material", "cluster_rank", mat_order_by_frac, mat_order_by_frac, "Target B: cluster_rank", "#d7191c"),
]
for ax, (feat, target_col, order, xlabels, title, color) in zip(axes.flat, panels):
    means = bc.groupby(feat, observed=True)[target_col].mean().reindex(order)
    counts = bc.groupby(feat, observed=True)[target_col].count().reindex(order)
    ax.bar(range(len(order)), means.values, color=color, edgecolor="white")
    for i, (v, n) in enumerate(zip(means.values, counts.values)):
        thin = n < 25
        ax.text(i, v, f"n={n}{'*' if thin else ''}", ha="center", va="bottom", fontsize=8,
                color="#d7191c" if thin else "#555", fontweight="bold" if thin else "normal")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xlabels, fontsize=8.5, rotation=20, ha="right")
    ax.set_ylabel(f"Mean {target_col}")
    ax.set_title(f"{feat} vs {title}", fontsize=11, fontweight="bold")
fig.suptitle("Why design_group and building_material are locked features: real category patterns",
             fontsize=14, fontweight="bold", y=1.00)
fig.text(0.5, -0.01, "* thin sample (n<25) -- descriptive only, not reliable for inference",
         ha="center", fontsize=9, color="#d7191c", style="italic")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage8_feature_selection_categorical.png", bbox_inches="tight")
plt.close(fig)
print("Saved: plots/stage8_feature_selection_categorical.png")

# ============================================================
# DESIGN MATRIX (shared for both targets)
# ============================================================
def make_X(df):
    d = df.copy()
    d["design_group"] = d["design_group"].astype(str)
    dg_dum = pd.get_dummies(d["design_group"], prefix="dg", drop_first=True)
    mat_dum = pd.get_dummies(d["building_material"].fillna("Unknown"), prefix="mat", drop_first=True)
    num = d[NUMERIC_7]
    return pd.concat([dg_dum, mat_dum, num], axis=1).astype(float)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
needed_cols = NUMERIC_7 + ["design_group", "building_material"]

print("\n" + "="*50)
print("9-FEATURE CV R2 BENCHMARK -- SAME DESIGN MATRIX, BOTH TARGETS")
print("="*50)
r2_rows = []
r2_data = {}
for name, target in [("Target A (frac_over)", "frac_over"), ("Target B (cluster_rank)", "cluster_rank")]:
    sub = bc.dropna(subset=[target] + needed_cols)
    X = make_X(sub)
    y = sub[target].astype(float)
    r2_cv_folds = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
    r2_cv = r2_cv_folds.mean()
    r2_in = LinearRegression().fit(X, y).score(X, y)
    print(f"\n{name}: n={len(sub)}")
    print(f"  in-sample R2 = {r2_in:.3f}")
    print(f"  5-fold CV R2 = {r2_cv:+.3f}   per-fold = {np.round(r2_cv_folds, 2).tolist()}")
    r2_rows.append({"target": name, "n": len(sub), "R2_insample": round(r2_in, 3), "R2_CV": round(r2_cv, 3)})
    r2_data[name] = {"cv": r2_cv, "insample": r2_in}

print("\n" + "="*50)
print("SUMMARY -- both targets show real, positive, roughly comparable out-of-sample signal")
print("="*50)
print(pd.DataFrame(r2_rows).to_string(index=False))
print("\nTarget A is somewhat more predictable; Target B has one unstable fold (negative R2 on one")
print("fold), consistent with the fuzzier Medium/High cluster boundary seen in Stage 7's figure --")
print("evidence Target B's feature-relationship is noisier / less stable than Target A's.")
print("\nThis is the LOCKED BENCHMARK for what the 9-feature point cards (Stage 9+) are approximating")
print("-- restate next to each card in the write-up as the honesty anchor (mirrors Stage")
print("5/11's R2~=0.17-0.22 framing).")

# ============================================================
# FIG -- R2 comparison, Target A vs Target B
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
names = list(r2_data.keys())
x = np.arange(len(names))
width = 0.32
cv_vals = [r2_data[n]["cv"] for n in names]
in_vals = [r2_data[n]["insample"] for n in names]
ax.bar(x - width/2, cv_vals, width=width, color=PAL5[0], label="5-fold CV R2 (out-of-sample)")
ax.bar(x + width/2, in_vals, width=width, color=PAL5[1], alpha=0.7, label="In-sample R2 (paler reference)")
for i, v in enumerate(cv_vals):
    ax.text(i - width/2, v + 0.005, f"{v:+.3f}", ha="center", fontsize=9, fontweight="bold")
for i, v in enumerate(in_vals):
    ax.text(i + width/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(["Target A\n(frac_over)", "Target B\n(cluster_rank)"])
ax.set_ylabel("R2 (9-feature LinearRegression)")
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_title("9-feature CV R2: comparable, modest signal for both targets")
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage8_r2_comparison.png")
plt.close(fig)

print("\nSaved: plots/stage8_r2_comparison.png")
print(">>> STAGE 8 complete.")
