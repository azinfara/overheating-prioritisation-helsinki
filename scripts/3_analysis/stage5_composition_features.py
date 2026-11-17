"""
STAGE 5 -- Composition x features (the core)
Within-building overheating spread in the Helsinki residential stock.

For each classification (overheating, lag, swing, behavior):
  (a) profile UNIFORM ("all_X") buildings by construction/urban features
  (b) predict composition LEVEL (fraction) from features -- sklearn R2, VIF-checked
  (c) predict SPREAD (within-building spread) from the same features
  (d) predict building MEAN and PEAK (real units, not population-relative) from features
  (e) classify composition CLASS (all_neg / in_between / all_pos) from features
KEY thesis check: LEVEL/MEAN/PEAK should be feature-predictable (R2~0.1-0.2),
SPREAD should NOT (R2 near zero / negative out-of-sample).
Material effect via Cramer's V (steel excluded, n=4).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from common import (
    PLOTS, DG_ORDER, DG_LABELS, CONSTRUCTION_FEATURES, URBAN_FEATURES, PAL5,
    load_buildings_metadata, load_apartment_metrics, load_hourly_indoor,
    compute_swing_range, build_apt_table, build_building_comp, attach_building_features, set_style,
)

set_style()

print(f"""
{'='*72}
STAGE 5 -- DEFINITIONS
{'='*72}
LEVEL       : the composition fraction itself (frac_over / frac_fast / frac_vol /
              frac of the building's dominant behavior type) -- 'how bad on average',
              population-relative (fast/volatile are defined vs the global median)
SPREAD      : within-building spread of the matching metric (spread_temp for
              overheating, spread_lag for lag, spread_swing for swing, behavior
              entropy for behavior) -- 'how unequal apartments are' (renamed
              from MIXEDNESS -- that word collided with the Stage-1 'mixed'
              composition class and implied the opposite of what it measured)
MEAN / PEAK : building-level mean and max of the RAW metric in real units
              (e.g. mean/max dh_27 per building) -- not population-relative,
              a second, more direct way to ask 'is the overall building bad?'
UNIFORM     : buildings where frac==0 (all_neg) or frac==1 (all_pos)
CLASS       : the 5-bucket composition label collapsed to 3: all_neg / in_between
              (mostly_neg + mixed + mostly_pos) / all_pos -- can features tell
              a uniformly-resilient building from a uniformly-overheating one?
Feature blocks (guardrail #7, lat/lon excluded, steel excluded from material tests):
  CONSTRUCTION = {CONSTRUCTION_FEATURES}
  URBAN        = {URBAN_FEATURES}
Model        : sklearn LinearRegression/LogisticRegression (NOT patsy/statsmodels formulas)
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

# behavior: per-building fraction of the DOMINANT type (purity) + 4-way entropy (diversity)
type_frac = (apt.groupby("building_id")["behavior_type"]
             .value_counts(normalize=True).unstack(fill_value=0))
bc["frac_behavior"] = type_frac.max(axis=1).reindex(bc.index)  # purity of dominant type
bc["entropy_behavior"] = (-(type_frac * np.log(type_frac.replace(0, np.nan))).sum(axis=1)).reindex(bc.index)

# ============================================================
# VIF CHECK -- drop collinear features per guardrail #5
# ============================================================
print("\n" + "="*50)
print("VIF CHECK")
print("="*50)
numeric_feats = ["n_floors", "gross_floor_area_m2", "n_apartments_hsy",
                  "impervious_fraction", "tree_fraction", "canyon_hw_ratio",
                  "distance_to_sea_m", "road_density_100m", "dist_major_road_m", "ALUETEHOK"]
vif_df = bc[numeric_feats].dropna()
r_canyon_floors = vif_df["canyon_hw_ratio"].corr(vif_df["n_floors"])
print(f"corr(canyon_hw_ratio, n_floors) = {r_canyon_floors:.2f}   [guardrail flags drop if r>0.9]")
vifs = pd.Series(
    [variance_inflation_factor(vif_df.values, i) for i in range(vif_df.shape[1])],
    index=numeric_feats).sort_values(ascending=False)
print(vifs.round(2).to_string())
DROP_COLLINEAR = r_canyon_floors > 0.9 or vifs["canyon_hw_ratio"] > 10
if DROP_COLLINEAR:
    numeric_feats.remove("canyon_hw_ratio")
    print("-> dropping canyon_hw_ratio (collinear with n_floors)")
else:
    print(f"-> canyon_hw_ratio kept (r={r_canyon_floors:.2f}, VIF={vifs['canyon_hw_ratio']:.1f}, below drop threshold)")

# ============================================================
# FEATURE MATRIX for regressions (one-hot design_group + material, numeric rest)
# ============================================================
NON_STEEL = bc["building_material"].fillna("Unknown") != "Steel"


def make_X(df):
    d = df.copy()
    d["design_group"] = d["design_group"].astype(str)
    dg_dum = pd.get_dummies(d["design_group"], prefix="dg", drop_first=True)
    mat_dum = pd.get_dummies(d["building_material"].fillna("Unknown"), prefix="mat", drop_first=True)
    num = d[numeric_feats]
    X = pd.concat([dg_dum, mat_dum, num], axis=1)
    return X.astype(float)


CLASSIFICATIONS = [
    ("overheating", "frac_over", "spread_temp", "class_over", "dh_27", "resilient", "overheating"),
    ("lag",         "frac_fast", "spread_lag",  "class_fast", "peak_lag_hot", "slow", "fast"),
    ("swing",       "frac_vol",  "spread_swing","class_vol",  "swing_range", "stable", "volatile"),
    ("behavior",    "frac_behavior", "entropy_behavior", None, None, None, None),
]

profile_feats = ["n_floors", "gross_floor_area_m2", "impervious_fraction", "tree_fraction",
                  "canyon_hw_ratio", "distance_to_sea_m"]

r2_summary = []
meanpeak_summary = []
class_summary = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, frac_col, spread_col, class_col, raw_col, neg_label, pos_label in CLASSIFICATIONS:
    print("\n" + "="*50)
    print(f"CLASSIFICATION: {name.upper()}")
    print("="*50)

    # (a) profile uniform buildings
    if class_col is not None:
        print(f"\n(a) Feature means by composition class:")
        prof = bc.groupby(class_col, observed=True)[profile_feats].mean()
        print(prof.round(2))
    else:
        uniform = bc["frac_behavior"] >= 0.9
        print(f"\n(a) Feature means, uniform (purity>=0.9, n={uniform.sum()}) vs mixed (n={(~uniform).sum()}):")
        print(bc.groupby(uniform)[profile_feats].mean().round(2).rename(index={True: "uniform", False: "mixed"}))

    # univariate effect sizes vs LEVEL (frac_col), sorted by |r|
    print(f"\n    Univariate Spearman r vs {frac_col} (LEVEL), sorted by |r|:")
    rows = []
    for feat in numeric_feats:
        d = bc[[feat, frac_col]].dropna()
        if len(d) < 10:
            continue
        r, p = stats.spearmanr(d[feat], d[frac_col])
        rows.append({"feature": feat, "r": round(r, 3), "meaningful": abs(r) >= 0.1 and p < 0.05})
    print(pd.DataFrame(rows).sort_values("r", key=abs, ascending=False).to_string(index=False))

    # (b) LEVEL model
    sub = bc.dropna(subset=[frac_col] + numeric_feats + ["design_group", "building_material"])
    X = make_X(sub)
    y_level = sub[frac_col].astype(float)
    r2_level = LinearRegression().fit(X, y_level).score(X, y_level)
    r2_level_cv = cross_val_score(LinearRegression(), X, y_level, cv=kf, scoring="r2").mean()
    print(f"\n(b) LEVEL model: R2({frac_col} ~ design_group + n_floors + size + material + urban)")
    print(f"    in-sample = {r2_level:.3f}   5-fold CV = {r2_level_cv:+.3f}   [expect ~0.1-0.2]")

    # (c) OVERHEATING SPREAD model
    sub_i = bc.dropna(subset=[spread_col] + numeric_feats + ["design_group", "building_material"])
    Xi = make_X(sub_i)
    y_ineq = sub_i[spread_col].astype(float)
    r2_ineq = LinearRegression().fit(Xi, y_ineq).score(Xi, y_ineq)
    r2_ineq_cv = cross_val_score(LinearRegression(), Xi, y_ineq, cv=kf, scoring="r2").mean()
    print(f"(c) OVERHEATING SPREAD model: R2({spread_col} ~ same features)")
    print(f"    in-sample = {r2_ineq:.3f}   5-fold CV = {r2_ineq_cv:+.3f}   [expect near-zero / negative out-of-sample]")

    r2_summary.append({"classification": name, "R2_level_insample": round(r2_level, 3),
                        "R2_level_CV": round(r2_level_cv, 3), "R2_ineq_insample": round(r2_ineq, 3),
                        "R2_ineq_CV": round(r2_ineq_cv, 3), "n_level": len(sub), "n_inequality": len(sub_i)})

    # (d) BUILDING MEAN & PEAK (raw units, not population-relative) -- skipped for behavior (no scalar raw column)
    if raw_col is not None:
        bld_mean = apt.groupby("building_id")[raw_col].mean().rename("bld_mean")
        bld_peak = apt.groupby("building_id")[raw_col].max().rename("bld_peak")
        mp = bc.join(bld_mean).join(bld_peak)

        sub_mean = mp.dropna(subset=["bld_mean"] + numeric_feats + ["design_group", "building_material"])
        Xmn = make_X(sub_mean)
        r2_mean_cv = cross_val_score(LinearRegression(), Xmn, sub_mean["bld_mean"].astype(float), cv=kf, scoring="r2").mean()

        sub_peak = mp.dropna(subset=["bld_peak"] + numeric_feats + ["design_group", "building_material"])
        Xpk = make_X(sub_peak)
        r2_peak_cv = cross_val_score(LinearRegression(), Xpk, sub_peak["bld_peak"].astype(float), cv=kf, scoring="r2").mean()

        print(f"\n(d) BUILDING MEAN/PEAK (raw {raw_col}, real units):")
        print(f"    5-fold CV R2, mean-per-building = {r2_mean_cv:+.3f}")
        print(f"    5-fold CV R2, peak(max)-per-building = {r2_peak_cv:+.3f}")
        meanpeak_summary.append({"classification": name, "raw_col": raw_col,
                                  "R2_mean_CV": round(r2_mean_cv, 3), "R2_peak_CV": round(r2_peak_cv, 3)})

    # (e) COMPOSITION CLASS (3-way: all_neg / in_between / all_pos) -- classification, not regression
    if class_col is not None:
        def collapse3(c):
            if c == f"all_{neg_label}":
                return "all_neg"
            if c == f"all_{pos_label}":
                return "all_pos"
            return "in_between"

        subc = bc.dropna(subset=[class_col] + numeric_feats + ["design_group", "building_material"])
        y3_labels = subc[class_col].map(collapse3)
        y3, uniques = pd.factorize(y3_labels)
        Xc = make_X(subc)
        baseline_acc = y3_labels.value_counts(normalize=True).max()
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
        cv_acc = cross_val_score(clf, Xc, y3, cv=kf, scoring="accuracy").mean()
        print(f"\n(e) COMPOSITION CLASS (3-way: all_{neg_label} / in_between / all_{pos_label}), n={len(subc)}:")
        print(f"    class shares: {y3_labels.value_counts(normalize=True).round(3).to_dict()}")
        print(f"    baseline (always-majority-class) accuracy = {baseline_acc:.3f}")
        print(f"    5-fold CV logistic regression accuracy   = {cv_acc:.3f}   "
              f"(improvement over baseline: {cv_acc - baseline_acc:+.3f})")
        class_summary.append({"classification": name, "baseline_acc": round(baseline_acc, 3),
                               "cv_acc": round(cv_acc, 3), "improvement": round(cv_acc - baseline_acc, 3), "n": len(subc)})

    # material Cramer's V (steel excluded)
    if class_col is not None:
        ct = pd.crosstab(bc.loc[NON_STEEL, "building_material"], bc.loc[NON_STEEL, class_col])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, exp = stats.chi2_contingency(ct)
            v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
            print(f"\n    Material x {class_col} (steel excluded): chi2={chi2:.1f}, p={p:.3g}, Cramer's V={v:.3f}")

print("\n" + "="*50)
print("SUMMARY 1 -- LEVEL vs SPREAD  (the key thesis check)")
print("="*50)
print(pd.DataFrame(r2_summary).to_string(index=False))
print("\nThesis: LEVEL should be clearly feature-predictable; SPREAD should not be.")

print("\n" + "="*50)
print("SUMMARY 2 -- BUILDING MEAN vs PEAK (raw units)")
print("="*50)
print(pd.DataFrame(meanpeak_summary).to_string(index=False))
print("\nIf mean/peak R2 look like LEVEL's R2, the population-relative fraction wasn't hiding anything;")
print("if they diverge, the choice of 'fraction vs raw units' matters for what features can predict.")

print("\n" + "="*50)
print("SUMMARY 3 -- COMPOSITION CLASS (all_neg / in_between / all_pos) explainability")
print("="*50)
print(pd.DataFrame(class_summary).to_string(index=False))
print("\nIf CV accuracy clears baseline by a real margin, features CAN tell uniform-good from uniform-bad buildings.")

# ============================================================
# FIG A -- overheating composition by era (stacked bars)
# ============================================================
order_c = ["all_resilient", "mostly_resilient", "mixed", "mostly_overheating", "all_overheating"]
comp_era = (bc.reset_index()[["building_id", "class_over"]]
            .merge(buildings_meta[["building_id", "design_group"]], on="building_id", how="left")
            .groupby("design_group", observed=True)["class_over"].value_counts(normalize=True)
            .unstack().reindex(DG_ORDER).reindex(columns=order_c).fillna(0) * 100)
fig, ax = plt.subplots(figsize=(8, 4.5))
bottom = np.zeros(len(comp_era))
for cls, color in zip(order_c, PAL5):
    ax.bar([DG_LABELS[g] for g in comp_era.index], comp_era[cls], bottom=bottom,
           label=cls.replace("_", " "), color=color, edgecolor="white", linewidth=0.5)
    bottom += comp_era[cls].values
ax.set_ylabel("% of buildings"); ax.set_xlabel("Construction era")
ax.set_title("Overheating composition of buildings, by era")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=9)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage5_overheating_by_era.png")
plt.close(fig)

# ============================================================
# FIG B -- feature effect sizes: LEVEL vs SPREAD (overheating classification)
# ============================================================
rows_level, rows_ineq = [], []
for feat in numeric_feats:
    d1 = bc[[feat, "frac_over"]].dropna()
    r1, _ = stats.spearmanr(d1[feat], d1["frac_over"])
    rows_level.append((feat, r1))
    d2 = bc[[feat, "spread_temp"]].dropna()
    r2, _ = stats.spearmanr(d2[feat], d2["spread_temp"])
    rows_ineq.append((feat, r2))
lvl = pd.Series(dict(rows_level)).sort_values(key=abs, ascending=False)
ineq = pd.Series(dict(rows_ineq)).reindex(lvl.index)

fig, ax = plt.subplots(figsize=(8, 5))
ypos = np.arange(len(lvl))
ax.barh(ypos - 0.2, lvl.values, height=0.4, color="#2c7bb6", label="LEVEL (frac_over)")
ax.barh(ypos + 0.2, ineq.values, height=0.4, color="#d7191c", label="SPREAD (spread_temp)")
ax.set_yticks(ypos); ax.set_yticklabels(lvl.index, fontsize=9)
ax.axvline(0, color="grey", linewidth=0.8)
ax.set_xlabel("Spearman r")
ax.set_title("Features predict overheating LEVEL, not within-building SPREAD")
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage5_level_vs_inequality_effects.png")
plt.close(fig)

print("\nSaved: plots/stage5_overheating_by_era.png, plots/stage5_level_vs_inequality_effects.png")
print(">>> STAGE 5 complete.")
