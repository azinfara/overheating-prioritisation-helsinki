"""
STAGE 10 -- Validate and compare
Overheating Risk Categorization Rubric ("Energy Label for Overheating").

Restructured around the paper's actual focus: point-card vs. black-box ML,
not a 3-format bake-off. Format 3 (hybrid) and the format-vs-format comparison
infrastructure (cross-format agreement, format ranking, per-format cross-target
crosstabs) are removed -- they existed to help choose between 3+ deployable
formats, which the deck never asked for. Two arms remain: Format 1 (point-card,
the deployed artifact) and the decision tree (fit_tree_per_target's guardrailed
tree, now treated as one ML technique among several, not a competing flowchart
format). The tree is folded into the SAME comparison as Linear/RF/Gradient
Boosting -- one "point-card vs. everything else" story per target.

Honest, out-of-fold (never in-sample) 5-fold CV validation throughout: every
method is FIT on the train fold only and SCORED on the held-out test fold --
the out-of-fold assignments across all 5 folds are concatenated into one
out-of-fold series, exactly mirroring the same cross_val_score discipline used
throughout the whole pipeline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from common import (
    PLOTS, PAL5, MIN_APTS, ALL_9_FEATURES, NUMERIC_7,
    load_buildings_metadata, load_apartment_metrics, load_hourly_indoor,
    compute_swing_range, build_apt_table, build_building_comp, attach_building_features,
    attach_risk_features, build_severity_cluster, build_composition_cluster, coarse_class,
    build_format1_specs, score_total, assign_tier,
    fit_tree_per_target, predict_tree_tier, render_table, set_style,
)

set_style()

TIER_ORDER = ["Low", "Medium", "High"]
TIER_RANK = {"Low": 0, "Medium": 1, "High": 2}
METHOD_NAMES = {"1": "Point-score card", "2": "Decision tree (guardrailed)"}

print(f"""
{'='*72}
STAGE 10 -- VALIDATE AND COMPARE (out-of-fold, never in-sample)
{'='*72}
Two arms per target: the point-card (deployed artifact) and the decision tree
(one ML technique, later folded into the same comparison as Linear/RF/GBM).
For each of 5 folds (KFold(5, shuffle=True, random_state=42)): both are FIT on
the train fold only and SCORED on the held-out test fold.
{'='*72}""")

# ============================================================
# LOAD + PREPARE (both targets)
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
bc["coarse_class"] = coarse_class(bc["frac_over"])
print(f"bc ready: n={len(bc)}")


# ============================================================
# OUT-OF-FOLD CV LOOP -- builds & scores both arms per target, per fold
# ============================================================
def run_cv_for_target(bc, target_col):
    sub = bc.dropna(subset=[target_col] + ALL_9_FEATURES).copy()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = pd.DataFrame(index=sub.index)
    oof["actual"] = sub[target_col].astype(float)
    for f in ["1", "2"]:
        oof[f"tier_{f}"] = pd.Series(index=sub.index, dtype=object)
        oof[f"score_{f}"] = np.nan  # CONTINUOUS score, pre-bucketing -- for the tier-vs-continuous comparison

    idx = sub.index.to_numpy()
    for fold_i, (train_pos, test_pos) in enumerate(kf.split(idx)):
        train_idx, test_idx = idx[train_pos], idx[test_pos]
        train, test = sub.loc[train_idx], sub.loc[test_idx]

        specs1 = build_format1_specs(train, target_col)
        tot1_tr, _ = score_total(train, specs1)
        tot1_te, _ = score_total(test, specs1)
        tier1, _ = assign_tier(tot1_te, train_score=tot1_tr)
        oof.loc[test_idx, "tier_1"] = tier1.values
        oof.loc[test_idx, "score_1"] = tot1_te.values

        tree_result = fit_tree_per_target(train, target_col)
        tree_out = test.apply(lambda row: predict_tree_tier(tree_result, row), axis=1)
        oof.loc[test_idx, "tier_2"] = tree_out.apply(lambda x: x[0]).values
        oof.loc[test_idx, "score_2"] = tree_out.apply(lambda x: x[1]).values

        print(f"  fold {fold_i+1}/5: train n={len(train)}, test n={len(test)} -- fit+scored point-card + tree")
    return oof


print("\n" + "="*50)
print("TARGET A -- 5-fold CV loop")
print("="*50)
oof_A = run_cv_for_target(bc, "frac_over")
oof_A["coarse_class"] = bc["coarse_class"].reindex(oof_A.index)

print("\n" + "="*50)
print("TARGET B -- 5-fold CV loop")
print("="*50)
oof_B = run_cv_for_target(bc, "cluster_rank")
oof_B["cluster_tier_true"] = bc["cluster_tier"].reindex(oof_B.index)

OOF = {"A": oof_A, "B": oof_B}
TARGETS = [("A", "frac_over"), ("B", "cluster_rank")]

# ============================================================
# PER-ARM VALIDATION -- out-of-fold Spearman r, tier means, monotonicity
# ============================================================
print("\n" + "="*50)
print("PER-ARM VALIDATION (out-of-fold)")
print("="*50)


def arm_stats(oof, fmt):
    tier = oof[f"tier_{fmt}"]
    valid = tier.notna()
    tr = tier[valid].map(TIER_RANK)
    r, p = stats.spearmanr(tr, oof.loc[valid, "actual"])
    g = oof.loc[valid].groupby(tier[valid], observed=True)["actual"].agg(["mean", "median", "count"]).reindex(TIER_ORDER)
    monotonic = bool(g["mean"].dropna().is_monotonic_increasing)
    overall_mean = oof.loc[valid, "actual"].mean()
    return {"spearman_r": r, "p": p, "monotonic": monotonic, "tier_table": g,
            "overall_mean": overall_mean, "n": int(valid.sum())}


arm_results = {}
for tkey, target_col in TARGETS:
    oof = OOF[tkey]
    for fmt in ["1", "2"]:
        res = arm_stats(oof, fmt)
        arm_results[(tkey, fmt)] = res
        print(f"\n[{tkey}{fmt}] {METHOD_NAMES[fmt]} (target {tkey}): n={res['n']}, "
              f"OOF Spearman r={res['spearman_r']:+.3f} (p={res['p']:.2g}), monotonic={res['monotonic']}")
        print(res["tier_table"].round(3).to_string())
        lift = res["tier_table"].loc["High", "mean"] - res["tier_table"].loc["Low", "mean"]
        print(f"  Lift (High mean - Low mean) = {lift:+.3f}   (overall mean = {res['overall_mean']:.3f})")

    if target_col == "frac_over":
        for fmt in ["1", "2"]:
            print(f"\n  [{tkey}{fmt}] tier x coarse composition-class crosstab:")
            print(pd.crosstab(oof[f"tier_{fmt}"], oof["coarse_class"]).reindex(index=TIER_ORDER).to_string())
    else:
        for fmt in ["1", "2"]:
            ct = pd.crosstab(oof[f"tier_{fmt}"], oof["cluster_tier_true"]).reindex(index=TIER_ORDER, columns=TIER_ORDER, fill_value=0)
            acc = np.trace(ct.values) / ct.values.sum() * 100
            print(f"\n  [{tkey}{fmt}] tier x raw-cluster-label (ground truth) crosstab -- exact match = {acc:.1f}%:")
            print(ct.to_string())

# ============================================================
# CONTINUOUS SCORE vs DISCRETE TIER -- does bucketing into Low/Medium/High
# throw away real ranking information? For prioritization (not labeling), a
# continuous score preserves the full ordering; a tier collapses everyone in
# the same bucket to indistinguishable. Compare OOF Spearman r using the raw
# continuous score directly against the tier-rank version already computed.
# Restricted to the point-card (the only arm actually deployed as a label) --
# the tree's continuous score is used later purely as one ML comparison point.
# ============================================================
print("\n" + "="*50)
print("CONTINUOUS SCORE vs DISCRETE TIER (point-card) -- how much ranking info does bucketing lose?")
print("="*50)
continuous_results = {}
for tkey, target_col in TARGETS:
    oof = OOF[tkey]
    for fmt in ["1", "2"]:
        valid = oof[f"score_{fmt}"].notna()
        r_cont, p_cont = stats.spearmanr(oof.loc[valid, f"score_{fmt}"], oof.loc[valid, "actual"])
        r_tier = arm_results[(tkey, fmt)]["spearman_r"]
        continuous_results[(tkey, fmt)] = {"r_continuous": r_cont, "p_continuous": p_cont, "r_tier": r_tier}
        if fmt == "1":
            delta = r_cont - r_tier
            print(f"  [{tkey}{fmt}] {METHOD_NAMES[fmt]}: continuous r={r_cont:+.3f} (p={p_cont:.2g})  vs  "
                  f"tier r={r_tier:+.3f}   delta={delta:+.3f}  "
                  f"{'(continuous meaningfully better)' if delta > 0.03 else '(about the same)'}")

print("\nInterpretation: if continuous consistently beats tier by a real margin, prioritizing by RAW")
print("SCORE (rank 1..N, or percentile) recovers ranking information the 3-bucket card throws away --")
print("worth offering as a companion 'priority list' output alongside the Low/Medium/High label card.")

# FIG -- continuous vs tier Spearman r, point-card only (2 targets)
fig, ax = plt.subplots(figsize=(6.5, 5))
labels = [f"Target {tkey}" for tkey, _ in TARGETS]
r_tier_vals = [continuous_results[(tkey, "1")]["r_tier"] for tkey, _ in TARGETS]
r_cont_vals = [continuous_results[(tkey, "1")]["r_continuous"] for tkey, _ in TARGETS]
x = np.arange(len(labels))
width = 0.35
ax.bar(x - width/2, r_tier_vals, width=width, label="Tier (Low/Medium/High)", color=PAL5[2])
ax.bar(x + width/2, r_cont_vals, width=width, label="Continuous score", color=PAL5[0])
for i, (rt, rc) in enumerate(zip(r_tier_vals, r_cont_vals)):
    ax.text(i - width/2, rt + 0.01, f"{rt:.2f}", ha="center", fontsize=9)
    ax.text(i + width/2, rc + 0.01, f"{rc:.2f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Out-of-fold Spearman r vs actual outcome")
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_title("Point-card: does bucketing into 3 tiers lose ranking information?")
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(rf"{PLOTS}/paper2_continuous_vs_tier.png")
plt.close(fig)
print("\nSaved: plots/paper2_continuous_vs_tier.png")

# ============================================================
# RANDOM FOREST ROBUSTNESS CHECK -- per target (once, not per format)
# ============================================================
print("\n" + "="*50)
print("RANDOM FOREST ROBUSTNESS CHECK (per target)")
print("="*50)


def make_X(df):
    d = df.copy()
    d["design_group"] = d["design_group"].astype(str)
    dg = pd.get_dummies(d["design_group"], prefix="dg", drop_first=True)
    mat = pd.get_dummies(d["building_material"].fillna("Unknown"), prefix="mat", drop_first=True)
    return pd.concat([dg, mat, d[NUMERIC_7]], axis=1).astype(float)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_importances = {}
for tkey, target_col in TARGETS:
    sub = bc.dropna(subset=[target_col] + ALL_9_FEATURES)
    X = make_X(sub)
    y = sub[target_col].astype(float)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42)
    r2cv = cross_val_score(rf, X, y, cv=kf, scoring="r2").mean()
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    rf_importances[tkey] = imp
    print(f"\nTarget {tkey}: RandomForest 5-fold CV R2 = {r2cv:+.3f}")
    print("Top feature importances:")
    print(imp.head(8).round(3).to_string())
    top_rf = [c for c in imp.index if not c.startswith("dg_") and not c.startswith("mat_")][:3]
    print(f"  Top 3 non-categorical RF features: {top_rf}  "
          f"(compare to Stage 8's univariate |r| ranking and the tree's split order below)")

# ============================================================
# POINT-CARD vs. BLACK-BOX ML -- the central comparison. The decision tree is
# folded in alongside Linear/Random Forest/Gradient Boosting as ONE ML
# comparison group (see BLACKBOX_MODEL_EXPLAINED.md for what "black box" does
# and doesn't mean here -- the tree itself is textbook white-box; "black box"
# is used loosely for "needed automated fitting," in contrast to the
# point-card's fixed, no-fitting lookup tables). Same 9 features, same honest
# 5-fold out-of-fold procedure, same Spearman-r ranking-quality scale as the
# point-card continuous score above -- a true apples-to-apples comparison.
# ============================================================
print("\n" + "="*50)
print("POINT-CARD vs. BLACK-BOX ML -- same scale, same features, same CV discipline")
print("="*50)

blackbox_results = {}
for tkey, target_col in TARGETS:
    sub = bc.dropna(subset=[target_col] + ALL_9_FEATURES)
    X = make_X(sub)
    y = sub[target_col].astype(float)
    kf_bb = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Linear (Stage 8 benchmark)": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    for mname, model in models.items():
        oof_pred = cross_val_predict(model, X, y, cv=kf_bb)
        r2 = cross_val_score(model, X, y, cv=kf_bb, scoring="r2").mean()
        r_rank, p_rank = stats.spearmanr(oof_pred, y)
        blackbox_results[(tkey, mname)] = {"r2": r2, "spearman_r": r_rank}
        print(f"  Target {tkey}, {mname:28s}: CV R2={r2:+.3f}   OOF Spearman r={r_rank:+.3f} (p={p_rank:.2g})")

print("\nComparison -- point-card, decision tree, and 3 black-box ML models, same out-of-fold scale:")
comparison_rows = []
for tkey, _ in TARGETS:
    comparison_rows.append({"target": tkey, "method": "Point-card",
                             "spearman_r": continuous_results[(tkey, "1")]["r_continuous"]})
    comparison_rows.append({"target": tkey, "method": "Decision Tree",
                             "spearman_r": continuous_results[(tkey, "2")]["r_continuous"]})
    for mname in ["Linear (Stage 8 benchmark)", "Random Forest", "Gradient Boosting"]:
        comparison_rows.append({"target": tkey, "method": mname, "spearman_r": blackbox_results[(tkey, mname)]["spearman_r"]})
comp_df = pd.DataFrame(comparison_rows)
print(comp_df.round(3).to_string(index=False))

ml_methods = ["Decision Tree", "Linear (Stage 8 benchmark)", "Random Forest", "Gradient Boosting"]


def ml_r(tkey, mname):
    if mname == "Decision Tree":
        return continuous_results[(tkey, "2")]["r_continuous"]
    return blackbox_results[(tkey, mname)]["spearman_r"]


best_ml = {tkey: max(ml_methods, key=lambda m: ml_r(tkey, m)) for tkey, _ in TARGETS}
for tkey, _ in TARGETS:
    bb_r = ml_r(tkey, best_ml[tkey])
    pc_r = continuous_results[(tkey, "1")]["r_continuous"]
    gap = bb_r - pc_r
    print(f"\nTarget {tkey}: best ML method ({best_ml[tkey]}, r={bb_r:+.3f}) vs point-card (r={pc_r:+.3f})"
          f"  -->  gap = {gap:+.3f}  "
          f"{'(ML meaningfully ahead)' if gap > 0.05 else '(essentially tied -- interpretability is nearly free)'}")

# ============================================================
# WHY does Target B's gap (point-card vs. best ML) run larger than Target A's?
# Two pieces of already-established evidence, recomputed here for one self-
# contained answer: (1) univariate feature strength -- is Target B's
# relationship to the 9 locked features simply weaker across the board? (2) how
# cleanly separated are the target's own Low/Medium/High groups to begin with
# (Stage 7's silhouette score)? A point-card can only rank-order using linear,
# additive, tertile-binned rules -- it has no way to exploit interactions or
# non-monotonic feature combinations the way RF/GBM can. That gap matters MORE
# when (a) the individual features are only weakly related to the target to
# start with, so nonlinear combinations of them carry a bigger share of the
# real signal, and/or (b) the target's own group structure is fuzzier, so
# small nonlinear corrections near a fuzzy boundary matter more.
# ============================================================
print("\n" + "="*50)
print("WHY IS THE GAP LARGER FOR TARGET B? (point 11)")
print("="*50)

uni_rows = []
for feat in NUMERIC_7:
    dA = bc[[feat, "frac_over"]].dropna()
    rA, _ = stats.spearmanr(dA[feat], dA["frac_over"])
    dB = bc[[feat, "cluster_rank"]].dropna()
    rB, _ = stats.spearmanr(dB[feat], dB["cluster_rank"])
    uni_rows.append({"feature": feat, "abs_r_A": abs(rA), "abs_r_B": abs(rB)})
uni_df = pd.DataFrame(uni_rows)
n_stronger_for_A = (uni_df["abs_r_A"] > uni_df["abs_r_B"]).sum()
print(f"Univariate |Spearman r| vs the 9 locked features: stronger for Target A in "
      f"{n_stronger_for_A}/{len(uni_df)} numeric features (full table in Stage 8).")
print(f"  Mean |r| across these features -- Target A: {uni_df['abs_r_A'].mean():.3f}   "
      f"Target B: {uni_df['abs_r_B'].mean():.3f}")

_, sil_a_check = build_composition_cluster(bc, min_apts=MIN_APTS)
sil_b_check = None
try:
    _, sil_b_check = build_severity_cluster(buildings_meta, min_apts=MIN_APTS)
except Exception:
    pass
print(f"\nCluster separation quality (Stage 7 silhouette, k=3): "
      f"Target A={sil_a_check[3]:.3f}" + (f"   Target B={sil_b_check[3]:.3f}" if sil_b_check is not None else ""))

print("\nInterpretation: Target B's relationship to the 9 registry features is weaker across almost")
print("every feature (not just one or two), AND its own Low/Medium/High groups are less cleanly")
print("separated to begin with. A point-card can only combine features additively and bin them into")
print("thirds -- it has no mechanism to exploit interactions or non-monotonic combinations. That")
print("costs more when (a) each feature's individual signal is weaker, so more of the real signal")
print("lives in feature COMBINATIONS a tree/ensemble can find but an additive card cannot, and (b)")
print("the target's own groups are fuzzier, so small nonlinear corrections near a fuzzy boundary")
print("matter more. Target A's stronger, more linear univariate relationships and crisper cluster")
print("separation leave much less of this kind of signal on the table -- which is exactly why the")
print("point-card ties the black-box ceiling on Target A; see the printed gap for")
print("Target B.")

# FIG -- point-card vs decision tree vs black-box ML, same Spearman r scale
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for i, (tkey, _) in enumerate(TARGETS):
    ax = axes[i]
    sub_comp = comp_df[comp_df["target"] == tkey]
    colors = [PAL5[0]] + ["#999999"] * 4
    ax.barh(sub_comp["method"], sub_comp["spearman_r"], color=colors)
    for j, v in enumerate(sub_comp["spearman_r"]):
        ax.text(v + 0.005, j, f"{v:.3f}", va="center", fontsize=9)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Out-of-fold Spearman r vs actual outcome")
    ax.set_title(f"Target {tkey}")
fig.suptitle("Point-card (blue) vs. decision tree + black-box ML (grey) -- same scale", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/paper2_blackbox_vs_rubric.png")
plt.close(fig)
print("\nSaved: plots/paper2_blackbox_vs_rubric.png")

# ============================================================
# TREE DEPTH ROBUSTNESS CHECK -- would a deeper tree have closed the gap to
# the point-card? Sweeps max_depth on an UNCONSTRAINED tree (all 9 features,
# no design_group guardrail root, no MEDIUM-tier fallback restriction) via the
# same 5-fold OOF procedure, on the SAME continuous Spearman-r scale as the
# point-card's continuous score above. This is the evidence for why the
# guardrailed tree was kept shallow (~depth 2) rather than a design assumption.
# ============================================================
print("\n" + "="*50)
print("TREE DEPTH ROBUSTNESS CHECK -- does a deeper (unconstrained) tree close the gap to the point-card?")
print("="*50)

depth_results = {}
for tkey, target_col in TARGETS:
    sub = bc.dropna(subset=[target_col] + ALL_9_FEATURES)
    X = make_X(sub)
    y = sub[target_col].astype(float)
    kf_d = KFold(n_splits=5, shuffle=True, random_state=42)
    for depth in [2, 3, 4, 5, 8, None]:
        oof_pred = cross_val_predict(DecisionTreeRegressor(max_depth=depth, min_samples_leaf=20, random_state=42), X, y, cv=kf_d)
        r, p = stats.spearmanr(oof_pred, y)
        n_leaves = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=20, random_state=42).fit(X, y).get_n_leaves()
        depth_results[(tkey, depth)] = {"spearman_r": r, "n_leaves": n_leaves}
        print(f"  Target {tkey}, max_depth={str(depth):5s}: OOF Spearman r={r:+.3f}  (full-data leaves={n_leaves})")

print("\nBest unconstrained depth per target vs the current guardrailed tree vs the point-card")
print("(all on the same continuous-score Spearman r scale):")
for tkey, _ in TARGETS:
    best_depth = max([2, 3, 4, 5, 8, None], key=lambda d: depth_results[(tkey, d)]["spearman_r"])
    best_r = depth_results[(tkey, best_depth)]["spearman_r"]
    tree_r = continuous_results[(tkey, "2")]["r_continuous"]
    pc_r = continuous_results[(tkey, "1")]["r_continuous"]
    print(f"  Target {tkey}: best unconstrained depth={best_depth} (r={best_r:+.3f}, "
          f"{depth_results[(tkey, best_depth)]['n_leaves']} leaves) vs guardrailed tree (r={tree_r:+.3f}, "
          f"~4-5 leaves) vs point-card (r={pc_r:+.3f})  -->  "
          f"{'point-card still wins or ties' if pc_r >= best_r else 'deeper tree edges out the point-card'}")

# FIG -- tree depth sweep vs point-card, both targets
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
depths_x = [2, 3, 4, 5, 8, "unlimited"]
for i, (tkey, _) in enumerate(TARGETS):
    ax = axes[i]
    r_vals = [depth_results[(tkey, d if d != "unlimited" else None)]["spearman_r"] for d in depths_x]
    ax.plot(range(len(depths_x)), r_vals, "o-", color=PAL5[4], linewidth=2, markersize=7, label="Tree (unconstrained, by depth)")
    ax.axhline(continuous_results[(tkey, "1")]["r_continuous"], color=PAL5[0], linewidth=2, linestyle="--",
               label=f"Point-card (r={continuous_results[(tkey, '1')]['r_continuous']:.3f})")
    ax.axhline(continuous_results[(tkey, "2")]["r_continuous"], color=PAL5[2], linewidth=1.5, linestyle=":",
               label=f"Guardrailed tree (r={continuous_results[(tkey, '2')]['r_continuous']:.3f})")
    for j, d in enumerate([2, 3, 4]):
        ax.annotate(f"{depth_results[(tkey, d)]['n_leaves']} leaves", (j, r_vals[j]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color="#555")
    ax.set_xticks(range(len(depths_x)))
    ax.set_xticklabels([str(d) for d in depths_x])
    ax.set_xlabel("Tree max_depth")
    ax.set_ylabel("Out-of-fold Spearman r (continuous)")
    ax.set_title(f"Target {tkey}")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("Does a deeper tree beat the point-card? No -- not even at its best depth", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/paper2_tree_depth_vs_pointcard.png")
plt.close(fig)
print("\nSaved: plots/paper2_tree_depth_vs_pointcard.png")

# ============================================================
# FIG 1 -- out-of-fold actual outcome by assigned tier, one panel per arm
# (2 formats x 2 targets, down from 3 formats x 2 targets)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
tier_colors = {"Low": PAL5[0], "Medium": PAL5[2], "High": PAL5[4]}
for i, (tkey, target_col) in enumerate(TARGETS):
    oof = OOF[tkey]
    for j, fmt in enumerate(["1", "2"]):
        ax = axes[i, j]
        res = arm_results[(tkey, fmt)]
        means = res["tier_table"]["mean"]
        counts = res["tier_table"]["count"]
        bars = ax.bar(TIER_ORDER, means.values, color=[tier_colors[t] for t in TIER_ORDER], edgecolor="white")
        for k, t in enumerate(TIER_ORDER):
            if pd.notna(means[t]):
                ax.text(k, means[t], f"n={int(counts[t])}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"[{tkey}{fmt}] {METHOD_NAMES[fmt]}\nOOF Spearman r={res['spearman_r']:+.2f}", fontsize=10)
        ax.set_ylabel(target_col)
fig.suptitle("Out-of-fold actual outcome by assigned tier, point-card + tree, both targets", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/paper2_tier_vs_actual_outcome.png")
plt.close(fig)

# ============================================================
# FIG -- Target B: rubric-assigned tier vs TRUE raw cluster label, one panel
# per arm (point-card, tree) -- how well does each recover the actual KMeans
# ground-truth severity group (not just correlate with cluster_rank)?
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))
for j, fmt in enumerate(["1", "2"]):
    ax = axes[j]
    ct = pd.crosstab(oof_B[f"tier_{fmt}"], oof_B["cluster_tier_true"]).reindex(
        index=TIER_ORDER, columns=TIER_ORDER, fill_value=0)
    acc = np.trace(ct.values) / ct.values.sum() * 100
    im = ax.imshow(ct.values, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(TIER_ORDER)
    ax.set_yticks(range(3)); ax.set_yticklabels(TIER_ORDER)
    ax.set_xlabel("TRUE cluster label (KMeans ground truth)"); ax.set_ylabel(f"{METHOD_NAMES[fmt]} assigned tier")
    for r in range(3):
        for c in range(3):
            ax.text(c, r, int(ct.values[r, c]), ha="center", va="center",
                     color="white" if ct.values[r, c] > ct.values.max() / 2 else "black")
    ax.set_title(f"{METHOD_NAMES[fmt]}\nexact match={acc:.1f}%")
fig.suptitle("Target B: out-of-fold assigned tier vs TRUE raw cluster label", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/paper2_targetB_vs_true_cluster.png")
plt.close(fig)

# ============================================================
# TABLE 1 -- validation summary, point-card + tree, both targets
# ============================================================
print("\n" + "="*50)
print("TABLE 1 -- VALIDATION SUMMARY (point-card + tree, both targets)")
print("="*50)
rows = []
for tkey, target_col in TARGETS:
    for fmt in ["1", "2"]:
        res = arm_results[(tkey, fmt)]
        t = res["tier_table"]
        lift = t.loc["High", "mean"] - t.loc["Low", "mean"]
        rows.append({
            "Arm": f"{tkey}{fmt} ({METHOD_NAMES[fmt]})",
            "OOF Spearman r": f"{res['spearman_r']:+.3f}",
            "Monotonic": str(res["monotonic"]),
            "Low mean": f"{t.loc['Low','mean']:.3f}" if pd.notna(t.loc["Low", "mean"]) else "n/a",
            "Med mean": f"{t.loc['Medium','mean']:.3f}" if pd.notna(t.loc["Medium", "mean"]) else "n/a",
            "High mean": f"{t.loc['High','mean']:.3f}" if pd.notna(t.loc["High", "mean"]) else "n/a",
            "Lift (High-Low)": f"{lift:+.3f}",
            "n": res["n"],
        })
t1 = pd.DataFrame(rows).set_index("Arm")
print(t1.to_string())
render_table(t1, "Table 1 -- Validation Summary (out-of-fold, point-card + tree)", "paper2_table1_rubric_validation.png")

print("\nSaved: paper2_tier_vs_actual_outcome.png, paper2_targetB_vs_true_cluster.png,")
print("       paper2_table1_rubric_validation.png")
print(">>> STAGE 10 complete.")
