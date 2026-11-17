"""
STAGE 11 -- Assemble final figures/tables (whole pipeline)

This file is the merge of the former paper1/stage7_assemble_final.py
(Part A: Stages 0-6 overview/age/predictability tables) and paper2/
stage4_assemble_final.py (Part B: Stages 7-10 building-level risk artifacts),
combined into one final-assembly stage for the whole pipeline.

Restructured around point-card vs. black-box ML (Format 3/hybrid and the
decision-tree-as-flowchart rendering are removed -- see BLACKBOX_MODEL_EXPLAINED.md
for why the tree is now treated as one ML technique, not a competing deployable
format). Only the point-card is rendered as a deployable one-page artifact; the
tree is inspected via its printed structure in Stage 9 and compared numerically
in Stage 10.

PART A -- verifies the Stage 0-6 (apartment-level spread / predictability)
figure set and produces 3 summary tables.
PART B -- verifies the Stage 7-10 (building-level risk) figure set, renders
the point-card as a final one-page printable artifact for both targets (fit on
the FULL bc table -- Stage 10 already did the honest out-of-fold validation
separately), and one worked example: a real building scored by both the
point-card and the tree, side by side, for both targets.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
from common import (
    PLOTS, DG_ORDER, DG_LABELS, MIN_APTS, NUMERIC_7,
    load_buildings_metadata, load_apartment_metrics, load_hourly_indoor,
    compute_swing_range, build_apt_table, build_building_comp, attach_building_features,
    attach_risk_features, build_severity_cluster,
    build_format1_specs, score_total, assign_tier,
    fit_tree_per_target, predict_tree_tier, render_table, set_style,
)

set_style()

print(f"""
{'='*72}
STAGE 11 -- ASSEMBLE FINAL FIGURES/TABLES (WHOLE PIPELINE)
{'='*72}""")

# ============================================================
# FULL FIGURE INVENTORY -- whole unified pipeline (Stages 0-12)
# ============================================================
EXPECTED_FIGS_PART_A = [
    "stage1_mean_vs_peak_spread.png", "stage1_coolest_apartment_still_overheated.png",
    "stage1_coolest_apartment_summary.png", "stage1_four_compositions.png",
    "stage2_accum_vs_recovery_slopes.png", "stage2_ratchet_trajectory.png",
    "stage3_worst_apartment_stability.png", "stage3_heatwave_worsening_scatter.png",
    "stage3_heatwave_worsening_stability_comparison.png", "stage3_summer_excl_heatwaves_comparison.png",
    "stage3_accumulation_slope_by_age.png", "stage3_material_effect_on_accumulation.png",
    "stage4_hump_vs_monotonic.png", "stage4_material_effect_on_lag.png",
    "stage5_overheating_by_era.png", "stage5_level_vs_inequality_effects.png",
    "stage6_icc_between_within.png", "stage6_variance_partition.png", "stage6_ceiling_vs_achieved.png",
]
EXPECTED_FIGS_PART_B = [
    "paper2_stage0_targetA_overview.png", "paper2_stage0_targetB_clusters.png",
    "paper2_stage0_targetB_severity_range.png",
    "stage8_feature_selection_numeric.png", "stage8_feature_selection_categorical.png",
    "stage8_r2_comparison.png",
    "paper2_tier_vs_actual_outcome.png",
    "paper2_targetB_vs_true_cluster.png", "paper2_continuous_vs_tier.png",
    "paper2_blackbox_vs_rubric.png", "paper2_tree_depth_vs_pointcard.png",
    "paper2_table1_rubric_validation.png",
    "paper2_rubric_pointcard_targetA.png", "paper2_rubric_pointcard_targetB.png",
    "paper2_worked_example.png",
]
EXPECTED_FIGS = EXPECTED_FIGS_PART_A + EXPECTED_FIGS_PART_B

print("\n" + "="*50)
print("0. FULL FIGURE SET CHECK (Stages 0-10 must already have run)")
print("="*50)
missing = []
for f in EXPECTED_FIGS:
    path = os.path.join(PLOTS, f)
    if os.path.exists(path):
        print(f"  OK   {f}  ({os.path.getsize(path)/1024:.0f} KB)")
    else:
        print(f"  MISSING   {f}")
        missing.append(f)
if missing:
    print(f"\nWARNING: {len(missing)} figure(s) missing -- re-run the corresponding stage script(s).")
else:
    print(f"\nAll {len(EXPECTED_FIGS)} publication figures present in plots/.")

# ============================================================
# PART A -- LOAD + PREPARE (apartment-level spread tables)
# ============================================================
buildings_meta = load_buildings_metadata()
apt_metrics = load_apartment_metrics(merge_area_type=True)
print("\nLoading hourly indoor temperature to compute swing_range ...")
hourly = load_hourly_indoor()
swing_df = compute_swing_range(hourly)
del hourly

apt = build_apt_table(apt_metrics, buildings_meta, swing_df)
building_comp, apt, lag_med, swing_med = build_building_comp(apt)
bc = attach_building_features(building_comp, buildings_meta)

# ============================================================
# TABLE 1 -- Overview & overheating spread
# ============================================================
print("\n" + "="*50)
print("TABLE 1 -- OVERVIEW & OVERHEATING SPREAD")
print("="*50)

comp_over = building_comp["class_over"].value_counts(normalize=True) * 100
coarse = np.select([building_comp["frac_over"] == 0, building_comp["frac_over"] == 1],
                    ["all_resilient", "all_overheating"], default="mixed")
coarse_pct = pd.Series(coarse).value_counts(normalize=True) * 100

a = apt[["device", "building_id", "temp_mean", "temp_max"]].dropna()
rank_rows = []
for bid, g in a.groupby("building_id"):
    if len(g) < MIN_APTS:
        continue
    rho, _ = stats.spearmanr(g["temp_mean"], g["temp_max"])
    same = g.loc[g["temp_mean"].idxmax(), "device"] == g.loc[g["temp_max"].idxmax(), "device"]
    rank_rows.append({"building_id": bid, "rho": rho, "same_worst": same})
rank_df = pd.DataFrame(rank_rows)

hw_spread = apt.groupby("building_id").agg(hw_spread=("hw_temp_mean", lambda x: x.max() - x.min()),
                                            n=("hw_temp_mean", "count")).query("n >= @MIN_APTS")
worsen = building_comp[["spread_temp"]].join(hw_spread[["hw_spread"]], how="inner").dropna()
pct_worsens = (worsen["hw_spread"] > worsen["spread_temp"]).mean() * 100

t1 = pd.DataFrame({
    "Value": [
        f"{apt['building_id'].nunique()}",
        f"{len(apt)}",
        f"{coarse_pct.get('all_resilient', 0):.1f}% / {coarse_pct.get('mixed', 0):.1f}% / {coarse_pct.get('all_overheating', 0):.1f}%",
        f"{building_comp['spread_temp'].median():.2f} C",
        f"{building_comp['spread_peak'].median():.2f} C",
        f"{rank_df['rho'].median():.2f}",
        f"{rank_df['same_worst'].mean()*100:.0f}%",
        f"{building_comp['spread_temp'].median():.2f} C -> {worsen['hw_spread'].median():.2f} C ({pct_worsens:.0f}% worsen)",
    ]
}, index=[
    "Buildings (n)", "Apartments (n)",
    "Composition (all-resilient / mixed / all-overheating)",
    "Median mean-based spread (within-building)",
    "Median peak-based spread (within-building)",
    "Median rank-consistency rho (mean vs peak ranking)",
    "same_worst (#1 apt identical by mean & peak)",
    "Heatwave worsening (summer -> heatwave spread)",
])
print(t1.to_string())
render_table(t1, "Table 1 -- Overview & Within-Building Overheating Spread", "stage11_table1_overview.png")

# ============================================================
# TABLE 2 -- Age structure
# ============================================================
print("\n" + "="*50)
print("TABLE 2 -- AGE STRUCTURE")
print("="*50)
OUTCOMES = ["dh_27", "temp_mean", "temp_max", "peak_lag_hot", "swing_range"]
med_table = apt.groupby("design_group", observed=True)[OUTCOMES].median().reindex(DG_ORDER)
shapes = []
for col in OUTCOMES:
    vals = med_table[col]
    shapes.append("HUMP (2004-2009)" if vals.idxmax() == "y2004_2009" else f"peak={vals.idxmax()}")
t2 = med_table.round(1).T
t2.index = ["dh_27 (deg-hrs)", "temp_mean (C)", "temp_max (C)", "peak_lag_hot (h)", "swing_range (C)"]
t2.columns = [DG_LABELS[g] for g in DG_ORDER]
t2["Shape"] = shapes
print(t2.to_string())
render_table(t2, "Table 2 -- Age Structure (apartment-level medians)", "stage11_table2_age_structure.png")

# ============================================================
# TABLE 3 -- Feature predictability (Stage 5 + 6 keystone numbers)
# ============================================================
print("\n" + "="*50)
print("TABLE 3 -- FEATURE PREDICTABILITY (cross-validated R2)")
print("="*50)

numeric_feats_a = ["n_floors", "gross_floor_area_m2", "n_apartments_hsy",
                    "impervious_fraction", "tree_fraction",
                    "distance_to_sea_m", "road_density_100m", "dist_major_road_m", "ALUETEHOK"]


def make_X_a(df):
    d = df.copy()
    d["design_group"] = d["design_group"].astype(str)
    dg = pd.get_dummies(d["design_group"], prefix="dg", drop_first=True)
    mat = pd.get_dummies(d["building_material"].fillna("Unknown"), prefix="mat", drop_first=True)
    return pd.concat([dg, mat, d[numeric_feats_a]], axis=1).astype(float)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for name, frac_col, spread_col in [("Overheating", "frac_over", "spread_temp"),
                                    ("Lag", "frac_fast", "spread_lag"),
                                    ("Swing", "frac_vol", "spread_swing")]:
    needed = [frac_col, spread_col] + numeric_feats_a + ["design_group", "building_material"]
    sub = bc.dropna(subset=needed)
    X = make_X_a(sub)
    r2_level = cross_val_score(LinearRegression(), X, sub[frac_col].astype(float), cv=kf, scoring="r2").mean()
    r2_spread = cross_val_score(LinearRegression(), X, sub[spread_col].astype(float), cv=kf, scoring="r2").mean()
    rows.append({"Outcome": name, "LEVEL R2 (CV)": f"{r2_level:+.3f}", "SPREAD R2 (CV)": f"{r2_spread:+.3f}"})

icc_vals = {}
for outcome in ["temp_mean", "temp_max"]:
    d = apt[[outcome, "building_id"]].dropna()
    fit = smf.mixedlm(f"{outcome} ~ 1", d, groups=d["building_id"]).fit(reml=True)
    icc_vals[outcome] = float(fit.cov_re.iloc[0, 0]) / (float(fit.cov_re.iloc[0, 0]) + float(fit.scale))

t3a = pd.DataFrame(rows).set_index("Outcome")
print(t3a.to_string())
print(f"\nICC: temp_mean={icc_vals['temp_mean']:.3f}, temp_max={icc_vals['temp_max']:.3f}  "
      f"(~{100*(1-np.mean(list(icc_vals.values()))):.0f}% of variance is within-building)")
render_table(t3a, "Table 3 -- Feature Predictability: LEVEL vs SPREAD (5-fold CV R2)", "stage11_table3_predictability.png")

print("\nSaved: stage11_table1_overview.png, stage11_table2_age_structure.png, stage11_table3_predictability.png")
print(">>> PART A complete (Stages 0-6 apartment-level spread pipeline).")

# ============================================================
# PART B -- LOAD + PREPARE (building-level risk rubric, both targets)
# FULL SAMPLE fits (not CV) for the deployed cards
# ============================================================
bc = attach_risk_features(bc, buildings_meta)
cluster_df, sil = build_severity_cluster(buildings_meta, min_apts=MIN_APTS)
bc["cluster_rank"] = cluster_df["cluster_rank"].reindex(bc.index)
bc["cluster_tier"] = cluster_df["cluster_tier"].reindex(bc.index)
print(f"\nbc ready: n={len(bc)}")

TARGETS = [("A", "frac_over", "Target A -- Composition risk (frac_over)"),
           ("B", "cluster_rank", "Target B -- Severity risk (cluster_rank)")]

ARM_DATA = {}  # tkey -> dict with specs1/order1/tier_edges1 (point-card), tree_result/tier2 (tree)
for tkey, target_col, _ in TARGETS:
    specs1 = build_format1_specs(bc, target_col)
    order1 = ["design_group"] + NUMERIC_7 + ["building_material"]
    total1, _ = score_total(bc, specs1)
    tier1, edges1 = assign_tier(total1)

    tree_result = fit_tree_per_target(bc, target_col)
    valid2 = bc.dropna(subset=[target_col] + NUMERIC_7 + ["design_group", "building_material"])
    tier2 = valid2.apply(lambda row: predict_tree_tier(tree_result, row)[0], axis=1)

    ARM_DATA[tkey] = {"specs1": specs1, "order1": order1, "edges1": edges1, "tier1": tier1, "total1": total1,
                       "tree_result": tree_result, "tier2": tier2}

FEAT_LABELS = {
    "design_group": "Construction era", "n_floors": "Number of floors",
    "building_material": "Building material", "distance_to_sea_m": "Distance from sea (m)",
    "dist_major_road_m": "Distance from major road (m)", "n_apartments": "Number of apartments",
    "greenery_score": "Greenery score", "ALUETEHOK": "Land-use efficiency (ALUETEHOK)",
    "road_density_100m": "Road density (100m)",
}


# ============================================================
# POINT-CARD RENDERER (the one deployed, printable artifact)
# ============================================================
def cell_descriptions(spec):
    if spec["kind"] == "numeric":
        e1, e2 = spec["edges"]
        bin_desc = {0: f"<= {e1:,.2f}", 1: f"{e1:,.2f} - {e2:,.2f}", 2: f"> {e2:,.2f}"}
        out = {0: "", 1: "", 2: ""}
        for bin_idx, pts in spec["points"].items():
            out[pts] = bin_desc[bin_idx]
        return out
    out = {0: [], 1: [], 2: []}
    for cat, pts in spec["points"].items():
        out[pts].append(str(cat))
    return {k: (", ".join(v) if v else "") for k, v in out.items()}


def render_point_card(specs, order, tier_edges, title, subtitle, fname, small_n_flags=None):
    small_n_flags = small_n_flags or []
    n = len(order)
    fig, ax = plt.subplots(figsize=(10.5, 1.9 + 0.62 * n))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n + 2.8)

    top = n + 2.4
    ax.text(5, top, title, ha="center", fontsize=15, fontweight="bold")
    ax.text(5, top - 0.55, subtitle, ha="center", fontsize=9.5, style="italic", color="#444")

    header_y = n + 1.0
    col_x = {0: 4.1, 1: 6.4, 2: 8.7}
    col_label = {0: "0 pts (Low risk)", 1: "1 pt (Medium risk)", 2: "2 pts (High risk)"}
    col_color = {0: "#2c7bb6", 1: "#666666", 2: "#d7191c"}
    ax.text(0.15, header_y, "Feature", fontsize=10.5, fontweight="bold", ha="left")
    for p in [0, 1, 2]:
        ax.text(col_x[p], header_y, col_label[p], fontsize=9.5, fontweight="bold", ha="center", color=col_color[p])
    ax.plot([0.1, 9.9], [header_y - 0.32, header_y - 0.32], color="black", lw=1.1)

    row_h = 1.0
    for i, feat in enumerate(order):
        yi = header_y - 0.85 - i * row_h
        spec = specs[feat]
        descs = cell_descriptions(spec)
        flag = "  *" if feat in ("design_group",) else ""
        ax.text(0.15, yi, FEAT_LABELS.get(feat, feat) + flag, fontsize=9.5, va="center", ha="left")
        for p in [0, 1, 2]:
            ax.text(col_x[p], yi, descs[p], fontsize=8.3, va="center", ha="center", wrap=True)
        if i < n - 1:
            ax.plot([0.1, 9.9], [yi - row_h / 2, yi - row_h / 2], color="#ddd", lw=0.7)

    bottom_y = header_y - 0.85 - (n - 1) * row_h - row_h / 2 - 0.15
    ax.plot([0.1, 9.9], [bottom_y, bottom_y], color="black", lw=1.3)
    e1, e2 = tier_edges
    ax.text(5, bottom_y - 0.45,
            f"TOTAL SCORE  ->  Low: score <= {e1:.0f}    Medium: {e1:.0f} < score <= {e2:.0f}    High: score > {e2:.0f}",
            ha="center", fontsize=10.5, fontweight="bold")
    ax.text(5, bottom_y - 0.85,
            "* construction era: y2004_2009 (n=21) / y2010_2011 (n=19) rest on thin samples -- descriptive only",
            ha="center", fontsize=7.5, color="#777")

    plt.tight_layout()
    plt.savefig(rf"{PLOTS}/{fname}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# RENDER THE POINT-CARD -- the one deployable artifact, both targets
# ============================================================
print("\n" + "="*50)
print("RENDER THE POINT-CARD (both targets)")
print("="*50)
for tkey, target_col, target_label in TARGETS:
    d = ARM_DATA[tkey]
    render_point_card(
        d["specs1"], d["order1"], d["edges1"],
        f"Overheating Risk Card -- {target_label}", "Additive point-score card (no interactions)",
        f"paper2_rubric_pointcard_target{tkey}.png")
    print(f"  Target {tkey}: rendered point-card")

# ============================================================
# WORKED EXAMPLE -- one building scored on all 6 arms
# ============================================================
print("\n" + "="*50)
print("WORKED EXAMPLE")
print("="*50)
tiers = pd.DataFrame({
    "A1_point": ARM_DATA["A"]["tier1"], "A2_tree": ARM_DATA["A"]["tier2"],
    "B1_point": ARM_DATA["B"]["tier1"], "B2_tree": ARM_DATA["B"]["tier2"],
})
tiers = tiers.dropna()
tiers["n_distinct_tiers"] = tiers.apply(lambda r: r.nunique(), axis=1)
example_id = tiers.sort_values(["n_distinct_tiers"], ascending=False).index[0]
print(f"Worked example building_id = {example_id}  "
      f"(chosen for max tier-label spread across point-card/tree x 2 targets: "
      f"{tiers.loc[example_id, 'n_distinct_tiers']} distinct tiers)")

row = bc.loc[example_id]
example_facts = pd.DataFrame({"Value": [
    row["design_group"], f"{row['n_floors']:.0f}", row["building_material"],
    f"{row['distance_to_sea_m']:.0f} m", f"{row['dist_major_road_m']:.0f} m", f"{row['n_apartments']:.0f}",
    f"{row['greenery_score']:.2f}", f"{row['ALUETEHOK']:.2f}", f"{row['road_density_100m']:.0f}",
    f"{row['frac_over']:.3f}", f"{row['cluster_rank']:.0f} ({row['cluster_tier']})",
]}, index=["Construction era", "Floors", "Material", "Distance to sea", "Distance to major road",
           "Apartments", "Greenery score", "ALUETEHOK", "Road density",
           "TRUE frac_over (Target A)", "TRUE cluster_rank (Target B)"])
print(f"\nBuilding {example_id} facts:")
print(example_facts.to_string())

example_tiers = pd.DataFrame({"Assigned tier": [
    tiers.loc[example_id, "A1_point"], tiers.loc[example_id, "A2_tree"],
    tiers.loc[example_id, "B1_point"], tiers.loc[example_id, "B2_tree"],
]}, index=["Target A -- Point-card", "Target A -- Decision tree",
           "Target B -- Point-card", "Target B -- Decision tree"])
print(f"\nBuilding {example_id} scored by both methods, both targets:")
print(example_tiers.to_string())

worked = pd.DataFrame({"Value": pd.concat([example_facts["Value"], example_tiers["Assigned tier"]])})
render_table(worked, f"Worked Example -- Building {example_id}, point-card vs. tree, both targets",
             "paper2_worked_example.png")

print(f"\nThis building lands in {tiers.loc[example_id, 'n_distinct_tiers']} different tiers across "
      "point-card/tree x 2 targets -- concrete evidence that which TARGET you pick (composition vs. "
      "severity) can change a building's assigned risk category, even holding the method fixed.")

# ============================================================
# FULL FIGURE SET RE-CHECK
# ============================================================
print("\n" + "="*50)
print("FULL FIGURE SET RE-CHECK (whole pipeline)")
print("="*50)
missing = []
for f in EXPECTED_FIGS:
    path = os.path.join(PLOTS, f)
    if os.path.exists(path):
        print(f"  OK   {f}  ({os.path.getsize(path)/1024:.0f} KB)")
    else:
        print(f"  MISSING   {f}")
        missing.append(f)
if missing:
    print(f"\nWARNING: {len(missing)} figure(s) missing.")
else:
    print(f"\nAll {len(EXPECTED_FIGS)} figures present in plots/.")

print("\n>>> STAGE 11 complete. Whole unified pipeline finished.")
