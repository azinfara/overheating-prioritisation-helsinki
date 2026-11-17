"""
STAGE 9 -- Build the point-card and the decision tree
Overheating Risk Categorization Rubric ("Energy Label for Overheating").

Restructured around the paper's actual focus: point-card vs. black-box ML.
Format 3 (hybrid) is removed -- it never appeared in the deck and existed only
to compare 3+ deployable formats against each other. The decision tree is kept,
but reframed: it is no longer "Format 2, a competing flowchart artifact" --
it is one ML technique, later compared on equal footing against Linear/Random
Forest/Gradient Boosting in Stage 10. See BLACKBOX_MODEL_EXPLAINED.md for the
full definitions (what a split search does, depth vs. leaf count, why "black
box" is used loosely here).

Builds BOTH arms for BOTH targets (Target A frac_over, Target B cluster_rank):
A1 (point-card/A), A2 (tree/A), B1 (point-card/B), B2 (tree/B).

Fit here on the FULL bc table (n~516/489) -- these are the published,
inspectable point/split tables. Stage 10 refits the SAME builders per CV fold
(train-only) for honest out-of-fold validation; this stage never validates,
it only builds and prints for inspection.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from common import (
    PLOTS, DG_ORDER, DG_LABELS, MIN_APTS, NUMERIC_7,
    load_buildings_metadata, load_apartment_metrics, load_hourly_indoor,
    compute_swing_range, build_apt_table, build_building_comp, attach_building_features,
    attach_risk_features, build_severity_cluster,
    build_format1_specs, score_total, assign_tier,
    fit_tree_per_target, set_style,
)

set_style()

print(f"""
{'='*72}
STAGE 9 -- BUILD THE POINT-CARD AND THE DECISION TREE
{'='*72}
Point-card: numeric features -> tertile points (0/1/2) from sub-group mean of
  the target's outcome; design_group -> fixed hump lookup table; building_material
  -> lookup table (steel folded into 'Other'); total score -> Low/Medium/High
  via tertiles of the score distribution. Additive, no interactions.
Decision tree: DecisionTreeRegressor(max_depth<=2, min_samples_leaf=20), root
  hardcoded to design_group==y2004_2009 (Stage 4 hump guardrail), depth-1 split
  per branch on the 5 remaining EASY features, MEDIUM-tier fallback split only
  on leaves straddling a tier boundary. One ML technique, compared against
  Linear/RF/GBM in Stage 10 -- not a deployable flowchart candidate.
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
print(f"bc ready: n={len(bc)}")

TARGETS = [("A", "frac_over"), ("B", "cluster_rank")]
arms = {}  # (target_key, arm_key) -> dict with tier Series + diagnostics

for tkey, target_col in TARGETS:
    print("\n" + "="*72)
    print(f"TARGET {tkey}  ({target_col})")
    print("="*72)

    # ------------------------------------------------------------
    # POINT-CARD
    # ------------------------------------------------------------
    print("\n" + "-"*50)
    print(f"[{tkey}1] POINT-CARD")
    print("-"*50)
    specs1 = build_format1_specs(bc, target_col)
    for feat in NUMERIC_7:
        s = specs1[feat]
        print(f"  {feat:20s} edges={tuple(round(e, 2) for e in s['edges'])}  "
              f"bin_means={s['bin_means'].round(3).to_dict()}  points={s['points']}")
    dg = specs1["design_group"]
    dg_n = bc.groupby("design_group", observed=True)[target_col].count()
    print(f"\n  design_group lookup table (5-row hump):")
    for cat in dg["means"].index:
        flag = "  [DESCRIPTIVE ONLY, thin n]" if dg_n.get(cat, 0) < 25 else ""
        print(f"    {cat:14s} n={dg_n.get(cat, 0):3d}  mean={dg['means'][cat]:.3f}  "
              f"points={dg['points'][cat]}{flag}")
    mat = specs1["building_material"]
    mat_n = bc["building_material"].where(~bc["building_material"].isin(["Steel"]), "Other").value_counts()
    print(f"\n  building_material lookup table (steel folded into 'Other'):")
    for cat in mat["means"].index:
        print(f"    {cat:34s} n={mat_n.get(cat, 0):3d}  mean={mat['means'][cat]:.3f}  points={mat['points'][cat]}")

    total1, pts1 = score_total(bc, specs1)
    tier1, edges1 = assign_tier(total1)
    print(f"\n  Total score range: [{total1.min():.0f}, {total1.max():.0f}]  tertile edges={tuple(round(e,2) for e in edges1)}")
    print(f"  Tier sizes: {tier1.value_counts().reindex(['Low','Medium','High']).to_dict()}")
    arms[(tkey, "1")] = {"specs": specs1, "total": total1, "tier": tier1, "edges": edges1}

    # ------------------------------------------------------------
    # DECISION TREE
    # ------------------------------------------------------------
    print("\n" + "-"*50)
    print(f"[{tkey}2] DECISION TREE")
    print("-"*50)
    tree_result = fit_tree_per_target(bc, target_col)
    print(f"  Guardrail cross-check: unconstrained depth-2 tree on one-hot design_group alone "
          f"top split = {tree_result['check_top_feature']}  "
          f"(confirmed dg_y2004_2009 = {tree_result['guardrail_confirmed']})")
    for bname, branch in tree_result["branches"].items():
        print(f"\n  Branch '{bname}' (n={branch['n']}):")
        print("    " + branch["text"].replace("\n", "\n    "))
        if branch["fallback"]:
            for lid, fb in branch["fallback"].items():
                print(f"    -- fallback split on leaf {lid} (straddled a tier boundary):")
                print("    " + fb["text"].replace("\n", "\n    "))
        else:
            print("    (no leaves straddled a tier boundary -- no fallback split needed)")

    def _tier2(row):
        from common import predict_tree_tier
        t, _ = predict_tree_tier(tree_result, row)
        return t
    valid_rows = bc.dropna(subset=[target_col] + ["design_group", "n_floors", "building_material",
                                                    "distance_to_sea_m", "dist_major_road_m", "n_apartments",
                                                    "greenery_score", "ALUETEHOK", "road_density_100m"])
    tier2 = valid_rows.apply(_tier2, axis=1)
    print(f"\n  Tier sizes (all buildings scored via the tree, n={len(tier2)}): "
          f"{tier2.value_counts().reindex(['Low','Medium','High']).to_dict()}")
    arms[(tkey, "2")] = {"tree_result": tree_result, "tier": tier2}

# ============================================================
# WITHIN-TARGET, WITHIN-FORMAT SANITY: design_group hump preserved in every arm?
# ============================================================
print("\n" + "="*50)
print("SANITY CHECK -- design_group hump preserved across both targets?")
print("="*50)
for tkey, target_col in TARGETS:
    dg_means = bc.groupby("design_group", observed=True)[target_col].mean().reindex(DG_ORDER)
    hump = dg_means.idxmax() == "y2004_2009"
    print(f"  Target {tkey}: peak design_group = {dg_means.idxmax()}  (hump={hump})")

print("\n>>> STAGE 9 complete. Both arms built for both targets -- see printed point/split tables above.")
