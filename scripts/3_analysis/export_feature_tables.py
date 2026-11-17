"""
Export feature scoring tables to Excel for reports/slides.
Reproduces the point-card scoring tables (percentile groups /
categories, N, average real overheating outcome, points awarded) for all 9
locked features, against BOTH targets, as clean spreadsheets a report can
cite directly. Not part of the numbered stage pipeline -- a reporting
utility, safe to re-run any time; always recomputes from the raw sources.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from common import (
    NUMERIC_7, MIN_APTS,
    load_buildings_metadata, load_apartment_metrics, load_hourly_indoor,
    compute_swing_range, build_apt_table, build_building_comp, attach_building_features,
    attach_risk_features, build_severity_cluster, build_format1_specs,
)

import os
BASE = os.environ.get("OVERHEATING_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
OUT_XLSX = rf"{BASE}/exports/paper2_feature_scoring_tables.xlsx"

print("Loading data and rebuilding the point-card (Format 1) specs ...")
buildings_meta = load_buildings_metadata()
apt_metrics = load_apartment_metrics(merge_area_type=True)
hourly = load_hourly_indoor()
swing_df = compute_swing_range(hourly)
del hourly
apt = build_apt_table(apt_metrics, buildings_meta, swing_df)
building_comp, apt, lag_med, swing_med = build_building_comp(apt)
bc = attach_building_features(building_comp, buildings_meta)
bc = attach_risk_features(bc, buildings_meta)
cluster_df, sil = build_severity_cluster(buildings_meta, min_apts=MIN_APTS)
bc["cluster_rank"] = cluster_df["cluster_rank"].reindex(bc.index)

TARGET_LABELS = {"frac_over": "Target A -- frac_over (composition risk)",
                  "cluster_rank": "Target B -- cluster_rank (severity risk)"}


def numeric_rows(feat, target_col, specs):
    spec = specs[feat]
    e1, e2 = spec["edges"]
    binned = pd.cut(bc[feat], bins=[-float("inf"), e1, e2, float("inf")], labels=[0, 1, 2])
    g = bc.assign(_bin=binned).groupby("_bin", observed=True)[target_col].agg(["count", "mean"])
    bin_desc = {0: f"<= {e1:,.2f}", 1: f"{e1:,.2f} to {e2:,.2f}", 2: f"> {e2:,.2f}"}
    rows = []
    for bin_idx in [0, 1, 2]:
        rows.append({
            "Feature": feat, "Type": "numeric (percentile thirds)",
            "Group": bin_desc[bin_idx],
            "N buildings": int(g.loc[bin_idx, "count"]) if bin_idx in g.index else 0,
            "Average outcome": round(float(g.loc[bin_idx, "mean"]), 3) if bin_idx in g.index else None,
            "Points": spec["points"][bin_idx],
        })
    return rows


def categorical_rows(feat, target_col, specs):
    spec = specs[feat]
    means = spec["means"]
    fold = spec["fold_into_other"]
    s = bc[feat].astype(object)
    if fold:
        s = s.where(~s.isin(fold), "Other")
    counts = s.value_counts()
    rows = []
    for cat in means.index:
        rows.append({
            "Feature": feat, "Type": "categorical",
            "Group": cat,
            "N buildings": int(counts.get(cat, 0)),
            "Average outcome": round(float(means[cat]), 3),
            "Points": spec["points"][cat],
        })
    return rows


with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    for target_col in ["frac_over", "cluster_rank"]:
        specs = build_format1_specs(bc, target_col)
        rows = []
        for feat in NUMERIC_7:
            rows += numeric_rows(feat, target_col, specs)
        rows += categorical_rows("design_group", target_col, specs)
        rows += categorical_rows("building_material", target_col, specs)
        df = pd.DataFrame(rows)
        sheet = "Target A - frac_over" if target_col == "frac_over" else "Target B - cluster_rank"
        df.to_excel(writer, sheet_name=sheet, index=False)
        print(f"Wrote sheet '{sheet}': {len(df)} rows ({df['Feature'].nunique()} features).")

    notes = pd.DataFrame({"Notes": [
        "Point-score card (Format 1) scoring tables -- reproduces the exact tables used to build",
        "every rubric card. n=516 buildings (>=3 monitored apartments, has_temp_data for Target B).",
        "",
        "How to read each row:",
        "- 'Group' = a percentile third (numeric features, cut at the 33rd/67th percentile of that",
        "  feature's values) or a real category (design_group era / building_material).",
        "- 'Average outcome' = the real average value of the target (frac_over, or cluster_rank",
        "  0/1/2) for buildings falling in that group.",
        "- 'Points' = 0/1/2, assigned by RANKING the groups' average outcomes low-to-high -- NOT by",
        "  the raw feature value's direction. This is why some features (dist_major_road_m,",
        "  greenery_score) show their points running in the opposite direction from their group",
        "  order: the group with the lowest real overheating average always gets 0 points.",
        "",
        "Target A = frac_over (fraction of apartments individually exceeding dh_27>150).",
        "Target B = cluster_rank (0=Low/1=Medium/2=High KMeans severity cluster).",
        "",
        "Source: scripts/overheating/export_feature_tables.py, recomputed from final/*.csv + parquet.",
    ]})
    notes.to_excel(writer, sheet_name="Notes", index=False, header=False)

print(f"\nSaved: {OUT_XLSX}")
