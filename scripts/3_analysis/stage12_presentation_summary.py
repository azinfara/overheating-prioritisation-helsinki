"""
STAGE 12 -- Presentation summary
Overheating Risk Categorization Rubric ("Energy Label for Overheating").

Renders the small set of presentation/communication figures built during
post-hoc review of Stages 7-11's results (not part of the original locked
plan, added afterward so every figure in plots/ has a reproducible source):
  1. Point-card vs. decision tree at the TREE'S BEST POSSIBLE DEPTH (a
     depth-sweep robustness check, not just the guardrailed ~depth-2 tree) --
     settles whether more tree complexity would change the point-card
     recommendation. It does not, on either target.
  2. Cumulative gains ("does the ranking work even though the 3 tiers
     overlap?") for BOTH targets -- inspecting the top X% of buildings by
     score catches more true-High buildings than picking X% at random, at
     every point along the list.
  3. A concrete accuracy-vs-complexity decision table: point-card vs. the
     guardrailed tree vs. the tree at its best depth.
  4. "Does the label match reality?" -- out-of-fold point-card tier vs. TRUE
     severity tier, for BOTH targets. Shows the tiers are not clean categories
     (they overlap, especially Low/Medium). The false-alarm rate for "High" is
     genuinely near-zero but NOT identically 0% on both targets -- computed
     per-target below, not hardcoded (Target B lands at 0%, Target A at a
     small but nonzero rate); "Low" is not a safe clearance either way (a real
     share of true-High buildings still get filed there).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from common import (
    PLOTS, PAL5, MIN_APTS, ALL_9_FEATURES, NUMERIC_7,
    load_buildings_metadata, load_apartment_metrics, load_hourly_indoor,
    compute_swing_range, build_apt_table, build_building_comp, attach_building_features,
    attach_risk_features, build_severity_cluster, build_composition_cluster,
    build_format1_specs, score_total, assign_tier, set_style,
)

set_style()
tier_colors = {"Low": PAL5[0], "Medium": PAL5[2], "High": PAL5[4]}

print(f"""
{'='*72}
STAGE 12 -- PRESENTATION SUMMARY (post-hoc, not part of the original locked plan)
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
cluster_df_b, sil_b = build_severity_cluster(buildings_meta, min_apts=MIN_APTS)
bc["cluster_rank"] = cluster_df_b["cluster_rank"].reindex(bc.index)
bc["cluster_tier"] = cluster_df_b["cluster_tier"].reindex(bc.index)
cluster_df_a, sil_a = build_composition_cluster(bc, min_apts=MIN_APTS)
bc["cluster_tier_A"] = cluster_df_a["cluster_tier"].reindex(bc.index)

TARGETS = [("A", "frac_over", "cluster_tier_A"), ("B", "cluster_rank", "cluster_tier")]


def make_X(df):
    d = df.copy()
    dg = pd.get_dummies(d["design_group"].astype(str), prefix="dg")
    mat = pd.get_dummies(d["building_material"].fillna("Unknown"), prefix="mat")
    return pd.concat([dg, mat, d[NUMERIC_7]], axis=1).astype(float)


# ============================================================
# OOF point-card score + OOF tier + OOF best-depth-tree score, both targets
# ============================================================
print("\n" + "="*50)
print("OUT-OF-FOLD SCORING -- point-card (tier + continuous) and depth-4 tree")
print("="*50)
results = {}
for tkey, target_col, true_tier_col in TARGETS:
    sub = bc.dropna(subset=[target_col, true_tier_col] + ALL_9_FEATURES).copy()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    idx = sub.index.to_numpy()

    pc_score = pd.Series(index=sub.index, dtype=float)
    pc_tier = pd.Series(index=sub.index, dtype=object)
    for tr, te in kf.split(idx):
        train, test = sub.loc[idx[tr]], sub.loc[idx[te]]
        specs1 = build_format1_specs(train, target_col)
        tot_tr, _ = score_total(train, specs1)
        tot_te, _ = score_total(test, specs1)
        tier_te, _ = assign_tier(tot_te, train_score=tot_tr)
        pc_score.loc[idx[te]] = tot_te.values
        pc_tier.loc[idx[te]] = tier_te.values

    X = make_X(sub)
    y = sub[target_col].astype(float)
    tree_score = pd.Series(cross_val_predict(
        DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=42), X, y, cv=kf), index=sub.index)
    n_leaves = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=42).fit(X, y).get_n_leaves()

    r_pc, _ = stats.spearmanr(pc_score, y)
    r_tree, _ = stats.spearmanr(tree_score, y)
    ct = pd.crosstab(pc_tier, sub[true_tier_col]).reindex(index=["Low", "Medium", "High"], columns=["Low", "Medium", "High"], fill_value=0)

    results[tkey] = {"sub": sub, "pc_score": pc_score, "pc_tier": pc_tier, "tree_score": tree_score,
                      "true_tier_col": true_tier_col, "r_pc": r_pc, "r_tree": r_tree, "n_leaves": n_leaves, "crosstab": ct}
    print(f"Target {tkey}: point-card r={r_pc:+.3f}, depth-4 tree r={r_tree:+.3f} ({n_leaves} leaves)")
    print(ct)
    acc = np.trace(ct.values) / ct.values.sum() * 100
    miss = ct.loc["Low", "High"] / ct.loc["Low"].sum() * 100
    false_alarm = ct.loc["High", "Low"] / ct.loc["High"].sum() * 100
    print(f"  exact match={acc:.1f}%, Low-labeled-but-truly-High={miss:.1f}%, High-labeled-but-truly-Low={false_alarm:.1f}%")

# ============================================================
# FIG 1 -- point-card vs decision tree at its best depth, both targets
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5.5))
labels = ["Target A\n(frac_over)", "Target B\n(cluster_rank)"]
x = np.arange(2)
width = 0.32
pc_vals = [results["A"]["r_pc"], results["B"]["r_pc"]]
tree_vals = [results["A"]["r_tree"], results["B"]["r_tree"]]
ax.bar(x - width/2, pc_vals, width=width, color=PAL5[0], label="Point-score card")
ax.bar(x + width/2, tree_vals, width=width, color=PAL5[4], label="Decision tree (best depth = 4)")
for i, v in enumerate(pc_vals):
    ax.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
for i, v in enumerate(tree_vals):
    ax.text(i + width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Out-of-fold Spearman r (validation accuracy)")
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_title("Validation: point-card vs decision tree (at its best depth)", fontsize=14, fontweight="bold")
ax.legend(frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig(rf"{PLOTS}/paper2_final_pointcard_vs_tree.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nSaved: plots/paper2_final_pointcard_vs_tree.png")

# ============================================================
# FIG 2 -- cumulative gains, both targets side by side
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, tkey in zip(axes, ["A", "B"]):
    r = results[tkey]
    df = pd.DataFrame({"score": r["pc_score"], "true_tier": r["sub"][r["true_tier_col"]]}).dropna()
    df["is_high"] = (df["true_tier"] == "High").astype(int)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    total_high = df["is_high"].sum()
    n = len(df)
    pct_inspected = np.arange(1, n + 1) / n * 100
    pct_high_caught = df["is_high"].cumsum() / total_high * 100

    ax.plot(pct_inspected, pct_high_caught, color=PAL5[4], linewidth=2.5, label="Point-card ranking (OOF)")
    ax.plot([0, 100], [0, 100], "--", color="grey", linewidth=1.5, label="Random selection")
    for mark_pct in [20, 50]:
        im = int(n * mark_pct / 100) - 1
        caught = pct_high_caught[im]
        ax.plot([mark_pct, mark_pct], [0, caught], ":", color="#888", linewidth=1)
        ax.plot([0, mark_pct], [caught, caught], ":", color="#888", linewidth=1)
        ax.text(mark_pct + 3, caught - 6, f"top {mark_pct}% -> {caught:.0f}%", fontsize=9.5, fontweight="bold", color="#333")
    ax.set_xlabel("% of building stock inspected")
    ax.set_ylabel("% of TRUE High-risk buildings caught" if tkey == "A" else "")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_title(f"Target {tkey}", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("The ranking has real value, even though the 3 tiers overlap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/paper2_final_cumulative_gains_both_targets.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: plots/paper2_final_cumulative_gains_both_targets.png")

# ============================================================
# FIG 3 -- final decision table: point-card vs current tree vs depth-4 tree
# ============================================================
t = pd.DataFrame({
    "Steps per\nbuilding": ["9 additions,\nalways", "2-3\nquestions", "up to 4\nquestions"],
    "End-states to\ndocument": ["1 table\n(9 rows)", "~5\nbranches", "~12\nbranches"],
    "Validation r\nTarget A": [f"{results['A']['r_pc']:.3f}", "0.307", f"{results['A']['r_tree']:.3f}"],
    "Validation r\nTarget B": [f"{results['B']['r_pc']:.3f}", "0.190", f"{results['B']['r_tree']:.3f}"],
    "Recommended?": ["YES -- deployed", "No -- weakest,\nesp. Target B", "No -- more complex,\nstill loses"],
}, index=["Point-score card", "Decision tree\n(current, ~depth 2)", "Decision tree\n(best depth = 4)"])

fig, ax = plt.subplots(figsize=(15, 3.6))
ax.axis("off")
tbl = ax.table(cellText=t.values, colLabels=t.columns, rowLabels=t.index, loc="center", cellLoc="center",
               colWidths=[0.14, 0.14, 0.13, 0.13, 0.22])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.8)
for (r_, c_), cell in tbl.get_celld().items():
    if r_ == 0:
        cell.set_facecolor("#2c7bb6"); cell.set_text_props(color="white", fontweight="bold")
    elif c_ == -1:
        cell.set_text_props(fontweight="bold")
    elif r_ == 1 and c_ >= 0:
        cell.set_facecolor("#e8f4ea")
ax.set_title("Final format decision: accuracy vs complexity", fontweight="bold", fontsize=14, pad=18)
plt.tight_layout()
plt.savefig(rf"{PLOTS}/paper2_final_decision_table.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: plots/paper2_final_decision_table.png")

# ============================================================
# FIG 4a/4b -- does the label match reality? one per target
# ============================================================
for tkey, target_col, true_tier_col in TARGETS:
    ct = results[tkey]["crosstab"]
    data = {f"Labeled\n{t.upper()}": {tt: int(ct.loc[t, tt]) for tt in ["Low", "Medium", "High"]} for t in ["Low", "Medium", "High"]}

    fig, ax = plt.subplots(figsize=(11.5, 5))
    labels = list(data.keys())
    y = np.arange(len(labels))
    for i, lab in enumerate(labels):
        total = sum(data[lab].values())
        left = 0
        for tier in ["Low", "Medium", "High"]:
            v = data[lab][tier] / total * 100
            ax.barh(i, v, left=left, color=tier_colors[tier], edgecolor="white", height=0.62)
            if v > 6:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
            left += v

    miss_pct = data["Labeled\nLOW"]["High"] / sum(data["Labeled\nLOW"].values()) * 100
    false_alarm_pct = data["Labeled\nHIGH"]["Low"] / sum(data["Labeled\nHIGH"].values()) * 100
    callouts = {
        0: (f"{miss_pct:.0f}% of \"Low\" buildings\nare actually truly High", "#b0301f"),
        2: (f"only {false_alarm_pct:.1f}% false alarms" if false_alarm_pct > 0 else "0% false alarms --\nnever truly-Low", "#2c5a7a"),
    }
    for i, (text, color) in callouts.items():
        ax.text(103, i, text, fontsize=10, color=color, ha="left", va="center", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 165)
    ax.set_xlabel(f"% of buildings, by their TRUE {'composition-risk' if tkey == 'A' else 'severity'} severity")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.invert_yaxis()
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.axvline(100, color="#ccc", linewidth=0.8)

    handles = [Patch(color=tier_colors[t], label=f"Truly {t}") for t in ["Low", "Medium", "High"]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.35, 1.16), ncol=3, frameon=False, fontsize=11)
    ax.set_title(f"Does the label match reality?  (Target {tkey}, point-score card)", fontsize=14, fontweight="bold", pad=38, x=0.35)

    plt.tight_layout()
    fname = f"paper2_label_vs_truth_summary_target{tkey}.png"
    plt.savefig(rf"{PLOTS}/{fname}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: plots/{fname}")

print("\n>>> STAGE 12 complete.")
