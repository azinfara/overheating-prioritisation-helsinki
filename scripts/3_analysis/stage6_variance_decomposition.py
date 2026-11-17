"""
STAGE 6 -- Variance decomposition (keystone)
Within-building overheating spread in the Helsinki residential stock.

1. ICC (mixed model, building random effect) for temp_mean and temp_max --
   expect ~0.34 both (66% of variance is WITHIN a building, not between buildings).
2. Three-way R2 partition (construction-unique / urban-unique / shared) for
   5 outcomes: overheating, prevalence, peak-tendency, overheating spread, worst-apt.
   Expect between-building outcomes R2~0.1-0.2, near-zero/negative for
   overheating spread and worst-apt (confirmed with a Random Forest robustness check).
3. Apartment area/type as WITHIN-building predictors: how much of the
   variance left over after removing building-to-building differences do
   apartment size/room-count explain?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
from common import (
    PLOTS, MIN_APTS, CONSTRUCTION_FEATURES, URBAN_FEATURES,
    TABLE2_CONSTRUCTION_NUM, TABLE2_URBAN_NUM, attach_risk_features,
    load_buildings_metadata, load_apartment_metrics, load_hourly_indoor,
    compute_swing_range, build_apt_table, build_building_comp, attach_building_features, set_style,
)

set_style()

print(f"""
{'='*72}
STAGE 6 -- DEFINITIONS
{'='*72}
ICC          : intraclass correlation from a random-intercept mixed model
               (outcome ~ 1 + (1 | building_id)), NO fixed-effect covariates.
               ICC = between-building variance / total variance.
               High ICC = apartments in the same building resemble each other;
               (1 - ICC) = share of variance that is WITHIN a building.
3-way partition : for construction-only, urban-only, and full (both) feature
               sets, fit sklearn LinearRegression and report 5-fold CV R2.
                 unique_construction = R2_full - R2_urban_only
                 unique_urban        = R2_full - R2_construction_only
                 shared              = R2_construction_only + R2_urban_only - R2_full
               (all R2 are cross-validated, matching Stage 5's finding that
               in-sample R2 is misleadingly optimistic for weak outcomes)
RF check     : RandomForestRegressor, same CV -- a nonlinear model should
               ALSO fail on overheating spread/worst-apt if the null result is real
               and not just a linear-model blind spot ('RF negative').
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

# ============================================================
# 1. ICC -- random-intercept mixed model, NO covariates
# ============================================================
print("\n" + "="*50)
print("1. ICC (building random effect)")
print("="*50)
icc_vals = {}
for outcome in ["temp_mean", "temp_max"]:
    d = apt[[outcome, "building_id"]].dropna()
    md = smf.mixedlm(f"{outcome} ~ 1", d, groups=d["building_id"])
    fit = md.fit(reml=True)
    between_var = float(fit.cov_re.iloc[0, 0])
    within_var = float(fit.scale)
    icc = between_var / (between_var + within_var)
    icc_vals[outcome] = icc
    print(f"\n{outcome}: n_apartments={len(d)}, n_buildings={d['building_id'].nunique()}")
    print(f"  between-building variance = {between_var:.3f}")
    print(f"  within-building variance  = {within_var:.3f}")
    print(f"  ICC = {icc:.3f}   [expect ~0.34]   -> {1-icc:.0%} of variance is WITHIN buildings")

# ============================================================
# Feature blocks = the nine features of the paper (Table 2), 13 November 2026.
# Construction: design period, material (dummies), floors, apartments.
# Urban: greenery score, land-use efficiency, road density, distance to major road, distance to sea.
# ============================================================
bc = attach_risk_features(bc, buildings_meta)
numeric_construction = TABLE2_CONSTRUCTION_NUM
numeric_urban = TABLE2_URBAN_NUM


def make_X(df, blocks):
    d = df.copy()
    parts = []
    if "construction" in blocks:
        d["design_group"] = d["design_group"].astype(str)
        parts.append(pd.get_dummies(d["design_group"], prefix="dg", drop_first=True))
        parts.append(pd.get_dummies(d["building_material"].fillna("Unknown"), prefix="mat", drop_first=True))
        parts.append(d[numeric_construction])
    if "urban" in blocks:
        parts.append(d[numeric_urban])
    X = pd.concat(parts, axis=1)
    return X.astype(float)


# ============================================================
# 2. THREE-WAY VARIANCE PARTITION, 5 outcomes
# ============================================================
print("\n" + "="*50)
print("2. THREE-WAY R2 PARTITION (5-fold CV)")
print("="*50)

bld_mean_dh27 = apt.groupby("building_id")["dh_27"].mean().rename("bld_mean_dh27")
bld_peak_dh27 = apt.groupby("building_id")["dh_27"].max().rename("bld_peak_dh27")
bld_mean_peak_temp = apt.groupby("building_id")["temp_max"].mean().rename("bld_mean_peak_temp")
bld_mean_tempmean = apt.groupby("building_id")["temp_mean"].mean().rename("bld_mean_tempmean")
bc2 = bc.join(bld_mean_dh27).join(bld_peak_dh27).join(bld_mean_peak_temp).join(bld_mean_tempmean)

OUTCOMES = {
    "overheating (bld mean dh_27)": "bld_mean_dh27",
    "prevalence (frac exceed criteria)": "frac_over",
    "mean apt temp (bld mean of temp_mean)": "bld_mean_tempmean",
    "peak-tendency (mean apt peak)": "bld_mean_peak_temp",
    "overheating spread (spread_temp)": "spread_temp",
    "worst-apt (bld max dh_27)": "bld_peak_dh27",
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
needed_cols = numeric_construction + numeric_urban + ["design_group", "building_material"]

partition_rows = []
rf_rows = []
for label, col in OUTCOMES.items():
    sub = bc2.dropna(subset=[col] + needed_cols)
    y = sub[col].astype(float)
    Xc = make_X(sub, ["construction"])
    Xu = make_X(sub, ["urban"])
    Xf = make_X(sub, ["construction", "urban"])

    r2_c = cross_val_score(LinearRegression(), Xc, y, cv=kf, scoring="r2").mean()
    r2_u = cross_val_score(LinearRegression(), Xu, y, cv=kf, scoring="r2").mean()
    r2_f = cross_val_score(LinearRegression(), Xf, y, cv=kf, scoring="r2").mean()

    unique_c = r2_f - r2_u
    unique_u = r2_f - r2_c
    shared = r2_c + r2_u - r2_f

    print(f"\n{label} (n={len(sub)}):")
    print(f"  R2_construction_only={r2_c:+.3f}  R2_urban_only={r2_u:+.3f}  R2_full={r2_f:+.3f}")
    print(f"  -> unique_construction={unique_c:+.3f}  unique_urban={unique_u:+.3f}  shared={shared:+.3f}")

    rf = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=5)
    r2_rf = cross_val_score(rf, Xf, y, cv=kf, scoring="r2").mean()
    verdict = "confirms null (RF also fails)" if r2_rf < 0.05 else "RF finds signal linear missed"
    print(f"  Random Forest R2 (same features, CV) = {r2_rf:+.3f}   -> {verdict}")

    partition_rows.append({"outcome": label, "R2_full": round(r2_f, 3), "unique_construction": round(unique_c, 3),
                            "unique_urban": round(unique_u, 3), "shared": round(shared, 3), "n": len(sub)})
    rf_rows.append({"outcome": label, "R2_linear": round(r2_f, 3), "R2_RF": round(r2_rf, 3)})

print("\n" + "="*50)
print("PARTITION SUMMARY")
print("="*50)
print(pd.DataFrame(partition_rows).to_string(index=False))
print("\nExpect: overheating/prevalence/peak-tendency clearly R2>0; overheating spread/worst-apt near-zero or negative,")
print("and confirmed by Random Forest also failing on those two (not just a linear-model limitation).")
print("\n" + pd.DataFrame(rf_rows).to_string(index=False))
# saved for the Results figures (fig_results.py)
os.makedirs(os.path.join(os.path.dirname(PLOTS), "exports"), exist_ok=True)
pd.DataFrame(partition_rows).merge(pd.DataFrame(rf_rows)[["outcome", "R2_RF"]], on="outcome").assign(
    col=list(OUTCOMES.values())).to_csv(os.path.join(os.path.dirname(PLOTS), "exports", "step2_r2.csv"), index=False)

# ============================================================
# 2b. VARIANCE PARTITION EXPLAINED -- ICC AS CEILING, R2 AS ACHIEVED
# ICC (step 1 above) answers "how much of apartment-to-apartment variance
# COULD ever be explained by ANY building-level factor" -- it says nothing about
# whether the SPECIFIC features in this registry actually capture that room.
# R2 (step 2 above) answers exactly that: what construction+urban features
# actually explain, out of the same variance. Pairing the SAME outcome computed
# both ways (building-mean temp_mean vs its ICC; building-mean temp_max, i.e.
# "peak-tendency", vs its ICC) makes the ceiling-vs-achieved comparison exact,
# not just a vague analogy across different outcome definitions.
# ============================================================
print("\n" + "="*50)
print("2b. VARIANCE PARTITION EXPLAINED -- ICC (ceiling) vs R2 (achieved)")
print("="*50)
pdict = {row["outcome"]: row["R2_full"] for row in partition_rows}
r2_mean_tempmean = pdict["mean apt temp (bld mean of temp_mean)"]
r2_peak_tendency = pdict["peak-tendency (mean apt peak)"]

ceiling_rows = [
    ("temp_mean", icc_vals["temp_mean"], r2_mean_tempmean),
    ("temp_max (peak-tendency)", icc_vals["temp_max"], r2_peak_tendency),
]
print("Outcome                    ICC (ceiling)   R2 (achieved)   % of ceiling captured")
for name, icc, r2 in ceiling_rows:
    pct_captured = 100 * r2 / icc if icc > 0 else float("nan")
    print(f"  {name:24s} {icc:.3f}           {r2:+.3f}          {pct_captured:.0f}%")

print("\nInterpretation: ICC says ~34-36% of RAW apartment temperature variance sits between")
print("buildings at all -- the most any building-level factor (known or unknown) could ever explain.")
print("Construction+urban features capture roughly HALF of that ceiling for both temp_mean and")
print("temp_max (~44-48%) -- real, substantial, but leaving the other half of the explainable room")
print("to factors not in this registry (insulation quality, glazing, occupant behavior, HVAC).")
print("By contrast, overheating SPREAD and the single worst-apartment value (Stage 5/6 above) show")
print("R2 <= 0 -- not 'half the ceiling missed', but NO measurable ceiling captured at all for those")
print("two outcomes specifically, a qualitatively different (harder) result than 'mean' or 'peak-tendency'.")

# ---- FIG C -- ceiling (ICC) vs achieved (R2), 3-way stacked: within / explained-between / unexplained-between
fig, ax = plt.subplots(figsize=(7.5, 5.5))
labels_c = [c[0] for c in ceiling_rows]
within_pct_c = [(1 - c[1]) * 100 for c in ceiling_rows]
explained_pct_c = [max(c[2], 0) * 100 for c in ceiling_rows]
unexplained_pct_c = [max(c[1] - max(c[2], 0), 0) * 100 for c in ceiling_rows]
bottom = np.zeros(len(labels_c))
for vals, label, color in [
    (within_pct_c, "Within-building (no building feature can explain this)", "#bdbdbd"),
    (explained_pct_c, "Between-building, EXPLAINED by construction+urban features", "#2c7bb6"),
    (unexplained_pct_c, "Between-building, UNEXPLAINED (real, but not in this registry)", "#d7191c"),
]:
    ax.bar(labels_c, vals, bottom=bottom, label=label, color=color, edgecolor="white")
    for i, v in enumerate(vals):
        if v > 2:
            ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center",
                    color="white" if color != "#bdbdbd" else "black", fontsize=9.5, fontweight="bold")
    bottom += np.array(vals)
ax.set_ylabel("% of total apartment-temperature variance")
ax.set_title("ICC is the ceiling; registry features explain about half of it", fontsize=13, fontweight="bold")
ax.legend(frameon=False, bbox_to_anchor=(0.5, -0.18), loc="upper center", fontsize=8.5)
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage6_ceiling_vs_achieved.png")
plt.close(fig)
print("Saved: plots/stage6_ceiling_vs_achieved.png")

# ============================================================
# 3. APARTMENT AREA/TYPE AS WITHIN-BUILDING PREDICTORS
# ============================================================
print("\n" + "="*50)
print("3. APARTMENT AREA/TYPE -- WITHIN-BUILDING VARIANCE EXPLAINED")
print("="*50)
apt_b = apt[apt["building_id"].isin(building_comp.index)].copy()  # restrict to >=3-apt buildings, consistent unit
for outcome in ["dh_27", "temp_mean"]:
    d = apt_b.dropna(subset=[outcome, "area", "type", "building_id"]).copy()
    d["within"] = d[outcome] - d.groupby("building_id")[outcome].transform("mean")
    X = d[["area", "type"]].astype(float)
    r2_within_cv = cross_val_score(LinearRegression(), X, d["within"], cv=kf, scoring="r2").mean()
    r2_within_insample = LinearRegression().fit(X, d["within"]).score(X, d["within"])
    print(f"\n{outcome}: within-building variance ~ area + type (n={len(d)} apartments, "
          f"{d['building_id'].nunique()} buildings)")
    print(f"  in-sample R2 = {r2_within_insample:.3f}   5-fold CV R2 = {r2_within_cv:+.3f}")
print("\n(Low R2 here would mean apartment size/room-count barely explain WHY one apartment in a")
print(" building runs hotter than another -- pointing to unmeasured factors: orientation, floor, layout.)")

# ============================================================
# FIG A -- ICC bar (between vs within)
# ============================================================
icc_vals = {}
for outcome in ["temp_mean", "temp_max"]:
    d = apt[[outcome, "building_id"]].dropna()
    fit = smf.mixedlm(f"{outcome} ~ 1", d, groups=d["building_id"]).fit(reml=True)
    between_var = float(fit.cov_re.iloc[0, 0])
    within_var = float(fit.scale)
    icc_vals[outcome] = between_var / (between_var + within_var)

fig, ax = plt.subplots(figsize=(6, 4.5))
labels = list(icc_vals.keys())
between_pct = [icc_vals[k] * 100 for k in labels]
within_pct = [100 - v for v in between_pct]
ax.bar(labels, between_pct, label="Between-building", color="#2c7bb6")
ax.bar(labels, within_pct, bottom=between_pct, label="Within-building", color="#fdae61")
for i, k in enumerate(labels):
    ax.text(i, between_pct[i] / 2, f"{between_pct[i]:.0f}%", ha="center", va="center", color="white", fontweight="bold")
    ax.text(i, between_pct[i] + within_pct[i] / 2, f"{within_pct[i]:.0f}%", ha="center", va="center", fontweight="bold")
ax.set_ylabel("% of total variance")
ax.set_title("Most temperature variance is WITHIN buildings, not between them")
ax.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage6_icc_between_within.png")
plt.close(fig)

# ============================================================
# FIG B -- variance partition by outcome (stacked bar)
# ============================================================
pdf = pd.DataFrame(partition_rows).set_index("outcome")
pdf["unexplained"] = 1 - pdf["R2_full"]
comp_cols = ["unique_construction", "unique_urban", "shared", "unexplained"]
colors = ["#2c7bb6", "#2ca25f", "#abd9e9", "#bdbdbd"]
fig, ax = plt.subplots(figsize=(10, 5))
bottom = np.zeros(len(pdf))
for col, color in zip(comp_cols, colors):
    vals = pdf[col].clip(lower=0).values  # negative unique/shared slivers clipped for display, noted in caption
    ax.bar(pdf.index, vals, bottom=bottom, label=col.replace("_", " "), color=color, edgecolor="white")
    bottom += vals
ax.set_ylabel("Share of variance (R2, clipped at 0)")
ax.set_title("Variance partition: construction vs urban features, by outcome")
ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage6_variance_partition.png")
plt.close(fig)

print("\nSaved: plots/stage6_icc_between_within.png, plots/stage6_variance_partition.png")

# ============================================================
# SIMPLE FIGURES (presentation versions of FIG A and FIG B)
# ============================================================
C_BETWEEN, C_WITHIN, C_NONE = "#2a78d6", "#eb6834", "#c8c7c2"
INK, INK_2 = "#0b0b0b", "#52514e"

# ---- SIMPLE 1 -- where do apartment temperature differences come from?
fig, ax = plt.subplots(figsize=(9, 4))
rows_s1 = [("Average temperature", icc_vals["temp_mean"]), ("Peak temperature", icc_vals["temp_max"])]
for i, (name, icc) in enumerate(rows_s1):
    b, w = icc * 100, (1 - icc) * 100
    ax.barh(i, b, color=C_BETWEEN, height=0.55, edgecolor="white", linewidth=2)
    ax.barh(i, w, left=b, color=C_WITHIN, height=0.55, edgecolor="white", linewidth=2)
    ax.text(b / 2, i, f"{b:.0f}%", ha="center", va="center", color="white", fontsize=13, fontweight="bold")
    ax.text(b + w / 2, i, f"{w:.0f}%", ha="center", va="center", color="white", fontsize=13, fontweight="bold")
ax.set_yticks(range(len(rows_s1)), [r[0] for r in rows_s1], fontsize=11, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], color=INK_2)
ax.grid(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Share of all differences between apartments", color=INK_2)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_BETWEEN), plt.Rectangle((0, 0), 1, 1, color=C_WITHIN)],
          labels=["Between buildings (one building warmer than another)",
                  "Within the same building (neighbours differ)"],
          frameon=False, loc="upper left", bbox_to_anchor=(0, -0.28), ncol=2, fontsize=10)
ax.set_title("Two-thirds of the temperature differences are between neighbours",
             fontsize=13, fontweight="bold", color=INK, loc="left", pad=12)
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage6_simple_where_differences.png")
plt.close(fig)

# ---- SIMPLE 2 -- what can building data (construction + urban) predict?
plain_names = {
    "overheating (bld mean dh_27)": "Average overheating (degree-hours > 27 C)",
    "prevalence (frac exceed criteria)": "Share of apartments overheating",
    "mean apt temp (bld mean of temp_mean)": "Average apartment temperature",
    "peak-tendency (mean apt peak)": "Typical peak temperature",
    "overheating spread (spread_temp)": "Temperature gap between apartments",
    "worst-apt (bld max dh_27)": "Worst apartment's overheating",
}
r2_simple = pd.Series({plain_names[r["outcome"]]: r["R2_full"] for r in partition_rows}).reindex(plain_names.values())
fig, ax = plt.subplots(figsize=(9, 4.4))
for i, (name, r2) in enumerate(r2_simple.items()):
    pct = max(r2, 0) * 100
    ax.barh(i, 100, color="#f1f0ec", height=0.6)  # track = everything that could be predicted
    if pct > 0:
        ax.barh(i, pct, color=C_BETWEEN, height=0.6)
        ax.text(pct + 1.5, i, f"{pct:.0f}%", va="center", fontsize=11, fontweight="bold", color=INK)
    else:
        ax.text(1.5, i, "0%  -  cannot be predicted", va="center", fontsize=11, fontweight="bold", color=INK)
ax.set_yticks(range(len(r2_simple)), r2_simple.index, fontsize=11, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], color=INK_2)
ax.grid(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.axhline(3.5, color=INK_2, linewidth=0.8, linestyle=":")
ax.set_xlabel("How much building data predicts (cross-validated R2, negative shown as 0)", color=INK_2)
ax.set_title("Building data can't predict the gap between apartments",
             fontsize=13, fontweight="bold", color=INK, loc="left", pad=26)
ax.text(0, 1.025,"Predictors: age, material, floors, size, trees, paved surface, sea & road distance",
        transform=ax.transAxes, fontsize=9.5, color=INK_2)
plt.tight_layout()
plt.savefig(rf"{PLOTS}/stage6_simple_predictability.png")
plt.close(fig)
print("Saved: plots/stage6_simple_where_differences.png, plots/stage6_simple_predictability.png")
print(">>> STAGE 6 complete.")
