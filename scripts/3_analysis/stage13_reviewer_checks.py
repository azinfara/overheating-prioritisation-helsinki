"""
Stage 13 -- reviewer-requested checks (added 9 November 2026).

1. Does within-building spread depend on the number of monitored apartments?
   Compares the raw max-min spread with two size-robust versions:
   spread_fixed3 = expected max-min of 3 randomly drawn apartments per building;
   sd_temp       = standard deviation between a building's apartments.
   Repeats the cross-validated predictability test for each.
2. Top-10% and top-20% capture of truly High buildings by the out-of-fold
   point-card score, with bootstrap 95% confidence intervals.
3. Sensitivity of the 150 degree-hour classification to the June-August window.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from common import (
    MIN_APTS, ALL_9_FEATURES, CONSTRUCTION_FEATURES, URBAN_FEATURES,
    TABLE2_CONSTRUCTION_NUM, TABLE2_URBAN_NUM,
    load_buildings_metadata, load_apartment_metrics, load_hourly_indoor,
    compute_swing_range, build_apt_table, build_building_comp, attach_building_features,
    attach_risk_features, build_severity_cluster, build_composition_cluster,
    build_format1_specs, score_total,
)

SEED = 42
N_DRAWS = 500
N_BOOT = 2000
rng = np.random.default_rng(SEED)

meta = load_buildings_metadata()
apt_metrics = load_apartment_metrics(merge_area_type=True)
print("Loading hourly indoor temperature ...")
hourly = load_hourly_indoor()[["device", "timestamp", "temperature"]]
import gc
gc.collect()

# ------------------------------------------------------------
# 3. June-August window for dh_27
# ------------------------------------------------------------
print("\n" + "=" * 60 + "\n3. dh_27 > 150: full monitoring period vs June-August\n" + "=" * 60)
ts = hourly["timestamp"]
jja = hourly[(ts >= "2021-06-01") & (ts < "2021-09-01")]
dh = (jja["temperature"] - 27).clip(lower=0).groupby(jja["device"]).sum().rename("dh_27_jja")
del jja, ts
gc.collect()
swing_df = compute_swing_range(hourly)
del hourly
gc.collect()
w = apt_metrics[["device", "building_id", "dh_27"]].merge(dh, on="device", how="left")
full, summer = w["dh_27"] > 150, w["dh_27_jja"] > 150
print(f"apartments n={len(w)}")
print(f"over 150, whole period : {full.mean()*100:.1f}%")
print(f"over 150, June-August  : {summer.mean()*100:.1f}%")
print(f"classification changes for {(full != summer).sum()} apartments ({(full != summer).mean()*100:.1f}%)")
print(f"median dh_27 whole={w['dh_27'].median():.0f}, June-August={w['dh_27_jja'].median():.0f}")

# ------------------------------------------------------------
# Tables
# ------------------------------------------------------------
apt = build_apt_table(apt_metrics, meta, swing_df)
building_comp, apt, lag_med, swing_med = build_building_comp(apt)
bc = attach_building_features(building_comp, meta)
bc = attach_risk_features(bc, meta)
cb, _ = build_severity_cluster(meta, min_apts=MIN_APTS)
bc["cluster_rank"] = cb["cluster_rank"].reindex(bc.index)
bc["cluster_tier"] = cb["cluster_tier"].reindex(bc.index)
ca, _ = build_composition_cluster(bc, min_apts=MIN_APTS)
bc["cluster_tier_A"] = ca["cluster_tier"].reindex(bc.index)

# ------------------------------------------------------------
# 1. Spread and sensor count
# ------------------------------------------------------------
print("\n" + "=" * 60 + "\n1. Spread vs number of monitored apartments\n" + "=" * 60)
rows = {}
for b, g in apt_metrics.dropna(subset=["temp_mean"]).groupby("building_id"):
    v = g["temp_mean"].to_numpy()
    if len(v) < MIN_APTS:
        continue
    draws = np.array([np.ptp(rng.choice(v, 3, replace=False)) for _ in range(N_DRAWS)])
    rows[b] = {"n_mon": len(v), "spread_raw": np.ptp(v), "spread_fixed3": draws.mean(), "sd_temp": v.std(ddof=1)}
s = pd.DataFrame.from_dict(rows, orient="index").rename_axis("building_id")
print(f"buildings n={len(s)}; monitored apartments per building median={s['n_mon'].median():.0f}, "
      f"IQR=[{s['n_mon'].quantile(.25):.0f}, {s['n_mon'].quantile(.75):.0f}], max={s['n_mon'].max():.0f}")
METRICS = ["spread_raw", "spread_fixed3", "sd_temp"]
for m in METRICS:
    r, p = stats.spearmanr(s["n_mon"], s[m])
    print(f"  {m:14s} median={s[m].median():.2f} C   Spearman r with n apartments={r:+.3f} (p={p:.2g})")
for lo, hi in [(3, 5), (6, 9), (10, 14), (15, 999)]:
    d = s[(s["n_mon"] >= lo) & (s["n_mon"] <= hi)]
    print(f"  n={lo}-{hi}: buildings={len(d):3d}  raw={d['spread_raw'].median():.2f}  "
          f"fixed3={d['spread_fixed3'].median():.2f}  sd={d['sd_temp'].median():.2f}")

feat = bc.join(s, how="inner", rsuffix="_s")
cat = ["design_group", "building_material"]
num = TABLE2_CONSTRUCTION_NUM + TABLE2_URBAN_NUM  # the nine Table 2 features (with the two dummies)
X = pd.concat([pd.get_dummies(feat["design_group"].astype(str), prefix="dg", drop_first=True),
               pd.get_dummies(feat["building_material"].fillna("Unknown"), prefix="mat", drop_first=True),
               feat[num]], axis=1).astype(float)
ok = X.notna().all(axis=1)
X, feat = X[ok], feat[ok]
kf = KFold(5, shuffle=True, random_state=SEED)
print(f"\nPredictability, same construction + urban features, 5-fold CV (n={len(X)}):")
R2_ROWS = []
for m in METRICS:
    lin = cross_val_score(LinearRegression(), X, feat[m], cv=kf, scoring="r2").mean()
    rf = cross_val_score(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1), X, feat[m], cv=kf, scoring="r2").mean()
    print(f"  {m:14s} linear R2={lin:+.3f}   random forest R2={rf:+.3f}")
    R2_ROWS.append({"measure": m, "R2_linear": round(lin, 3), "R2_RF": round(rf, 3)})
import os
from common import DATA
pd.DataFrame(R2_ROWS).to_csv(os.path.join(os.path.dirname(DATA), "exports", "table4_r2.csv"), index=False)
X2 = X.assign(n_mon=feat["n_mon"])
lin = cross_val_score(LinearRegression(), X2, feat["spread_raw"], cv=kf, scoring="r2").mean()
print(f"  spread_raw with n apartments added as a predictor: linear R2={lin:+.3f}")
lin = cross_val_score(LinearRegression(), feat[["n_mon"]], feat["spread_raw"], cv=kf, scoring="r2").mean()
print(f"  spread_raw from n apartments alone               : linear R2={lin:+.3f}")

# ------------------------------------------------------------
# 2. Capture at top X%
# ------------------------------------------------------------
print("\n" + "=" * 60 + f"\n2. Capture of truly High buildings (bootstrap n={N_BOOT})\n" + "=" * 60)


def capture(score, high, frac, r):
    """Share of High caught in the top frac, ties broken at random."""
    order = np.lexsort((r.random(len(score)), -score))
    k = int(round(len(score) * frac))
    return high[order[:k]].sum() / high.sum(), high[order[:k]].mean()


for key, tcol, tier in [("A", "frac_over", "cluster_tier_A"), ("B", "cluster_rank", "cluster_tier")]:
    sub = bc.dropna(subset=[tcol, tier] + ALL_9_FEATURES).copy()
    idx = sub.index.to_numpy()
    oof = pd.Series(index=sub.index, dtype=float)
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(idx):
        specs = build_format1_specs(sub.loc[idx[tr]], tcol)
        oof.loc[idx[te]] = score_total(sub.loc[idx[te]], specs)[0].values
    y = (sub[tier] == "High").to_numpy().astype(int)
    sc = oof.to_numpy()
    print(f"\nTarget {key}: n={len(y)}, truly High={y.sum()} ({y.mean()*100:.1f}%)")
    for frac in (0.10, 0.20):
        pts = [capture(sc, y, frac, rng) for _ in range(200)]
        cap = np.mean([p[0] for p in pts])
        prec = np.mean([p[1] for p in pts])
        bc_cap, bc_prec = [], []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            if y[i].sum() == 0:
                continue
            c, p = capture(sc[i], y[i], frac, rng)
            bc_cap.append(c)
            bc_prec.append(p)
        lo, hi = np.percentile(bc_cap, [2.5, 97.5])
        plo, phi = np.percentile(bc_prec, [2.5, 97.5])
        print(f"  top {int(frac*100)}%: captures {cap*100:.1f}% of High [95% CI {lo*100:.1f}-{hi*100:.1f}] vs random {frac*100:.0f}% "
              f"(lift {cap/frac:.2f}); {prec*100:.1f}% of selected are High [{plo*100:.1f}-{phi*100:.1f}] vs base rate {y.mean()*100:.1f}%")
print("\n>>> STAGE 13 complete.")
