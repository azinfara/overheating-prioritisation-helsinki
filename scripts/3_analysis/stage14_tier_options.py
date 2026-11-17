"""
Stage 14 -- do different risk groupings improve separation? (added 9 November 2026)
Out-of-fold point-card scores only.
 a. AUC for High vs everyone else, and for High vs Low with Medium removed.
 b. Two-group operating rule: priority = top 30% of score.
 c. A rarer High: worst 25% of the continuous outcome.
"""
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from common import (
    MIN_APTS, ALL_9_FEATURES, load_buildings_metadata, load_apartment_metrics, load_hourly_indoor,
    compute_swing_range, build_apt_table, build_building_comp, attach_building_features,
    attach_risk_features, build_severity_cluster, build_composition_cluster,
    build_format1_specs, score_total,
)

rng = np.random.default_rng(42)
meta = load_buildings_metadata()
am = load_apartment_metrics(merge_area_type=True)
h = load_hourly_indoor()[["device", "timestamp", "temperature"]]
gc.collect()
sw = compute_swing_range(h)
del h
gc.collect()
apt = build_apt_table(am, meta, sw)
comp, apt, _, _ = build_building_comp(apt)
bc = attach_risk_features(attach_building_features(comp, meta), meta)
cb, _ = build_severity_cluster(meta, min_apts=MIN_APTS)
bc["cluster_rank"] = cb["cluster_rank"].reindex(bc.index)
bc["cluster_tier"] = cb["cluster_tier"].reindex(bc.index)
ca, _ = build_composition_cluster(bc, min_apts=MIN_APTS)
bc["cluster_tier_A"] = ca["cluster_tier"].reindex(bc.index)
bc["bld_dh_27_mean"] = meta.set_index("building_id")["bld_dh_27_mean"].reindex(bc.index)


def boot_auc(y, s, n=2000):
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            out.append(roc_auc_score(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


def ranked(score):
    return np.lexsort((rng.random(len(score)), -score))


OOF_ROWS = []
for key, tcol, tier, cont in [("A", "frac_over", "cluster_tier_A", "frac_over"), ("B", "cluster_rank", "cluster_tier", "bld_dh_27_mean")]:
    sub = bc.dropna(subset=[tcol, tier, cont] + ALL_9_FEATURES).copy()
    idx = sub.index.to_numpy()
    oof = pd.Series(index=sub.index, dtype=float)
    for tr, te in KFold(5, shuffle=True, random_state=42).split(idx):
        specs = build_format1_specs(sub.loc[idx[tr]], tcol)
        oof.loc[idx[te]] = score_total(sub.loc[idx[te]], specs)[0].values
    s = oof.to_numpy()
    t = sub[tier].astype(str).to_numpy()
    y = (t == "High").astype(int)
    print(f"\n===== Target {key}: n={len(y)}  Low/Medium/High = {(t=='Low').sum()}/{(t=='Medium').sum()}/{y.sum()}")
    lo, hi = boot_auc(y, s)
    print(f"a. AUC High vs everyone else       = {roc_auc_score(y, s):.3f} [{lo:.3f}-{hi:.3f}]")
    m = t != "Medium"
    lo, hi = boot_auc(y[m], s[m])
    print(f"   AUC High vs Low (Medium dropped) = {roc_auc_score(y[m], s[m]):.3f} [{lo:.3f}-{hi:.3f}] n={m.sum()}")
    o = ranked(s)
    k = int(round(.30 * len(y)))
    pri, rest = o[:k], o[k:]
    print(f"b. Priority top 30%: {y[pri].mean()*100:.1f}% truly High; remaining 70%: {y[rest].mean()*100:.1f}% truly High; "
          f"priority captures {y[pri].sum()/y.sum()*100:.1f}% of High")
    OOF_ROWS.append(pd.DataFrame({"building_id": sub.index, "target": key, "score": s, "tier": t,
                                  "priority": np.isin(np.arange(len(y)), pri)}))
    v = sub[cont].to_numpy()
    y25 = (v >= np.quantile(v, .75)).astype(int)
    lo, hi = boot_auc(y25, s)
    print(f"c. Worst 25% of {cont}: n={y25.sum()} ({y25.mean()*100:.1f}%)  AUC={roc_auc_score(y25, s):.3f} [{lo:.3f}-{hi:.3f}]")
    for f in (.10, .20):
        k = int(round(f * len(y)))
        vals = [(y25[ranked(s)[:k]].sum() / y25.sum(), y25[ranked(s)[:k]].mean()) for _ in range(200)]
        cap, pre = np.mean(vals, axis=0)
        print(f"   top {int(f*100)}%: {pre*100:.1f}% are in the worst group (base {y25.mean()*100:.1f}%), capture {cap*100:.1f}% "
              f"(max possible {min(1, f/y25.mean())*100:.0f}%, lift {cap/f:.2f})")
import os
from common import DATA
pd.concat(OOF_ROWS).to_csv(os.path.join(os.path.dirname(DATA), "exports", "oof_point_scores.csv"), index=False)
print("\n>>> STAGE 14 complete.")
