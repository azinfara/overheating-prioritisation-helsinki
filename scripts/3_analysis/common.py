"""
Shared loader/helpers for the unified Helsinki residential overheating pipeline
(one paper: apartment-level overheating spread -> predictability -> building-
level risk classification -> point-card vs. decision-tree comparison). Every
loader here reads from the raw source files each time it's called -- no cached
intermediate CSVs are produced or consumed. Each stage script imports what it
needs from this module and is otherwise self-contained.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
BASE = os.environ.get("OVERHEATING_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
DATA   = rf"{BASE}/final"
SOURCE = rf"{BASE}/source"
PLOTS  = rf"{BASE}/plots"

MIN_APTS = 3

PERIODS = {
    "short_hw": ("2021-06-18", "2021-06-23"),
    "gap":      ("2021-06-24", "2021-07-02"),
    "long_hw":  ("2021-07-03", "2021-07-18"),
    "recovery": ("2021-07-19", "2021-07-24"),
}

DG_ORDER = ["pre1977", "y1977_2003", "y2004_2009", "y2010_2011", "post2012"]
DG_LABELS = {"pre1977": "<1977", "y1977_2003": "1977-2003", "y2004_2009": "2004-2009",
             "y2010_2011": "2010-2011", "post2012": ">2012"}

CONSTRUCTION_FEATURES = ["design_group", "n_floors", "gross_floor_area_m2",
                          "n_apartments_hsy", "building_material"]
URBAN_FEATURES = ["impervious_fraction", "tree_fraction", "canyon_hw_ratio",
                   "distance_to_sea_m", "road_density_100m", "dist_major_road_m", "ALUETEHOK"]

# Feature set of the paper (Table 2), split into construction and urban groups for
# the Step 2 predictability analysis (stage 6) and the Table 4 checks (stage 13).
# Requires attach_building_features + attach_risk_features (greenery_score, n_apartments).
TABLE2_CONSTRUCTION_NUM = ["n_floors", "n_apartments"]          # + design_group, building_material dummies
TABLE2_URBAN_NUM = ["greenery_score", "ALUETEHOK", "road_density_100m",
                    "dist_major_road_m", "distance_to_sea_m"]

PAL5 = ["#2c7bb6", "#abd9e9", "#bdbdbd", "#fdae61", "#d7191c"]

# Fixed top-to-bottom order for every material plot; anything else (Steel, Other,
# Unknown) is too thin-sample and is excluded from material plots.
MATERIAL_ORDER = ["Concrete or lightweight concrete", "Brick", "Wood"]


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans",
    })


def design_group(y):
    if pd.isna(y):
        return np.nan
    if y < 1977:
        return "pre1977"
    elif y <= 2003:
        return "y1977_2003"
    elif y <= 2009:
        return "y2004_2009"
    elif y <= 2011:
        return "y2010_2011"
    else:
        return "post2012"


def load_buildings_metadata():
    bm = pd.read_csv(rf"{DATA}/buildings_metadata.csv")
    bm["design_group"] = pd.Categorical(
        bm["year_of_construction"].apply(design_group), categories=DG_ORDER, ordered=True)
    return bm


def load_apartment_metrics(merge_area_type=True):
    apt = pd.read_csv(rf"{DATA}/apartment_metrics.csv")
    if merge_area_type:
        big = pd.read_csv(rf"{SOURCE}/building big data 1(in).csv")
        bfeat = big.groupby("device").agg(area=("area", "first"), type=("type", "first")).reset_index()
        apt = apt.merge(bfeat, on="device", how="left")
    return apt


def load_hourly_indoor():
    h = pd.read_parquet(rf"{DATA}/hourly_indoor_temperature.parquet")
    h["timestamp"] = pd.to_datetime(h["timestamp"])
    return h


def load_hourly_weather():
    w = pd.read_csv(rf"{DATA}/hourly_weather.csv")
    w["timestamp"] = pd.to_datetime(w["timestamp"])
    return w


def assign_night_date(h):
    """Add date/hour/night columns. Night [22:00-06:00) is labeled by the date it
    STARTS on (22:00-23:59 -> that date; 00:00-05:59 -> the previous date)."""
    h = h.copy()
    h["date"] = h["timestamp"].dt.normalize()
    h["hour"] = h["timestamp"].dt.hour
    is_late = h["hour"] >= 22
    is_early = h["hour"] < 6
    night = pd.Series(pd.NaT, index=h.index, dtype="datetime64[ns]")
    night.loc[is_late] = h.loc[is_late, "date"]
    night.loc[is_early] = h.loc[is_early, "date"] - pd.Timedelta(days=1)
    h["night"] = night
    return h


def compute_swing_range(hourly):
    """Per device: mean over summer of (daytime[08-20] max - nighttime[22-06] min).
    Vectorized (not .apply) because the source table is ~15.8M rows."""
    h = assign_night_date(hourly[["device", "timestamp", "temperature"]])

    day = h[(h["hour"] >= 8) & (h["hour"] <= 20)]
    day_max = day.groupby(["device", "date"])["temperature"].max().rename("day_max").reset_index()

    night_min = (h.dropna(subset=["night"]).groupby(["device", "night"])["temperature"]
                 .min().rename("night_min").reset_index().rename(columns={"night": "date"}))

    sw = day_max.merge(night_min, on=["device", "date"], how="inner")
    sw["swing"] = sw["day_max"] - sw["night_min"]
    return sw.groupby("device")["swing"].mean().rename("swing_range").reset_index()


def compute_nightly_min(hourly, id_col="device"):
    """Per id_col (device or building_id), per night [labeled by start date]: min temperature."""
    h = assign_night_date(hourly[[id_col, "timestamp", "temperature"]])
    nmin = (h.dropna(subset=["night"]).groupby([id_col, "night"])["temperature"]
            .min().rename("night_min").reset_index().rename(columns={"night": "date"}))
    return nmin


def compute_period_mean(hourly, start, end, id_col="device"):
    """Per id_col: mean temperature within the inclusive calendar-date range
    [start, end] (both 'YYYY-MM-DD' strings). Used to build baselines/windows
    that are NOT just 'whole summer' or a single named PERIODS entry -- e.g. a
    summer-excluding-heatwaves baseline, or a specific pre-heatwave ramp window."""
    mask = (hourly["timestamp"] >= pd.Timestamp(start)) & (hourly["timestamp"] < pd.Timestamp(end) + pd.Timedelta(days=1))
    return hourly.loc[mask].groupby(id_col)["temperature"].mean().rename("period_mean")


def outside_heatwave_mask(timestamps, periods=PERIODS):
    """Boolean mask, True for timestamps OUTSIDE both short_hw and long_hw (i.e. the
    'gap' lull, the 'recovery' tail, and any shoulder days before/after the named
    PERIODS window all count as baseline -- only the two heatwave blocks themselves
    are excluded)."""
    in_short = (timestamps >= pd.Timestamp(periods["short_hw"][0])) & (timestamps < pd.Timestamp(periods["short_hw"][1]) + pd.Timedelta(days=1))
    in_long = (timestamps >= pd.Timestamp(periods["long_hw"][0])) & (timestamps < pd.Timestamp(periods["long_hw"][1]) + pd.Timedelta(days=1))
    return ~(in_short | in_long)


def build_apt_table(apt_metrics=None, buildings_meta=None, swing_df=None):
    """Apartment-level working table: metrics + building attrs + exceeds_code flag (+ swing_range if given)."""
    if apt_metrics is None:
        apt_metrics = load_apartment_metrics()
    if buildings_meta is None:
        buildings_meta = load_buildings_metadata()
    apt = apt_metrics.merge(
        buildings_meta[["building_id", "design_group", "n_floors", "building_material", "year_of_construction"]],
        on="building_id", how="left")
    apt["exceeds_code"] = apt["dh_27"] > 150
    apt["code_status"] = np.where(apt["exceeds_code"], "overheating", "resilient")
    if swing_df is not None:
        apt = apt.merge(swing_df, on="device", how="left")
    return apt


def material_table(bld_values, buildings_meta):
    """Building-level Series (index=building_id) -> median/mean/count per material,
    restricted to MATERIAL_ORDER and in that order. count = number of buildings."""
    mat_map = buildings_meta.set_index("building_id")["building_material"]
    d = bld_values.dropna().to_frame("value").join(mat_map, how="inner")
    d = d[d["building_material"].isin(MATERIAL_ORDER)]
    return d.groupby("building_material")["value"].agg(["median", "mean", "count"]).reindex(MATERIAL_ORDER)


def plot_material_bars(table, fmt, xlabel, title, color, path):
    """Horizontal bars of table['median'] in MATERIAL_ORDER (first material on top),
    each labelled with its value and number of buildings."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y = np.arange(len(table))
    ax.barh(y, table["median"], color=color, edgecolor="white")
    ax.set_yticks(y, table.index)
    ax.invert_yaxis()
    pad = 0.01 * np.nanmax(table["median"].abs())
    for i, (v, n) in enumerate(zip(table["median"], table["count"])):
        if pd.notna(v):
            ax.text(v + pad, i, f"{fmt.format(v)} (n={int(n)} buildings)", va="center", fontsize=9.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def composition(df, flag_col, pos, neg, min_apts=MIN_APTS):
    """Building-level composition class from the fraction of apartments flagged positive."""
    g = df.groupby("building_id")[flag_col].agg(["sum", "count"])
    g = g[g["count"] >= min_apts]
    g["frac"] = g["sum"] / g["count"]
    g["class"] = np.select(
        [g["frac"] == 0, g["frac"] == 1, g["frac"] < 0.4, g["frac"] > 0.6],
        [f"all_{neg}", f"all_{pos}", f"mostly_{neg}", f"mostly_{pos}"], default="mixed")
    return g[["frac", "class"]]


def add_behavior_type(apt, lag_col="peak_lag_hot", swing_col="swing_range"):
    """2x2 typology: (fast/slow lag) x (volatile/stable swing), split at population medians.
    Returns (apt with is_fast/is_volatile/behavior_type columns added, lag_med, swing_med)."""
    apt = apt.copy()
    lag_med = apt[lag_col].median()
    swing_med = apt[swing_col].median()
    apt["is_fast"] = apt[lag_col] < lag_med
    apt["is_volatile"] = apt[swing_col] >= swing_med
    slow = ~apt["is_fast"]
    stable = ~apt["is_volatile"]
    apt["behavior_type"] = np.select(
        [slow & stable, slow & ~stable, ~slow & stable],
        ["buffered_stable", "buffered_volatile", "fast_stable"],
        default="fast_volatile")
    return apt, lag_med, swing_med


def compute_sensitivity(hourly, weather, min_days=10):
    """Per device: OLS slope of daily mean indoor temp regressed on daily mean
    outdoor temp ('envelope sensitivity'). Used only for the Stage 3 independence
    test -- this metric fails independence vs dh_27 and is excluded elsewhere."""
    from scipy import stats as _stats

    h = hourly[["device", "timestamp", "temperature"]].copy()
    h["date"] = h["timestamp"].dt.normalize()
    daily_indoor = h.groupby(["device", "date"])["temperature"].mean().rename("indoor_mean").reset_index()

    w = weather.copy()
    w["date"] = w["timestamp"].dt.normalize()
    daily_outdoor = w.groupby("date")["outdoor_temp"].mean().rename("outdoor_mean").reset_index()

    md = daily_indoor.merge(daily_outdoor, on="date", how="inner")
    out = {}
    for dev, g in md.groupby("device"):
        if len(g) < min_days:
            continue
        s, *_ = _stats.linregress(g["outdoor_mean"], g["indoor_mean"])
        out[dev] = s
    return pd.Series(out, name="sensitivity").rename_axis("device").reset_index()


def attach_building_features(building_comp, buildings_meta):
    """Join CONSTRUCTION_FEATURES + URBAN_FEATURES + building_material onto a
    building-indexed table (e.g. building_comp), excluding lat/lon (guardrail #7)."""
    cols = list(dict.fromkeys(CONSTRUCTION_FEATURES + URBAN_FEATURES + ["building_material"]))
    feat = buildings_meta.set_index("building_id")[cols]
    return building_comp.join(feat, how="left")


def build_building_comp(apt, lag_col="peak_lag_hot", swing_col="swing_range", min_apts=MIN_APTS):
    """Master building-level table: 3 composition classes (overheating/lag/swing),
    dominant behavior type, and within-building spread of 5 metrics.
    Returns (building_comp, apt_with_behavior_cols, lag_med, swing_med)."""
    apt, lag_med, swing_med = add_behavior_type(apt, lag_col, swing_col)

    comp_over = composition(apt, "exceeds_code", "overheating", "resilient", min_apts)
    comp_fast = composition(apt, "is_fast", "fast", "slow", min_apts)
    comp_vol = composition(apt, "is_volatile", "volatile", "stable", min_apts)

    spread = apt.groupby("building_id").agg(
        spread_dh=("dh_27", lambda x: x.max() - x.min()),
        spread_temp=("temp_mean", lambda x: x.max() - x.min()),
        spread_peak=("temp_max", lambda x: x.max() - x.min()),
        spread_lag=(lag_col, lambda x: x.max() - x.min()),
        spread_swing=(swing_col, lambda x: x.max() - x.min()),
        spread_area=("area", lambda x: x.max() - x.min()),
        n_apts=("dh_27", "count"),
    ).query("n_apts >= @min_apts")

    dominant = (apt.groupby("building_id")["behavior_type"]
                .agg(lambda x: x.value_counts().idxmax()).rename("dominant_behavior"))

    building_comp = (
        comp_over.rename(columns={"frac": "frac_over", "class": "class_over"})
        .join(comp_fast.rename(columns={"frac": "frac_fast", "class": "class_fast"}))
        .join(comp_vol.rename(columns={"frac": "frac_vol", "class": "class_vol"}))
        .join(spread)
        .join(dominant)
    )
    return building_comp, apt, lag_med, swing_med


# ============================================================
# FEATURE SET -- locked to 9 roles (see plan), tested against BOTH targets
# ============================================================
FEATURE_TIERS = {
    "EASY": ["design_group", "n_floors", "building_material", "distance_to_sea_m",
             "dist_major_road_m", "n_apartments", "road_density_100m"],
    "MEDIUM": ["greenery_score", "ALUETEHOK"],
}
ALL_9_FEATURES = FEATURE_TIERS["EASY"] + FEATURE_TIERS["MEDIUM"]

# the 7 numeric-tertile roles for Format 1/3 point cards (design_group and
# building_material are scored categorically, not by plain numeric tertile)
NUMERIC_7 = ["n_floors", "distance_to_sea_m", "dist_major_road_m", "greenery_score",
             "ALUETEHOK", "road_density_100m", "n_apartments"]

# tree-discovery feature groups (Design detail #1 / Format 2 & 3)
# design_group is NOT re-included at branch level -- it is consumed entirely by
# the hardcoded root split (Stage 4 hump guardrail), so re-splitting on it again
# inside a branch would be redundant.
TREE_BRANCH_FEATS = ["n_floors", "building_material", "distance_to_sea_m",
                      "dist_major_road_m", "n_apartments"]
TREE_FALLBACK_FEATS = ["greenery_score", "ALUETEHOK", "road_density_100m"]


# ============================================================
# FEATURE CONSTRUCTION
# ============================================================
def greenery_score(buildings_meta):
    """Mean of z-scored tree_fraction + veg_fraction + tall_tree_fraction, indexed
    by building_id. tree_fraction = trees layer (vegetation >2m height, 100m
    buffer); veg_fraction = trees + low vegetation combined (100m buffer);
    tall_tree_fraction = tall trees layer (>10m, 100m buffer). A composite of 3
    correlated-but-not-identical vegetation indices (r=0.84-0.91, not 1.0) --
    see plan limitations."""
    cols = ["tree_fraction", "veg_fraction", "tall_tree_fraction"]
    z = buildings_meta.set_index("building_id")[cols].apply(lambda c: (c - c.mean()) / c.std())
    return z.mean(axis=1).rename("greenery_score")


def attach_risk_features(bc, buildings_meta):
    """Adds greenery_score + n_apartments (RAW registry count, not the
    n_apartments_hsy convention used elsewhere -- see plan for why) onto
    a building-level table already run through attach_building_features
    (which supplies design_group, n_floors, building_material, distance_to_sea_m,
    dist_major_road_m, ALUETEHOK, road_density_100m)."""
    bc = bc.copy()
    bc["greenery_score"] = greenery_score(buildings_meta).reindex(bc.index)
    bc["n_apartments"] = buildings_meta.set_index("building_id")["n_apartments"].reindex(bc.index)
    return bc


def coarse_class(frac_over):
    """3-bucket composition class from frac_over: all_resilient (frac==0) /
    mixed / all_overheating (frac==1)."""
    return np.select([frac_over == 0, frac_over == 1],
                      ["all_resilient", "all_overheating"], default="mixed")


# ============================================================
# TARGET B -- severity cluster (KMeans k=3), the Target-B ground-truth builder
# ============================================================
def build_severity_cluster(buildings_meta, min_apts=MIN_APTS):
    """Target B: KMeans(n_clusters=3, random_state=42, n_init=10) on standardized
    [bld_hw_temp_mean, bld_hw_temp_max, log1p(bld_dh_27_mean), log1p(bld_dh_30_mean)], filtered to has_temp_data & bld_n_apartments >=
    min_apts (n=516). dh_27/dh_30 are log1p-transformed before standardizing --
    both are heavily zero-inflated (19%/75% exactly zero) and, fed in raw,
    dominate the Euclidean distance and collapse High down to 2 buildings;
    log1p fixes that (silhouette 0.434 vs 0.422 for the 3-variable version)
    while still folding in the 30 C threshold, not
    just 27 C. Cluster ranks ordered 0=Low/1=Medium/2=High by mean
    bld_hw_temp_mean per cluster.
    Returns (cluster_df indexed by building_id with cluster_raw/cluster_rank/
    cluster_tier columns, silhouette Series for k=2..6). NEVER used as a deployed
    model -- only to define the ground-truth label for the 570 monitored buildings."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    cols = ["bld_hw_temp_mean", "bld_hw_temp_max", "bld_dh_27_mean", "bld_dh_30_mean"]
    log_cols = ["bld_dh_27_mean", "bld_dh_30_mean"]
    d = buildings_meta[(buildings_meta["has_temp_data"] == True) &
                        (buildings_meta["bld_n_apartments"] >= min_apts)].copy()
    X = d[cols].dropna()
    d = d.loc[X.index].set_index("building_id")
    Xt = X.copy()
    for c in log_cols:
        Xt[c] = np.log1p(Xt[c])
    Xs = StandardScaler().fit_transform(Xt)

    sil = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        sil[k] = silhouette_score(Xs, km.labels_)
    sil = pd.Series(sil, name="silhouette")

    km3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs)
    d["cluster_raw"] = km3.labels_
    order = d.groupby("cluster_raw")["bld_hw_temp_mean"].mean().sort_values().index.tolist()
    rank_map = {c: i for i, c in enumerate(order)}
    d["cluster_rank"] = d["cluster_raw"].map(rank_map)
    d["cluster_tier"] = d["cluster_rank"].map({0: "Low", 1: "Medium", 2: "High"})
    return d, sil


def build_composition_cluster(bc, min_apts=MIN_APTS):
    """Target A's Low/Medium/High: KMeans(n_clusters=3, random_state=42,
    n_init=10) on standardized frac_over ALONE (1-dimensional -- mirrors
    build_severity_cluster's method, applied to Target A's own metric instead
    of plain equal-sized tertiles). Tertiles force exactly 1/3 of buildings
    into each bucket regardless of where the data's real breaks fall -- e.g.
    the tertile 'Low' bucket included buildings with up to 62.5% of apartments
    overheating. A natural-break cluster instead finds a much stricter, more
    honest Low group (mean ~13% overheating) at the cost of very unequal
    sizes (~10%/36%/54% instead of 33/33/33%).
    Returns (cluster_df indexed by building_id with cluster_raw/cluster_rank/
    cluster_tier columns, silhouette Series for k=2..6)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    x = bc[["frac_over"]].dropna()
    Xs = StandardScaler().fit_transform(x)

    sil = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        sil[k] = silhouette_score(Xs, km.labels_)
    sil = pd.Series(sil, name="silhouette")

    km3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs)
    d = x.copy()
    d["cluster_raw"] = km3.labels_
    order = d.groupby("cluster_raw")["frac_over"].mean().sort_values().index.tolist()
    rank_map = {c: i for i, c in enumerate(order)}
    d["cluster_rank"] = d["cluster_raw"].map(rank_map)
    d["cluster_tier"] = d["cluster_rank"].map({0: "Low", 1: "Medium", 2: "High"})
    return d, sil


def build_composition_cluster_2var(bc, extra_col="frac_over_30", min_apts=MIN_APTS):
    """Test variant of Target A's clustering (point 10 in the plan): KMeans(k=3)
    on standardized [frac_over, extra_col] instead of frac_over alone -- does
    adding a dh_30-based composition signal (frac_over_30 = fraction of a
    building's apartments with dh_30>0, a MUCH stricter bar than dh_27>150)
    sharpen the Low/Medium/High separation? `bc` must already have `extra_col`
    attached (join it on before calling). Same KMeans/StandardScaler/silhouette
    convention as build_composition_cluster, so silhouette scores are directly
    comparable between the two. NOT used unless it wins on silhouette --
    see stage7's printed comparison for the actual decision."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    x = bc[["frac_over", extra_col]].dropna()
    Xs = StandardScaler().fit_transform(x)

    sil = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        sil[k] = silhouette_score(Xs, km.labels_)
    sil = pd.Series(sil, name="silhouette")

    km3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs)
    d = x.copy()
    d["cluster_raw"] = km3.labels_
    order = d.groupby("cluster_raw")["frac_over"].mean().sort_values().index.tolist()
    rank_map = {c: i for i, c in enumerate(order)}
    d["cluster_rank"] = d["cluster_raw"].map(rank_map)
    d["cluster_tier"] = d["cluster_rank"].map({0: "Low", 1: "Medium", 2: "High"})
    return d, sil


# ============================================================
# POINT-CARD SCORING PRIMITIVES -- fit on a TRAIN fold, applied to any fold
# (never in-sample cutpoints; the 5-fold CV loop in stage9/stage10 calls these
# once per fold)
# ============================================================
def fit_numeric_tertile(train, feat, target, edges=None):
    """Numeric feature -> tertile points (0/1/2) from the SUB-GROUP MEAN of the
    target's outcome (not just raw quantile order -- direction-agnostic by
    construction). Returns a spec dict fit on `train`, apply with apply_score().
    Optional `edges` override skips the plain train-quantile split in favor of
    externally supplied cut values -- everything downstream (bin-mean ranking ->
    points) works identically either way; unused by the current point-card."""
    x = train[feat]
    if edges is None:
        e1, e2 = np.quantile(x.dropna(), [1 / 3, 2 / 3])
    else:
        e1, e2 = edges
    binned = pd.cut(x, bins=[-np.inf, e1, e2, np.inf], labels=[0, 1, 2])
    bin_mean = train.assign(_bin=binned).groupby("_bin", observed=True)[target].mean()
    order = bin_mean.sort_values().index.tolist()
    points = {b: i for i, b in enumerate(order)}
    return {"kind": "numeric", "feat": feat, "edges": (float(e1), float(e2)),
            "points": points, "bin_means": bin_mean}


def score_tertile(train, feat, target, edges=None):
    """Alias matching the plan's naming -- see fit_numeric_tertile."""
    return fit_numeric_tertile(train, feat, target, edges=edges)


def fit_categorical_score(train, feat, target, fold_into_other=None):
    """Categorical feature (design_group / building_material) -> points (0/1/2) by
    ranking category means of the target ascending and splitting the ORDERED
    category list into 3 contiguous point-buckets (np.array_split -- handles
    design_group's 5 categories as a '5-row lookup table' collapsing to 3 point
    values, and building_material's categories with steel folded into 'Other')."""
    d = train.copy()
    if fold_into_other:
        d[feat] = d[feat].where(~d[feat].isin(fold_into_other), "Other")
    means = d.groupby(feat, observed=True)[target].mean().sort_values()
    order = means.index.tolist()
    groups = np.array_split(order, 3)
    points = {}
    for pts, grp in enumerate(groups):
        for cat in grp:
            points[cat] = pts
    return {"kind": "categorical", "feat": feat, "points": points, "means": means,
            "fold_into_other": list(fold_into_other) if fold_into_other else None}


def score_design_group(train, target):
    """design_group -> fixed lookup table (5 rows collapsing to 0/1/2 points)."""
    return fit_categorical_score(train, "design_group", target)


def score_material(train, target, fold_into_other=("Steel",)):
    """building_material -> lookup table, steel folded into 'Other' (n too small)."""
    return fit_categorical_score(train, "building_material", target, fold_into_other=list(fold_into_other))


def apply_score(spec, series):
    """Applies a fitted numeric or categorical scoring spec to any series (fit on
    train, applied out-of-fold to validation/test)."""
    if spec["kind"] == "numeric":
        e1, e2 = spec["edges"]
        binned = pd.cut(series, bins=[-np.inf, e1, e2, np.inf], labels=[0, 1, 2])
        return binned.map(spec["points"]).astype(float)
    s = series.astype(object)
    if spec["fold_into_other"]:
        s = s.where(~s.isin(spec["fold_into_other"]), "Other")
    return s.map(spec["points"]).astype(float)


def assign_tier(score, train_score=None):
    """Total score -> Low/Medium/High via tertiles of the score distribution (fit
    on train_score, applied to `score` -- out-of-fold when the two differ)."""
    if train_score is None:
        train_score = score
    e1, e2 = np.quantile(train_score.dropna(), [1 / 3, 2 / 3])
    tier = pd.cut(score, bins=[-np.inf, e1, e2, np.inf], labels=["Low", "Medium", "High"])
    return tier, (float(e1), float(e2))


# ============================================================
# DECISION-TREE HELPER (Design detail #1 -- calibration only, never shipped as
# a loaded/pickled model; the tree is one ML technique compared to the
# point-card in Stage 10, not a competing deployable format)
# ============================================================
def _make_branch_dummies(df, feats):
    """One-hot encodes building_material within a branch feature subset; numeric
    features pass through as float columns."""
    parts = []
    for f in feats:
        if f == "building_material":
            parts.append(pd.get_dummies(df[f].fillna("Unknown"), prefix="mat"))
        else:
            parts.append(df[[f]].astype(float))
    return pd.concat(parts, axis=1)


def fit_tree_per_target(df, target_col, min_samples_leaf=20, random_state=42):
    """Internal tree-discovery step (calibration only, NEVER shipped as a loaded/
    pickled model). Design-group guardrail: never feed continuous
    year_of_construction, and never let the tree freely pick its root -- hardcode
    the top-level split as design_group == 'y2004_2009' (peak-era vs rest, Stage
    4's hump finding). Within each branch, fit a depth-1 DecisionTreeRegressor on
    the remaining 5 EASY-tier features (TREE_BRANCH_FEATS) -> an effective
    depth-2 tree with a guardrailed root. Any leaf whose [min,max] target range
    straddles an overall tertile boundary gets a depth-1 FALLBACK split fit on
    the MEDIUM-tier features (TREE_FALLBACK_FEATS), restricted to that leaf's
    rows only. Also cross-checks that an UNCONSTRAINED depth-2 tree fit on
    one-hot design_group dummies ALONE independently favors dg_y2004_2009 as its
    top split (confirms the guardrail isn't fighting the data)."""
    from sklearn.tree import DecisionTreeRegressor, export_text

    needed = [target_col, "design_group"] + TREE_BRANCH_FEATS + TREE_FALLBACK_FEATS
    d = df.dropna(subset=needed).copy()
    tier_edges = tuple(np.quantile(d[target_col].astype(float), [1 / 3, 2 / 3]))

    is_peak = d["design_group"].astype(str) == "y2004_2009"
    branches = {}
    for bname, mask in [("y2004_2009", is_peak), ("rest", ~is_peak)]:
        sub = d[mask]
        Xb = _make_branch_dummies(sub, TREE_BRANCH_FEATS)
        yb = sub[target_col].astype(float)
        tree = DecisionTreeRegressor(max_depth=1, min_samples_leaf=min_samples_leaf, random_state=random_state)
        tree.fit(Xb, yb)
        leaf = tree.apply(Xb)
        leaf_stats = (pd.DataFrame({"_leaf": leaf, "_y": yb.values})
                      .groupby("_leaf")["_y"].agg(["min", "max", "mean", "count"]))

        fallback = {}
        for lid, row in leaf_stats.iterrows():
            straddles = any(row["min"] < e < row["max"] for e in tier_edges)
            leaf_rows = sub[leaf == lid]
            if straddles and len(leaf_rows) >= 2 * min_samples_leaf:
                Xf = leaf_rows[TREE_FALLBACK_FEATS].astype(float)
                yf = leaf_rows[target_col].astype(float)
                ftree = DecisionTreeRegressor(max_depth=1, min_samples_leaf=min_samples_leaf, random_state=random_state)
                ftree.fit(Xf, yf)
                fallback[lid] = {"tree": ftree, "feats": TREE_FALLBACK_FEATS,
                                  "text": export_text(ftree, feature_names=TREE_FALLBACK_FEATS)}

        pred = tree.predict(Xb)
        for lid, fb in fallback.items():
            sel = leaf == lid
            Xf = sub.loc[sel, TREE_FALLBACK_FEATS].astype(float)
            pred[sel] = fb["tree"].predict(Xf)

        branches[bname] = {"tree": tree, "columns": list(Xb.columns), "n": len(sub),
                            "text": export_text(tree, feature_names=list(Xb.columns)),
                            "leaf_stats": leaf_stats, "fallback": fallback,
                            "fitted_pred": pd.Series(pred, index=sub.index)}

    # tier labels are read off tertiles of the tree's OWN fitted-value distribution
    # (mirrors Format 1's "score -> Low/Medium/High via tertiles of the score
    # distribution") -- NOT tertiles of the raw target. For a skewed/discrete
    # target like Target B's cluster_rank (0/1/2, ~55% at 1), raw-target tertile
    # edges can collapse such that no leaf MEAN ever clears the upper edge,
    # leaving the High tier empty; tertiles of the fitted predictions can't do
    # that by construction (they're always ~1/3 of buildings each).
    all_fitted = pd.concat([b["fitted_pred"] for b in branches.values()])
    predicted_tier_edges = tuple(np.quantile(all_fitted, [1 / 3, 2 / 3]))

    dg_dum = pd.get_dummies(d["design_group"].astype(str), prefix="dg")
    check_tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=min_samples_leaf, random_state=random_state)
    check_tree.fit(dg_dum, d[target_col].astype(float))
    top_feat = dg_dum.columns[check_tree.tree_.feature[0]]

    return {"branches": branches, "tier_edges": tier_edges, "predicted_tier_edges": predicted_tier_edges,
            "n": len(d), "guardrail_confirmed": bool(top_feat == "dg_y2004_2009"), "check_top_feature": top_feat}


def _column_value(row, col):
    """Evaluates a single tree design-matrix column (numeric passthrough or a
    'mat_<Category>' one-hot dummy) for one building row, without needing a full
    one-hot DataFrame -- used to score one building at a time (Stage 3 CV /
    Stage 4 worked example) against a tree fit on a train fold."""
    if col.startswith("mat_"):
        cat = col[len("mat_"):]
        val = row["building_material"] if pd.notna(row["building_material"]) else "Unknown"
        return 1.0 if val == cat else 0.0
    return float(row[col])


def _apply_tree_single(tree, columns, row):
    """Walks a fitted sklearn tree for a single row (Series), returns the leaf node id."""
    t = tree.tree_
    node = 0
    while t.children_left[node] != -1:
        feat = columns[t.feature[node]]
        thr = t.threshold[node]
        node = t.children_left[node] if _column_value(row, feat) <= thr else t.children_right[node]
    return node


def predict_tree_tier(tree_result, row):
    """Format 2/3 scoring function: given a fitted tree_result (from
    fit_tree_per_target) and one building's feature row, walks the guardrailed
    tree (hardcoded design_group root -> branch depth-1 split -> optional
    fallback split) and returns the assigned Low/Medium/High tier, read off
    tertiles of the tree's fitted-value distribution (predicted_tier_edges --
    NOT tertiles of the raw target; see fit_tree_per_target)."""
    branch_name = "y2004_2009" if str(row["design_group"]) == "y2004_2009" else "rest"
    branch = tree_result["branches"][branch_name]
    leaf_id = _apply_tree_single(branch["tree"], branch["columns"], row)
    if leaf_id in branch["fallback"]:
        fb = branch["fallback"][leaf_id]
        sub_leaf = _apply_tree_single(fb["tree"], fb["feats"], row)
        mean_val = fb["tree"].tree_.value[sub_leaf][0][0]
    else:
        mean_val = branch["leaf_stats"].loc[leaf_id, "mean"]
    e1, e2 = tree_result["predicted_tier_edges"]
    tier = "Low" if mean_val <= e1 else ("Medium" if mean_val <= e2 else "High")
    return tier, float(mean_val)


# ============================================================
# ARM BUILDERS -- shared by Stage 9 (build/print), Stage 10 (per-fold CV refit),
# Stage 11 (final render + worked example) so none of them duplicate this logic
# ============================================================
def build_format1_specs(train, target_col):
    """Format 1 (point-score card): plain train-quantile tertile edges for the 7
    numeric roles + categorical lookup tables for design_group/building_material.
    Fit on `train` -- pass a CV train fold for out-of-fold scoring, or the full
    bc table for the published Stage 9/11 card."""
    specs = {feat: fit_numeric_tertile(train, feat, target_col) for feat in NUMERIC_7}
    specs["design_group"] = score_design_group(train, target_col)
    specs["building_material"] = score_material(train, target_col)
    return specs


def score_total(df, specs):
    """Applies a fitted point-card specs dict to any df (fit fold or
    scoring fold) -> (total_score Series, per-feature points DataFrame). A
    building missing ANY of the 9 features gets total_score=NaN (excluded from
    tier assignment), matching the dropna-based exclusion used everywhere else
    in this pipeline."""
    pts = pd.DataFrame(index=df.index)
    for feat, spec in specs.items():
        pts[feat] = apply_score(spec, df[feat])
    return pts.sum(axis=1, skipna=False), pts


# ============================================================
# TABLE RENDERING (used by stage11_assemble_final.py)
# ============================================================
def render_table(df, title, fname, plots_dir=None):
    plots_dir = plots_dir or PLOTS
    fig, ax = plt.subplots(figsize=(min(14, 1.6 + 1.7 * df.shape[1]), 0.6 + 0.45 * (len(df) + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c7bb6")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == -1:
            cell.set_text_props(fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=12, pad=14)
    plt.tight_layout()
    plt.savefig(rf"{plots_dir}/{fname}", dpi=300, bbox_inches="tight")
    plt.close(fig)
