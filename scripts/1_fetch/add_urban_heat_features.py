"""Add urban-heat / microenvironment features to apartments_final_with_hsy.csv.

Three new building-level feature sets are computed for the year 2020:

  Feature 1 — impervious_fraction
      Sealed-surface area / 100 m buffer area, from HSY Land Cover 2020.
      Impervious classes: paved road (maanpeite_paallystetty_tie_2020),
      buildings (maanpeite_rakennus_2020), and other impervious surface
      (maanpeite_muu_vetta_lapaisematon_pinta_2020).

  Feature 2 — dist_major_road_m and road_density_100m
      From Digiroad WFS layer `digiroad:dr_tielinkki_toim_lk`. Major roads =
      functional class (toiminn_lk) in {1, 2, 3}. Density counts every class.

  Feature 3 — building_height_m and canyon_hw_ratio
      Height estimated from n_floors * 3.0 m (Helsinki 3D CityGML not
      downloaded; the n_floors fallback works uniformly for every
      municipality in the dataset, including Espoo/Vantaa/Kerava/…).
      Nearest-street width estimated from the road's functional class
      (class 1 → 15 m … class 5 → 5 m) since Digiroad's `dr_leveys` layer
      is only sparsely populated for state roads.

Output: apartments_final_with_urban_heat.csv (UTF-8 with BOM).
"""

from __future__ import annotations

import io
import os
import sys
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point
from shapely.strtree import STRtree

# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------

HERE = Path(os.environ.get("OVERHEATING_ROOT", Path(__file__).resolve().parents[2]))
CSV_IN = HERE / "intermediate" / "apartments_final_with_hsy.csv"
CSV_OUT = HERE / "intermediate" / "apartments_final_with_urban_heat.csv"

HSY_CACHE = HERE / "fetched" / "hsy_cache"
HSY_CACHE.mkdir(parents=True, exist_ok=True)
DR_CACHE = HERE / "fetched" / "digiroad_cache"
DR_CACHE.mkdir(parents=True, exist_ok=True)

BUFFER_M = 100
FLOOR_HEIGHT_M = 3.0

# HSY WFS
HSY_WFS = "https://kartta.hsy.fi/geoserver/asuminen_ja_maankaytto/wfs"
HSY_CRS = "EPSG:3879"
HSY_IMPERV_LAYERS = {
    "paallystetty_tie":         "asuminen_ja_maankaytto:maanpeite_paallystetty_tie_2020",
    "rakennus":                 "asuminen_ja_maankaytto:maanpeite_rakennus_2020",
    "muu_vetta_lapaisematon":   "asuminen_ja_maankaytto:maanpeite_muu_vetta_lapaisematon_pinta_2020",
}
HSY_TILE_M = 5000
HSY_MAX_RETRIES = 3

# Digiroad WFS
DR_WFS = "https://avoinapi.vaylapilvi.fi/vaylatiedot/digiroad/wfs"
DR_CRS = "EPSG:3067"
DR_LAYER = "digiroad:dr_tielinkki_toim_lk"
DR_PAGE_SIZE = 5000
DR_TILE_M = 4000  # small enough to comfortably page-through per tile

# Nearest-street width by functional class (metres). Values chosen per user spec.
WIDTH_BY_CLASS = {1: 15.0, 2: 12.0, 3: 10.0, 4: 7.0, 5: 5.0}
DEFAULT_WIDTH = 6.0  # fallback when functional class is missing/unknown

HEADERS = {"User-Agent": "urban-heat-pipeline/1.0"}


# =============================================================================
# HSY WFS — tiled shape-zip download (mirrors compute_greenery.py behaviour)
# =============================================================================

def _hsy_fetch_tile(typename: str, tile: tuple, attempt: int = 1) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = tile
    bbox_str = f"{miny},{minx},{maxy},{maxx},urn:ogc:def:crs:EPSG::3879"
    params = {
        "service": "WFS", "version": "1.1.0", "request": "GetFeature",
        "typeName": typename,
        "outputFormat": "shape-zip",
        "srsName": "EPSG:3879",
        "bbox": bbox_str,
    }
    try:
        r = requests.get(HSY_WFS, params=params, headers=HEADERS,
                         timeout=600, stream=True)
        r.raise_for_status()
    except (requests.exceptions.HTTPError, requests.exceptions.ReadTimeout) as e:
        if attempt < HSY_MAX_RETRIES:
            wait = 5 * attempt
            print(f"      retry {attempt+1}/{HSY_MAX_RETRIES} after {wait}s ({e.__class__.__name__})")
            time.sleep(wait)
            return _hsy_fetch_tile(typename, tile, attempt + 1)
        raise

    raw = r.content
    safe = typename.split(":")[-1]
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        tmpdir = HSY_CACHE / f"_tmp_{safe}_{int(minx)}_{int(miny)}"
        tmpdir.mkdir(exist_ok=True)
        zf.extractall(tmpdir)
        shp = next(tmpdir.glob("*.shp"), None)
        if shp is None:
            gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:3879")
        else:
            gdf = gpd.read_file(shp).set_crs("EPSG:3879", allow_override=True)
        for p in tmpdir.glob("*"):
            p.unlink()
        tmpdir.rmdir()
    return gdf


def hsy_fetch_layer(typename: str, buffers_3879: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fetch a WFS layer covering the extent of `buffers_3879`, cached on disk.

    Only tiles that intersect at least one buffer are actually downloaded.
    """
    safe = typename.split(":")[-1]
    cache = HSY_CACHE / f"{safe}.gpkg"
    if cache.exists():
        gdf = gpd.read_file(cache)
        print(f"  cached {safe}: {len(gdf):,} features")
        return gdf

    bb = buffers_3879.total_bounds
    minx = float(bb[0]) - 200
    miny = float(bb[1]) - 200
    maxx = float(bb[2]) + 200
    maxy = float(bb[3]) + 200
    xs = list(range(int(minx), int(maxx), HSY_TILE_M)) + [int(maxx)]
    ys = list(range(int(miny), int(maxy), HSY_TILE_M)) + [int(maxy)]

    cx = buffers_3879.geometry.centroid.x.values
    cy = buffers_3879.geometry.centroid.y.values
    active = set()
    for k in range(len(buffers_3879)):
        for px in (cx[k] - BUFFER_M, cx[k] + BUFFER_M):
            for py in (cy[k] - BUFFER_M, cy[k] + BUFFER_M):
                i = max(0, min(len(xs) - 2, int((px - xs[0]) // HSY_TILE_M)))
                j = max(0, min(len(ys) - 2, int((py - ys[0]) // HSY_TILE_M)))
                active.add((i, j))

    tiles = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            if (i, j) in active:
                tiles.append((xs[i], ys[j], xs[i + 1], ys[j + 1]))

    print(f"  fetching {safe} in {len(tiles)} tiles ...", flush=True)
    t0 = time.time()
    parts = []
    for k, tile in enumerate(tiles, 1):
        gdf = _hsy_fetch_tile(typename, tile)
        parts.append(gdf)
        print(f"    [{k}/{len(tiles)}] {len(gdf):,} feats", flush=True)

    if any(not p.empty for p in parts):
        merged = gpd.GeoDataFrame(
            pd.concat([p[["geometry"]] for p in parts if not p.empty],
                      ignore_index=True),
            crs="EPSG:3879",
        )
        from shapely import to_wkb
        keys = pd.Series(to_wkb(merged.geometry.values, hex=False))
        merged = merged.loc[~keys.duplicated().values].reset_index(drop=True)
        merged.to_file(cache, driver="GPKG")
    else:
        merged = gpd.GeoDataFrame(geometry=[], crs="EPSG:3879")

    print(f"    total {len(merged):,} features in {time.time()-t0:.1f}s -> {cache.name}")
    return merged


def buffer_intersection_area(
    layer_gdf: gpd.GeoDataFrame,
    buffers: gpd.GeoDataFrame,
) -> pd.Series:
    """For each buffer, return the total intersected area with `layer_gdf`."""
    if layer_gdf.empty:
        return pd.Series(0.0, index=buffers.index)
    joined = gpd.sjoin(
        layer_gdf[["geometry"]], buffers[["geometry"]],
        how="inner", predicate="intersects",
    )
    if joined.empty:
        return pd.Series(0.0, index=buffers.index)
    left_geom = joined.geometry.values
    right_geom = buffers.geometry.loc[joined["index_right"]].values
    from shapely import intersection, area as sh_area
    inter = intersection(left_geom, right_geom)
    areas = pd.Series(sh_area(inter), index=joined["index_right"].values)
    return areas.groupby(level=0).sum().reindex(buffers.index, fill_value=0.0)


# =============================================================================
# Digiroad WFS — tiled GeoJSON download with pagination
# =============================================================================

def _dr_fetch_tile(tile: tuple, attempt: int = 1) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = tile
    bbox_str = f"{minx},{miny},{maxx},{maxy},urn:ogc:def:crs:EPSG::3067"
    pages = []
    start = 0
    while True:
        params = {
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": DR_LAYER,
            "srsName": "EPSG:3067",
            "outputFormat": "application/json",
            "bbox": bbox_str,
            "count": DR_PAGE_SIZE,
            "startIndex": start,
        }
        try:
            r = requests.get(DR_WFS, params=params, headers=HEADERS, timeout=300)
            r.raise_for_status()
        except (requests.exceptions.HTTPError,
                requests.exceptions.ReadTimeout) as e:
            if attempt < HSY_MAX_RETRIES:
                wait = 5 * attempt
                print(f"      retry {attempt+1}/{HSY_MAX_RETRIES} after {wait}s ({e.__class__.__name__})")
                time.sleep(wait)
                return _dr_fetch_tile(tile, attempt + 1)
            raise

        try:
            gj = r.json()
        except ValueError:
            # Server returned non-JSON (probably an XML exception); treat as empty
            return gpd.GeoDataFrame(
                columns=["toiminn_lk", "geometry"], geometry="geometry",
                crs="EPSG:3067",
            )

        feats = gj.get("features", [])
        if not feats:
            break
        pages.append(gpd.GeoDataFrame.from_features(feats, crs="EPSG:3067"))
        n = len(feats)
        if n < DR_PAGE_SIZE:
            break
        start += DR_PAGE_SIZE

    if not pages:
        return gpd.GeoDataFrame(
            columns=["toiminn_lk", "geometry"], geometry="geometry",
            crs="EPSG:3067",
        )
    gdf = pd.concat(pages, ignore_index=True)
    keep_cols = [c for c in ("toiminn_lk", "link_id") if c in gdf.columns]
    return gpd.GeoDataFrame(gdf[keep_cols + ["geometry"]], crs="EPSG:3067")


def dr_fetch_links(buffers_3067: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fetch Digiroad road links across the bbox of `buffers_3067`."""
    cache = DR_CACHE / "dr_tielinkki_toim_lk.gpkg"
    if cache.exists():
        gdf = gpd.read_file(cache)
        print(f"  cached digiroad links: {len(gdf):,} features")
        return gdf

    bb = buffers_3067.total_bounds
    minx = float(bb[0]) - 300
    miny = float(bb[1]) - 300
    maxx = float(bb[2]) + 300
    maxy = float(bb[3]) + 300
    xs = list(range(int(minx), int(maxx), DR_TILE_M)) + [int(maxx)]
    ys = list(range(int(miny), int(maxy), DR_TILE_M)) + [int(maxy)]

    cx = buffers_3067.geometry.centroid.x.values
    cy = buffers_3067.geometry.centroid.y.values
    active = set()
    for k in range(len(buffers_3067)):
        for px in (cx[k] - BUFFER_M, cx[k] + BUFFER_M):
            for py in (cy[k] - BUFFER_M, cy[k] + BUFFER_M):
                i = max(0, min(len(xs) - 2, int((px - xs[0]) // DR_TILE_M)))
                j = max(0, min(len(ys) - 2, int((py - ys[0]) // DR_TILE_M)))
                active.add((i, j))

    # Also expand by 1 tile ring so nearest-major-road search finds links
    # slightly outside the pure buffer envelope (needed for dist_major_road_m).
    expanded = set(active)
    for (i, j) in list(active):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < len(xs) - 1 and 0 <= nj < len(ys) - 1:
                    expanded.add((ni, nj))

    tiles = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            if (i, j) in expanded:
                tiles.append((xs[i], ys[j], xs[i + 1], ys[j + 1]))

    print(f"  fetching digiroad in {len(tiles)} tiles ...", flush=True)
    t0 = time.time()
    parts = []
    for k, tile in enumerate(tiles, 1):
        gdf = _dr_fetch_tile(tile)
        parts.append(gdf)
        if k % 25 == 0 or k == len(tiles):
            print(f"    [{k}/{len(tiles)}] running total "
                  f"{sum(len(p) for p in parts):,} feats", flush=True)

    if any(not p.empty for p in parts):
        merged = gpd.GeoDataFrame(
            pd.concat([p for p in parts if not p.empty], ignore_index=True),
            crs="EPSG:3067",
        )
        # Deduplicate by link_id (bordering tiles return the same link twice).
        if "link_id" in merged.columns:
            merged = merged.drop_duplicates(subset=["link_id"]).reset_index(drop=True)
        merged.to_file(cache, driver="GPKG")
    else:
        merged = gpd.GeoDataFrame(
            columns=["toiminn_lk", "geometry"], geometry="geometry",
            crs="EPSG:3067",
        )

    print(f"    total {len(merged):,} links in {time.time()-t0:.1f}s -> {cache.name}")
    return merged


# =============================================================================
# Feature computation
# =============================================================================

def compute_impervious_fraction(
    buildings_3879: gpd.GeoDataFrame,
) -> pd.Series:
    """impervious_fraction per building, using 100 m buffers in EPSG:3879."""
    buffers = gpd.GeoDataFrame(
        {"building_id": buildings_3879["building_id"].values},
        geometry=buildings_3879.geometry.buffer(BUFFER_M),
        crs="EPSG:3879",
    ).reset_index(drop=True)
    buffer_area = float(buffers.geometry.area.iloc[0])
    print(f"  buffer area (m^2): {buffer_area:.1f}")

    total_area = pd.Series(0.0, index=buffers.index)
    for key, typename in HSY_IMPERV_LAYERS.items():
        layer = hsy_fetch_layer(typename, buffers)
        print(f"  computing {key} intersection ({len(layer):,} feats) ...", flush=True)
        t0 = time.time()
        area = buffer_intersection_area(layer, buffers)
        total_area = total_area.add(area, fill_value=0.0)
        print(f"    done in {time.time()-t0:.1f}s")

    frac = (total_area / buffer_area).clip(0, 1)
    frac.index = buffers["building_id"].values
    return frac


def compute_road_features(
    buildings_3067: gpd.GeoDataFrame,
    links_3067: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Return DataFrame indexed by building_id with dist_major_road_m,
    road_density_100m, nearest_link_class."""
    ids = buildings_3067["building_id"].to_numpy()
    pts = buildings_3067.geometry.values

    # -- Distance to nearest major road (functional class 1-3) --
    major_mask = links_3067["toiminn_lk"].isin([1, 2, 3]) if "toiminn_lk" in links_3067.columns else pd.Series(False, index=links_3067.index)
    major = links_3067.loc[major_mask].reset_index(drop=True)
    if major.empty:
        dist_major = np.full(len(ids), np.nan)
    else:
        tree = STRtree(major.geometry.values)
        dist_major = np.empty(len(ids), dtype=float)
        for i, pt in enumerate(pts):
            idx = tree.nearest(pt)
            dist_major[i] = pt.distance(major.geometry.iloc[int(idx)])

    # -- Nearest link overall (for canyon H/W width lookup) --
    all_tree = STRtree(links_3067.geometry.values)
    nearest_class = np.empty(len(ids), dtype=float)
    for i, pt in enumerate(pts):
        idx = all_tree.nearest(pt)
        cls = links_3067["toiminn_lk"].iloc[int(idx)] if "toiminn_lk" in links_3067.columns else np.nan
        nearest_class[i] = float(cls) if pd.notna(cls) else np.nan

    # -- Road density: total link length inside 100 m buffer --
    buffers = gpd.GeoDataFrame(
        {"building_id": ids},
        geometry=buildings_3067.geometry.buffer(BUFFER_M),
        crs="EPSG:3067",
    ).reset_index(drop=True)
    joined = gpd.sjoin(
        links_3067[["geometry"]], buffers[["geometry"]],
        how="inner", predicate="intersects",
    )
    if joined.empty:
        density = pd.Series(0.0, index=range(len(ids)))
    else:
        left_geom = joined.geometry.values
        right_geom = buffers.geometry.loc[joined["index_right"]].values
        from shapely import intersection, length as sh_length
        inter = intersection(left_geom, right_geom)
        lens = pd.Series(sh_length(inter), index=joined["index_right"].values)
        density = lens.groupby(level=0).sum().reindex(range(len(ids)), fill_value=0.0)

    out = pd.DataFrame({
        "building_id": ids,
        "dist_major_road_m": dist_major,
        "road_density_100m": density.values,
        "nearest_link_class": nearest_class,
    }).set_index("building_id")
    return out


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    print(f"Reading {CSV_IN.name} ...")
    df = pd.read_csv(CSV_IN)
    print(f"  {len(df):,} apartment rows, {df['building_id'].nunique():,} unique buildings")

    # ---- One row per building --------------------------------------------
    per_bldg = (
        df.groupby("building_id", as_index=False)
          .agg(
              building_lat=("building_lat", "first"),
              building_lon=("building_lon", "first"),
              n_floors=("n_floors", "first"),
              city=("city", "first"),
          )
    )
    per_bldg = per_bldg.dropna(subset=["building_lat", "building_lon"]).reset_index(drop=True)
    print(f"  buildings with coords: {len(per_bldg):,}")

    # ---- GeoDataFrames in the two working CRS ---------------------------
    b_wgs = gpd.GeoDataFrame(
        per_bldg.copy(),
        geometry=[Point(x, y) for x, y in
                  zip(per_bldg["building_lon"], per_bldg["building_lat"])],
        crs="EPSG:4326",
    )
    b_3879 = b_wgs.to_crs(HSY_CRS).reset_index(drop=True)
    b_3067 = b_wgs.to_crs(DR_CRS).reset_index(drop=True)

    # =====================================================================
    # Feature 1 — impervious_fraction (HSY 2020)
    # =====================================================================
    print("\n=== Feature 1: impervious_fraction (HSY Land Cover 2020) ===")
    imperv = compute_impervious_fraction(b_3879)
    per_bldg["impervious_fraction"] = per_bldg["building_id"].map(imperv)

    # =====================================================================
    # Feature 2 — dist_major_road_m + road_density_100m (Digiroad 2020)
    # =====================================================================
    print("\n=== Feature 2: Digiroad distance + density ===")
    links = dr_fetch_links(b_3067)
    if links.crs is None:
        links = links.set_crs(DR_CRS, allow_override=True)
    elif str(links.crs) != DR_CRS:
        links = links.to_crs(DR_CRS)
    road_df = compute_road_features(b_3067, links)
    per_bldg = per_bldg.merge(
        road_df.reset_index(),
        on="building_id", how="left",
    )

    # =====================================================================
    # Feature 3 — building_height_m + canyon_hw_ratio
    # =====================================================================
    print("\n=== Feature 3: building height + canyon H/W ratio ===")
    # HSY registry uses 999999999 as a "missing" sentinel; treat any implausible
    # floor count (>100) or non-positive value as unknown.
    floors = pd.to_numeric(per_bldg["n_floors"], errors="coerce")
    floors = floors.where((floors > 0) & (floors <= 100))
    per_bldg["building_height_m"] = floors * FLOOR_HEIGHT_M
    width = per_bldg["nearest_link_class"].map(WIDTH_BY_CLASS).fillna(DEFAULT_WIDTH)
    per_bldg["nearest_street_width_m"] = width.values
    per_bldg["canyon_hw_ratio"] = per_bldg["building_height_m"] / width

    # =====================================================================
    # Merge back to apartment-level and save
    # =====================================================================
    new_cols = [
        "impervious_fraction",
        "dist_major_road_m",
        "road_density_100m",
        "building_height_m",
        "canyon_hw_ratio",
    ]
    bldg_feat = per_bldg[["building_id"] + new_cols]

    # Drop pre-existing columns of the same name if any (defensive re-run)
    df = df.drop(columns=[c for c in new_cols if c in df.columns])
    out = df.merge(bldg_feat, on="building_id", how="left")
    out.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"\nWrote {CSV_OUT.name}: {len(out):,} rows, {out.shape[1]} cols")

    # ---- Summary ---------------------------------------------------------
    print("\nPer-building summary (one value per building):")
    print(per_bldg[new_cols].describe().round(3).to_string())

    print("\nCoverage (non-null buildings):")
    for c in new_cols:
        v = per_bldg[c]
        print(f"  {c:<22s}: valid {v.notna().sum():>4d} / {len(per_bldg)}  "
              f"missing {v.isna().sum():>3d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
