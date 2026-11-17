"""
Fetch HSY Metropolitan Area Building Register data for buildings in a CSV.

Uses the HSY WFS API to find building registry records by spatial proximity
(BBOX around each building's lat/lon coordinates).

Requirements:
    pip install requests pandas

Usage:
    python fetch_hsy_buildings.py buildings_combined_with_sea.csv

Output:
    buildings_with_hsy_registry.csv — original data enriched with HSY fields
"""

import sys
import time
import json
import math
import requests
import pandas as pd

# --- Configuration ---
WFS_URL = "https://kartta.hsy.fi/geoserver/wfs"
LAYER = "asuminen_ja_maankaytto:pks_rakennukset_paivittyva"
BBOX_RADIUS_DEG = 0.0005  # ~50m at Helsinki latitude
REQUEST_DELAY = 0.3        # seconds between requests (be polite to the API)
MAX_RETRIES = 3

# HSY coverage: Helsinki, Espoo, Vantaa, Kauniainen
HSY_CITIES = {"Helsinki", "Espoo", "Vantaa", "Kauniainen"}


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two WGS84 points."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def query_wfs_bbox(lat, lon, radius_deg=BBOX_RADIUS_DEG):
    """Query HSY WFS for buildings within a small BBOX around (lat, lon)."""
    minx = lon - radius_deg
    maxx = lon + radius_deg
    miny = lat - radius_deg
    maxy = lat + radius_deg

    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": LAYER,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "BBOX": f"{minx},{miny},{maxx},{maxy},EPSG:4326",
        "maxFeatures": "10",
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(WFS_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("features", [])
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [ERROR] Failed after {MAX_RETRIES} attempts: {e}")
                return []


def pick_nearest(features, lat, lon):
    """From a list of GeoJSON features, return the one nearest to (lat, lon)."""
    best = None
    best_dist = float("inf")
    for f in features:
        geom = f.get("geometry")
        if not geom:
            continue
        coords = geom.get("coordinates")
        if not coords:
            continue
        if geom["type"] == "Point":
            flon, flat = coords[0], coords[1]
        elif geom["type"] in ("Polygon", "MultiPolygon"):
            ring = coords[0] if geom["type"] == "Polygon" else coords[0][0]
            flon = sum(c[0] for c in ring) / len(ring)
            flat = sum(c[1] for c in ring) / len(ring)
        else:
            continue
        d = haversine_m(lat, lon, flat, flon)
        if d < best_dist:
            best_dist = d
            best = f
    return best, best_dist


def extract_properties(feature):
    """Extract a flat dict of building registry properties from a GeoJSON feature."""
    if not feature:
        return {}
    props = feature.get("properties", {})
    return {f"hsy_{k}": v for k, v in props.items()}


def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_hsy_buildings.py <input_csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "buildings_with_hsy_registry.csv"

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} buildings from {input_file}")

    all_hsy_keys = set()
    results = []

    for idx, row in df.iterrows():
        building_id = row.get("building_id", idx)
        address = row.get("building", "")
        city = row.get("city", "")
        lat = row.get("building_lat")
        lon = row.get("building_lon")

        if city not in HSY_CITIES:
            print(f"  [{building_id}] {address}, {city} — outside HSY area, skipping")
            results.append({})
            continue

        if pd.isna(lat) or pd.isna(lon):
            print(f"  [{building_id}] {address} — missing coordinates, skipping")
            results.append({})
            continue

        features = query_wfs_bbox(lat, lon)

        if not features:
            features = query_wfs_bbox(lat, lon, radius_deg=BBOX_RADIUS_DEG * 2)

        if not features:
            print(f"  [{building_id}] {address} — no HSY match found")
            results.append({})
        else:
            best, dist = pick_nearest(features, lat, lon)
            props = extract_properties(best)
            props["hsy_match_distance_m"] = round(dist, 1)
            all_hsy_keys.update(props.keys())
            addr_hsy = props.get("hsy_osoite", props.get("hsy_katuosoite", "?"))
            print(f"  [{building_id}] {address} → HSY: {addr_hsy} ({dist:.0f}m)")
            results.append(props)

        time.sleep(REQUEST_DELAY)

        if (idx + 1) % 50 == 0:
            print(f"--- Progress: {idx + 1}/{len(df)} ---")

    hsy_df = pd.DataFrame(results)
    merged = pd.concat([df.reset_index(drop=True), hsy_df.reset_index(drop=True)], axis=1)
    merged.to_csv(output_file, index=False)

    matched = hsy_df.apply(lambda r: any(pd.notna(v) and v != "" for v in r), axis=1).sum()
    print(f"\nDone! {matched}/{len(df)} buildings matched.")
    print(f"Output: {output_file}")
    print(f"HSY fields added: {sorted(all_hsy_keys)}")


if __name__ == "__main__":
    main()
