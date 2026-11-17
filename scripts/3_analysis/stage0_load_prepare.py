"""
STAGE 0 -- Load & prepare
Within-building overheating spread in the Helsinki residential stock.

Loads the 4 source tables, merges apartment area/type, builds design_group,
computes swing_range from hourly data, and runs sanity checks against the
locked benchmark numbers from prior validated analysis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from common import (
    DATA, SOURCE, DG_ORDER, MIN_APTS,
    load_buildings_metadata, load_apartment_metrics, load_hourly_indoor,
    compute_swing_range, build_apt_table, composition,
)

print(f"""
{'='*72}
STAGE 0 -- DEFINITIONS
{'='*72}
overheating apartment : dh_27 > 150 (Finnish criteria limit; ABSOLUTE threshold)
design_group          : non-linear age bins tied to Finnish code revisions
                         pre1977 | 1977-2003 | 2004-2009 | 2010-2011 | post2012
swing_range           : per apartment, mean over summer of
                         (daytime[08-20] max - nighttime[22-06] min)
area/type             : merged from source/'building big data 1(in).csv' by device
                         (99.6% match; NOT buildinginfo(in).csv [69%] or
                         device_2021_daily_enriched [43%])
{'='*72}""")

# ============================================================
# LOAD
# ============================================================
buildings_meta = load_buildings_metadata()
apt_metrics = load_apartment_metrics(merge_area_type=True)
print(f"Loaded: apartment_metrics {apt_metrics.shape}, buildings_metadata {buildings_meta.shape}")

area_matched = apt_metrics["area"].notna().sum()
print(f"\nArea/type matched: {area_matched}/{len(apt_metrics)} ({100*area_matched/len(apt_metrics):.1f}%)  "
      f"[expect ~99.6%]")

print("\n=== apartment_metrics columns ===")
print(list(apt_metrics.columns))
print("\n=== buildings_metadata columns ===")
print(list(buildings_meta.columns))

# ============================================================
# swing_range from hourly (15.8M rows -- vectorized)
# ============================================================
print("\nLoading hourly indoor temperature (15.8M rows) to compute swing_range ...")
hourly = load_hourly_indoor()
swing_df = compute_swing_range(hourly)
print(f"swing_range computed for {len(swing_df)} devices; "
      f"median={swing_df['swing_range'].median():.2f} C, "
      f"mean={swing_df['swing_range'].mean():.2f} C")
del hourly  # free ~15.8M-row frame before continuing

# ============================================================
# Apartment working table
# ============================================================
apt = build_apt_table(apt_metrics, buildings_meta, swing_df)

# ============================================================
# SANITY CHECKS
# ============================================================
print("\n" + "="*50)
print("SANITY CHECKS")
print("="*50)
print(f"Apartments: {len(apt)}, buildings: {apt['building_id'].nunique()}")
print(f"Apartments with design_group: {apt['design_group'].notna().sum()}")

bld = buildings_meta.dropna(subset=["design_group"])
print(f"\nDesign-group counts (BUILDINGS, n={len(bld)}) -- expect 166/306/26/20/61:")
print(bld["design_group"].value_counts().reindex(DG_ORDER))

print(f"\nOverall apartment criteria exceedance: {(apt['dh_27']>150).mean()*100:.0f}%  [expect ~71%]")

comp = composition(apt, "exceeds_code", "overheating", "resilient")
print(f"\nBuilding overheating composition (>={MIN_APTS} apts, n={len(comp)}) -- expect ~4.7/19.6/17.8 tails,"
      f" collapsing to ~5/78/18 all-resilient/mixed/all-overheating:")
print((comp["class"].value_counts(normalize=True)*100).round(1))
coarse = np.select([comp["frac"]==0, comp["frac"]==1], ["all_resilient","all_overheating"], default="mixed")
print("\nCoarse 3-bucket collapse (expect ~5/78/18):")
print((pd.Series(coarse).value_counts(normalize=True)*100).round(1))

apts_per_bld = apt.groupby("building_id").size()
print(f"\nApartments per building: median={apts_per_bld.median():.0f}, mean={apts_per_bld.mean():.1f}, "
      f"min={apts_per_bld.min()}, max={apts_per_bld.max()}")

print("\n>>> STAGE 0 complete.")
