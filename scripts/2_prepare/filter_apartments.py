"""
Apartment filter (published cleaning rules, Farahani et al., 2023a).

Removes apartments with more than 15% missing hours (fewer than 85% of the
maximum number of hourly records), a minimum temperature below 18 C or a
maximum above 40 C. Reads final/apartment_metrics.csv (from
compute_overheating.py), writes the excluded list to logs/excluded_apartments.csv
and removes those apartments from final/hourly_indoor_temperature.parquet.
Run compute_overheating.py again afterwards. Re-running on filtered data
removes nothing.
"""
import os
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.environ.get("OVERHEATING_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
a = pd.read_csv(os.path.join(ROOT, "final", "apartment_metrics.csv"),
                usecols=["device", "building_id", "n_hours", "temp_min", "temp_max"])
N = a["n_hours"].max()
a["low_coverage"] = a["n_hours"] < 0.85 * N
a["below_18"] = a["temp_min"] < 18
a["above_40"] = a["temp_max"] > 40
bad = a[a[["low_coverage", "below_18", "above_40"]].any(axis=1)].copy()
print("full hours", N, "low coverage", int(a.low_coverage.sum()), "below 18", int(a.below_18.sum()),
      "above 40", int(a.above_40.sum()))
print("excluded", len(bad), "remaining", len(a) - len(bad))
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
if len(bad):
    bad.to_csv(os.path.join(ROOT, "logs", "excluded_apartments.csv"), index=False, encoding="utf-8-sig")
    p = os.path.join(ROOT, "final", "hourly_indoor_temperature.parquet")
    t = pq.read_table(p)
    mask = pc.invert(pc.is_in(t["device"], value_set=pa.array(bad["device"].astype(str).tolist())))
    t2 = t.filter(mask)
    print("rows", t.num_rows, "->", t2.num_rows, "devices", len(pc.unique(t2["device"])))
    del t
    pq.write_table(t2, p, compression="snappy")
