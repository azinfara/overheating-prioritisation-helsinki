# Prioritising apartment buildings for overheating renovation

Analysis code for:

> Farahani, A.V., Kosonen, R. Prioritising apartment buildings for overheating retrofits: A point-score method using registry and microclimate data, benchmarked with machine learning in Helsinki. *(manuscript in preparation)*

The code analyses measured indoor temperatures in 5,978 apartments in 568 apartment buildings in the Helsinki region during summer 2021 (15 May–31 August). It:

1. quantifies how much summer indoor temperatures differ between apartments of the same building,
2. tests how much of the building-level overheating can be predicted from public building-register and microclimate data,
3. defines two building-level risk targets (share of overheated apartments; heatwave severity), and
4. builds a point-score card for ranking buildings for renovation assessment, and benchmarks it against a decision tree, linear regression, random forest and gradient boosting.

## Data

**The indoor temperature data cannot be shared and are not included in this repository.** The dataset is described in Farahani et al. (2023a). The public data sources are:

| Data | Source |
|---|---|
| Building register (construction year, floors, floor area, material, …) | HSY Helsinki region building register, WFS `kartta.hsy.fi/geoserver/wfs` |
| Land cover 2020 (impervious surfaces, trees) | HSY land cover data |
| Building information grid 2021 (`ALUETEHOK`) | HSY `Rakennustietoruudukko_2021` |
| Road network | Väylävirasto Digiroad, WFS `avoinapi.vaylapilvi.fi` |
| Coastline | National Land Survey of Finland, Topographic Database |
| Outdoor weather | Finnish Meteorological Institute, Helsinki-Vantaa airport station |

The scripts expect a data folder (set with the environment variable `OVERHEATING_ROOT`; by default the repository folder itself) with this layout:

```
source/    raw tables (indoor data, apartment area and type, weather)
fetched/   downloaded public data (created by scripts/1_fetch)
intermediate/
final/     hourly_indoor_temperature.parquet, hourly_weather.csv, buildings_metadata.csv
```

Outputs are written to `logs/`, `plots/` and `exports/` in the same folder. All data folders are excluded by `.gitignore`.

## Pipeline

| Folder | Script | What it does |
|---|---|---|
| `scripts/1_fetch` | `fetch_hsy_buildings.py` | Matches each building to the HSY building register by location |
| | `add_urban_heat_features.py` | Impervious fraction (100 m buffer), road density and distance to major roads |
| `scripts/2_prepare` | `compute_overheating.py` | Hot days (max > 25 °C), heatwaves (> 4 consecutive hot days), apartment and building indicators: degree-hours above 27 and 30 °C, mean and maximum temperatures |
| | `filter_apartments.py` | Published cleaning filter: removes apartments with > 15% missing hours, minimum < 18 °C or maximum > 40 °C |
| `scripts/3_analysis` | `common.py` | Shared paths, periods, feature definitions and helper functions |
| | `stage0_load_prepare.py` | Loads data, design periods, sanity checks |
| | `stage1_apartment_spread.py`, `stage3_apartment_stability.py`, `stage13_reviewer_checks.py` | Step 1: within-building variation (ranges, three-apartment range, standard deviation, coolest apartment, heatwave vs non-heatwave range) |
| | `stage4_age_structure.py` | Overheating by construction period |
| | `stage5_composition_features.py`, `stage6_variance_decomposition.py` | Step 2: ICC from a mixed-effects model; cross-validated R² and its partition into construction, urban and shared parts; random-forest check |
| | `stage7_building_risk_targets.py` | Step 3: Target A and Target B (k-means, silhouette) |
| | `stage8_feature_scoring.py`, `stage9_build_rubrics.py` | Step 4: point card and decision tree |
| | `stage10_validate_and_compare.py`, `stage12_presentation_summary.py`, `stage13_reviewer_checks.py`, `stage14_tier_options.py` | Benchmark models, five-fold cross-validation, Spearman r, AUC, precision, capture, lift, bootstrap confidence intervals, tree depth sweep, Priority/Standard groups |
| | `stage11_assemble_final.py` | Summary tables |
| | `export_feature_tables.py` | Point-card tables to Excel |
| | `fig_results.py` | Results figures (Figs. 3–5) and the risk-group table (Table 4); run after stages 6, 13 and 14 |
| `scripts/4_figures` | `fig_workflow.py`, `fig_outdoor_conditions.py` | Fig. 1 and Fig. 2 |

Stage numbers follow the order in which the analyses were developed; stage 2 (night-time build-up and recovery) is not part of this paper and is not included.

## Running

```bash
pip install -r requirements.txt
OVERHEATING_ROOT=/path/to/data bash run_all.sh   # set PYTHON=... to choose the interpreter
```

`run_all.sh` runs steps 2–4. The fetch scripts in `scripts/1_fetch` call public web services and are run separately:

```bash
python scripts/1_fetch/fetch_hsy_buildings.py buildings_combined_with_sea.csv fetched/buildings_with_hsy_registry.csv
python scripts/1_fetch/add_urban_heat_features.py
```

All random processes use seed 42. Tested with Python 3.12 (package versions in `requirements.txt`).

## Licence

MIT (see `LICENSE`). The indoor temperature data are not covered by this licence and are not distributed.
