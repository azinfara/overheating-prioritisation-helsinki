#!/usr/bin/env bash
# Reruns the full analysis from the prepared data. Usage, from the repository root:
#   OVERHEATING_ROOT=/path/to/data bash run_all.sh
# OVERHEATING_ROOT must contain source/ and final/ (see README). Defaults to this folder.
set -u
PY="${PYTHON:-python}"
ROOT="${OVERHEATING_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
export OVERHEATING_ROOT="$ROOT" MPLBACKEND=Agg PYTHONIOENCODING=utf-8
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p logs plots exports
run() {
  name=$(basename "$1" .py)
  s=$(date +%s)
  "$PY" "$REPO/$1" > "logs/$name.log" 2>&1
  echo "$name exit=$? seconds=$(( $(date +%s) - s ))" | tee -a logs/_status.txt
}
: > logs/_status.txt
# 2. Prepare: indicators, apartment filter, indicators again on the filtered data
run scripts/2_prepare/compute_overheating.py
run scripts/2_prepare/filter_apartments.py
run scripts/2_prepare/compute_overheating.py
# 3. Analysis (Steps 1-4, benchmarks, validation)
run scripts/3_analysis/stage0_load_prepare.py
run scripts/3_analysis/stage1_apartment_spread.py
run scripts/3_analysis/stage3_apartment_stability.py
run scripts/3_analysis/stage4_age_structure.py
run scripts/3_analysis/stage5_composition_features.py
run scripts/3_analysis/stage6_variance_decomposition.py
run scripts/3_analysis/stage7_building_risk_targets.py
run scripts/3_analysis/stage8_feature_scoring.py
run scripts/3_analysis/stage9_build_rubrics.py
run scripts/3_analysis/stage10_validate_and_compare.py
run scripts/3_analysis/stage11_assemble_final.py
run scripts/3_analysis/stage12_presentation_summary.py
run scripts/3_analysis/stage13_reviewer_checks.py
run scripts/3_analysis/stage14_tier_options.py
run scripts/3_analysis/export_feature_tables.py
run scripts/3_analysis/fig_results.py
# 4. Paper figures
run scripts/4_figures/fig_workflow.py
run scripts/4_figures/fig_outdoor_conditions.py
echo ALL_DONE | tee -a logs/_status.txt
