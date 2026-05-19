#!/bin/bash
#SBATCH --job-name=shap_per_sample_nometa
#SBATCH --output=joblog/shap_per_sample_nometa_%j.out
#SBATCH --error=joblog/shap_per_sample_nometa_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=30000M
#SBATCH --time=3-00:00:00

set -euo pipefail

cd /home/i055ueno/k_data/reserch
mkdir -p joblog

PYTHON=/home/i055ueno/k_data/reserch/.venv/bin/python

DATA_ROOT=${DATA_ROOT:-data/avast_ctu_cape/ngram_dataset_family_104}
FEATURE_ROOT=${FEATURE_ROOT:-logs/cape_ngram_desc_full_lgbm_seed42_106}
MODEL_DIR=${MODEL_DIR:-logs/cape_ngram_desc_full_lgbm_seed42_106/lgbm/models}
OUTPUT_DIR=${OUTPUT_DIR:-shap_analysis/ngram_desc_lgbm_group_nometa_106/per_sample_plots}

SPLIT=${SPLIT:-test}
LABELS=${LABELS:-}
MAX_SAMPLES=${MAX_SAMPLES:-0}
BATCH_SIZE=${BATCH_SIZE:-64}
OUTPUT_TYPE=${OUTPUT_TYPE:-both}
MAX_DISPLAY=${MAX_DISPLAY:-20}
MAX_FEATURE_NAME_LEN=${MAX_FEATURE_NAME_LEN:-32}
FEATURE_NAME_TAIL_LEN=${FEATURE_NAME_TAIL_LEN:-8}
SEQ_DEDUPE=${SEQ_DEDUPE:-1}
API_COUNT_LOG1P=${API_COUNT_LOG1P:-1}

CMD=(
  "${PYTHON}" /home/i055ueno/k_data/reserch/src/export_shap_ngram_desc_lgbm_per_sample_plots.py
  --feature-root "${FEATURE_ROOT}"
  --model-dir "${MODEL_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --data-root "${DATA_ROOT}"
  --split "${SPLIT}"
  --batch-size "${BATCH_SIZE}"
  --output-type "${OUTPUT_TYPE}"
  --max-display "${MAX_DISPLAY}"
  --max-feature-name-len "${MAX_FEATURE_NAME_LEN}"
  --feature-name-tail-len "${FEATURE_NAME_TAIL_LEN}"
)

if [[ -n "${LABELS}" ]]; then
  CMD+=(--labels "${LABELS}")
fi
if [[ "${SEQ_DEDUPE}" == "1" ]]; then
  CMD+=(--seq-dedupe-consecutive)
fi
if [[ "${API_COUNT_LOG1P}" == "1" ]]; then
  CMD+=(--api-count-log1p)
fi
if [[ "${MAX_SAMPLES}" -gt 0 ]]; then
  CMD+=(--max-samples "${MAX_SAMPLES}")
fi

printf '[INFO] Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

printf '[INFO] Completed: %s\n' "${OUTPUT_DIR}"
