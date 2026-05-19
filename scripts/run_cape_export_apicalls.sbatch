#!/bin/bash
#SBATCH --job-name=cape_export
#SBATCH --output=joblog/cape_export_%j.out
#SBATCH --error=joblog/cape_export_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=30000M
#SBATCH --time=2-00:00:00

set -euo pipefail

cd /home/i055ueno/k_data/reserch
mkdir -p joblog

PYTHON=/home/i055ueno/k_data/reserch/.venv/bin/python

REPORTS_ZIP=${REPORTS_ZIP:-data/avast_ctu_cape/public_full_reports.zip}
INDEX_JSONL=${INDEX_JSONL:-data/avast_ctu_cape/cape_index.jsonl}
LABEL_KEY=${LABEL_KEY:-family}
TRAIN_RATIO=${TRAIN_RATIO:-0.8}
YEAR_SPLIT=${YEAR_SPLIT:-}
MAX_SAMPLES=${MAX_SAMPLES:-0}
MIN_LABEL_COUNT=${MIN_LABEL_COUNT:-0}
API_MAX_LEN=${API_MAX_LEN:-2000}
DROP_EMPTY=${DROP_EMPTY:-0}
API_LIST_SOURCE=${API_LIST_SOURCE:-train}
INCLUDE_META=${INCLUDE_META:-0}
OUT_DIR=${OUT_DIR:-data/avast_ctu_cape/ngram_dataset_${LABEL_KEY}_${SLURM_JOB_ID}}

CMD=(
  "${PYTHON}" src/cape_export_apicalls.py
  --reports-zip "${REPORTS_ZIP}"
  --index-jsonl "${INDEX_JSONL}"
  --label-key "${LABEL_KEY}"
  --train-ratio "${TRAIN_RATIO}"
  --api-max-len "${API_MAX_LEN}"
  --api-list-source "${API_LIST_SOURCE}"
  --output-root "${OUT_DIR}"
)

if [[ "${MAX_SAMPLES}" -gt 0 ]]; then
  CMD+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${YEAR_SPLIT}" ]]; then
  CMD+=(--year-split "${YEAR_SPLIT}")
fi
if [[ "${MIN_LABEL_COUNT}" -gt 0 ]]; then
  CMD+=(--min-label-count "${MIN_LABEL_COUNT}")
fi
if [[ "${DROP_EMPTY}" == "1" ]]; then
  CMD+=(--drop-empty)
fi
if [[ "${INCLUDE_META}" == "1" ]]; then
  CMD+=(--include-meta)
fi

printf '[INFO] Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

printf '[INFO] Completed. Output root: %s\n' "${OUT_DIR}"
