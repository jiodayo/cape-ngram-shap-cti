#!/bin/bash
#SBATCH --job-name=cape_ngram_desc_full
#SBATCH --output=joblog/cape_ngram_desc_full_%j.out
#SBATCH --error=joblog/cape_ngram_desc_full_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=30000M
#SBATCH --time=3-00:00:00

set -euo pipefail

cd /home/i055ueno/k_data/reserch
mkdir -p joblog logs

PYTHON=/home/i055ueno/k_data/reserch/.venv/bin/python

DATA_ROOT=${DATA_ROOT:-data/avast_ctu_cape/ngram_dataset_family_102}
TRAIN_DIR=${TRAIN_DIR:-${DATA_ROOT}/train}
TEST_DIR=${TEST_DIR:-${DATA_ROOT}/test}
LABEL_SET=${LABEL_SET:-${DATA_ROOT}/label_set.json}
API_LIST=${API_LIST:-${DATA_ROOT}/api.json}
API_DESCRIPTIONS=${API_DESCRIPTIONS:-${DATA_ROOT}/api_descriptions.json}

MAX_TRAIN=${MAX_TRAIN:-0}
MAX_TEST=${MAX_TEST:-0}
MAX_SEQ_FEATURES=${MAX_SEQ_FEATURES:-200000}
MAX_SKIP_FEATURES=${MAX_SKIP_FEATURES:-100000}
DESC_MAX_FEATURES=${DESC_MAX_FEATURES:-4096}
DESC_MODE=${DESC_MODE:-tfidf}
SKIP_MAX_GAP=${SKIP_MAX_GAP:-2}
SKIP_WINDOW=${SKIP_WINDOW:-4}
SEQ_DEDUPE=${SEQ_DEDUPE:-1}
API_COUNT_LOG1P=${API_COUNT_LOG1P:-1}
HASH_SEQ=${HASH_SEQ:-0}
HASH_SKIP=${HASH_SKIP:-0}
HASH_N_FEATURES=${HASH_N_FEATURES:-1048576}
HASH_BINARY=${HASH_BINARY:-0}
HASH_ALT_SIGN=${HASH_ALT_SIGN:-0}
LGBM_N_ESTIMATORS=${LGBM_N_ESTIMATORS:-400}
LGBM_NUM_LEAVES=${LGBM_NUM_LEAVES:-63}
LGBM_LEARNING_RATE=${LGBM_LEARNING_RATE:-0.05}
N_JOBS=${N_JOBS:--1}
RANDOM_STATE=${RANDOM_STATE:-42}

OUT_DIR=${OUT_DIR:-logs/cape_ngram_desc_full_lgbm_seed${RANDOM_STATE}_${SLURM_JOB_ID}}

printf '[INFO] Job ID: %s\n' "${SLURM_JOB_ID}"
printf '[INFO] Output dir: %s\n' "${OUT_DIR}"
printf '[INFO] Train: %s\n' "${TRAIN_DIR}"
printf '[INFO] Test:  %s\n' "${TEST_DIR}"

CMD=(
  "${PYTHON}" src/2026_learning_ngram_desc_br.py
  --train-dir "${TRAIN_DIR}"
  --test-dir "${TEST_DIR}"
  --label-set "${LABEL_SET}"
  --api-list "${API_LIST}"
  --api-descriptions "${API_DESCRIPTIONS}"
  --model lgbm
  --use-skipgram
  --skip-max-gap "${SKIP_MAX_GAP}"
  --skip-window "${SKIP_WINDOW}"
  --max-seq-features "${MAX_SEQ_FEATURES}"
  --max-skip-features "${MAX_SKIP_FEATURES}"
  --desc-max-features "${DESC_MAX_FEATURES}"
  --desc-mode "${DESC_MODE}"
  --lgbm-n-estimators "${LGBM_N_ESTIMATORS}"
  --lgbm-num-leaves "${LGBM_NUM_LEAVES}"
  --lgbm-learning-rate "${LGBM_LEARNING_RATE}"
  --random-state "${RANDOM_STATE}"
  --n-jobs "${N_JOBS}"
  --output-root "${OUT_DIR}"
)

if [[ "${MAX_TRAIN}" -gt 0 ]]; then
  CMD+=(--max-train-files "${MAX_TRAIN}")
fi
if [[ "${MAX_TEST}" -gt 0 ]]; then
  CMD+=(--max-test-files "${MAX_TEST}")
fi
if [[ "${SEQ_DEDUPE}" == "1" ]]; then
  CMD+=(--seq-dedupe-consecutive)
fi
if [[ "${API_COUNT_LOG1P}" == "1" ]]; then
  CMD+=(--api-count-log1p)
fi
if [[ "${HASH_SEQ}" == "1" ]]; then
  CMD+=(--seq-hash)
fi
if [[ "${HASH_SKIP}" == "1" ]]; then
  CMD+=(--skip-hash)
fi
if [[ "${HASH_SEQ}" == "1" || "${HASH_SKIP}" == "1" ]]; then
  CMD+=(--hash-n-features "${HASH_N_FEATURES}")
  if [[ "${HASH_BINARY}" == "1" ]]; then
    CMD+=(--hash-binary)
  fi
  if [[ "${HASH_ALT_SIGN}" == "1" ]]; then
    CMD+=(--hash-alternate-sign)
  fi
fi

printf '[INFO] Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

printf '[INFO] Completed: %s\n' "${OUT_DIR}"
