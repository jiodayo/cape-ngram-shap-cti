#!/bin/bash
##############################################################################
# 複数seed一括実験スクリプト
#
# seed を変えて学習→全パイプラインを連続実行し、
# リーダーボードで自動比較する。
#
# 使い方:
#   # seed 42, 123, 456 で3回実験（デフォルト）
#   sbatch scripts/run_multi_seed.sbatch
#
#   # seed 指定
#   SEEDS="42 100 200 300 400" sbatch scripts/run_multi_seed.sbatch
#
#   # パイプラインのみ（学習済みの場合）
#   SKIP_TRAINING=1 sbatch scripts/run_multi_seed.sbatch
##############################################################################

#SBATCH --job-name=multi_seed
#SBATCH --output=joblog/multi_seed_%j.out
#SBATCH --error=joblog/multi_seed_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=30000M
#SBATCH --time=7-00:00:00

set -euo pipefail

cd /home/i055ueno/k_data/reserch
mkdir -p joblog

PYTHON=/home/i055ueno/k_data/reserch/.venv/bin/python

# ==========================================================================
# 設定
# ==========================================================================

SEEDS=${SEEDS:-"42 123 456"}
SKIP_TRAINING=${SKIP_TRAINING:-0}
DATA_ROOT=${DATA_ROOT:-data/avast_ctu_cape/ngram_dataset_family_104}
MIN_CONFIDENCE=${MIN_CONFIDENCE:-0.3}
LEADERBOARD_DB=${LEADERBOARD_DB:-leaderboard.sqlite}

echo "========================================================================"
echo " Multi-Seed Experiment"
echo "========================================================================"
echo "[INFO] Seeds: ${SEEDS}"
echo "[INFO] Skip training: ${SKIP_TRAINING}"
echo "[INFO] Data root: ${DATA_ROOT}"
echo ""

COMPLETED=0
FAILED=0

for SEED in ${SEEDS}; do
    echo ""
    echo "################################################################"
    echo " SEED = ${SEED}"
    echo "################################################################"

    FEAT="logs/cape_ngram_desc_full_lgbm_seed${SEED}"
    EXP_NAME="lgbm_seed${SEED}"

    # --- 学習 ---
    if [[ "${SKIP_TRAINING}" != "1" ]]; then
        echo "[INFO] Training with seed=${SEED}..."

        RANDOM_STATE="${SEED}" \
        OUT_DIR="${FEAT}" \
        DATA_ROOT="${DATA_ROOT}" \
        "${PYTHON}" src/2026_learning_ngram_desc_br.py \
            --train-dir "${DATA_ROOT}/train" \
            --test-dir "${DATA_ROOT}/test" \
            --label-set "${DATA_ROOT}/label_set.json" \
            --api-list "${DATA_ROOT}/api.json" \
            --api-descriptions "${DATA_ROOT}/api_descriptions.json" \
            --model lgbm \
            --use-skipgram \
            --random-state "${SEED}" \
            --output-root "${FEAT}" \
        || {
            echo "[ERROR] Training failed for seed=${SEED}"
            FAILED=$((FAILED + 1))
            continue
        }
        echo "[DONE] Training completed for seed=${SEED}"
    fi

    # --- パイプライン ---
    echo "[INFO] Running pipeline for seed=${SEED}..."

    FEAT="${FEAT}" \
    EXP_NAME="${EXP_NAME}" \
    RANDOM_STATE="${SEED}" \
    MIN_CONFIDENCE="${MIN_CONFIDENCE}" \
    REGISTER_LEADERBOARD=1 \
    LEADERBOARD_DB="${LEADERBOARD_DB}" \
    bash scripts/run_full_pipeline.sbatch \
    || {
        echo "[ERROR] Pipeline failed for seed=${SEED}"
        FAILED=$((FAILED + 1))
        continue
    }

    COMPLETED=$((COMPLETED + 1))
    echo "[DONE] Seed ${SEED} completed."
done

# ==========================================================================
# 最終リーダーボード
# ==========================================================================

echo ""
echo "========================================================================"
echo " MULTI-SEED RESULTS"
echo "========================================================================"
echo "[INFO] Completed: ${COMPLETED} / Failed: ${FAILED}"
echo ""

"${PYTHON}" src/leaderboard.py --db "${LEADERBOARD_DB}" show

# HTML更新
"${PYTHON}" src/leaderboard.py \
    --db "${LEADERBOARD_DB}" \
    html \
    --output "reports/leaderboard.html" \
    --title "Multi-Seed Experiment Leaderboard"

echo ""
echo "[INFO] Leaderboard: reports/leaderboard.html"
echo "[INFO] All done at $(date '+%Y-%m-%d %H:%M:%S')"
