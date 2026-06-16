#!/bin/bash
#SBATCH --job-name=ffri_keyword_bow
#SBATCH --output=joblog/ffri_keyword_bow_%j.out
#SBATCH --error=joblog/ffri_keyword_bow_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=30000M
#SBATCH --time=1-00:00:00

set -euo pipefail

cd /home/i055ueno/k_data/reserch

# ログディレクトリの作成
mkdir -p joblog logs_keyword features_keyword

# 仮想環境のPythonを使う場合（適宜変更してください）
PYTHON=${PYTHON:-/home/i055ueno/k_data/reserch/.venv/bin/python}

# --- デフォルト設定（環境変数で上書き可能） ---
API_DESC="${API_DESC:-data/api_descriptions.json}"
API_KEYWORDS="${API_KEYWORDS:-data/api_keywords_single.json}"
API_LIST="${API_LIST:-data/api.json}"
TRAIN_DIR="${TRAIN_DIR:-2024/Dataset_Extract/2016}"
TEST_DIR="${TEST_DIR:-2024/Dataset_Extract/2017}"
FEATURES_DIR="${FEATURES_DIR:-features_keyword}"
LOGS_DIR="${LOGS_DIR:-logs_keyword}"

# RF パラメータ
N_ESTIMATORS="${N_ESTIMATORS:-100}"
N_FOLDS="${N_FOLDS:-4}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
KEYWORD_TOP_N="${KEYWORD_TOP_N:-3}"

echo "============================================="
echo " FFRI Dataset: キーワードBoW機能推定パイプライン (SLURM)"
echo "============================================="
echo ""
echo "設定:"
echo "  Python:          ${PYTHON}"
echo "  API説明文:       ${API_DESC}"
echo "  キーワードJSON:  ${API_KEYWORDS}"
echo "  APIリスト:       ${API_LIST}"
echo "  訓練データ:      ${TRAIN_DIR}"
echo "  テストデータ:    ${TEST_DIR}"
echo "  特徴量出力:      ${FEATURES_DIR}"
echo "  ログ出力:        ${LOGS_DIR}"
echo "  RF 決定木数:     ${N_ESTIMATORS}"
echo "  CV 分割数:       ${N_FOLDS}"
echo "  ジョブ並列数:    ${N_JOBS}"
echo ""

# =============================================================================
# Step 1: キーワード抽出（api_keywords_single.json が無い場合のみ実行）
# =============================================================================
echo "--- Step 1: キーワード抽出 ---"
${PYTHON} src/make_keyword.py \
    --input "${API_DESC}" \
    --output "${API_KEYWORDS}" \
    --top-n "${KEYWORD_TOP_N}"
echo ""

# =============================================================================
# Step 2: BoW特徴量生成
# =============================================================================
echo "--- Step 2: BoW特徴量生成 ---"
${PYTHON} src/make_bagofwords.py \
    --api-keywords "${API_KEYWORDS}" \
    --api-list "${API_LIST}" \
    --train-dir "${TRAIN_DIR}" \
    --test-dir "${TEST_DIR}" \
    --output-dir "${FEATURES_DIR}"
echo ""

# =============================================================================
# Step 3: 学習・評価（RF Binary Relevance + 層化CV）
# =============================================================================
echo "--- Step 3: 学習・評価 ---"
${PYTHON} src/train_bagofwords.py \
    --features-dir "${FEATURES_DIR}" \
    --output-dir "${LOGS_DIR}" \
    --n-estimators "${N_ESTIMATORS}" \
    --n-folds "${N_FOLDS}" \
    --n-jobs "${N_JOBS}"
echo ""

echo "============================================="
echo " パイプライン完了！"
echo " レポート: ${LOGS_DIR}/final_reports_br_+exist_rf/"
echo " モデル:   ${LOGS_DIR}/models_br_+exist_rf/"
echo "============================================="
