#!/bin/bash
# =============================================================================
# FFRI Dataset: キーワードBoW機能推定実験パイプライン
# =============================================================================
#
# 使い方:
#   cd /home/i055ueno/k_data/reserch
#   bash scripts/run_ffri_keyword_bow.sh
#
# 環境変数でパスを上書きできます:
#   API_DESC=data/api_descriptions.json \
#   API_KEYWORDS=data/api_keywords_single.json \
#   API_LIST=data/api.json \
#   TRAIN_DIR=2024/Dataset_Extract/2016 \
#   TEST_DIR=2024/Dataset_Extract/2017 \
#   FEATURES_DIR=features_keyword \
#   LOGS_DIR=logs_keyword \
#   bash scripts/run_ffri_keyword_bow.sh
#
# =============================================================================

set -euo pipefail

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
N_JOBS="${N_JOBS:--1}"
KEYWORD_TOP_N="${KEYWORD_TOP_N:-3}"

echo "============================================="
echo " FFRI Dataset: キーワードBoW機能推定パイプライン"
echo "============================================="
echo ""
echo "設定:"
echo "  API説明文:       ${API_DESC}"
echo "  キーワードJSON:  ${API_KEYWORDS}"
echo "  APIリスト:       ${API_LIST}"
echo "  訓練データ:      ${TRAIN_DIR}"
echo "  テストデータ:    ${TEST_DIR}"
echo "  特徴量出力:      ${FEATURES_DIR}"
echo "  ログ出力:        ${LOGS_DIR}"
echo "  RF 決定木数:     ${N_ESTIMATORS}"
echo "  CV 分割数:       ${N_FOLDS}"
echo ""

# =============================================================================
# Step 1: キーワード抽出（api_keywords_single.json が無い場合のみ実行）
# =============================================================================
echo "--- Step 1: キーワード抽出 ---"
python3 src/make_keyword.py \
    --input "${API_DESC}" \
    --output "${API_KEYWORDS}" \
    --top-n "${KEYWORD_TOP_N}"
echo ""

# =============================================================================
# Step 2: BoW特徴量生成
# =============================================================================
echo "--- Step 2: BoW特徴量生成 ---"
python3 src/make_bagofwords.py \
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
python3 src/train_bagofwords.py \
    --features-dir "${FEATURES_DIR}" \
    --output-dir "${LOGS_DIR}" \
    --n-estimators "${N_ESTIMATORS}" \
    --n-folds "${N_FOLDS}" \
    --n-jobs "${N_JOBS}"
echo ""

echo "============================================="
echo " パイプライン完了！"
echo " レポート: ${LOGS_DIR}/final_reports_br_+freq_rf/"
echo " モデル:   ${LOGS_DIR}/models_br_+freq_rf/"
echo "============================================="
