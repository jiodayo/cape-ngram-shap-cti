#!/bin/bash
#SBATCH --job-name=full_pipeline
#SBATCH --output=joblog/full_pipeline_%j.out
#SBATCH --error=joblog/full_pipeline_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=30000M
#SBATCH --time=3-00:00:00

##############################################################################
# 全パイプライン一括実行スクリプト
#
# Step 3〜10 + リーダーボード登録 を一発で実行する。
# 学習（Step 2）は別ジョブで事前に完了している前提。
#
# 使い方:
#   sbatch scripts/run_full_pipeline.sbatch
#
#   # パラメータを変更して実行:
#   FEAT=logs/my_experiment RANDOM_STATE=123 sbatch scripts/run_full_pipeline.sbatch
#
#   # 特定ラベルだけ処理:
#   LABELS="Trojan,Ransomware" sbatch scripts/run_full_pipeline.sbatch
##############################################################################

set -euo pipefail

cd /home/i055ueno/k_data/reserch
mkdir -p joblog

PYTHON=/home/i055ueno/k_data/reserch/.venv/bin/python

# ==========================================================================
# 設定（環境変数で上書き可能）
# ==========================================================================

# --- 入力パス ---
FEAT=${FEAT:-logs/cape_ngram_desc_full_lgbm_seed42_106}
MODEL_DIR=${MODEL_DIR:-${FEAT}/lgbm/models}
DATA_ROOT=${DATA_ROOT:-data/avast_ctu_cape/ngram_dataset_family_104}

# --- 出力パス ---
RANDOM_STATE=${RANDOM_STATE:-42}
EXP_NAME=${EXP_NAME:-lgbm_seed${RANDOM_STATE}_$(date +%Y%m%d)}
OUT=${OUT:-shap_analysis/${EXP_NAME}}

# --- 各種設定 ---
TOP_K=${TOP_K:-10}                        # per-sample SHAP上位k個
MIN_CONFIDENCE=${MIN_CONFIDENCE:-0.3}     # CTIルールの最低confidence
TOP_TECHNIQUES=${TOP_TECHNIQUES:-5}       # 検体あたりの最大技術数
TOP_SAMPLES=${TOP_SAMPLES:-50}            # 攻撃チェーン出力の最大検体数
LABELS=${LABELS:-}                        # 特定ラベルのみ処理（空=全ラベル）

# --- ATT&CK ---
ATTACK_JSON=${ATTACK_JSON:-reference/mitre/enterprise-attack.json}
ATTACK_DB=${ATTACK_DB:-reference/mitre/attack.sqlite}
RULES_CSV=${RULES_CSV:-reference/mitre/api_to_attack_rules.csv}
TEMPLATES=${TEMPLATES:-reference/mitre/explanation_templates.json}

# --- Leaderboard ---
REGISTER_LEADERBOARD=${REGISTER_LEADERBOARD:-1}  # 1=登録する 0=スキップ
LEADERBOARD_DB=${LEADERBOARD_DB:-leaderboard.sqlite}

# ==========================================================================
# ヘルパー関数
# ==========================================================================

step_start() {
    echo ""
    echo "========================================================================"
    echo "[STEP $1] $2"
    echo "========================================================================"
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S')"
}

step_done() {
    echo "[DONE] Step $1 completed at $(date '+%Y-%m-%d %H:%M:%S')"
}

# LABELS引数の組み立て
LABEL_ARGS=()
if [[ -n "${LABELS}" ]]; then
    LABEL_ARGS=(--labels "${LABELS}")
fi

# ==========================================================================
# 事前チェック
# ==========================================================================

echo "========================================================================"
echo " Full Pipeline: ${EXP_NAME}"
echo "========================================================================"
echo "[INFO] Feature root : ${FEAT}"
echo "[INFO] Model dir    : ${MODEL_DIR}"
echo "[INFO] Output dir   : ${OUT}"
echo "[INFO] Labels       : ${LABELS:-ALL}"
echo "[INFO] min_confidence: ${MIN_CONFIDENCE}"
echo "[INFO] SLURM Job ID : ${SLURM_JOB_ID:-local}"
echo ""

# モデルの存在確認
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "[ERROR] Model directory not found: ${MODEL_DIR}"
    echo "        学習（Step 2）を先に実行してください。"
    exit 1
fi

# ATT&CK DB がなければ作成
if [[ ! -f "${ATTACK_DB}" ]]; then
    echo "[INFO] ATT&CK DB not found. Building..."
    if [[ ! -f "${ATTACK_JSON}" ]]; then
        echo "[INFO] Downloading MITRE ATT&CK STIX data..."
        mkdir -p "$(dirname "${ATTACK_JSON}")"
        curl -sL -o "${ATTACK_JSON}" \
            "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    fi
    "${PYTHON}" src/cti_build_attack_db.py \
        --attack-json "${ATTACK_JSON}" \
        --output-db "${ATTACK_DB}"
fi

# ==========================================================================
# Step 3: SHAP グループ分析
# ==========================================================================
step_start 3 "SHAP Group Analysis"

"${PYTHON}" src/analyze_shap_ngram_desc_lgbm_group.py \
    --feature-root "${FEAT}" \
    --model-dir "${MODEL_DIR}" \
    --output-dir "${OUT}" \
    "${LABEL_ARGS[@]+"${LABEL_ARGS[@]}"}"

step_done 3

# ==========================================================================
# Step 4: Per-Sample SHAP出力
# ==========================================================================
step_start 4 "Per-Sample SHAP Export"

"${PYTHON}" src/export_shap_ngram_desc_lgbm_per_sample.py \
    --feature-root "${FEAT}" \
    --model-dir "${MODEL_DIR}" \
    --output-dir "${OUT}/per_sample" \
    --top-k "${TOP_K}" \
    "${LABEL_ARGS[@]+"${LABEL_ARGS[@]}"}"

step_done 4

# ==========================================================================
# Step 5: CTI マッピング（ATT&CK連携）
# ==========================================================================
step_start 5 "CTI Mapping (ATT&CK)"

"${PYTHON}" src/cti_attach_shap_explanations.py \
    --shap-per-sample-dir "${OUT}/per_sample" \
    --output-dir "${OUT}/cti" \
    --output-db "${OUT}/cti/cti_results.sqlite" \
    --attack-db "${ATTACK_DB}" \
    --rules-csv "${RULES_CSV}" \
    --templates "${TEMPLATES}" \
    --use-confidence-weight \
    --min-confidence "${MIN_CONFIDENCE}" \
    --top-techniques "${TOP_TECHNIQUES}"

step_done 5

# ==========================================================================
# Step 6a: スライド用3点セット
# ==========================================================================
step_start 6a "Slide Visualization (3-piece set)"

"${PYTHON}" src/visualize_shap_slide_set.py \
    --group-csv "${OUT}/shap_group_contributions.csv" \
    --top-csv "${OUT}/shap_top_features.csv" \
    --per-sample-dir "${OUT}/per_sample" \
    --output-dir "figures/${EXP_NAME}/slide_set" \
    --top-k 20

step_done 6a

# ==========================================================================
# Step 6b: SHAP+CTI統合可視化
# ==========================================================================
step_start 6b "SHAP+CTI Integrated Visualization"

"${PYTHON}" src/visualize_shap_cti_integrated.py \
    --cti-db "${OUT}/cti/cti_results.sqlite" \
    --attack-db "${ATTACK_DB}" \
    --group-csv "${OUT}/shap_group_contributions.csv" \
    --top-csv "${OUT}/shap_top_features.csv" \
    --output-dir "figures/${EXP_NAME}/cti_integrated" \
    "${LABEL_ARGS[@]+"${LABEL_ARGS[@]}"}"

step_done 6b

# ==========================================================================
# Step 7: CTI深層分析
# ==========================================================================
step_start 7 "CTI Advanced Analysis"

"${PYTHON}" src/cti_advanced_analysis.py \
    --cti-db "${OUT}/cti/cti_results.sqlite" \
    --attack-db "${ATTACK_DB}" \
    --templates "${TEMPLATES}" \
    --output-dir "reports/${EXP_NAME}/cti_analysis" \
    --top-samples "${TOP_SAMPLES}" \
    "${LABEL_ARGS[@]+"${LABEL_ARGS[@]}"}"

step_done 7

# ==========================================================================
# Step 8: CTI分析可視化
# ==========================================================================
step_start 8 "CTI Analysis Visualization"

"${PYTHON}" src/visualize_cti_analysis.py \
    --input-dir "reports/${EXP_NAME}/cti_analysis" \
    --output-dir "figures/${EXP_NAME}/cti_analysis"

step_done 8

# ==========================================================================
# Step 9: 説明品質評価
# ==========================================================================
step_start 9 "Explanation Quality Evaluation"

EVAL_DIR="evaluation/${EXP_NAME}"
mkdir -p "${EVAL_DIR}"

"${PYTHON}" src/evaluate_explanation_quality.py \
    --cti-db "${OUT}/cti/cti_results.sqlite" \
    --group-csv "${OUT}/shap_group_contributions.csv" \
    --per-sample-dir "${OUT}/per_sample" \
    --output-dir "${EVAL_DIR}"

step_done 9

# ==========================================================================
# Step 10: HTMLレポート生成
# ==========================================================================
step_start 10 "HTML Report Generation"

REPORT_DIR="reports/${EXP_NAME}"
mkdir -p "${REPORT_DIR}"

"${PYTHON}" src/generate_html_report.py \
    --group-csv "${OUT}/shap_group_contributions.csv" \
    --top-csv "${OUT}/shap_top_features.csv" \
    --per-sample-dir "${OUT}/per_sample" \
    --cti-dir "${OUT}/cti" \
    --eval-json "${EVAL_DIR}/explanation_quality.json" \
    --figures-dir "figures/${EXP_NAME}/slide_set" \
    --output "${REPORT_DIR}/shap_cti_report.html" \
    --title "SHAP/CTI Report — ${EXP_NAME}"

step_done 10

# ==========================================================================
# リーダーボード登録
# ==========================================================================

if [[ "${REGISTER_LEADERBOARD}" == "1" ]]; then
    step_start "LB" "Leaderboard Registration"

    LB_ARGS=(
        "${PYTHON}" src/leaderboard.py
        --db "${LEADERBOARD_DB}"
        register
        --name "${EXP_NAME}"
        --eval-json "${EVAL_DIR}/explanation_quality.json"
        --notes "job=${SLURM_JOB_ID:-local} min_conf=${MIN_CONFIDENCE} top_k=${TOP_K}"
    )

    # metrics_overall.json があれば追加
    METRICS_JSON="${FEAT}/lgbm/metrics_overall.json"
    if [[ -f "${METRICS_JSON}" ]]; then
        LB_ARGS+=(--metrics-json "${METRICS_JSON}")
    else
        echo "[WARN] metrics_overall.json not found: ${METRICS_JSON}"
    fi

    # per_label_metrics.csv があれば追加
    PER_LABEL_CSV="${FEAT}/lgbm/per_label_metrics.csv"
    if [[ -f "${PER_LABEL_CSV}" ]]; then
        LB_ARGS+=(--per-label-csv "${PER_LABEL_CSV}")
    else
        echo "[WARN] per_label_metrics.csv not found: ${PER_LABEL_CSV}"
    fi

    # 設定情報
    CONFIG_JSON=$(cat <<EOF
{"seed":${RANDOM_STATE},"model":"lgbm","min_confidence":${MIN_CONFIDENCE},"top_k":${TOP_K},"feature_root":"${FEAT}","top_techniques":${TOP_TECHNIQUES}}
EOF
    )
    LB_ARGS+=(--config "${CONFIG_JSON}")

    "${LB_ARGS[@]}"

    step_done "LB"

    # リーダーボードHTML更新
    "${PYTHON}" src/leaderboard.py \
        --db "${LEADERBOARD_DB}" \
        html \
        --output "reports/leaderboard.html" \
        --title "Experiment Leaderboard"
    echo "[INFO] Leaderboard HTML updated: reports/leaderboard.html"
fi

# ==========================================================================
# 完了サマリ
# ==========================================================================

echo ""
echo "========================================================================"
echo " PIPELINE COMPLETE"
echo "========================================================================"
echo "[INFO] Experiment   : ${EXP_NAME}"
echo "[INFO] Completed at : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo " 生成物一覧:"
echo "   SHAP分析      : ${OUT}/"
echo "   スライド図    : figures/${EXP_NAME}/slide_set/"
echo "   統合可視化    : figures/${EXP_NAME}/cti_integrated/"
echo "   CTI分析図    : figures/${EXP_NAME}/cti_analysis/"
echo "   攻撃チェーン  : reports/${EXP_NAME}/cti_analysis/"
echo "   Navigator     : reports/${EXP_NAME}/cti_analysis/navigator_layers/"
echo "   評価          : ${EVAL_DIR}/"
echo "   HTMLレポート  : ${REPORT_DIR}/shap_cti_report.html"
echo "   リーダーボード: reports/leaderboard.html"
echo ""
echo " ローカルにダウンロード:"
echo "   scp server:~/k_data/reserch/${REPORT_DIR}/shap_cti_report.html ./"
echo "   scp server:~/k_data/reserch/reports/leaderboard.html ./"
echo "========================================================================"
