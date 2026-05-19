#!/bin/bash
#SBATCH --job-name=cti_attach_shap
#SBATCH --output=joblog/cti_attach_shap_%j.out
#SBATCH --error=joblog/cti_attach_shap_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000M
#SBATCH --time=1-00:00:00

set -euo pipefail

cd /home/i055ueno/k_data/reserch
mkdir -p joblog

PYTHON=/home/i055ueno/k_data/reserch/.venv/bin/python

ATTACK_JSON=${ATTACK_JSON:-reference/mitre/enterprise-attack.json}
ATTACK_DB=${ATTACK_DB:-reference/mitre/attack.sqlite}
RULES=${RULES:-reference/mitre/api_to_attack_rules.csv}
SHAP_DIR=${SHAP_DIR:-shap_analysis/ngram_desc_lgbm_group_nometa_106/per_sample}
OUT_DIR=${OUT_DIR:-shap_analysis/ngram_desc_lgbm_group_nometa_106/cti}
OUT_DB=${OUT_DB:-shap_analysis/ngram_desc_lgbm_group_nometa_106/cti/cti_results.sqlite}
TOP_TECHNIQUES=${TOP_TECHNIQUES:-3}

"${PYTHON}" /home/i055ueno/k_data/reserch/src/cti_build_attack_db.py \
  --attack-json "${ATTACK_JSON}" \
  --output-db "${ATTACK_DB}"

"${PYTHON}" /home/i055ueno/k_data/reserch/src/cti_attach_shap_explanations.py \
  --shap-per-sample-dir "${SHAP_DIR}" \
  --attack-db "${ATTACK_DB}" \
  --rules "${RULES}" \
  --output-dir "${OUT_DIR}" \
  --output-db "${OUT_DB}" \
  --top-techniques "${TOP_TECHNIQUES}"

printf '[INFO] Completed CTI attach: %s\n' "${OUT_DIR}"
