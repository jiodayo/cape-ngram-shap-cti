#!/bin/bash
#SBATCH -J shap_hybrid
#SBATCH -p GPU801
#SBATCH -D /k_data1/i055ueno/reserch
#SBATCH -o logs/shap_hybrid_%j.log
#SBATCH -e logs/shap_hybrid_%j.err

# エラーが発生したら即座にスクリプトを停止する
set -e

echo "Job started at: $(date)"

# 仮想環境の有効化
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "=== ハイブリッドSHAP分析 (keyword モデル) ==="
python3 src/analyze_shap_hybrid.py --model-type keyword

echo "Job finished at: $(date)"
