#!/bin/bash
#SBATCH -J MBC_PIPELINE
#SBATCH -p GPU1
#SBATCH -w gpu801
#SBATCH -D /k_data1/i055ueno/reserch
#SBATCH -o logs/MBC_PIPELINE_%j.log
#SBATCH -e logs/MBC_PIPELINE_%j.err

# エラーが発生したら即座にスクリプトを停止する
set -e

echo "Job started at: $(date)"

# 仮想環境の有効化（環境に合わせて変更）
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "=== 1. KeyBERTによるキーワード抽出 ==="
python3 src/make_keyword.py --force

echo "=== 2. BoW特徴量の生成 ==="
python3 src/make_bagofwords.py

echo "=== 3. MBC Micro-Behaviorsへのマッピング ==="
python3 src/build_mbc_mapping.py

echo "=== 4. Random Forestの学習 (Keywordモデル) ==="
python3 src/train_bagofwords.py

echo "=== 5. ハイブリッドSHAP分析 ==="
python3 src/analyze_shap_hybrid.py --model-type keyword

echo "Job finished at: $(date)"
