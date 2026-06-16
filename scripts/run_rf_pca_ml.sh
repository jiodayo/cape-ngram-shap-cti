#!/bin/bash
#SBATCH -J PCA_ML
#SBATCH -p GPU1
#SBATCH -D /k_data1/i055ueno/reserch
#SBATCH -o logs/PCA_ML_%j.log
#SBATCH -e logs/PCA_ML_%j.err

set -euo pipefail
cd /k_data1/i055ueno/reserch

# 仮想環境を有効化 (パスは適宜調整)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=== PCA特徴量を用いたRandom Forestの学習を開始します ==="
python3 src/2025_RF_PCA.py train_pca

echo "=== 学習完了。生成物をコピーします ==="
TODAY=$(date +%Y-%m-%d)
TODAY_DIR="daily/${TODAY}/pca"
mkdir -p "$TODAY_DIR"

# 2025_RF_PCA.py は logs/models_br_rf, logs/fold_reports_br_rf, logs/final_reports_br_rf に結果を出力します
cp -r logs/models_br_rf "$TODAY_DIR/" || true
cp -r logs/fold_reports_br_rf "$TODAY_DIR/" || true
cp -r logs/final_reports_br_rf "$TODAY_DIR/" || true

echo "=== すべての処理が完了しました ==="