#!/bin/bash
#SBATCH -J PCA_FULL
#SBATCH -p GPU1
#SBATCH -w gpu801
#SBATCH -D /k_data1/i055ueno/reserch
#SBATCH -o logs/PCA_FULL_%j.log
#SBATCH -e logs/PCA_FULL_%j.err

# エラーが発生したら即座にスクリプトを停止する
set -e

# 日付を取得してエコー（後でログの追跡に便利）
echo "Job started at: $(date)"

# 仮想環境の有効化（環境に合わせて変更）
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "=== 1. 特徴量とラベルの完全再構築 (prepare_data) ==="
python3 src/2025_RF_PCA.py prepare_data

echo "=== 2. PCA特徴量の再生成 (IncrementalPCA) ==="
python3 src/2025_reduce_features.py

echo "=== 3. PCA特徴量を用いたRandom Forestの学習 (train_pca) ==="
python3 src/2025_RF_PCA.py train_pca

echo "Job finished at: $(date)"
