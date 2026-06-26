#!/bin/bash
#SBATCH --job-name=build_mbc
#SBATCH --output=joblog/build_mbc_%j.out
#SBATCH --error=joblog/build_mbc_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=02:00:00

set -euo pipefail

# プロジェクトルートへ移動（環境に合わせて修正してください）
cd /home/i055ueno/k_data/reserch

# ログディレクトリの作成
mkdir -p joblog

# 仮想環境のPythonを使う場合
PYTHON=${PYTHON:-/home/i055ueno/k_data/reserch/.venv/bin/python}

echo "=================================================="
echo "Job Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $SLURMD_NODENAME"
echo "=================================================="

# MBCマッピングスクリプトの実行
echo "Running build_mbc_mapping.py..."
$PYTHON src/build_mbc_mapping.py

echo "=================================================="
echo "Job End: $(date)"
echo "=================================================="
