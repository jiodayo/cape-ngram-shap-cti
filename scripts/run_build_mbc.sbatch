#!/bin/bash
#SBATCH -J build_mbc
#SBATCH -p GPU801
#SBATCH -D /home/i055ueno/k_data/reserch
#SBATCH -o joblog/%x_%j.log
#SBATCH -e joblog/%x_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=200G

set -euo pipefail
ROOT_DIR="/home/i055ueno/k_data/reserch"
cd "$ROOT_DIR"
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
