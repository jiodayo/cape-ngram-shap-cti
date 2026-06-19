#!/bin/bash
#SBATCH --job-name=mbc_pipeline
#SBATCH --output=mbc_pipeline_%j.out
#SBATCH --error=mbc_pipeline_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# ==========================================
# GPUを使用する場合は以下のコメントアウトを外してください
# #SBATCH --gpus=1
# #SBATCH --partition=gpu
# ==========================================

echo "=== MBC Keyword Pipeline Started at $(date) ==="

# ==========================================
# 環境のロードが必要な場合はここに追記してください
# 例:
# source ~/.bashrc
# conda activate cape-env
# ==========================================

echo "Installing required packages if needed..."
pip install -r requirements.txt

echo "[1/5] Extracting keywords with KeyBERT..."
python src/make_keyword.py --force
if [ $? -ne 0 ]; then echo "Error in make_keyword.py"; exit 1; fi

echo "[2/5] Building Bag-of-Words features..."
python src/make_bagofwords.py
if [ $? -ne 0 ]; then echo "Error in make_bagofwords.py"; exit 1; fi

echo "[3/5] Mapping keywords to MBC..."
python src/build_mbc_mapping.py
if [ $? -ne 0 ]; then echo "Error in build_mbc_mapping.py"; exit 1; fi

echo "[4/5] Training Random Forest model..."
python src/2025_RF_PCA.py --mode keyword
if [ $? -ne 0 ]; then echo "Error in 2025_RF_PCA.py"; exit 1; fi

echo "[5/5] Analyzing SHAP values..."
python src/analyze_shap_hybrid.py --model-type keyword
if [ $? -ne 0 ]; then echo "Error in analyze_shap_hybrid.py"; exit 1; fi

echo "=== Pipeline Completed Successfully at $(date) ==="
