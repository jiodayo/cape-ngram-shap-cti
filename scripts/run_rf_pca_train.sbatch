#!/bin/bash
#SBATCH --job-name=rf_pca_train
#SBATCH --output=logs/rf_pca_train_%j.log
#SBATCH --error=logs/rf_pca_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

echo "Starting Random Forest training on PCA features..."
date

source .venv/bin/activate

echo "Synchronizing labels (prepare_labels)..."
python src/2025_RF_PCA.py prepare_labels

python src/2025_RF_PCA.py train_pca

echo "Training completed."
date
