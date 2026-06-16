#!/bin/bash
#SBATCH -J BRjob
#SBATCH -p GPU1
#SBATCH -D /k_data1/i055ueno/reserch
#SBATCH -o joblog/%x_%j_withAPI.log
#SBATCH -e joblog/%x_%j_withAPI.err

set -euo pipefail
cd /k_data1/i055ueno/reserch
python3 src/analyze_pca.py
