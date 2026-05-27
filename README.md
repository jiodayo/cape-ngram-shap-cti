# マルウェア検出研究 - ディレクトリ構成

## 📁 ディレクトリ一覧

### メイン実行ディレクトリ

| ディレクトリ | 説明 | 主要ファイル |
|-----------|------|-----------|
| `src/` | Pythonスクリプト（学習・分析・処理） | `2025_learning_RF_Sen+Fre.py`, `task1_shap_contribution_analysis.py`, `analyze_SHAP.py` 等 27個 |
| `scripts/` | シェルスクリプト | `RF.sh`, `L_3.sh`, `analyze.sh` 等 10個 |
| `data/` | JSON設定・ラベルファイル | `api.json`, `labels.json`, `api_keywords.json` 等 |
| `reports/` | ドキュメント | `研究進捗報告書_論文投稿版.txt`, `詳細な研究進捗分析レポート.txt` 等 |

### 結果・出力ディレクトリ

| ディレクトリ | 説明 | 備考 |
|-----------|------|------|
| `logs/` | モデル・レポート出力 | 各学習実行時のモデル・結果ファイル |
| `logs_api_frequency/` | API频度ベース学習結果 | `train_api_frequency_rf.py` の出力 |
| `logs_keyword/` | キーワード（BoW）ベース学習結果 | 古いパイプライン結果 |
| `shap_analysis/` | SHAP解釈結果 | `contribution_analysis/shap_contribution_analysis.csv` 等 |
| `analysis_results/` | 分析結果 | `feature_contribution_analysis.csv` 等 |
| `tree_visualization/` | 決定木可視化 | `visualize_tree.py` の出力 |

### データファイル

| ディレクトリ | 説明 | ファイルサイズ |
|-----------|------|-------------|
| `encoded_train_npz/` | 訓練データ（NPZ形式） | 本体: 254GB（mmap化） |
| `encoded_test_npz/` | テストデータ（NPZ形式） | - |
| `encoded_train_features.mmap` | 訓練特徴量（メモリマップ） | 254GB |
| `encoded_train_features_pca.mmap` | 訓練特徴量（PCA削減後） | - |
| `encoded_train_labels.npy` | 訓練ラベル | - |
| `encoded_test_labels.npy` | テストラベル | - |

### アーカイブ

| ディレクトリ | 説明 |
|-----------|------|
| `archives/` | 古いファイル・非使用スクリプト |
| `archives/features_old/` | 古いBoW特徴量処理（deprecated） |
| `huyou/` | 一時的ファイル |
| `joblog/` | ジョブログ |

---

## 🚀 主要スクリプト実行例

### 現在のメインパイプライン（BERT + API frequency）

```bash
# ルートディレクトリから実行
cd /home/i055ueno/k_data/reserch

# 1. 訓練・モデル学習
python3 src/2025_learning_RF_Sen+Fre.py

# 2. タスク1：SHAP寄与度分析（機械学習可視化）
python3 src/task1_shap_contribution_analysis.py
# 出力: shap_analysis/contribution_analysis/shap_contribution_analysis.csv

# 3. SHAP分析（各ラベルの特徴量寄与度）
python3 src/analyze_SHAP.py

# 4. 保存モデルの評価
python3 src/2025_evaluate_saved_model.py
```

### 追加実験パイプライン（n-gram + API説明文 + BR）

```bash
# ルートディレクトリから実行
cd /home/i055ueno/k_data/reserch

# 1) LightGBMで学習（推奨）
python3 src/2026_learning_ngram_desc_br.py \
   --model lgbm

# 2) skip-gramを有効化して学習
python3 src/2026_learning_ngram_desc_br.py \
   --model lgbm \
   --use-skipgram \
   --skip-max-gap 2 \
   --skip-window 4

# 3) 同一特徴量で LightGBM と RandomForest を比較
python3 src/2026_learning_ngram_desc_br.py \
   --model lgbm \
   --compare-rf \
   --use-skipgram

# 4) 説明文埋め込み（Sentence-Transformers）を使用
python3 src/2026_learning_ngram_desc_br.py \
   --model lgbm \
   --desc-mode embedding \
   --desc-embedding-model sentence-transformers/all-MiniLM-L6-v2

# 5) TF-IDF と埋め込みを併用
python3 src/2026_learning_ngram_desc_br.py \
   --model lgbm \
   --desc-mode hybrid \
   --desc-embedding-model sentence-transformers/all-MiniLM-L6-v2
```

出力先:
- `logs/ngram_desc/feature_summary.json`（特徴量次元サマリ）
- `logs/ngram_desc/lgbm/metrics_overall.json`
- `logs/ngram_desc/rf/metrics_overall.json`（`--compare-rf` 時）
- `logs/ngram_desc/model_comparison.csv`（`--compare-rf` 時）

### サーバーバッチ実行（SLURM）

```bash
# ルートディレクトリ
cd /home/i055ueno/k_data/reserch

# 1) 中規模比較（LGBM vs RF）
sbatch scripts/run_ngram_desc_compare.sbatch

# 2) 本番LGBM
sbatch scripts/run_ngram_desc_lgbm_full.sbatch

# 3) 比較完了後に本番を依存実行
bash scripts/submit_ngram_desc_jobs.sh
```

主な可変パラメータ例:

```bash
# 例: 中規模比較のサンプル数と特徴次元を変更
MAX_TRAIN=500 MAX_TEST=250 MAX_SEQ_FEATURES=150000 MAX_SKIP_FEATURES=100000 \
sbatch scripts/run_ngram_desc_compare.sbatch

# 例: 本番LGBMの木数を調整
LGBM_N_ESTIMATORS=300 LGBM_NUM_LEAVES=31 \
sbatch scripts/run_ngram_desc_lgbm_full.sbatch

# 例: バッチで説明文埋め込みのみ利用
DESC_MODE=embedding DESC_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
sbatch scripts/run_ngram_desc_compare.sbatch

# 例: バッチでTF-IDFと埋め込みを併用
DESC_MODE=hybrid DESC_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
sbatch scripts/run_ngram_desc_lgbm_full.sbatch
```

### 古いパイプライン（参考）

```bash
# API频度ベース RF学習
python3 src/train_api_frequency_rf.py

# キーワード（BoW）ベース RF学習
# features/train_bagofwords.py, features/train_keyword_rf.py
```

---

## 📊 データ処理フロー

```
1. NPZ入力 (encoded_train_npz/sample_*.npz)
   ↓ [読み込み → BERT (768D) + API_freq (301D) = 1069D]
   ↓
2. メモリマップ (encoded_train_features.mmap, 254GB)
   ↓ [PCA削減: 1069D → 365D (99.997% 削減)]
   ↓
3. PCA後特徴量 (encoded_train_features_pca.mmap)
   ↓ [Binary Relevance RF モデル ×10ラベル]
   ↓
4. 学習済みモデル (logs/models_br_rf/*.joblib)
   ↓ [SHAP値計算]
   ↓
5. SHAP結果 (shap_analysis/*.csv)
   ↓ [寄与度分析]
   ↓
6. 最終レポート (shap_analysis/contribution_analysis/*)
```

---

## 🔗 CTI連携・XAI パイプライン

```
モデル学習完了後:

1. SHAP寄与度分析（グループ別）
   python3 src/analyze_shap_ngram_desc_lgbm_group.py
   → shap_analysis/.../shap_group_contributions.csv
   → shap_analysis/.../shap_top_features.csv

2. per-sample SHAP出力
   python3 src/export_shap_ngram_desc_lgbm_per_sample.py
   → shap_analysis/.../per_sample/shap_per_sample_topk_{label}.csv

3. ATT&CK技術マッピング（confidence重み付き）
   python3 src/cti_attach_shap_explanations.py --use-confidence-weight
   → shap_analysis/.../cti/cti_results.sqlite

4. スライド用可視化3点セット
   python3 src/visualize_shap_slide_set.py
   → figures/slide_set/{label}/summary_bar.png
   → figures/slide_set/{label}/group_donut.png
   → figures/slide_set/{label}/representative_waterfall.png

5. 説明品質評価
   python3 src/evaluate_explanation_quality.py
   → evaluation/explanation_quality.json

6. HTMLレポート生成（全結果を統合）
   python3 src/generate_html_report.py
   → reports/shap_cti_report.html

7. CTI深層分析（共起・攻撃チェーン・Navigatorレイヤー）
   python3 src/cti_advanced_analysis.py
   → reports/cti_analysis/attack_chains.json
   → reports/cti_analysis/navigator_layers/layer_{label}.json
   → reports/cti_analysis/threat_profiles.json

8. CTI分析可視化
   python3 src/visualize_cti_analysis.py
   → figures/cti_analysis/cooccurrence_heatmap.png
   → figures/cti_analysis/threat_level_distribution.png
```

### ツール一覧

| スクリプト | 用途 |
|-----------|------|
| `src/common.py` | 共通関数モジュール（全スクリプトが依存） |
| `src/cti_validate_rules.py` | ATT&CKルールの検証・カバレッジ確認 |
| `src/visualize_shap_slide_set.py` | スライド用SHAP可視化3点セット生成 |
| `src/query_shap_db.py` | SQLiteからの検体・技術検索CLI |
| `src/evaluate_explanation_quality.py` | 説明品質の定量評価 |
| `src/generate_html_report.py` | 全結果を統合したHTMLレポート生成 |
| `src/visualize_shap_cti_integrated.py` | SHAP+CTI統合可視化（ヒートマップ・キルチェーン・レーダー） |
| `src/generate_api_descriptions.py` | Windows API説明文の自動生成（desc特徴量の品質向上） |
| `src/cti_advanced_analysis.py` | CTI深層分析（共起・攻撃チェーン・Navigator・脅威プロファイル） |
| `src/visualize_cti_analysis.py` | CTI分析結果の可視化（共起ヒートマップ・脅威レベル分布等） |
| `src/leaderboard.py` | 実験リーダーボード（登録・比較・HTMLランキング生成） |

### 設定ファイル

| ファイル | 用途 |
|---------|------|
| `config/default_paths.json` | デフォルトパス設定（全スクリプト共通） |
| `reference/mitre/api_to_attack_rules.csv` | API→ATT&CK技術マッピングルール（98ルール） |
| `reference/mitre/explanation_templates.json` | 技術IDごとの自然言語説明テンプレート（31技術） |
| `requirements.txt` | Python依存パッケージ |

---

## 📝 重要な設定ファイル

| ファイル | 用途 |
|---------|------|
| `data/labels.json` | マルウェア挙動ラベル定義（15個） |
| `data/label_set.json` | 別形式のラベル定義 |
| `data/api.json` | API定義データ |
| `data/api_keywords.json` | API×キーワード対応表 |

---

## 💾 パス参照の注意

**相対パスは全てルートから解釈されます**

```python
# スクリプト内の例
SHAP_DIR = Path("shap_analysis")      # → /home/i055ueno/k_data/reserch/shap_analysis
OUTPUT_DIR = Path("shap_analysis/contribution_analysis")  # → 同上/contribution_analysis
```

**実行方法**
```bash
cd /home/i055ueno/k_data/reserch
python3 src/task1_shap_contribution_analysis.py  # ✓ 正常動作
```

❌ src/ディレクトリから直接実行しないでください
```bash
cd src
python3 task1_shap_contribution_analysis.py  # ✗ パス参照失敗
```

---

## 🔍 最新プロジェクト情報

- **研究テーマ**: 寄与度ギャップを利用したマルウェア検出XAI
- **現況**: タスク1（SHAP寄与度分析）✅ 完了
- **次のタスク**: Task 2（file_recreated 精度向上）、Task 3（XAI Dashboard）
- **詳細報告書**: `reports/研究進捗報告書_論文投稿版.txt`

