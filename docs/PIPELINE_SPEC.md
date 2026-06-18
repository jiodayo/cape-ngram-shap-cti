# マルウェア機能推定パイプライン — 技術仕様書

> **目的**: CAPEサンドボックスの動的解析ログからマルウェアの**17種類の機能ラベル**を推定し、SHAPによる説明可能性を付与するシステム。  
> **最終更新**: 2026-06-18

---

## 1. 研究の概要

マルウェアが実行時に呼び出すWindows APIコールのシーケンスを入力として、そのマルウェアが「どのような機能を持つか」（ファイル操作、レジストリ書き換え、ネットワーク通信など）を17種類のラベルでマルチラベル分類する。  
分類結果に対してSHAP（SHapley Additive exPlanations）を適用し、「なぜその機能ラベルが付与されたのか」を特徴量レベルで説明する。

### 対象の17ラベル

| # | ラベル名 | 意味 |
|---|---------|------|
| 0 | `command_line` | コマンドライン操作 |
| 1 | `connects_host` | ホストへの接続 |
| 2 | `connects_ip` | IPアドレスへの接続 |
| 3 | `directory_created` | ディレクトリの作成 |
| 4 | `directory_enumerated` | ディレクトリの列挙 |
| 5 | `file_copied` | ファイルのコピー |
| 6 | `file_created` | ファイルの作成 |
| 7 | `file_deleted` | ファイルの削除 |
| 8 | `file_failed` | ファイル操作の失敗 |
| 9 | `file_read` | ファイルの読み込み |
| 10 | `file_recreated` | ファイルの再作成 |
| 11 | `file_written` | ファイルへの書き込み |
| 12 | `guid` | GUID の生成・使用 |
| 13 | `mutex` | ミューテックスの操作 |
| 14 | `regkey_deleted` | レジストリキーの削除 |
| 15 | `regkey_written` | レジストリキーの書き込み |
| 16 | `resolves_host` | ホスト名の名前解決 |

---

## 2. 2つのモデルパイプライン

本システムでは、異なる特徴量表現に基づく **2つのモデル** を並行して学習・評価し、それぞれにSHAP分析を適用する。

### 2.1 Keyword モデル（BoW + API頻度）

API説明文から自然言語処理でキーワードを抽出し、Bag-of-Words として特徴量化するアプローチ。

```
CAPEサンドボックスJSONレポート
  │
  ├─ APIコール列の抽出
  │    各検体の apicalls: ["CreateFileW", "RegSetValueExW", ...]
  │
  ├─ API説明文の準備 (generate_api_descriptions.py)
  │    "CreateFileW" → "Create or open a file or device using Unicode name"
  │    ※ Microsoft公式ドキュメントに基づく ~300 API の説明文を手動キュレーション
  │
  ├─ TextRank でキーワード抽出 (make_keyword.py)
  │    spaCy + pytextrank を使用
  │    "CreateFileW" → ["file", "device", "name"]
  │    "RegSetValueExW" → ["registry", "value", "data"]
  │    出力: api_keywords_single.json
  │
  ├─ Bag-of-Words 特徴量生成 (make_bagofwords.py)
  │    各検体で呼ばれたAPIのキーワードをカウント
  │    + 各APIの呼び出し頻度(301次元) を結合
  │    出力: features/train_keyword_features.csv, test_keyword_features.csv
  │
  └─ Random Forest (Binary Relevance) で学習 (train_bagofwords.py)
       17ラベル × 独立した二値分類器
       4分割層化交差検証 + 全データ再学習
       出力: logs_keyword/models_br_+freq_rf/*.joblib
```

**特徴量の構成**:
- **キーワードBoW**: ~数百次元（APIキーワードのユニーク数に依存）
- **API頻度**: 301次元（`api__` プレフィックス付き）
- **合計**: ~数百次元のハイブリッド特徴量

---

### 2.2 PCA モデル（BERT Mean Pooling + API頻度）

マルウェアのAPIコール列をBERT系モデルで埋め込み、Mean Poolingで768次元に集約した後にPCAで圧縮するアプローチ。

```
CAPEサンドボックスJSONレポート
  │
  ├─ BERT埋め込みの事前計算 (外部で実行済み)
  │    各検体のAPIコール列をBERTでエンコード
  │    出力: encoded_train_npz/, encoded_test_npz/
  │    各ファイル: sample_XXXX.npz (embedding: [文数, トークン数, 768])
  │
  ├─ Mean Pooling (2025_RF_PCA.py prepare_data)
  │    embedding の全文・全トークン方向の平均を計算
  │    (文数, トークン数, 768) → (768,)
  │    出力: encoded_train_features.mmap (768次元 × サンプル数)
  │
  ├─ PCA次元削減 (2025_reduce_features.py)
  │    768次元 → 128次元
  │    オンメモリの標準PCA（約0.1秒で完了）
  │    出力: encoded_train_features_pca.mmap
  │
  ├─ API頻度とのハイブリッド化 (2025_RF_PCA.py train_pca)
  │    128次元(PCA) + 301次元(API頻度) = 429次元
  │    API頻度は features/train_keyword_features.csv から api__ 列を抽出
  │
  └─ Random Forest (Binary Relevance) で学習
       17ラベル × 独立した二値分類器
       4分割層化交差検証 + 全データ再学習
       出力: logs/models_br_rf/*.joblib
```

**特徴量の構成**:
- **BERT Mean Pooling → PCA**: 128次元（`pca_0` ~ `pca_127`）
- **API頻度**: 301次元（`api__` プレフィックス付き）
- **合計**: 429次元のハイブリッド特徴量

---

## 3. SHAP分析パイプライン

両モデルに対して統一的なSHAP分析を行い、特徴量重要度をカテゴリ別に可視化する。

```
学習済みモデル (*.joblib) × 17ラベル
  │
  ├─ shap.TreeExplainer で SHAP値を計算
  │    テストデータから500件をサンプリングして高速化
  │
  ├─ カテゴリ別重要度の算出
  │    特徴量を2カテゴリに分離:
  │      ① メイン特徴量 (Keyword or PCA)
  │      ② API頻度特徴量 (301次元)
  │    → 全ラベル平均: Keyword 65% / API頻度 35% (keywordモデルの実績値)
  │
  ├─ 方針A: キーワードのAPI出自逆引き
  │    api_keywords_single.json を逆引き
  │    "file" ← CreateFileW, ReadFile, WriteFile, DeleteFileW, ...
  │    → 各キーワードがどのAPIの説明文から抽出されたかを追跡可能に
  │
  ├─ 方針B: 機能直結キーワードフィルタ
  │    11カテゴリのホワイトリストで分類:
  │      File Operation / Registry / Process-Thread / Network /
  │      Memory / Service / Crypto / Security / System Info /
  │      Screen-Input / Anti-Analysis
  │    → "life_cycle", "option", "handle" 等の抽象語を自動除外
  │
  └─ 出力
       ├─ Waterfall プロット (各ラベル × 指定サンプル)
       ├─ shap_report.html (学会向けHTMLレポート)
       └─ overall_report.txt (テキストレポート)
```

---

## 4. 使用技術スタック

### 4.1 言語・フレームワーク

| カテゴリ | 技術 | 用途 |
|---------|------|------|
| 言語 | Python 3.12 | 全スクリプト |
| 機械学習 | scikit-learn | RandomForestClassifier, PCA |
| 機械学習 | LightGBM | 別パイプライン（N-gram + Desc実験）で使用 |
| マルチラベル | scikit-multilearn | IterativeStratification（層化K分割） |
| 説明可能AI | SHAP | TreeExplainer, Waterfall plot |
| 自然言語処理 | spaCy + pytextrank | TextRank によるキーワード抽出 |
| 深層学習 | BERT (事前計算済み) | APIコール列の埋め込み (768次元) |
| データ処理 | NumPy, Pandas | 特徴量の行列演算, CSV入出力 |
| 可視化 | Matplotlib | Waterfall, ヒートマップ |
| ジョブ管理 | SLURM (sbatch) | GPUサーバー上でのバッチ実行 |

### 4.2 データ形式

| ファイル | 形式 | 説明 |
|---------|------|------|
| `encoded_*_npz/sample_XXXX.npz` | NumPy圧縮 | BERT埋め込み (文数×トークン数×768) |
| `encoded_*_features.mmap` | NumPy memmap | Mean Pooling後の特徴量 (float32) |
| `encoded_*_features_pca.mmap` | NumPy memmap | PCA後の特徴量 (float32) |
| `features/*.csv` | CSV | Keyword + API頻度 特徴量 |
| `*.joblib` | joblib | 学習済みRandomForestモデル |
| `api_keywords_single.json` | JSON | API名 → キーワードリスト |
| `api_descriptions.json` | JSON | API名 → 英語説明文 |

---

## 5. データセット

| 項目 | 値 |
|------|-----|
| データソース | CAPE Sandbox 動的解析レポート (JSON) |
| 訓練データ | 2016年収集検体: 5,691件 |
| テストデータ | 2017年収集検体: 4,644件 |
| API種類数 | ~301種類 |
| ラベル数 | 17（マルチラベル） |
| 分割方式 | 年代別分割（時系列リーク防止） |

---

## 6. 実行手順

### 6.1 Keywordモデルのパイプライン

```bash
cd /k_data1/i055ueno/reserch

# Step 1: API説明文からキーワード抽出 → BoW特徴量生成 → RF学習
bash scripts/run_ffri_keyword_bow.sh

# Step 2: SHAP分析
python src/analyze_shap_hybrid.py \
    --model-type keyword \
    --api-keywords-path api_keywords_single.json
```

### 6.2 PCAモデルのパイプライン

```bash
cd /k_data1/i055ueno/reserch

# Step 1-3: prepare_data → PCA → train_pca (ハイブリッド学習)
sbatch scripts/run_rf_pca_full.sh

# Step 4: SHAP分析
python src/analyze_shap_hybrid.py \
    --model-type pca \
    --api-keywords-path api_keywords_single.json
```

---

## 7. ディレクトリ構成（主要ファイル）

```
/k_data1/i055ueno/reserch/
├── src/
│   ├── generate_api_descriptions.py  # API説明文の定義 (~300 API)
│   ├── make_keyword.py               # TextRankキーワード抽出
│   ├── make_bagofwords.py            # BoW + API頻度 特徴量生成
│   ├── train_bagofwords.py           # Keyword RF学習・評価
│   ├── 2025_RF_PCA.py                # PCAモデル: prepare_data / train_pca
│   ├── 2025_reduce_features.py       # PCA次元削減 (768→128)
│   ├── analyze_shap_hybrid.py        # SHAP分析 (A: API出自 + B: 機能フィルタ)
│   └── common.py                     # 共通ユーティリティ
├── scripts/
│   ├── run_ffri_keyword_bow.sh       # Keyword全パイプライン
│   ├── run_rf_pca_full.sh            # PCA全パイプライン (SLURM)
│   └── ...
├── features/                          # 生成される特徴量CSV
├── logs_keyword/                      # Keywordモデルの出力
├── logs/                              # PCAモデルの出力
├── logs_shap_analysis/                # SHAP分析結果
│   ├── keyword/
│   │   ├── shap_report.html
│   │   ├── overall_report.txt
│   │   └── waterfall_*.png
│   └── pca/
│       └── ...
├── encoded_train_npz/                 # BERT埋め込み (訓練)
├── encoded_test_npz/                  # BERT埋め込み (テスト)
└── reference/
    └── mitre/                         # MITRE ATT&CK関連データ
```

---

## 8. 今後の課題

1. **機能直結キーワードの精度向上**: ホワイトリストの拡充と、キーワード抽出手法自体の改善（TextRankの代替として、機能ラベルを直接教師信号にしたキーワード選定など）
2. **学会向けケーススタディの選定**: F1スコアが高く、SHAPの結果が直感的に解釈できるラベル（`file_read`, `regkey_written` など）を中心に分析を深掘り
3. **CTI連携の強化**: SHAPの結果を MITRE ATT&CK フレームワークと照合し、攻撃の戦術・技術として自動マッピング
