# 機械学習 → SHAP分析 の全フロー解説

## 全体像

```
  検体のAPIコール列（生データ）
         │
    ┌────┴────┐
    ▼         ▼
 Keyword    PCA
  モデル    モデル
    │         │
    ▼         ▼
 Random Forest × 17ラベル（Binary Relevance）
         │
         ▼
    SHAP分析
    「なぜこの機能ラベルが付いたか」を説明
```

---

## Step 1: 特徴量の構築

### 入力データ（両モデル共通）

CAPEサンドボックスで実行した各マルウェア検体のJSON:
```json
{
  "apicalls": ["CreateFileW", "WriteFile", "RegSetValueExW", "CreateFileW", ...],
  "functions": ["file_created", "file_written", "regkey_written"]
}
```
- `apicalls`: 実行時に呼ばれたWindows APIの列（時系列順）
- `functions`: 正解ラベル（17種類のうち該当するもの）

---

### Keywordモデルの特徴量構築

#### ① API説明文 → キーワード抽出

まず各APIに対して英語の説明文を用意してある:
```
CreateFileW    → "Create or open a file or device using Unicode name"
RegSetValueExW → "Set data for a registry value using Unicode name"
```

この説明文に **TextRank**（spaCy + pytextrank）をかけて、名詞・動詞のキーワードを抽出:
```
CreateFileW    → ["file", "device", "name"]
RegSetValueExW → ["registry", "value", "data"]
WriteFile      → ["file", "data", "write"]
DeleteFileW    → ["file", "delete"]
```

> **ポイント**: `"file"` は複数のAPIから共通して出てくる。
> これにより「ファイル操作系APIを多く呼ぶ検体」は `"file"` のカウントが大きくなる。

#### ② Bag-of-Words カウント

ある検体が以下のAPIを呼んだとする:
```
CreateFileW × 50回, WriteFile × 30回, RegSetValueExW × 10回
```

キーワードのカウント（API横断で合算）:
```
"file"     = 50(←CreateFileW) + 30(←WriteFile) = 80
"device"   = 50(←CreateFileW) = 50
"data"     = 30(←WriteFile) + 10(←RegSetValueExW) = 40
"write"    = 30(←WriteFile) = 30
"registry" = 10(←RegSetValueExW) = 10
"value"    = 10(←RegSetValueExW) = 10
"name"     = 50(←CreateFileW) + 10(←RegSetValueExW) = 60
...
```

#### ③ API頻度ベクトル（301次元）も結合

キーワードとは別に、各APIの呼び出し回数そのものも特徴量にする:
```
api__CreateFileW    = 50
api__WriteFile      = 30
api__RegSetValueExW = 10
api__ReadFile       = 0
...（301種類のAPI分）
```

#### ④ 最終的な特徴量ベクトル

```
[キーワードBoW (数百次元)] + [API頻度 (301次元)]
= 1つの検体あたり 数百次元 のベクトル
```

---

### PCAモデルの特徴量構築

#### ① BERT埋め込み（事前計算済み）

各検体のAPIコール列をBERTに通して、768次元の埋め込みベクトルを取得:
```
入力: "CreateFileW WriteFile RegSetValueExW ..."
     ↓ BERT
出力: (文数 × トークン数 × 768) のテンソル
```

#### ② Mean Pooling

テンソル全体の平均をとって 1検体 = 768次元 に圧縮:
```
(文数 × トークン数 × 768) → mean → (768,)
```

> **なぜ平均？**: flatten（平坦化）すると1100万次元に爆発し、
> ゼロパディングだらけで意味のある情報が埋もれてしまう。
> 平均をとることで「文脈全体の意味」を768次元に凝縮できる。

#### ③ PCA次元削減

768次元 → 128次元 に圧縮（累積寄与率 ~69%）

#### ④ API頻度と結合（ハイブリッド化）

Keywordモデルと同じ301次元のAPI頻度を結合:
```
[PCA 128次元] + [API頻度 301次元] = 429次元
```

---

## Step 2: Random Forest によるマルチラベル学習

### Binary Relevance 方式

17ラベルを**それぞれ独立した二値分類問題**として扱う:

```
ラベル "file_read"      → RF分類器 #1:  「file_readか否か」
ラベル "regkey_written"  → RF分類器 #2:  「regkey_writtenか否か」
ラベル "mutex"           → RF分類器 #3:  「mutexか否か」
...
合計 17個の独立した Random Forest を学習
```

### 学習の流れ（各ラベルごと）

```
訓練データ (5,691件)
  │
  ├─ 4分割の層化交差検証 (IterativeStratification)
  │    Fold 1: 訓練3/4 → 検証1/4 → F1スコア算出
  │    Fold 2: 訓練3/4 → 検証1/4 → F1スコア算出
  │    Fold 3: ...
  │    Fold 4: ...
  │    → 4つの平均F1を報告（モデルの汎化性能の推定）
  │
  └─ 全訓練データ (5,691件) で最終モデルを再学習
       → テストデータ (4,644件) で最終評価
       → モデルを .joblib として保存
```

### 各RFの中身

```
RandomForestClassifier(
    n_estimators=100,   # 100本の決定木
    random_state=42,    # 再現性
    n_jobs=-1           # 全CPUコアで並列
)
```

各決定木は特徴量のランダムなサブセットで分岐ルールを学習する。
100本の決定木の多数決で最終予測が決まる。

---

## Step 3: SHAP分析

### SHAPとは

**「各特徴量が予測にどれだけ貢献したか」** を数値化する手法。
ゲーム理論のShapley値に基づき、各特徴量の「公平な貢献度」を計算する。

### TreeExplainer

Random Forestのような木構造モデルに特化した高速SHAP計算アルゴリズム。
全決定木の全分岐パスを辿り、各特徴量のSHAP値を正確に計算する。

### 計算の流れ

```
学習済みRFモデル (file_read.joblib)
  +
テストデータ 500件（ランダムサンプリング）
  │
  ▼ shap.TreeExplainer
  │
  SHAP値の行列: (500サンプル × 特徴量数)
  │
  │ 例: あるサンプルのSHAP値
  │   "file"           = +0.15  ← file_readを「はい」に押し上げた
  │   "registry"       = -0.02  ← file_readを「いいえ」に少し押した
  │   api__CreateFileW = +0.08  ← file_readを「はい」に押し上げた
  │   api__RegOpenKeyExA = -0.01
  │   ...
  │
  ▼ 平均絶対値を計算
  │
  各特徴量の重要度（全サンプル平均）:
    "file"           : 0.0824  ← 全体的に重要
    api__CreateFileW : 0.0543
    "registry"       : 0.0321
    ...
```

### SHAP値の解釈

| SHAP値 | 意味 |
|--------|------|
| **正の値 (+)** | その特徴量が「このラベルに該当する」方向に押した |
| **負の値 (-)** | その特徴量が「このラベルに該当しない」方向に押した |
| **絶対値が大きい** | 予測への影響が大きい（重要な特徴量） |

### Waterfallプロット

1つのサンプルに対して「各特徴量がどれだけ予測を押し上げ/押し下げたか」を
滝グラフとして可視化したもの。SHAPの定番ビジュアライゼーション。

```
E[f(x)] = 0.35  (ベースライン: 全サンプルの平均予測値)
  │
  │  "file" = 80        → +0.15
  │  api__CreateFileW=50 → +0.08
  │  "write" = 30        → +0.05
  │  "registry" = 10     → -0.02
  │  ...
  │
  ▼
f(x) = 0.92  (この検体の最終予測値)
```

---

## Step 4: カテゴリ別の分析

### 2つのカテゴリ

特徴量を2つのグループに分けて、それぞれの重要度割合を算出:

```
特徴量全体
├─ メイン特徴量 (Keyword or PCA)  → 全体の 65%
└─ API頻度特徴量 (301次元)        → 全体の 35%
```

### 方針A: API出自の逆引き

`api_keywords_single.json` を逆方向に引くことで、
SHAPで重要と出たキーワードの「由来API」を特定:

```
SHAP重要キーワード: "file" (重要度: 0.0824)
  ← 由来API: CreateFileW, ReadFile, WriteFile, DeleteFileW, CopyFileW, ...
  → 解釈: この検体はファイル操作系APIを多用していた
```

### 方針B: 機能直結キーワードフィルタ

キーワードを11カテゴリに自動分類し、抽象語を除外:

```
✅ 採用（機能直結）:
  "file"     → File Operation
  "registry" → Registry
  "process"  → Process/Thread
  "connect"  → Network
  "encrypt"  → Crypto

❌ 除外（抽象語）:
  "life_cycle"  → マルウェア機能と無関係
  "option"      → 汎用的すぎる
  "handle"      → OS内部概念
  "parameter"   → 意味が曖昧
```

---

## まとめ: データの旅路

```
マルウェア検体
  │
  │ CAPEサンドボックスで動的解析
  ▼
APIコール列: [CreateFileW, WriteFile, RegSetValueExW, ...]
  │
  │ 特徴量化
  ├─ Keyword: API説明文→TextRank→BoW+API頻度 (数百次元)
  └─ PCA:     BERT→Mean Pooling→PCA+API頻度 (429次元)
  │
  │ 学習 (Binary Relevance × Random Forest × 17ラベル)
  ▼
学習済みモデル (*.joblib)
  │
  │ SHAP (TreeExplainer)
  ▼
各特徴量の貢献度 (SHAP値)
  │
  │ カテゴリ別集計 + API出自逆引き + 機能フィルタ
  ▼
最終レポート:
  「この検体はファイル操作(CreateFileW, WriteFile)と
   レジストリ書き込み(RegSetValueExW)が支配的だった」
```
