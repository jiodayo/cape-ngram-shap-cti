# 機能直結キーワードのカテゴライズ手法

## 概要

マルウェアの動作を解釈する際、API説明文から抽出されたキーワード群には「機能に直結する単語」と「抽象的で解釈に結びつかない単語（ノイズ）」が混在しています。

これを解決するため、本研究では**マルウェアの代表的な挙動に基づいた11の「機能カテゴリ」**を定義し、該当するキーワードのみを抽出する**ホワイトリスト方式（方針B）**を採用しています。

---

## カテゴリの決定方法（背景と根拠）

本研究で用いる「11の機能カテゴリ」は、単なる恣意的な分類ではなく、**サイバー脅威インテリジェンス（CTI）やマルウェア解析の標準的な枠組み（MITRE ATT&CKやMAEC等）と、OSのサブシステム構造**をベースに決定されています。

### 1. 決定の背景（NLPとCTIのギャップ）
TextRank等の自然言語処理（NLP）アルゴリズムは、コーパス内で「統計的に出現頻度やリンク性が高い単語」を重要キーワードとして抽出します。そのため、`life_cycle` や `option` といった単語が高く評価される傾向があります。
しかし、サイバーセキュリティアナリスト（CTIの観点）が知りたいのは「統計的な重要語」ではなく、**「そのマルウェアがOSのどのサブシステム（ファイル、ネットワーク、プロセス等）にどのような影響（作成、通信、インジェクション等）を及ぼしたか」という機能的・アクション志向な情報**です。
このギャップを埋めるため、NLPの抽出結果に対して「機能に基づくフィルタ」をかける必要がありました。

### 2. 具体的な決定プロセス

1. **マルウェア挙動のタクソノミー（分類体系）の策定**
   Windows OSの主要なサブシステム（File, Registry, Process, Memory, Networkなど）と、マルウェア特有の挙動（Anti-Analysis, Screen/Input盗聴など）を掛け合わせ、解析者が直感的に理解しやすい粒度として**11種類のカテゴリ**を策定しました。これは、動的解析ツール（CAPE SandboxやCuckoo）が採用している振る舞いシグネチャの分類にも準拠しています。

2. **キーワード群のボトムアップ検証**
   全API説明文からTextRankで抽出された数百種類のキーワード群を独自にレビューしました。
   - `read`, `write`, `delete` → **File Operation** へ
   - `inject`, `thread`, `execute` → **Process/Thread** へ
   - このように、CTIアナリストの視点で「明らかにマルウェアの特定の機能・目的に直結する単語」を手動でマッピングし、辞書（`FUNCTIONAL_KEYWORD_CATEGORIES`）として構築しました。

3. **自動フィルタリングによるノイズ除去**
   SHAP分析時、この辞書に登録されているキーワードのみを「機能直結キーワード」として評価に採用します。結果として、**辞書にマッピングされなかった単語（`life_cycle`, `handle`, `parameter` など）は「抽象語（ノイズ）」として自動的に棄却**される仕組みになっています。

---

## 11の機能カテゴリと該当キーワード一覧

| カテゴリ名 | 該当するキーワードの例 | 意味合い |
|:---|:---|:---|
| **File Operation**<br>（ファイル操作） | `file`, `read`, `write`, `delete`, `copy`, `create`, `open`, `directory`, `path`, `download` 等 | ファイルやディレクトリの作成・読み書き・削除・検索など |
| **Registry**<br>（レジストリ） | `registry`, `key`, `value`, `subkey`, `enumerate` | レジストリキーや値の読み書き・列挙 |
| **Process/Thread**<br>（プロセス/スレッド） | `process`, `thread`, `execute`, `terminate`, `inject`, `module`, `dll`, `load` 等 | プロセスの起動・停止、スレッド操作、DLLインジェクションなど |
| **Network**<br>（ネットワーク） | `connect`, `send`, `receive`, `socket`, `server`, `host`, `url`, `http`, `dns`, `port` 等 | 通信の確立、データ送受信、名前解決など |
| **Memory**<br>（メモリ） | `memory`, `allocate`, `protect`, `virtual`, `heap`, `map`, `page` 等 | メモリ領域の確保、保護属性の変更など |
| **Service**<br>（サービス） | `service`, `driver`, `install`, `start`, `control` | Windowsサービスやドライバのインストール・制御 |
| **Crypto**<br>（暗号化） | `encrypt`, `decrypt`, `hash`, `certificate`, `cipher` | データの暗号化・復号、ハッシュ計算、証明書操作 |
| **Security**<br>（セキュリティ/権限） | `token`, `privilege`, `credential`, `impersonate`, `account` | 権限昇格、アクセストークン操作、認証情報の窃取など |
| **System Info**<br>（システム情報） | `system`, `computer`, `user`, `version`, `environment`, `disk` | OSバージョン、環境変数、ディスク情報などの取得 |
| **Screen/Input**<br>（画面/入力） | `screen`, `capture`, `keyboard`, `hook`, `input`, `clipboard` | キーロガー（フック）、画面キャプチャ、クリップボード盗聴など |
| **Anti-Analysis**<br>（解析妨害） | `debugger`, `debug`, `evasion`, `detect` | デバッガの検知など、サンドボックスや解析環境を回避する挙動 |

---

## 除外される抽象語（ノイズ）の例

ホワイトリストに該当しない以下の様な単語は、SHAP分析のメイン結果からは除外され、「参考」として別枠で集計されます。

- **OS内部概念:** `handle`, `pointer`, `descriptor`
- **汎用的な名詞:** `option`, `parameter`, `information`, `state`
- **マルウェアの機能に直結しない語:** `life_cycle`, `interface`, `component`
