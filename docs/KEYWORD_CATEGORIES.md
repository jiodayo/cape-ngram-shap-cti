# 機能直結キーワードのカテゴライズ手法

## 概要

マルウェアの動作を解釈する際、API説明文から抽出されたキーワード群には「機能に直結する単語」と「抽象的で解釈に結びつかない単語（ノイズ）」が混在しています。

これを解決するため、本研究では**マルウェアの代表的な挙動に基づいた11の「機能カテゴリ」**を定義し、該当するキーワードのみを抽出する**ホワイトリスト方式（方針B）**を採用しています。

---

## カテゴリの決定方法（ホワイトリスト方式）

1. **マルウェアの代表的な挙動の洗い出し**
   ファイル操作、レジストリ書き換え、通信、プロセス操作など、マルウェア解析において着目すべき主要な動作をベースに**11カテゴリ**を定義。
2. **キーワードのマッピング（辞書化）**
   API説明文から抽出されたキーワード群を目視で確認し、マルウェアの動作を直接表す具体的な単語を各カテゴリに分類して辞書（`FUNCTIONAL_KEYWORD_CATEGORIES`）を構築。
3. **自動フィルタリングによるノイズ除去**
   SHAP分析時、この辞書に登録されているキーワードのみを「機能直結キーワード」として採用し、**辞書にない単語（`life_cycle`, `option`, `handle` など）は「抽象語」として自動的に除外**する。

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
