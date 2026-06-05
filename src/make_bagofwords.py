# -*- coding: utf-8 -*-
"""
マルウェア検体のデータセット(JSON形式)と、APIごとのキーワードリストから、
Bag-of-Words形式のキーワード特徴量と、マルチホットエンコードされたラベルを生成し、
ファイルに保存するプログラム。（固定ラベルセット使用バージョン）

■ 使い方:
1. このファイルと同じ階層に、以下のファイルを配置します。
   - "api_keywords_single.json" (キーワード抽出プログラムで作成)
   - "2024/Dataset_Extract/2016/" (学習用データセットのディレクトリ)
   - "2024/Dataset_Extract/2017/" (テスト用データセットのディレクトリ)
2. コマンドラインで `python make_bagofwords.py` を実行します。
3. "features/" ディレクトリ以下に、学習と評価に使用するファイルが出力されます。
"""
import json
import os
from collections import Counter
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- 設定 ---
API_KEYWORDS_PATH = Path("api_keywords_single.json")
API_LIST_PATH = Path("api.json")
TRAIN_DATA_DIR = Path("2024/Dataset_Extract/2016")
TEST_DATA_DIR = Path("2024/Dataset_Extract/2017")
TRAIN_FILTER_PATH = Path("filtered_indexes_2016.txt")
TEST_FILTER_PATH = Path("filtered_indexes_2017.txt")

# 出力ディレクトリ
OUTPUT_DIR = Path("features")
# 出力ファイル名
VOCABULARY_PATH = OUTPUT_DIR / "keyword_vocabulary.json"
API_FEATURES_PATH = OUTPUT_DIR / "api_presence_features.json"
FEATURE_COLUMNS_PATH = OUTPUT_DIR / "feature_columns.json"
LABEL_SET_PATH = OUTPUT_DIR / "label_set.json"
TRAIN_FEATURES_PATH = OUTPUT_DIR / "train_keyword_features.csv"
TRAIN_LABELS_PATH = OUTPUT_DIR / "train_labels.csv"
TEST_FEATURES_PATH = OUTPUT_DIR / "test_keyword_features.csv"
TEST_LABELS_PATH = OUTPUT_DIR / "test_labels.csv"

# --- 固定ラベルセット ---
# ユーザー指定の17個の機能ラベル
PREDEFINED_LABELS = {
    "command_line": 0, "connects_host": 1, "connects_ip": 2,
    "directory_created": 3, "directory_enumerated": 4, "file_copied": 5,
    "file_created": 6, "file_deleted": 7, "file_failed": 8, "file_read": 9,
    "file_recreated": 10, "file_written": 11, "guid": 12, "mutex": 13,
    "regkey_deleted": 14, "regkey_written": 15, "resolves_host": 16
}


def load_filtered_indexes(path):
    """番号リストファイルからインデックス集合を読み込む。存在しない場合はNoneを返す。"""
    if not path.exists():
        print(f"警告: '{path}' が見つからなかったため、全検体を使用します。")
        return None

    indexes = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                indexes.add(int(line))
            except ValueError:
                print(
                    f"  - 警告: フィルタ '{path}' に数値化できない行 '{line}' がありました。無視します。")

    print(f"フィルタ '{path}' から {len(indexes)} 件のインデックスを読み込みました。")
    return indexes


API_FEATURE_PREFIX = "api__"


def process_dataset(
    data_dir,
    api_keywords,
    vocabulary,
    api_list,
    feature_columns,
    label_set,
    filtered_indexes=None,
):
    """
    指定されたディレクトリのデータセットを処理し、
    特徴量ベクトルとラベルのDataFrameを返す。
    filtered_indexes が指定された場合、その番号の検体のみを対象とする。
    """
    feature_vectors = []
    label_vectors = []
    sample_names = []

    file_list = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".json")])

    skipped_count = 0
    fallback_count = 0

    for idx, filename in enumerate(tqdm(file_list, desc=f"Processing {data_dir.name}")):
        sample_path = data_dir / filename
        with open(sample_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # フィルタ適用
        if filtered_indexes is not None:
            sample_index = None
            stem = Path(filename).stem
            if stem.isdigit():
                sample_index = int(stem)
            elif stem.startswith("sample_"):
                num_part = stem[len("sample_"):]
                if num_part.isdigit():
                    sample_index = int(num_part)

            if sample_index is None:
                sample_index = idx
                fallback_count += 1

            if sample_index not in filtered_indexes:
                skipped_count += 1
                continue

        # --- 特徴量ベクトルの作成 ---
        called_apis = data.get("apicalls", [])
        sample_keyword_counts = Counter()
        called_api_set = set(called_apis)
        for api in called_apis:
            if api in api_keywords:
                sample_keyword_counts.update(api_keywords[api])

        feature_vector = [sample_keyword_counts.get(
            keyword, 0) for keyword in vocabulary]
        api_presence_vector = [
            1 if api in called_api_set else 0 for api in api_list]
        feature_vector.extend(api_presence_vector)
        feature_vectors.append(feature_vector)

        # --- ラベルベクトルの作成 ---
        sample_labels = data.get("functions", [])
        # PREDEFINED_LABELSに含まれる機能のみを対象とする
        label_vector = [
            1 if label in sample_labels else 0 for label in label_set]
        label_vectors.append(label_vector)

        # 検体名を保存（ファイル名から拡張子を除外）
        sample_names.append(Path(filename).stem)

    # DataFrameに変換
    features_df = pd.DataFrame(
        feature_vectors, columns=feature_columns, index=sample_names)
    labels_df = pd.DataFrame(
        label_vectors, columns=label_set, index=sample_names)

    if filtered_indexes is not None:
        print(
            f"  -> フィルタ適用により {skipped_count} 件を除外、残り {len(features_df)} 件を使用")
        if fallback_count > 0:
            print(
                f"     (補足: ファイル名に番号が含まれず {fallback_count} 件で列挙順インデックスを使用)")

    return features_df, labels_df


def main():
    """メイン処理"""
    # 出力ディレクトリを作成
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- 1. APIキーワードの読み込み ---
    print(f"'{API_KEYWORDS_PATH}' からAPIキーワードを読み込んでいます...")
    try:
        with open(API_KEYWORDS_PATH, 'r', encoding='utf-8') as f:
            api_keywords = json.load(f)
    except FileNotFoundError:
        print(f"エラー: {API_KEYWORDS_PATH} が見つかりません。")
        print("先にキーワード抽出プログラムを実行してください。")
        return

    print(f"'{API_LIST_PATH}' からAPIリストを読み込んでいます...")
    try:
        with open(API_LIST_PATH, 'r', encoding='utf-8') as f:
            api_list = json.load(f)
    except FileNotFoundError:
        print(f"エラー: {API_LIST_PATH} が見つかりません。")
        print("APIの存在特徴を追加するためにリストが必要です。")
        return

    # --- 2. 語彙リストの作成と、固定ラベルセットの使用 ---

    # 語彙リストの作成 (変更なし)
    vocabulary_set = set()
    for keywords in api_keywords.values():
        vocabulary_set.update(keywords)
    vocabulary = sorted(list(vocabulary_set))

    # 固定ラベルセットを使用
    sorted_labels = sorted(list(PREDEFINED_LABELS.keys()))

    # 保存
    with open(VOCABULARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocabulary, f, indent=4)
    # 固定ラベルセットをjsonファイルとして保存（後続のプログラムが参照するため）
    with open(LABEL_SET_PATH, 'w', encoding='utf-8') as f:
        json.dump(sorted_labels, f, indent=4)

    api_feature_names = [f"{API_FEATURE_PREFIX}{api}" for api in api_list]
    feature_columns = vocabulary + api_feature_names

    with open(API_FEATURES_PATH, 'w', encoding='utf-8') as f:
        json.dump(api_feature_names, f, indent=4)
    with open(FEATURE_COLUMNS_PATH, 'w', encoding='utf-8') as f:
        json.dump(feature_columns, f, indent=4)

    print(f"ユニークなキーワード数: {len(vocabulary)}")
    print(f"API存在特徴数: {len(api_feature_names)}")
    print(f"使用する固定ラベル数: {len(sorted_labels)}")

    # --- 3. 訓練データとテストデータの特徴量・ラベルを生成 ---
    print("\n--- 訓練データの処理 ---")
    train_filtered_indexes = load_filtered_indexes(TRAIN_FILTER_PATH)
    test_filtered_indexes = load_filtered_indexes(TEST_FILTER_PATH)

    train_features_df, train_labels_df = process_dataset(
        TRAIN_DATA_DIR,
        api_keywords,
        vocabulary,
        api_list,
        feature_columns,
        sorted_labels,
        train_filtered_indexes,
    )

    print("\n--- テストデータの処理 ---")
    test_features_df, test_labels_df = process_dataset(
        TEST_DATA_DIR,
        api_keywords,
        vocabulary,
        api_list,
        feature_columns,
        sorted_labels,
        test_filtered_indexes,
    )

    # --- 4. ファイルに保存 ---
    print("\n結果をCSVファイルに保存しています...")
    train_features_df.to_csv(TRAIN_FEATURES_PATH)
    train_labels_df.to_csv(TRAIN_LABELS_PATH)
    test_features_df.to_csv(TEST_FEATURES_PATH)
    test_labels_df.to_csv(TEST_LABELS_PATH)

    print("\n--- 前処理完了 ---")
    print(f"生成されたファイルは '{OUTPUT_DIR}' ディレクトリにあります。")
    print("次のステップ: '2_train_with_keywords.py' を実行してモデルを学習・評価してください。")


if __name__ == '__main__':
    main()
ain()
