# -*- coding: utf-8 -*-
"""
マルウェア検体のデータセット(JSON形式)と、APIごとのキーワードリストから、
Bag-of-Words形式のキーワード特徴量と、マルチホットエンコードされたラベルを生成し、
ファイルに保存するプログラム。（固定ラベルセット使用バージョン）

■ 使い方:
  python make_bagofwords.py [オプション]
  オプション無しの場合は従来のデフォルトパスで実行されます。
  --help で全オプションを確認できます。
"""
import argparse
import json
import os
from collections import Counter
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- 固定ラベルセット ---
# ユーザー指定の17個の機能ラベル
PREDEFINED_LABELS = {
    "command_line": 0, "connects_host": 1, "connects_ip": 2,
    "directory_created": 3, "directory_enumerated": 4, "file_copied": 5,
    "file_created": 6, "file_deleted": 7, "file_failed": 8, "file_read": 9,
    "file_recreated": 10, "file_written": 11, "guid": 12, "mutex": 13,
    "regkey_deleted": 14, "regkey_written": 15, "resolves_host": 16
}


API_FEATURE_PREFIX = "api__"


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
        api_call_counts = Counter(called_apis)
        for api in called_apis:
            if api in api_keywords:
                sample_keyword_counts.update(api_keywords[api])

        feature_vector = [sample_keyword_counts.get(
            keyword, 0) for keyword in vocabulary]
        api_freq_vector = [api_call_counts.get(api, 0) for api in api_list]
        feature_vector.extend(api_freq_vector)
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


def parse_args():
    """コマンドライン引数をパースする。"""
    parser = argparse.ArgumentParser(
        description="キーワードBoW特徴量とマルチホットラベルを生成する。")
    parser.add_argument(
        "--api-keywords", type=str, default="api_keywords_single.json",
        help="APIキーワードJSONファイルのパス (default: api_keywords_single.json)")
    parser.add_argument(
        "--api-list", type=str, default="api.json",
        help="APIリストJSONファイルのパス (default: api.json)")
    parser.add_argument(
        "--train-dir", type=str, default="2024/Dataset_Extract/2016",
        help="訓練データディレクトリ (default: 2024/Dataset_Extract/2016)")
    parser.add_argument(
        "--test-dir", type=str, default="2024/Dataset_Extract/2017",
        help="テストデータディレクトリ (default: 2024/Dataset_Extract/2017)")
    parser.add_argument(
        "--train-filter", type=str, default=None,
        help="訓練データフィルタファイル (default: None)")
    parser.add_argument(
        "--test-filter", type=str, default=None,
        help="テストデータフィルタファイル (default: None)")
    parser.add_argument(
        "--output-dir", type=str, default="features",
        help="出力ディレクトリ (default: features)")
    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_args()

    api_keywords_path = Path(args.api_keywords)
    api_list_path = Path(args.api_list)
    train_data_dir = Path(args.train_dir)
    test_data_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)

    # フィルタパス
    train_filter_path = Path(args.train_filter) if args.train_filter else None
    test_filter_path = Path(args.test_filter) if args.test_filter else None

    # 出力ファイル名
    vocabulary_path = output_dir / "keyword_vocabulary.json"
    api_features_path = output_dir / "api_presence_features.json"
    feature_columns_path = output_dir / "feature_columns.json"
    label_set_path = output_dir / "label_set.json"
    train_features_path = output_dir / "train_keyword_features.csv"
    train_labels_path = output_dir / "train_labels.csv"
    test_features_path = output_dir / "test_keyword_features.csv"
    test_labels_path = output_dir / "test_labels.csv"

    # 出力ディレクトリを作成
    output_dir.mkdir(exist_ok=True)

    # --- 1. APIキーワードの読み込み ---
    print(f"'{api_keywords_path}' からAPIキーワードを読み込んでいます...")
    try:
        with open(api_keywords_path, 'r', encoding='utf-8') as f:
            api_keywords = json.load(f)
    except FileNotFoundError:
        print(f"エラー: {api_keywords_path} が見つかりません。")
        print("先にキーワード抽出プログラムを実行してください。")
        return

    print(f"'{api_list_path}' からAPIリストを読み込んでいます...")
    try:
        with open(api_list_path, 'r', encoding='utf-8') as f:
            api_list = json.load(f)
    except FileNotFoundError:
        print(f"エラー: {api_list_path} が見つかりません。")
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
    with open(vocabulary_path, 'w', encoding='utf-8') as f:
        json.dump(vocabulary, f, indent=4)
    # 固定ラベルセットをjsonファイルとして保存（後続のプログラムが参照するため）
    with open(label_set_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_labels, f, indent=4)

    api_feature_names = [f"{API_FEATURE_PREFIX}{api}" for api in api_list]
    feature_columns = vocabulary + api_feature_names

    with open(api_features_path, 'w', encoding='utf-8') as f:
        json.dump(api_feature_names, f, indent=4)
    with open(feature_columns_path, 'w', encoding='utf-8') as f:
        json.dump(feature_columns, f, indent=4)

    print(f"ユニークなキーワード数: {len(vocabulary)}")
    print(f"API存在特徴数: {len(api_feature_names)}")
    print(f"使用する固定ラベル数: {len(sorted_labels)}")

    # --- 3. 訓練データとテストデータの特徴量・ラベルを生成 ---
    print("\n--- 訓練データの処理 ---")
    train_filtered_indexes = None
    test_filtered_indexes = None
    if train_filter_path is not None:
        train_filtered_indexes = load_filtered_indexes(train_filter_path)
    if test_filter_path is not None:
        test_filtered_indexes = load_filtered_indexes(test_filter_path)

    train_features_df, train_labels_df = process_dataset(
        train_data_dir,
        api_keywords,
        vocabulary,
        api_list,
        feature_columns,
        sorted_labels,
        train_filtered_indexes,
    )

    print("\n--- テストデータの処理 ---")
    test_features_df, test_labels_df = process_dataset(
        test_data_dir,
        api_keywords,
        vocabulary,
        api_list,
        feature_columns,
        sorted_labels,
        test_filtered_indexes,
    )

    # --- 4. ファイルに保存 ---
    print("\n結果をCSVファイルに保存しています...")
    train_features_df.to_csv(train_features_path)
    train_labels_df.to_csv(train_labels_path)
    test_features_df.to_csv(test_features_path)
    test_labels_df.to_csv(test_labels_path)

    print("\n--- 前処理完了 ---")
    print(f"生成されたファイルは '{output_dir}' ディレクトリにあります。")
    print("次のステップ: 'train_bagofwords.py' を実行してモデルを学習・評価してください。")


if __name__ == '__main__':
    main()

