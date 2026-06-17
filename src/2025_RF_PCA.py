import os
import numpy as np
import json
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from skmultilearn.model_selection import IterativeStratification
import joblib
import sys

# --- 設定 ---
# 2025_learning_br.py を参考に、最大文数を140に固定
MAX_SENTENCES = 140
NUM_FOLDS = 4  # 交差検証の分割数

# --- パス設定 ---
TRAIN_NPZ_DIR = "encoded_train_npz"
TEST_NPZ_DIR = "encoded_test_npz"
# ラベルセット生成とサンプル数確認のために使用
TRAIN_JSON_PATH = "./2024/Dataset_Extract/2016"
TEST_JSON_PATH = "./2024/Dataset_Extract/2017"

# --- 出力ファイル設定 ---
# prepareモードで生成されるファイル
TRAIN_MEMMAP_PATH = "encoded_train_features.mmap"
TEST_MEMMAP_PATH = "encoded_test_features.mmap"
TRAIN_LABELS_PATH = "encoded_train_labels.npy"
TEST_LABELS_PATH = "encoded_test_labels.npy"
LABEL_SET_PATH = "label_set_pca.json"

# train_pcaモードで使用するファイル (2025_reduce_features.pyが生成)
TRAIN_MEMMAP_PCA_PATH = "encoded_train_features_pca.mmap"
TEST_MEMMAP_PCA_PATH = "encoded_test_features_pca.mmap"


def load_dataset(data_path):
    """
    指定されたパスからJSONファイルを読み込み、データセットを返す関数
    (2025_learning_br.pyから流用)
    """
    data = []
    for filename in tqdm(os.listdir(data_path), desc=f"ファイルをロード中 ({os.path.basename(data_path)})"):
        if filename.endswith(".json"):
            with open(os.path.join(data_path, filename), "r", errors="ignore") as f:
                data.append(json.load(f))
    return data


def prepare_features_and_labels(npz_dir, json_dir, max_sentences, features_memmap_path, labels_npy_path, label_set, labels_only=False):
    """
    .npzファイルから特徴量とラベルを読み込み、memmapとnpyファイルに保存する関数
    - labels_only=Trueの場合、特徴量のmemmap生成をスキップし、ラベルのみを更新します（高速）
    """
    file_list = sorted(
        [f for f in os.listdir(npz_dir) if f.endswith(".npz")])
    num_samples = len(file_list)

    if num_samples == 0:
        print(f"警告: {npz_dir} に .npz ファイルが見つかりません。")
        return

    # ラベルの準備
    labels_array = np.zeros((num_samples, len(label_set)), dtype=np.int8)

    if not labels_only:
        # Mean Pooling により特徴量は常に 768次元 になる
        flattened_dim = 768

        features_memmap = np.memmap(
            features_memmap_path, dtype='float32', mode='w+', shape=(num_samples, flattened_dim))

    # JSONファイルのリストをアルファベット順にソートして事前に取得しておく (マッピング用)
    try:
        json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
    except FileNotFoundError:
        print(f"警告: {json_dir} が見つかりません。")
        json_files = []

    # 並列処理用の関数定義
    def process_and_write(i, filename):
        stem = filename.replace(".npz", "")
        if stem.startswith("sample_") and stem[len("sample_"):].isdigit():
            sample_idx = int(stem[len("sample_"):])
            if sample_idx < len(json_files):
                json_filename = json_files[sample_idx]
            else:
                json_filename = None
        else:
            json_filename = filename.replace(".npz", ".json")
            
        sample_labels = []
        if json_filename is not None:
            json_path = os.path.join(json_dir, json_filename)
            if os.path.exists(json_path):
                with open(json_path, "r", errors="ignore") as f:
                    json_data = json.load(f)
                    sample_labels = json_data.get("functions", [])
            else:
                print(f"警告: {json_path} が見つかりません。ラベルは空になります。")
        else:
            print(f"警告: {filename} に対応する JSON ファイルが特定できません。")

        if not labels_only:
            data = np.load(os.path.join(npz_dir, filename))
            embedding = data["embedding"]
            # Mean Pooling: (num_sentences, tokens, 768) -> (768,)
            # 全文・全トークンの平均をとることで、文脈全体の意味を表現する密なベクトルを得る
            pooled_embedding = np.mean(embedding, axis=(0, 1))
            features_memmap[i] = pooled_embedding

        for label in sample_labels:
            if label in label_set:
                labels_array[i, label_set[label]] = 1

    # 並列処理で全ファイルを処理 (I/Oバウンドなのでthreadingが有効)
    from joblib import Parallel, delayed
    Parallel(n_jobs=16, backend="threading")(
        delayed(process_and_write)(i, filename)
        for i, filename in enumerate(tqdm(file_list, desc=f"データを処理中 ({os.path.basename(npz_dir)})"))
    )

    # ラベル配列を.npyファイルに保存
    np.save(labels_npy_path, labels_array)
    if not labels_only:
        features_memmap.flush()
        print(f"特徴量を {features_memmap_path} に保存しました。Shape: {features_memmap.shape}")
    print(f"ラベルを {labels_npy_path} に保存しました。Shape: {labels_array.shape}")


def run_br_with_rf_cv(train_features, train_labels, test_features, test_labels, label_set, n_estimators=100, n_jobs=-1, num_folds=NUM_FOLDS):
    """
    Random Forestを用いたBinary Relevance分類器の学習と評価を、交差検証を付けて実行する
    """
    # 2025_learning_br.pyに倣ってログとモデルの保存先ディレクトリを作成
    os.makedirs("logs/models_br_rf", exist_ok=True)
    os.makedirs("logs/fold_reports_br_rf", exist_ok=True)
    os.makedirs("logs/final_reports_br_rf", exist_ok=True)

    # 全ラベルに対するテストの予測結果と正解ラベルを格納する
    y_true_all = []
    y_pred_all = []
    final_accuracies = {}  # 各ラベルの最終的な正解率を保存する辞書

    # ラベルごとにモデルを学習
    for label_name, label_idx in tqdm(label_set.items(), desc="各ラベルでRFを学習・評価中"):

        y_train_single = train_labels[:, label_idx]
        y_test_single = test_labels[:, label_idx]

        # --- 層化K分割交差検証 ---
        # skmultilearnはyが2次元配列であることを期待するため、reshapeする
        y_train_single_2d = y_train_single.reshape(-1, 1)
        kf = IterativeStratification(
            n_splits=num_folds, order=1)

        fold_f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_features, y_train_single_2d)):
            X_train_fold, X_val_fold = train_features[train_idx], train_features[val_idx]
            y_train_fold, y_val_fold = y_train_single[train_idx], y_train_single[val_idx]

            # モデルの学習 (パラメータを引数で指定)
            rf_fold_model = RandomForestClassifier(
                n_estimators=n_estimators, random_state=42, n_jobs=n_jobs)
            rf_fold_model.fit(X_train_fold, y_train_fold)

            # 検証データで評価
            y_val_pred = rf_fold_model.predict(X_val_fold)
            f1 = f1_score(y_val_fold, y_val_pred,
                          average="binary", zero_division=0)
            fold_f1_scores.append(f1)

            # 各foldのレポートを保存
            fold_report = classification_report(
                y_val_fold, y_val_pred,
                labels=[0, 1],
                target_names=[f"not_{label_name}", label_name],
                zero_division=0
            )
            with open(f"logs/fold_reports_br_rf/{label_name}_fold{fold}_report.txt", "w") as f:
                f.write(fold_report)

        print(
            f"  [BR CV] Label: {label_name}, Average F1 over {num_folds} folds: {np.mean(fold_f1_scores):.4f}")

        # --- 全訓練データで再学習し、最終評価 ---
        print(f"  [BR Final] Label: {label_name}, 全訓練データで再学習中...")
        final_rf_model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, n_jobs=n_jobs)
        final_rf_model.fit(train_features, y_train_single)

        # 学習済みモデルを保存
        model_path = f"logs/models_br_rf/{label_name}.joblib"
        joblib.dump(final_rf_model, model_path)

        # テストデータで予測
        y_pred_single = final_rf_model.predict(test_features)

        # 正解率を計算
        acc_single = accuracy_score(y_test_single, y_pred_single)
        final_accuracies[label_name] = acc_single

        # 最終評価レポートの生成と保存
        final_report = classification_report(
            y_test_single,
            y_pred_single,
            labels=[0, 1],
            target_names=[f"not_{label_name}", label_name],
            zero_division=0
        )

        with open(f"logs/final_reports_br_rf/{label_name}_final_report.txt", "w") as f:
            f.write(f"--- Final Report for Label: {label_name} ---\n")
            f.write(f"(CV Average F1: {np.mean(fold_f1_scores):.4f})\n")
            f.write(f"(Final Test Accuracy: {acc_single:.4f})\n\n")
            f.write(final_report)

        y_true_all.append(y_test_single)
        y_pred_all.append(y_pred_single)

    # 全ラベルをまとめた最終評価
    print("\n--- Overall Final Report ---")
    # y_true_allとy_pred_allを (サンプル数, ラベル数) の形状に変換
    y_true_all = np.array(y_true_all).T
    y_pred_all = np.array(y_pred_all).T

    overall_report = classification_report(
        y_true_all,
        y_pred_all,
        target_names=list(label_set.keys()),
        zero_division=0
    )
    print(overall_report)

    # --- 全ラベルの正解率をまとめて表示・保存 ---
    print("\n--- Overall Per-label Accuracy ---")
    overall_acc_report_str = ""
    for label_name, acc in final_accuracies.items():
        line = f"{label_name}: {acc:.4f}\n"
        print(line, end="")
        overall_acc_report_str += line

    with open("logs/final_reports_br_rf/overall_final_report.txt", "w") as f:
        f.write("--- Overall Classification Report ---\n")
        f.write(overall_report)
        f.write("\n\n--- Overall Per-label Accuracy ---\n")
        f.write(overall_acc_report_str)

    print("\n[BR with RF] 完了: 全ラベルの学習・評価が完了しました。")
    print("モデルは 'logs/models_br_rf/' に、レポートは 'logs/final_reports_br_rf/' に保存されました。")


def main_prepare_data(labels_only=False):
    """
    モード1: 特徴量とラベルの前処理と保存
    """
    print(f"--- モード: prepare_{'labels' if labels_only else 'data'} ---")
    # 2025_learning_br.pyからパスと設定を流用
    train_npz_dir = TRAIN_NPZ_DIR
    test_npz_dir = TEST_NPZ_DIR

    # ユーザー指定の17個の機能ラベル
    PREDEFINED_LABELS = {
        "command_line": 0, "connects_host": 1, "connects_ip": 2,
        "directory_created": 3, "directory_enumerated": 4, "file_copied": 5,
        "file_created": 6, "file_deleted": 7, "file_failed": 8, "file_read": 9,
        "file_recreated": 10, "file_written": 11, "guid": 12, "mutex": 13,
        "regkey_deleted": 14, "regkey_written": 15, "resolves_host": 16
    }
    
    label_set = PREDEFINED_LABELS

    print(f"対象ラベル数: {len(label_set)}")
    with open(LABEL_SET_PATH, 'w') as f:
        json.dump(label_set, f, indent=4)
    print(f"ラベルセットを {LABEL_SET_PATH} に保存しました。")

    # 訓練データとテストデータの特徴量とラベルを準備・保存
    print(f"\n訓練データの特徴量とラベルを準備中 (labels_only={labels_only})...")
    prepare_features_and_labels(
        train_npz_dir, TRAIN_JSON_PATH, MAX_SENTENCES, TRAIN_MEMMAP_PATH, TRAIN_LABELS_PATH, label_set, labels_only=labels_only)

    print(f"\nテストデータの特徴量とラベルを準備中 (labels_only={labels_only})...")
    prepare_features_and_labels(
        test_npz_dir, TEST_JSON_PATH, MAX_SENTENCES, TEST_MEMMAP_PATH, TEST_LABELS_PATH, label_set, labels_only=labels_only)

    print(f"\n--- prepare_{'labels' if labels_only else 'data'} モード完了 ---")
    print("次のステップ:")
    if labels_only:
        print(f"PCA特徴量が存在する場合は、そのまま学習・評価を実行できます: python {sys.argv[0]} train_pca")
    else:
        print(f"1. PCAによる特徴量削減を実行してください: python 2025_reduce_features.py")
        print(f"2. PCA適用後の特徴量で学習・評価を実行してください: python {sys.argv[0]} train_pca")


def main_train_pca():
    """
    モード2: PCA適用後の特徴量で学習・評価
    """
    print("--- モード: train_pca ---")
    # 必要なファイルが存在するかチェック
    required_files = [
        TRAIN_MEMMAP_PCA_PATH, TEST_MEMMAP_PCA_PATH,
        TRAIN_LABELS_PATH, TEST_LABELS_PATH, LABEL_SET_PATH
    ]
    for f_path in required_files:
        if not os.path.exists(f_path):
            print(f"エラー: 必要なファイルが見つかりません: {f_path}")
            print("先に 'prepare_data' モードと '2025_reduce_features.py' を実行してください。")
            return

    # ラベルセットをロード
    with open(LABEL_SET_PATH, 'r') as f:
        label_set = json.load(f)
    print(f"ラベルセットを {LABEL_SET_PATH} から読み込みました。")

    # ラベルをロード
    Y_train = np.load(TRAIN_LABELS_PATH)
    Y_test = np.load(TEST_LABELS_PATH)

    import pandas as pd

    # PCA適用済みの特徴量をmemmapでロード
    # まずは形状を特定するために、npyからサンプル数を取得
    num_train_samples = Y_train.shape[0]
    num_test_samples = Y_test.shape[0]

    # PCA後の次元数をファイルサイズから計算 (次元数は整数のはず)
    pca_feature_size = os.path.getsize(TRAIN_MEMMAP_PCA_PATH)
    pca_dim = pca_feature_size // (num_train_samples *
                                   np.dtype('float32').itemsize)
    if pca_feature_size % (num_train_samples * np.dtype('float32').itemsize) != 0:
        print("警告: PCA特徴量ファイルのサイズが不正です。次元数を正しく計算できない可能性があります。")

    print(f"PCA適用後の特徴量次元数 (PCA単体): {pca_dim}")

    X_train_pca = np.memmap(TRAIN_MEMMAP_PCA_PATH, dtype='float32',
                            mode='r', shape=(num_train_samples, pca_dim))
    X_test_pca = np.memmap(TEST_MEMMAP_PCA_PATH, dtype='float32',
                           mode='r', shape=(num_test_samples, pca_dim))

    # --- API頻度特徴量(301次元)とのハイブリッド化 ---
    train_csv_path = "features/train_keyword_features.csv"
    test_csv_path = "features/test_keyword_features.csv"

    if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
        print("API頻度特徴量 (301次元) を結合してハイブリッド化します...")
        train_df = pd.read_csv(train_csv_path, index_col=0)
        test_df = pd.read_csv(test_csv_path, index_col=0)
        
        # NPZファイルからサンプルのID（ファイル名の拡張子抜き）を取得し、結合順序を合わせる
        train_npz_files = sorted([f for f in os.listdir("encoded_train_npz") if f.endswith(".npz")])
        train_stems = [f.replace(".npz", "") for f in train_npz_files]
        
        test_npz_files = sorted([f for f in os.listdir("encoded_test_npz") if f.endswith(".npz")])
        test_stems = [f.replace(".npz", "") for f in test_npz_files]
        
        # フィルタリングされて欠落しているサンプルがある場合は 0 で埋める
        train_df = train_df.reindex(train_stems).fillna(0)
        test_df = test_df.reindex(test_stems).fillna(0)
        
        api_cols = [col for col in train_df.columns if col.startswith("api__")]
        print(f"抽出したAPI特徴量次元数: {len(api_cols)}")
        
        # PCA特徴量の名前を定義
        pca_feature_names = [f"pca_{i}" for i in range(pca_dim)]
        
        # 結合した特徴量名リストを作成し、後でSHAP分析に使えるように保存
        hybrid_feature_names = pca_feature_names + api_cols
        os.makedirs("logs/models_br_rf", exist_ok=True)
        with open("logs/models_br_rf/hybrid_feature_names.json", "w") as f:
            json.dump(hybrid_feature_names, f, indent=4)
        print("ハイブリッド特徴量名を logs/models_br_rf/hybrid_feature_names.json に保存しました。")
        
        # 結合処理 (メモリ上)
        X_train_api = train_df[api_cols].values.astype('float32')
        X_test_api = test_df[api_cols].values.astype('float32')
        
        X_train_hybrid = np.concatenate([X_train_pca, X_train_api], axis=1)
        X_test_hybrid = np.concatenate([X_test_pca, X_test_api], axis=1)
    else:
        print(f"警告: {train_csv_path} または {test_csv_path} が見つからないため、ハイブリッド化をスキップしPCA単独で実行します。")
        X_train_hybrid = np.array(X_train_pca)
        X_test_hybrid = np.array(X_test_pca)

    print(f"最終的な訓練データの形状: X={X_train_hybrid.shape}, Y={Y_train.shape}")
    print(f"最終的なテストデータの形状: X={X_test_hybrid.shape}, Y={Y_test.shape}")

    # Binary Relevance with Random Forest を実行 (PCA後は性能重視のパラメータで)
    run_br_with_rf_cv(X_train_hybrid, Y_train, X_test_hybrid, Y_test,
                      label_set, n_estimators=100, n_jobs=-1)

    print("\n--- train_pca モード完了 ---")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["prepare_data", "prepare_labels", "train_pca"]:
        print(f"使い方: python {sys.argv[0]} [mode]")
        print("  mode:")
        print("    prepare_data   : 特徴量とラベルを前処理してファイルに保存します。")
        print("    prepare_labels : 特徴量の生成をスキップし、ラベル(17種)のみを高速に更新します。")
        print("    train_pca      : PCAで次元削減された特徴量を使って学習・評価します。")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "prepare_data":
        main_prepare_data(labels_only=False)
    elif mode == "prepare_labels":
        main_prepare_data(labels_only=True)
    elif mode == "train_pca":
        main_train_pca()
