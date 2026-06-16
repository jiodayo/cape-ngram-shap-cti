import os
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import joblib

# --- 設定 ---
# 入力するmemmapファイル
TRAIN_MEMMAP_PATH = "encoded_train_features.mmap"
TEST_MEMMAP_PATH = "encoded_test_features.mmap"

# 出力するmemmapファイル
TRAIN_PCA_MEMMAP_PATH = "encoded_train_features_pca.mmap"
TEST_PCA_MEMMAP_PATH = "encoded_test_features_pca.mmap"

# PCAモデルの保存先
PCA_MODEL_PATH = "logs/ipca_model.joblib"

# 削減後の次元数
PCA_COMPONENTS = 256
BATCH_SIZE = 512  # 一度に処理するサンプル数


def get_memmap_shape(memmap_path):
    """
    ファイル名からmemmapの形状とデータ型を推測するヘルパー関数。
    注意: この方法は、ファイル名に '_{shape}_{dtype}.mmap' のような
    情報が含まれている場合にのみ機能します。
    今回は形状が不明なため、まず全データを読み込んで形状を取得します。
    より良い方法は、特徴量生成時に形状を保存しておくことです。
    """
    # 今回は形状が不明なため、一度読み込んで形状を取得します。
    # これは非効率ですが、現在のファイル構造から形状を知る唯一の方法です。
    temp_memmap = np.memmap(memmap_path, dtype='float32', mode='r')
    # 特徴量の次元数は、ファイルサイズをサンプル数で割ることで計算できるが、
    # サンプル数が不明。そのため、最初の特徴量ファイルから次元数を取得する。
    # ここでは、平坦化後の次元数が 140 * 32 * 768 = 3440640 であると仮定します。
    # (MAX_SENTENCES * トークン数 * 埋め込み次元)
    # この値は 2025_learning_RF.py の prepare_features_for_rf から取得するのが最も正確です。
    # 実際の次元数に合わせて調整してください。
    # 仮に訓練データが10000サンプルあるとすると...
    # このアプローチは不安定なので、学習スクリプトから形状を直接参照するのが望ましい。
    # 今回は、学習スクリプト側で形状が分かっているので、それを直接使います。
    # このファイルは単独で実行されるため、形状を知る必要があります。
    # 最初の .npz ファイルから次元を計算します。
    train_npz_dir = "encoded_train_npz"
    file_list = sorted([f for f in os.listdir(
        train_npz_dir) if f.endswith(".npz")])
    data = np.load(os.path.join(train_npz_dir, file_list[0]))
    embedding = data["embedding"]
    # (MAX_SENTENCES, トークン数, 埋め込み次元)
    feature_shape = (140, embedding.shape[1], embedding.shape[2])
    flattened_dim = np.prod(feature_shape)

    num_samples = len(os.listdir(train_npz_dir))  # サンプル数
    return (num_samples, flattened_dim)


def main():
    """
    メイン関数
    """
    os.makedirs("logs", exist_ok=True)

    # --- 訓練データでPCAを学習し、変換する ---
    print("訓練データの特徴量を読み込みます...")
    # 形状を正しく取得するために、一度読み込みます。
    # 注意：巨大なファイルの場合、ここが遅くなる可能性があります。
    X_train_original = np.memmap(TRAIN_MEMMAP_PATH, dtype='float32', mode='r')
    # 形状を (サンプル数, 特徴量次元数) にリシェイプ
    # サンプル数はファイルリストから取得
    num_train_samples = len([f for f in os.listdir(
        "encoded_train_npz") if f.endswith(".npz")])
    X_train = X_train_original.reshape(num_train_samples, -1)
    print(f"訓練データの形状: {X_train.shape}")

    print(f"IncrementalPCAの学習を開始 (n_components={PCA_COMPONENTS})...")
    ipca = IncrementalPCA(n_components=PCA_COMPONENTS, batch_size=BATCH_SIZE)

    # バッチ処理でPCAモデルを学習
    for i in tqdm(range(0, X_train.shape[0], BATCH_SIZE), desc="IPCA fitting"):
        ipca.partial_fit(X_train[i:i + BATCH_SIZE])

    print("学習済みPCAモデルを保存します...")
    joblib.dump(ipca, PCA_MODEL_PATH)
    print(f"モデルを {PCA_MODEL_PATH} に保存しました。")

    print("訓練データをPCAで変換します...")
    # 変換後のデータを保存する新しいmemmapファイルを作成
    X_train_pca = np.memmap(TRAIN_PCA_MEMMAP_PATH, dtype='float32',
                            mode='w+', shape=(X_train.shape[0], PCA_COMPONENTS))

    # バッチ処理でデータを変換し、新しいmemmapファイルに書き込む
    for i in tqdm(range(0, X_train.shape[0], BATCH_SIZE), desc="IPCA transforming train"):
        transformed_batch = ipca.transform(X_train[i:i + BATCH_SIZE])
        X_train_pca[i:i + len(transformed_batch)] = transformed_batch
    
    X_train_pca.flush()
    print(f"次元削減後の訓練データを {TRAIN_PCA_MEMMAP_PATH} に保存しました。")

    # --- 学習済みPCAモデルを使ってテストデータを変換する ---
    print("\nテストデータの特徴量を読み込みます...")
    X_test_original = np.memmap(TEST_MEMMAP_PATH, dtype='float32', mode='r')
    num_test_samples = len([f for f in os.listdir(
        "encoded_test_npz") if f.endswith(".npz")])
    X_test = X_test_original.reshape(num_test_samples, -1)
    print(f"テストデータの形状: {X_test.shape}")

    print("学習済みPCAモデルを読み込みます...")
    ipca_trained = joblib.load(PCA_MODEL_PATH)

    print("テストデータをPCAで変換します...")
    X_test_pca = np.memmap(TEST_PCA_MEMMAP_PATH, dtype='float32',
                           mode='w+', shape=(X_test.shape[0], PCA_COMPONENTS))

    for i in tqdm(range(0, X_test.shape[0], BATCH_SIZE), desc="IPCA transforming test"):
        transformed_batch = ipca_trained.transform(X_test[i:i + BATCH_SIZE])
        X_test_pca[i:i + len(transformed_batch)] = transformed_batch
    
    X_test_pca.flush()
    print(f"次元削減後のテストデータを {TEST_PCA_MEMMAP_PATH} に保存しました。")
    print("すべての処理が完了しました。")


if __name__ == "__main__":
    main()
