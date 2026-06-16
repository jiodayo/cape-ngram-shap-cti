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

# 削減後の次元数 (希望値。実際の特徴量次元に基づいて安全な値に調整される)
PCA_COMPONENTS_DESIRED = 128


def compute_safe_params(feature_dim, desired_components):
    """
    LAPACKの32bit整数オーバーフローを回避するため、
    実際の特徴量次元数から安全なn_componentsとbatch_sizeを自動計算する。

    IncrementalPCAの2回目以降のpartial_fitでは、内部で以下の行列を結合してSVDにかける:
      [singular_values * components]  ... (n_components, feature_dim)
      [new_batch]                     ... (batch_size, feature_dim)
      [mean_correction]              ... (1, feature_dim)
    合計行数 = n_components + batch_size + 1

    scipy.linalg.svd は行列の要素数が int32 の上限 (2^31-1) を超えるとエラーになるため、
    (n_components + batch_size + 1) * feature_dim < 2^31 - 1 を満たす必要がある。
    """
    max_elements = np.iinfo(np.int32).max  # 2,147,483,647
    max_total_rows = max_elements // feature_dim

    # 安全マージンを10%取る
    max_total_rows = int(max_total_rows * 0.9)

    if max_total_rows < 3:
        raise ValueError(
            f"特徴量次元数 ({feature_dim}) が大きすぎて、PCAを適用できません。"
            f"特徴量の次元削減を先に行ってください。"
        )

    # batch_size >= n_components が必要 (IncrementalPCAの制約)
    # total_rows = n_components + batch_size + 1
    # batch_size = n_components とすると: total = 2 * n_components + 1
    # → n_components = (max_total_rows - 1) // 2
    safe_max_components = (max_total_rows - 1) // 2

    n_components = min(desired_components, safe_max_components)
    # batch_size は n_components 以上で、かつ total が max_total_rows 以下
    batch_size = min(max_total_rows - n_components - 1, n_components * 2)
    # batch_size は最低でも n_components 必要
    batch_size = max(batch_size, n_components)

    return n_components, batch_size


def main():
    """
    メイン関数
    """
    os.makedirs("logs", exist_ok=True)

    # --- 訓練データでPCAを学習し、変換する ---
    print("訓練データの特徴量を読み込みます...")
    X_train_original = np.memmap(TRAIN_MEMMAP_PATH, dtype='float32', mode='r')
    # 形状を (サンプル数, 特徴量次元数) にリシェイプ
    num_train_samples = len([f for f in os.listdir(
        "encoded_train_npz") if f.endswith(".npz")])
    X_train = X_train_original.reshape(num_train_samples, -1)
    feature_dim = X_train.shape[1]
    print(f"訓練データの形状: {X_train.shape}")
    print(f"特徴量次元数: {feature_dim:,}")

    # LAPACKオーバーフローを回避する安全なパラメータを計算
    n_components, batch_size = compute_safe_params(feature_dim, PCA_COMPONENTS_DESIRED)

    if n_components < PCA_COMPONENTS_DESIRED:
        print(f"警告: LAPACKのint32制限により、n_components を {PCA_COMPONENTS_DESIRED} → {n_components} に縮小しました。")
    print(f"PCA設定: n_components={n_components}, batch_size={batch_size}")

    max_elements = np.iinfo(np.int32).max
    total_rows = n_components + batch_size + 1
    actual_elements = total_rows * feature_dim
    print(f"SVD行列の最大要素数: {actual_elements:,} / {max_elements:,} (使用率: {actual_elements/max_elements*100:.1f}%)")

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    print(f"IncrementalPCAの学習を開始 (n_components={n_components})...")
    # バッチ処理でPCAモデルを学習
    for i in tqdm(range(0, X_train.shape[0], batch_size), desc="IPCA fitting"):
        ipca.partial_fit(X_train[i:i + batch_size])

    print("学習済みPCAモデルを保存します...")
    joblib.dump(ipca, PCA_MODEL_PATH)
    print(f"モデルを {PCA_MODEL_PATH} に保存しました。")

    print("訓練データをPCAで変換します...")
    # 変換後のデータを保存する新しいmemmapファイルを作成
    X_train_pca = np.memmap(TRAIN_PCA_MEMMAP_PATH, dtype='float32',
                            mode='w+', shape=(X_train.shape[0], n_components))

    # バッチ処理でデータを変換し、新しいmemmapファイルに書き込む
    for i in tqdm(range(0, X_train.shape[0], batch_size), desc="IPCA transforming train"):
        transformed_batch = ipca.transform(X_train[i:i + batch_size])
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
                           mode='w+', shape=(X_test.shape[0], n_components))

    for i in tqdm(range(0, X_test.shape[0], batch_size), desc="IPCA transforming test"):
        transformed_batch = ipca_trained.transform(X_test[i:i + batch_size])
        X_test_pca[i:i + len(transformed_batch)] = transformed_batch

    X_test_pca.flush()
    print(f"次元削減後のテストデータを {TEST_PCA_MEMMAP_PATH} に保存しました。")
    print("すべての処理が完了しました。")


if __name__ == "__main__":
    main()
