import os
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA
from tqdm import tqdm
import joblib
import time

# --- 設定 ---
# 入力するmemmapファイル
TRAIN_MEMMAP_PATH = "encoded_train_features.mmap"
TEST_MEMMAP_PATH = "encoded_test_features.mmap"

# 出力するmemmapファイル
TRAIN_PCA_MEMMAP_PATH = "encoded_train_features_pca.mmap"
TEST_PCA_MEMMAP_PATH = "encoded_test_features_pca.mmap"

# モデルの保存先
MODEL_PATH = "logs/ipca_model.joblib"

# --- パラメータ ---
FINAL_COMPONENTS = 128   # 最終的な出力次元数
RP_DIM = 1000            # ランダム射影による中間次元数
BATCH_SIZE = 64          # memmap読み込みのバッチサイズ


def transform_batched(transformer, X_memmap, n_samples, out_dim, batch_size, desc):
    """
    巨大なmemmapデータをバッチ単位で読み込みながら変換する。
    各バッチをメモリにコピーしてからtransformするため、メモリ使用量を制御できる。
    """
    result = np.zeros((n_samples, out_dim), dtype=np.float32)
    for i in tqdm(range(0, n_samples, batch_size), desc=desc):
        end = min(i + batch_size, n_samples)
        batch = np.array(X_memmap[i:end])  # memmap → dense配列 (明示的コピー)
        result[i:end] = transformer.transform(batch).astype(np.float32)
    return result


def main():
    """
    2段階の次元削減を行う:
      Step 1: SparseRandomProjection (数百万次元 → 1000次元)
        - Johnson-Lindenstrauss補題に基づき、ペアワイズ距離を近似的に保持
        - スパース行列を使うため、メモリ効率が高く高速
        - LAPACKのint32オーバーフロー問題が発生しない
      Step 2: PCA (1000次元 → 128次元)
        - ランダム射影後の小さな行列に対して通常のPCAを適用
        - 分散最大化の方向を求める (最適な低次元表現)

    従来の IncrementalPCA (数時間) → この手法 (数分~十数分) に短縮。
    """
    os.makedirs("logs", exist_ok=True)
    t_start = time.time()

    # --- 訓練データの読み込み ---
    print("訓練データの特徴量を読み込みます...")
    X_train_raw = np.memmap(TRAIN_MEMMAP_PATH, dtype='float32', mode='r')
    num_train = len([f for f in os.listdir("encoded_train_npz") if f.endswith(".npz")])
    X_train = X_train_raw.reshape(num_train, -1)
    n_features = X_train.shape[1]
    print(f"訓練データの形状: {X_train.shape}")
    print(f"特徴量次元数: {n_features:,}")

    # --- Step 1: SparseRandomProjection ---
    rp_dim = min(RP_DIM, n_features)
    print(f"\n{'='*60}")
    print(f"Step 1: SparseRandomProjection ({n_features:,} → {rp_dim})")
    print(f"{'='*60}")

    rp = SparseRandomProjection(n_components=rp_dim, random_state=42)
    # fitにはn_featuresの情報だけが必要 (データの値は使わない)
    rp.fit(np.zeros((1, n_features), dtype=np.float32))

    sparse_nnz = rp.components_.nnz
    density = sparse_nnz / (rp.components_.shape[0] * rp.components_.shape[1])
    print(f"射影行列: shape={rp.components_.shape}, "
          f"非ゼロ要素数={sparse_nnz:,} (密度={density:.6f})")

    t1 = time.time()
    X_train_rp = transform_batched(
        rp, X_train, num_train, rp_dim, BATCH_SIZE, "RP変換 (train)")
    print(f"ランダム射影後: {X_train_rp.shape} ({time.time()-t1:.1f}秒)")

    # --- Step 2: PCA ---
    final_dim = min(FINAL_COMPONENTS, rp_dim, num_train)
    print(f"\n{'='*60}")
    print(f"Step 2: PCA ({rp_dim} → {final_dim})")
    print(f"{'='*60}")

    t2 = time.time()
    pca = PCA(n_components=final_dim, random_state=42)
    X_train_final = pca.fit_transform(X_train_rp).astype(np.float32)
    del X_train_rp  # メモリ解放

    cumvar = pca.explained_variance_ratio_.sum()
    print(f"PCA後: {X_train_final.shape} ({time.time()-t2:.1f}秒)")
    print(f"累積寄与率 (射影後空間): {cumvar:.4f}")

    # --- モデルの保存 ---
    joblib.dump({"rp": rp, "pca": pca}, MODEL_PATH)
    print(f"\nモデルを {MODEL_PATH} に保存しました。")

    # --- 訓練データPCA特徴量をmemmapに保存 ---
    out_train = np.memmap(TRAIN_PCA_MEMMAP_PATH, dtype='float32',
                          mode='w+', shape=X_train_final.shape)
    out_train[:] = X_train_final
    out_train.flush()
    del X_train_final
    print(f"訓練PCA特徴量を {TRAIN_PCA_MEMMAP_PATH} に保存しました。")

    # --- テストデータの変換 ---
    print(f"\n{'='*60}")
    print(f"テストデータの変換")
    print(f"{'='*60}")

    X_test_raw = np.memmap(TEST_MEMMAP_PATH, dtype='float32', mode='r')
    num_test = len([f for f in os.listdir("encoded_test_npz") if f.endswith(".npz")])
    X_test = X_test_raw.reshape(num_test, -1)
    print(f"テストデータの形状: {X_test.shape}")

    t3 = time.time()
    X_test_rp = transform_batched(
        rp, X_test, num_test, rp_dim, BATCH_SIZE, "RP変換 (test)")
    X_test_final = pca.transform(X_test_rp).astype(np.float32)
    del X_test_rp
    print(f"テストデータ変換完了: {X_test_final.shape} ({time.time()-t3:.1f}秒)")

    out_test = np.memmap(TEST_PCA_MEMMAP_PATH, dtype='float32',
                         mode='w+', shape=X_test_final.shape)
    out_test[:] = X_test_final
    out_test.flush()
    del X_test_final
    print(f"テストPCA特徴量を {TEST_PCA_MEMMAP_PATH} に保存しました。")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"すべての処理が完了しました。総処理時間: {elapsed:.1f}秒 ({elapsed/60:.1f}分)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
