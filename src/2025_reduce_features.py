import os
import numpy as np
from sklearn.decomposition import PCA
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

def main():
    """
    Mean Pooling適用後の特徴量 (768次元) に対してPCAを実行する。
    データサイズが非常に小さくなった(約17MB)ため、メモリ上で高速な標準PCAを使用する。
    """
    os.makedirs("logs", exist_ok=True)
    t_start = time.time()

    # --- 訓練データの読み込み ---
    print("訓練データの特徴量を読み込みます...")
    X_train_raw = np.memmap(TRAIN_MEMMAP_PATH, dtype='float32', mode='r')
    num_train = len([f for f in os.listdir("encoded_train_npz") if f.endswith(".npz")])
    
    # 768次元にリシェイプ
    X_train = X_train_raw.reshape(num_train, -1)
    n_features = X_train.shape[1]
    print(f"訓練データの形状: {X_train.shape}")
    print(f"特徴量次元数: {n_features}")

    # --- PCAの学習と変換 ---
    final_dim = min(FINAL_COMPONENTS, n_features, num_train)
    print(f"\n{'='*60}")
    print(f"PCA ({n_features} → {final_dim}) を実行中...")
    print(f"{'='*60}")

    pca = PCA(n_components=final_dim, random_state=42)
    # メモリ上で一気に計算 (データが小さいので一瞬で終わる)
    X_train_final = pca.fit_transform(X_train).astype(np.float32)

    cumvar = pca.explained_variance_ratio_.sum()
    print(f"PCA後: {X_train_final.shape}")
    print(f"累積寄与率: {cumvar:.4f}")

    # --- モデルの保存 ---
    joblib.dump(pca, MODEL_PATH)
    print(f"\nモデルを {MODEL_PATH} に保存しました。")

    # --- 訓練データPCA特徴量をmemmapに保存 ---
    out_train = np.memmap(TRAIN_PCA_MEMMAP_PATH, dtype='float32',
                          mode='w+', shape=X_train_final.shape)
    out_train[:] = X_train_final
    out_train.flush()
    print(f"訓練PCA特徴量を {TRAIN_PCA_MEMMAP_PATH} に保存しました。")

    # --- テストデータの変換 ---
    print(f"\n{'='*60}")
    print(f"テストデータの変換")
    print(f"{'='*60}")

    X_test_raw = np.memmap(TEST_MEMMAP_PATH, dtype='float32', mode='r')
    num_test = len([f for f in os.listdir("encoded_test_npz") if f.endswith(".npz")])
    X_test = X_test_raw.reshape(num_test, -1)
    print(f"テストデータの形状: {X_test.shape}")

    X_test_final = pca.transform(X_test).astype(np.float32)
    print(f"テストデータ変換完了: {X_test_final.shape}")

    out_test = np.memmap(TEST_PCA_MEMMAP_PATH, dtype='float32',
                         mode='w+', shape=X_test_final.shape)
    out_test[:] = X_test_final
    out_test.flush()
    print(f"テストPCA特徴量を {TEST_PCA_MEMMAP_PATH} に保存しました。")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"すべての処理が完了しました。総処理時間: {elapsed:.3f}秒")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
