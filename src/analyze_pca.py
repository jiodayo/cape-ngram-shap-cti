import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import sys


def analyze_pca_model(model_path="logs/ipca_model.joblib"):
    """
    学習済みのPCAモデルをロードし、各主成分の寄与率を可視化して保存する関数。
    """
    print("--- PCAモデルの分析を開始 ---")

    # PCAモデルファイルの存在確認
    if not os.path.exists(model_path):
        print(f"エラー: PCAモデルファイルが見つかりません: {model_path}")
        print("先に '2025_learning_RF.py' の 'prepare' または 'all' モードを実行して、モデルを生成してください。")
        sys.exit(1)

    # モデルをロード
    try:
        ipca = joblib.load(model_path)
        print(f"PCAモデルを正常に読み込みました: {model_path}")
    except Exception as e:
        print(f"エラー: PCAモデルの読み込みに失敗しました。詳細: {e}")
        sys.exit(1)

    # 寄与率を取得
    explained_variance_ratio = ipca.explained_variance_ratio_
    n_components = len(explained_variance_ratio)

    print(f"主成分の数: {n_components}")
    print(f"各主成分の寄与率: {explained_variance_ratio}")
    print(f"寄与率の合計 (累積寄与率): {np.sum(explained_variance_ratio):.4f}")

    # グラフ描画エリアの作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('PCA Component Analysis', fontsize=16)

    # 1. 棒グラフ（各主成分の寄与率）
    component_indices = np.arange(n_components)  # 0から始まるインデックスに変更
    ax1.bar(component_indices, explained_variance_ratio, alpha=0.7, align='center',
            label='Individual explained variance')
    ax1.set_ylabel('Explained variance ratio')
    ax1.set_xlabel('Principal Component Index')
    ax1.set_title('Explained Variance per Component')
    ax1.legend(loc='best')
    ax1.grid(True)
    ax1.set_xticks(component_indices)

    # 2. 折れ線グラフ（累積寄与率）
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    ax2.plot(component_indices, cumulative_variance_ratio, marker='o', linestyle='--',
             label='Cumulative explained variance')
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('Cumulative explained variance ratio')
    ax2.set_xlabel('Principal Component Index')
    ax2.set_title('Cumulative Explained Variance')
    ax2.legend(loc='best')
    ax2.grid(True)
    ax2.set_xticks(component_indices)

    # レイアウトを調整
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # グラフをファイルに保存
    save_dir = "logs"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "pca_analysis_report.png")
    plt.savefig(save_path)

    print(f"\nPCAの寄与率グラフを {save_path} に保存しました。")
    print("--- PCAモデルの分析完了 ---")


if __name__ == "__main__":
    analyze_pca_model()
