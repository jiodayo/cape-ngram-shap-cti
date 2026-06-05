"""
'make_bagofwords.py'で生成したBag-of-Words特徴量とラベルを使い、
ランダムフォレストを用いたBinary Relevanceでのマルウェア機能推定を行うプログラム。
訓練データ内で層化4分割交差検証も行うバージョン。

■ 使い方:
1. 'make_bagofwords.py' を実行して、'features/' ディレクトリを生成します。
2. コマンドラインで `python train_bagofwords.py` を実行します。
3. 'logs_keyword/' ディレクトリに学習済みモデルと評価レポートが出力されます。
"""
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from skmultilearn.model_selection import IterativeStratification
import joblib
from pathlib import Path

# --- 設定 ---
NUM_FOLDS = 4  # 交差検証の分割数
N_ESTIMATORS = 100  # ランダムフォレストの決定木の数
N_JOBS = -1      # 並列処理に使うCPUコア数 (-1は全て)

# --- パス設定 ---
FEATURES_DIR = Path("features")
LOGS_DIR = Path("logs_keyword")

# 入力ファイル
LABEL_SET_PATH = FEATURES_DIR / "label_set.json"
TRAIN_FEATURES_PATH = FEATURES_DIR / "train_keyword_features.csv"
TRAIN_LABELS_PATH = FEATURES_DIR / "train_labels.csv"
TEST_FEATURES_PATH = FEATURES_DIR / "test_keyword_features.csv"
TEST_LABELS_PATH = FEATURES_DIR / "test_labels.csv"

# 出力ディレクトリ
MODELS_DIR = LOGS_DIR / "models_br_+exist_rf"
REPORTS_DIR = LOGS_DIR / "final_reports_br_+exist_rf"


def run_br_with_rf_cv(train_features, train_labels, test_features, test_labels, label_names):
    """
    Random Forestを用いたBinary Relevance分類器の学習と評価を、交差検証付きで行う。
    """
    # 出力ディレクトリを作成
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    y_true_all = []
    y_pred_all = []
    final_accuracies = {}  # 各ラベルの正解率を保存する辞書

    # ラベルごとにモデルを学習
    for i, label_name in enumerate(tqdm(label_names, desc="各ラベルでRFを学習・評価中")):

        y_train_single = train_labels[:, i]
        y_test_single = test_labels[:, i]

        # --- 層化K分割交差検証 ---
        print(f"\n[CV] Label: {label_name} の交差検証を開始...")
        # skmultilearnはyが2次元配列であることを期待するため、reshapeする
        y_train_single_2d = y_train_single.reshape(-1, 1)
        kf = IterativeStratification(n_splits=NUM_FOLDS, order=1)

        fold_f1_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_features, y_train_single_2d)):
            X_train_fold, X_val_fold = train_features[train_idx], train_features[val_idx]
            y_train_fold, y_val_fold = y_train_single[train_idx], y_train_single[val_idx]

            # モデルの学習
            rf_fold_model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, random_state=42, n_jobs=N_JOBS)
            rf_fold_model.fit(X_train_fold, y_train_fold)

            # 検証データで評価
            y_val_pred = rf_fold_model.predict(X_val_fold)
            f1 = f1_score(y_val_fold, y_val_pred,
                          average="binary", zero_division=0)
            fold_f1_scores.append(f1)

        avg_cv_f1 = np.mean(fold_f1_scores)
        print(
            f"  [CV Result] Label: {label_name}, Average F1 over {NUM_FOLDS} folds: {avg_cv_f1:.4f}")

        # --- 全訓練データで再学習し、最終評価 ---
        print(f"  [Final Train] Label: {label_name}, 全訓練データで再学習中...")
        final_model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=42, n_jobs=N_JOBS)
        final_model.fit(train_features, y_train_single)

        # 学習済みモデルを保存
        model_path = MODELS_DIR / f"{label_name}.joblib"
        joblib.dump(final_model, model_path)

        # テストデータで予測
        y_pred_single = final_model.predict(test_features)

        # 正解率を計算して保存
        acc_single = accuracy_score(y_test_single, y_pred_single)
        final_accuracies[label_name] = acc_single

        # 評価レポートの生成と保存
        report = classification_report(
            y_test_single,
            y_pred_single,
            target_names=[f"not_{label_name}", label_name],
            zero_division=0
        )
        with open(REPORTS_DIR / f"{label_name}_final_report.txt", "w") as f:
            f.write(f"--- Final Report for Label: {label_name} ---\n")
            f.write(f"CV Average F1-score: {avg_cv_f1:.4f}\n")
            f.write(f"Test Accuracy: {acc_single:.4f}\n\n")
            f.write(report)

        y_true_all.append(y_test_single)
        y_pred_all.append(y_pred_single)

    # --- 全ラベルをまとめた最終評価 ---
    print("\n--- Overall Final Report ---")
    y_true_all = np.array(y_true_all).T
    y_pred_all = np.array(y_pred_all).T

    overall_report = classification_report(
        y_true_all,
        y_pred_all,
        target_names=label_names,
        zero_division=0
    )
    print(overall_report)

    # --- ラベルごとの正解率をまとめて表示・保存 ---
    print("\n--- Overall Per-label Accuracy ---")
    overall_acc_report_str = ""
    # 正解率の高い順にソートして表示
    sorted_accuracies = sorted(
        final_accuracies.items(), key=lambda item: item[1], reverse=True)
    for label_name, acc in sorted_accuracies:
        line = f"{label_name}: {acc:.4f}\n"
        print(line, end="")
        overall_acc_report_str += line

    with open(REPORTS_DIR / "overall_final_report.txt", "w") as f:
        f.write("--- Overall Classification Report ---\n")
        f.write(overall_report)
        f.write("\n\n--- Overall Per-label Accuracy ---\n")
        f.write(overall_acc_report_str)

    print(f"\n完了: モデルは '{MODELS_DIR}' に、レポートは '{REPORTS_DIR}' に保存されました。")


def main():
    """メイン処理"""
    # --- データの読み込み ---
    print("特徴量とラベルのファイルを読み込んでいます...")
    try:
        with open(LABEL_SET_PATH, 'r', encoding='utf-8') as f:
            label_names = json.load(f)

        # index_col=0 で最初の列（検体名）をインデックスとして使用
        X_train_df = pd.read_csv(TRAIN_FEATURES_PATH, index_col=0)
        Y_train_df = pd.read_csv(TRAIN_LABELS_PATH, index_col=0)
        X_test_df = pd.read_csv(TEST_FEATURES_PATH, index_col=0)
        Y_test_df = pd.read_csv(TEST_LABELS_PATH, index_col=0)

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません: {e.filename}")
        print("先に '1_prepare_keyword_features.py' を実行してください。")
        return

    # pandas DataFrameをNumpy配列に変換
    X_train = X_train_df.values
    Y_train = Y_train_df.values
    X_test = X_test_df.values
    Y_test = Y_test_df.values

    print(f"訓練データの形状: X={X_train.shape}, Y={Y_train.shape}")
    print(f"テストデータの形状: X={X_test.shape}, Y={Y_test.shape}")

    # 学習と評価を実行
    run_br_with_rf_cv(X_train, Y_train, X_test, Y_test, label_names)


if __name__ == '__main__':
    main()
__':
    main()
