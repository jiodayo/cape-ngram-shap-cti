"""
'make_bagofwords.py'で生成したBag-of-Words特徴量とラベルを使い、
ランダムフォレストを用いたBinary Relevanceでのマルウェア機能推定を行うプログラム。
訓練データ内で層化4分割交差検証も行うバージョン。

■ 使い方:
  python train_bagofwords.py [オプション]
  オプション無しの場合は従来のデフォルト設定で実行されます。
  --help で全オプションを確認できます。
"""
import argparse
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


def parse_args():
    """コマンドライン引数をパースする。"""
    parser = argparse.ArgumentParser(
        description="キーワードBoW特徴量を使ったRF Binary Relevance学習・評価。")
    parser.add_argument(
        "--features-dir", type=str, default="features",
        help="特徴量ディレクトリ (default: features)")
    parser.add_argument(
        "--output-dir", type=str, default="logs_keyword",
        help="ログ・モデル出力ディレクトリ (default: logs_keyword)")
    parser.add_argument(
        "--n-estimators", type=int, default=100,
        help="ランダムフォレストの決定木の数 (default: 100)")
    parser.add_argument(
        "--n-folds", type=int, default=4,
        help="交差検証の分割数 (default: 4)")
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="並列処理に使うCPUコア数 (default: -1 = 全て)")
    parser.add_argument(
        "--model-subdir", type=str, default="models_br_+freq_rf",
        help="モデル保存サブディレクトリ名 (default: models_br_+freq_rf)")
    parser.add_argument(
        "--report-subdir", type=str, default="final_reports_br_+freq_rf",
        help="レポート保存サブディレクトリ名 (default: final_reports_br_+freq_rf)")
    return parser.parse_args()


def run_br_with_rf_cv(train_features, train_labels, test_features, test_labels,
                      label_names, n_estimators, n_folds, n_jobs,
                      models_dir, reports_dir):
    """
    Random Forestを用いたBinary Relevance分類器の学習と評価を、交差検証付きで行う。
    """
    # 出力ディレクトリを作成
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

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
        kf = IterativeStratification(n_splits=n_folds, order=1)

        fold_f1_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_features, y_train_single_2d)):
            X_train_fold, X_val_fold = train_features[train_idx], train_features[val_idx]
            y_train_fold, y_val_fold = y_train_single[train_idx], y_train_single[val_idx]

            # モデルの学習
            rf_fold_model = RandomForestClassifier(
                n_estimators=n_estimators, random_state=42, n_jobs=n_jobs)
            rf_fold_model.fit(X_train_fold, y_train_fold)

            # 検証データで評価
            y_val_pred = rf_fold_model.predict(X_val_fold)
            f1 = f1_score(y_val_fold, y_val_pred,
                          average="binary", zero_division=0)
            fold_f1_scores.append(f1)

        avg_cv_f1 = np.mean(fold_f1_scores)
        print(
            f"  [CV Result] Label: {label_name}, Average F1 over {n_folds} folds: {avg_cv_f1:.4f}")

        # --- 全訓練データで再学習し、最終評価 ---
        print(f"  [Final Train] Label: {label_name}, 全訓練データで再学習中...")
        final_model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, n_jobs=n_jobs)
        final_model.fit(train_features, y_train_single)

        # 学習済みモデルを保存
        model_path = models_dir / f"{label_name}.joblib"
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
        with open(reports_dir / f"{label_name}_final_report.txt", "w") as f:
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

    with open(reports_dir / "overall_final_report.txt", "w") as f:
        f.write("--- Overall Classification Report ---\n")
        f.write(overall_report)
        f.write("\n\n--- Overall Per-label Accuracy ---\n")
        f.write(overall_acc_report_str)

    print(f"\n完了: モデルは '{models_dir}' に、レポートは '{reports_dir}' に保存されました。")


def main():
    """メイン処理"""
    args = parse_args()

    features_dir = Path(args.features_dir)
    logs_dir = Path(args.output_dir)
    models_dir = logs_dir / args.model_subdir
    reports_dir = logs_dir / args.report_subdir

    # --- パス設定 ---
    label_set_path = features_dir / "label_set.json"
    train_features_path = features_dir / "train_keyword_features.csv"
    train_labels_path = features_dir / "train_labels.csv"
    test_features_path = features_dir / "test_keyword_features.csv"
    test_labels_path = features_dir / "test_labels.csv"

    # --- データの読み込み ---
    print("特徴量とラベルのファイルを読み込んでいます...")
    try:
        with open(label_set_path, 'r', encoding='utf-8') as f:
            label_names = json.load(f)

        # index_col=0 で最初の列（検体名）をインデックスとして使用
        X_train_df = pd.read_csv(train_features_path, index_col=0)
        Y_train_df = pd.read_csv(train_labels_path, index_col=0)
        X_test_df = pd.read_csv(test_features_path, index_col=0)
        Y_test_df = pd.read_csv(test_labels_path, index_col=0)

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません: {e.filename}")
        print("先に 'make_bagofwords.py' を実行してください。")
        return

    # pandas DataFrameをNumpy配列に変換
    X_train = X_train_df.values
    Y_train = Y_train_df.values
    X_test = X_test_df.values
    Y_test = Y_test_df.values

    print(f"訓練データの形状: X={X_train.shape}, Y={Y_train.shape}")
    print(f"テストデータの形状: X={X_test.shape}, Y={Y_test.shape}")

    # 学習と評価を実行
    run_br_with_rf_cv(
        X_train, Y_train, X_test, Y_test, label_names,
        n_estimators=args.n_estimators,
        n_folds=args.n_folds,
        n_jobs=args.n_jobs,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )


if __name__ == '__main__':
    main()

