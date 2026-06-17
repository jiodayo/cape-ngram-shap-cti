"""
keywordモデルとpcaモデルの両方に対応したハイブリッド特徴量のSHAP分析スクリプト。
「メイン特徴量（KeywordまたはPCA）」と「301次元API頻度特徴量」のカテゴリ別に
重要度を算出し、それぞれのカテゴリ内でのトップ特徴量を抽出する。
さらに、指定されたサンプルのWaterfallプロットを出力する。
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="ハイブリッドモデルのカテゴリ別SHAP分析")
    parser.add_argument("--model-type", type=str, choices=["keyword", "pca"], required=True,
                        help="分析対象のモデル（keyword または pca）")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Waterfallプロットを出力するテストデータのインデックス (default: 0)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="各カテゴリで表示する上位特徴量の数 (default: 20)")
    return parser.parse_args()


def load_data(model_type):
    """モデルタイプに応じたデータと特徴量名をロードする"""
    features_dir = Path("features")
    label_set_path = features_dir / "label_set.json"
    
    with open(label_set_path, "r", encoding="utf-8") as f:
        labels_dict = json.load(f)
    if isinstance(labels_dict, dict):
        label_names = [k for k, v in sorted(labels_dict.items(), key=lambda item: item[1])]
    else:
        label_names = labels_dict

    if model_type == "keyword":
        test_features_path = features_dir / "test_keyword_features.csv"
        test_df = pd.read_csv(test_features_path, index_col=0)
        feature_names = test_df.columns.tolist()
        X_test = test_df.values
        sample_names = test_df.index.tolist()
        models_dir = Path("logs_keyword/models_br_+freq_rf")
    
    elif model_type == "pca":
        test_features_path = "encoded_test_features_pca.mmap"
        test_labels_path = "encoded_test_labels.npy"
        
        Y_test = np.load(test_labels_path)
        num_test_samples = Y_test.shape[0]
        
        # 128 (PCA)
        pca_dim = 128 
        X_test_pca = np.memmap(test_features_path, dtype='float32', mode='r', shape=(num_test_samples, pca_dim))
        
        # API頻度特徴量の読み込み
        test_csv_path = features_dir / "test_keyword_features.csv"
        test_df_api = pd.read_csv(test_csv_path, index_col=0)
        sample_names = test_df_api.index.tolist()
        
        api_cols = [col for col in test_df_api.columns if col.startswith("api__")]
        X_test_api = test_df_api[api_cols].values.astype('float32')
        
        X_test = np.concatenate([X_test_pca, X_test_api], axis=1)
        
        # 特徴量名リストのロード
        with open("logs/models_br_rf/hybrid_feature_names.json", "r") as f:
            feature_names = json.load(f)
            
        models_dir = Path("logs/models_br_rf")

    return X_test, feature_names, sample_names, label_names, models_dir


def analyze_shap(model_type, X_test, feature_names, sample_names, label_names, models_dir, sample_index, top_n):
    output_dir = Path(f"logs_shap_analysis/{model_type}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # カテゴリのインデックスを特定
    api_indices = [i for i, name in enumerate(feature_names) if name.startswith("api__")]
    main_indices = [i for i, name in enumerate(feature_names) if not name.startswith("api__")]
    
    main_category_name = "Keyword" if model_type == "keyword" else "PCA(Mean Pool)"
    print(f"[{model_type.upper()} Model] 特徴量の構成:")
    print(f"  - {main_category_name} 次元数: {len(main_indices)}")
    print(f"  - API Frequency 次元数: {len(api_indices)}")
    
    # 総合的な重要度を保存する辞書
    overall_importances = {
        "main_sum": 0.0,
        "api_sum": 0.0,
        "feature_importances": np.zeros(len(feature_names))
    }
    
    sample_vector = X_test[sample_index].reshape(1, -1)
    sample_name = sample_names[sample_index]
    
    for label in tqdm(label_names, desc="各ラベルのSHAPを計算中"):
        model_path = models_dir / f"{label}.joblib"
        if not model_path.exists():
            print(f"モデルが見つかりません: {model_path} (スキップ)")
            continue
            
        model = joblib.load(model_path)
        explainer = shap.TreeExplainer(model)
        
        # --- 全テストデータのSHAP値計算（重い場合はサンプリング）---
        # 計算を速くするため最大500件で評価する
        eval_size = min(500, X_test.shape[0])
        np.random.seed(42)
        idx = np.random.choice(X_test.shape[0], eval_size, replace=False)
        X_eval = X_test[idx]
        
        shap_values_obj = explainer(X_eval)
        
        # 二値分類の場合、[samples, features, classes]になることがあるので対処
        if len(shap_values_obj.shape) == 3:
            shap_values_matrix = shap_values_obj.values[:, :, 1]
        else:
            shap_values_matrix = shap_values_obj.values
            
        # 平均絶対SHAP値を計算
        mean_abs_shap = np.mean(np.abs(shap_values_matrix), axis=0)
        
        overall_importances["feature_importances"] += mean_abs_shap
        
        main_importance = np.sum(mean_abs_shap[main_indices])
        api_importance = np.sum(mean_abs_shap[api_indices])
        
        overall_importances["main_sum"] += main_importance
        overall_importances["api_sum"] += api_importance
        
        # --- Waterfall プロットの出力 ---
        # 指定されたサンプルに対するWaterfall plot
        shap_values_sample = explainer(sample_vector)
        if len(shap_values_sample.shape) == 3:
            shap_val = shap_values_sample[:, :, 1]
        else:
            shap_val = shap_values_sample
            
        shap_val.feature_names = feature_names
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_val[0], max_display=15, show=False)
        plt.title(f"Waterfall Plot - {label} ({sample_name})")
        plt.tight_layout()
        plt.savefig(output_dir / f"waterfall_{label}.png", dpi=150)
        plt.close()
        
    # --- 全ラベルの総合レポート ---
    total_importance = overall_importances["main_sum"] + overall_importances["api_sum"]
    if total_importance > 0:
        main_ratio = overall_importances["main_sum"] / total_importance * 100
        api_ratio = overall_importances["api_sum"] / total_importance * 100
    else:
        main_ratio = 0
        api_ratio = 0
        
    avg_feature_importances = overall_importances["feature_importances"] / len(label_names)
    
    print("\n" + "="*50)
    print("総合カテゴリ重要度割合 (全ラベル平均)")
    print("="*50)
    print(f"{main_category_name} Features: {main_ratio:.1f}%")
    print(f"API Frequency Features: {api_ratio:.1f}%")
    
    # カテゴリ内のトップ特徴量を抽出
    main_fi = [(feature_names[i], avg_feature_importances[i]) for i in main_indices]
    api_fi = [(feature_names[i], avg_feature_importances[i]) for i in api_indices]
    
    main_fi.sort(key=lambda x: x[1], reverse=True)
    api_fi.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*50)
    print(f"Top {top_n} Features in {main_category_name} Category")
    print("="*50)
    for i, (name, val) in enumerate(main_fi[:top_n]):
        print(f"{i+1:2d}. {name:40s} : {val:.6f}")
        
    print("\n" + "="*50)
    print(f"Top {top_n} Features in API Frequency Category")
    print("="*50)
    for i, (name, val) in enumerate(api_fi[:top_n]):
        clean_name = name.replace("api__", "")
        print(f"{i+1:2d}. {clean_name:40s} : {val:.6f}")
        
    # テキストレポートとして保存
    with open(output_dir / "overall_category_importance_report.txt", "w") as f:
        f.write("=== Overall Category Importance Ratio ===\n")
        f.write(f"{main_category_name} Features: {main_ratio:.1f}%\n")
        f.write(f"API Frequency Features: {api_ratio:.1f}%\n\n")
        
        f.write(f"=== Top {top_n} {main_category_name} Features ===\n")
        for i, (name, val) in enumerate(main_fi[:top_n]):
            f.write(f"{i+1:2d}. {name:40s} : {val:.6f}\n")
            
        f.write(f"\n=== Top {top_n} API Frequency Features ===\n")
        for i, (name, val) in enumerate(api_fi[:top_n]):
            clean_name = name.replace("api__", "")
            f.write(f"{i+1:2d}. {clean_name:40s} : {val:.6f}\n")
            
    print(f"\n結果を {output_dir} に保存しました。Waterfall plot も生成されています。")


if __name__ == "__main__":
    args = parse_args()
    
    print("データの読み込み中...")
    X_test, feature_names, sample_names, label_names, models_dir = load_data(args.model_type)
    
    print(f"SHAP分析の実行 ({args.model_type}モデル)...")
    analyze_shap(args.model_type, X_test, feature_names, sample_names, label_names, models_dir, args.sample_index, args.top_n)
