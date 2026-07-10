import pandas as pd
import numpy as np
import os

def verify_bias():
    feature_csv = "features/test_keyword_features.csv"
    label_csv = "features/test_labels.csv"
    
    if not os.path.exists(feature_csv) or not os.path.exists(label_csv):
        print(f"エラー: {feature_csv} または {label_csv} が見つかりません。")
        return

    print("データをロード中...")
    X_test = pd.read_csv(feature_csv, index_col=0)
    Y_test = pd.read_csv(label_csv, index_col=0)
    
    target_label = "resolves_host"
    target_features = ["dns", "sockets", "protocol"]
    
    if target_label not in Y_test.columns:
        print(f"ラベル '{target_label}' が見つかりません。")
        return
        
    y_true = Y_test[target_label] > 0
    total_pos = y_true.sum()
    total_neg = (~y_true).sum()
    
    print("="*50)
    print(f"ターゲットラベル: {target_label}")
    print(f" - 全体サンプル数: {len(y_true)}")
    print(f" - 正例 (resolves_host = 1): {total_pos} 件")
    print(f" - 負例 (resolves_host = 0): {total_neg} 件")
    print("="*50)
    
    for feat in target_features:
        if feat not in X_test.columns:
            print(f"特徴量 '{feat}' が見つかりません。")
            continue
            
        x_val = X_test[feat] > 0
        
        # クロス集計表
        ct = pd.crosstab(x_val, y_true, rownames=[f'Has "{feat}"?'], colnames=[f'Label: {target_label}?'])
        
        prob_given_pos = ct.loc[True, True] / total_pos if True in ct.index and True in ct.columns else 0
        prob_given_neg = ct.loc[True, False] / total_neg if True in ct.index and False in ct.columns else 0
        
        print(f"\n■ キーワード: {feat}")
        print("-" * 30)
        print(ct)
        print("-" * 30)
        print(f"resolves_host=1 (正例) のうち、'{feat}' を使う割合 : {prob_given_pos*100:.2f}%")
        print(f"resolves_host=0 (負例) のうち、'{feat}' を使う割合 : {prob_given_neg*100:.2f}%")
        
        if prob_given_pos < prob_given_neg:
            print(f"結論: 負例のほうが '{feat}' をよく使っているため、モデルは負の相関（マイナス要素）として学習します。")
        else:
            print(f"結論: 正例のほうが '{feat}' をよく使っているため、プラス要素として学習します。")
            
    print("\n検証完了！")

if __name__ == "__main__":
    verify_bias()
