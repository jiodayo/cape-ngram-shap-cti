import pandas as pd
import sys

def check_sample():
    # 引数で検体ハッシュを受け取れるようにする（デフォルトは対象のハッシュ）
    sample_id = sys.argv[1] if len(sys.argv) > 1 else "000c60fb7fb04267a1759dc58d0bf935b45637a717b77e0cdf62e20180cf58bb"
    feature_csv = "features/test_keyword_features.csv"
    
    try:
        print(f"データをロード中... ({feature_csv})")
        df = pd.read_csv(feature_csv, index_col=0)
    except FileNotFoundError:
        print(f"エラー: {feature_csv} が見つかりません。")
        return
    
    # インデックスに拡張子(.jsonなど)がついている可能性も考慮して部分一致で探す
    matching_indices = [idx for idx in df.index if sample_id in str(idx)]
    
    if not matching_indices:
        print(f"エラー: 検体 '{sample_id}' がテストデータ(CSVのインデックス)に見つかりませんでした。")
        return
        
    actual_idx = matching_indices[0]
    print("=" * 50)
    print(f"検体を発見しました: {actual_idx}")
    print("=" * 50)
    
    row = df.loc[actual_idx]
    
    # 先ほどのレポートに載っていた主要キーワード
    keywords = ["dns", "sockets", "protocol", "executable", "tcp", "udp", "connection"]
    
    for kw in keywords:
        if kw in row.index:
            val = row[kw]
            if val == 0:
                print(f" [ 0 ] キーワード '{kw}': 含まれていない（値={val}）")
            else:
                print(f" [*1*] キーワード '{kw}': 含まれている！（値={val}）")
        else:
            print(f" [-] キーワード '{kw}': 特徴量カラムに存在しません")
            
    print("=" * 50)

if __name__ == "__main__":
    check_sample()
