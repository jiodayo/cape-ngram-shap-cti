"""
抽出されたキーワードを、Sentence-BERTを用いたZero-shot分類によって
MBC (Malware Behavior Catalog) のMicro-Behaviorsへ自動マッピングするスクリプト。
"""
import json
import os
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def build_mapping(input_filepath, output_filepath):
    if not os.path.exists(input_filepath):
        print(f"エラー: 入力ファイル '{input_filepath}' が見つかりません。")
        return

    # MBCの全612カテゴリ辞書をロード
    mbc_json_path = os.path.join(os.path.dirname(__file__), 'mbc_full_categories.json')
    if not os.path.exists(mbc_json_path):
        print(f"エラー: MBCカテゴリファイル '{mbc_json_path}' が見つかりません。事前に extract_mbc_catalog.py を実行してください。")
        return
        
    with open(mbc_json_path, 'r', encoding='utf-8') as f:
        mbc_micro_behaviors = json.load(f)
        
    print(f"MBC公式から抽出された {len(mbc_micro_behaviors)} 個の全カテゴリを候補としてロードしました。")

    print(f"'{input_filepath}' から抽出されたキーワードを読み込んでいます...")
    with open(input_filepath, 'r', encoding='utf-8') as f:
        api_keywords = json.load(f)

    # 全APIから一意なキーワードのリストを作成
    unique_keywords = set()
    for kw_list in api_keywords.values():
        for kw in kw_list:
            unique_keywords.add(kw)
    unique_keywords = list(unique_keywords)
    print(f"合計 {len(unique_keywords)} 個のユニークなキーワードが見つかりました。")

    print("Sentence-BERTモデル (all-MiniLM-L6-v2) をロードしています...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # MBCカテゴリの埋め込みを計算
    print("MBC全カテゴリのベクトルを計算中...")
    mbc_categories = list(mbc_micro_behaviors.keys())
    mbc_descriptions = list(mbc_micro_behaviors.values())
    mbc_embeddings = model.encode(mbc_descriptions)

    # キーワードの埋め込みを計算
    print("キーワードのベクトルを計算中...")
    keyword_embeddings = model.encode(unique_keywords)

    print("総当たりZero-shot分類（コサイン類似度計算）を実行中...")
    similarities = cosine_similarity(keyword_embeddings, mbc_embeddings)

    # --- データドリブンな閾値の算出 ---
    # 各キーワードの「一番高い類似度スコア」を抽出
    max_similarities = np.max(similarities, axis=1)
    
    # 上位25%（75パーセンタイル）を動的閾値として設定
    dynamic_threshold = np.percentile(max_similarities, 75)
    print(f"\n[データドリブン選定] スコア分布の75パーセンタイルから動的閾値を算出: {dynamic_threshold:.4f}")

    # --- カテゴリの自動フィルタリング ---
    # 閾値を超えたキーワード数を数える
    above_threshold_count = int(np.sum(max_similarities >= dynamic_threshold))
    
    # 期待値の算出: もし均等に分配されたら各カテゴリに何個入るか
    expected_per_category = above_threshold_count / len(mbc_categories)
    # 期待値以上の集積があるカテゴリのみ採用（最低1）
    min_keywords = max(1, int(np.ceil(expected_per_category)))
    print(f"[データドリブン選定] 閾値超えキーワード数: {above_threshold_count}, カテゴリ数: {len(mbc_categories)}")
    print(f"[データドリブン選定] 期待値 = {expected_per_category:.2f} → 最小キーワード数(min_keywords) = {min_keywords}")
    
    print(f"\n各カテゴリに閾値以上のキーワードが {min_keywords} 個以上あるか検証中...")
    category_counts = {}
    for i, kw in enumerate(unique_keywords):
        best_idx = np.argmax(similarities[i])
        best_score = similarities[i][best_idx]
        if best_score >= dynamic_threshold:
            cat = mbc_categories[best_idx]
            category_counts[cat] = category_counts.get(cat, 0) + 1

    # 有効なカテゴリ（採用されたカテゴリ）を抽出
    adopted_categories = {cat for cat, count in category_counts.items() if count >= min_keywords}
    print(f"{len(mbc_categories)} 個の候補のうち、データによって客観的に {len(adopted_categories)} 個のカテゴリが自動採用されました。")

    # --- 最終マッピング ---
    mapping_result = {}
    noise_count = 0
    for i, kw in enumerate(unique_keywords):
        best_cat = None
        best_score = -1.0
        
        # 採用されたカテゴリの中だけで一番高い類似度を探す
        for j, cat in enumerate(mbc_categories):
            if cat in adopted_categories:
                score = similarities[i][j]
                if score > best_score:
                    best_score = score
                    best_cat = cat
                    
        # 最終的に採用カテゴリ内でのベストスコアが閾値を超えていれば紐付け
        if best_cat and best_score >= dynamic_threshold:
            mapping_result[kw] = {
                "category": best_cat,
                "similarity_score": float(best_score)
            }
        else:
            noise_count += 1
            mapping_result[kw] = {
                "category": "Uncategorized (Noise)",
                "similarity_score": float(best_score if best_score != -1.0 else 0.0)
            }

    # 保存
    os.makedirs(os.path.dirname(output_filepath) if os.path.dirname(output_filepath) else '.', exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(mapping_result, f, indent=4, ensure_ascii=False)

    print(f"\nマッピングが完了しました！結果を '{output_filepath}' に保存しました。")
    print(f" - 有効マッピング数: {len(unique_keywords) - noise_count}")
    print(f" - 除外（ノイズ）数: {noise_count}")
    
    # 簡単な統計を表示
    stats = {}
    for res in mapping_result.values():
        c = res['category']
        if c != "Uncategorized (Noise)":
            stats[c] = stats.get(c, 0) + 1
    
    print("\n【採用されたカテゴリごとのキーワード数】")
    for c, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {c}: {count} 個")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="キーワードをMBC Micro-Behaviorsにマッピングする。")
    parser.add_argument("--input", type=str, default="api_keywords_single.json", help="入力となるキーワードJSONファイル")
    parser.add_argument("--output", type=str, default="features/mbc_keyword_mapping.json", help="出力するマッピング結果JSONファイル")
    args = parser.parse_args()

    build_mapping(args.input, args.output)
