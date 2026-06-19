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

# MBC Micro-Behaviors の定義 (公式の定義をベースにシンプル化)
MBC_MICRO_BEHAVIORS = {
    "File System": "Behaviors related to creating, reading, writing, deleting, or modifying files, directories, paths, and attributes.",
    "Process/Thread": "Behaviors related to executing, injecting, terminating, or managing processes, threads, and modules.",
    "Registry": "Behaviors related to reading, writing, modifying, or deleting the Windows Registry keys and values.",
    "Network/Communication": "Behaviors related to establishing connections, sending or receiving data, sockets, and internet over the network.",
    "Cryptography": "Behaviors related to encrypting, decrypting, hashing data, keys, and certificates.",
    "Memory": "Behaviors related to allocating, freeing, protecting, or reading system virtual memory and heaps.",
    "Service": "Behaviors related to creating, starting, or managing Windows services, drivers, and control managers.",
    "System Info/Discovery": "Behaviors related to gathering information about the system environment, hardware, configuration, and time.",
    "Synchronization": "Behaviors related to mutexes, semaphores, events, and process synchronization.",
    "GUI/Input": "Behaviors related to the graphical user interface, windows, mouse, or capturing user input like keystrokes.",
    "COM": "Behaviors related to Component Object Model (COM) initialization, classes, and object instantiation."
}

def build_mapping(input_filepath, output_filepath, threshold=0.25):
    if not os.path.exists(input_filepath):
        print(f"エラー: 入力ファイル '{input_filepath}' が見つかりません。")
        return

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
    print("MBCカテゴリのベクトルを計算中...")
    mbc_categories = list(MBC_MICRO_BEHAVIORS.keys())
    mbc_descriptions = list(MBC_MICRO_BEHAVIORS.values())
    mbc_embeddings = model.encode(mbc_descriptions)

    # キーワードの埋め込みを計算
    print("キーワードのベクトルを計算中...")
    keyword_embeddings = model.encode(unique_keywords)

    print("Zero-shot分類（コサイン類似度計算）を実行中...")
    similarities = cosine_similarity(keyword_embeddings, mbc_embeddings)

    mapping_result = {}
    for i, kw in enumerate(unique_keywords):
        best_idx = np.argmax(similarities[i])
        best_score = similarities[i][best_idx]

        if best_score >= threshold:
            category = mbc_categories[best_idx]
        else:
            category = "Uncategorized (Noise)"

        mapping_result[kw] = {
            "category": category,
            "similarity_score": float(best_score)
        }

    # 保存
    os.makedirs(os.path.dirname(output_filepath) if os.path.dirname(output_filepath) else '.', exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(mapping_result, f, indent=4, ensure_ascii=False)

    print(f"\nマッピングが完了しました！結果を '{output_filepath}' に保存しました。")
    
    # 簡単な統計を表示
    stats = {}
    for res in mapping_result.values():
        c = res['category']
        stats[c] = stats.get(c, 0) + 1
    
    print("\n【カテゴライズ結果の統計】")
    for c, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {c}: {count} 個")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="キーワードをMBC Micro-Behaviorsにマッピングする。")
    parser.add_argument("--input", type=str, default="api_keywords_single.json", help="入力となるキーワードJSONファイル")
    parser.add_argument("--output", type=str, default="features/mbc_keyword_mapping.json", help="出力するマッピング結果JSONファイル")
    parser.add_argument("--threshold", type=float, default=0.25, help="カテゴリに割り当てる最小コサイン類似度（閾値）")
    args = parser.parse_args()

    build_mapping(args.input, args.output, threshold=args.threshold)
