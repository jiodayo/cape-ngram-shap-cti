"""
APIの説明文が記載されたJSONファイルから、KeyBERTを用いて
キーワードを抽出し、別のJSONファイルに出力するプログラム。

■ 必要なライブラリのインストール:
pip install spacy==3.7.2 keybert
python -m spacy download en_core_web_sm
"""
import json
import spacy
from keybert import KeyBERT
import argparse
import os

def extract_single_keywords(input_filepath, output_filepath, top_n_phrases=3):
    """
    JSONファイルからAPI説明文を読み込み、KeyBERTを用いてキーワードを抽出して保存する。
    """
    print("spaCyモデルを読み込んでいます...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("spaCyの英語モデル 'en_core_web_sm' が見つかりません。ダウンロードを開始します...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    print("KeyBERTモデル (all-MiniLM-L6-v2) を読み込んでいます...")
    kw_model = KeyBERT('all-MiniLM-L6-v2')

    print(f"'{input_filepath}' から説明文を読み込んでいます...")
    with open(input_filepath, 'r', encoding='utf-8') as f:
        api_descriptions = json.load(f)

    extracted_results = {}

    print("単一キーワード抽出（KeyBERT）を開始します...")
    for api_name, description in api_descriptions.items():
        doc = nlp(description)

        # 1. spaCyを用いて意味のある単語（名詞・固有名詞・動詞）のみを残し、基本形（レンマ）にする
        filtered_words = [
            token.lemma_.lower() for token in doc 
            if not token.is_stop and not token.is_punct and token.pos_ in ["NOUN", "PROPN", "VERB"]
        ]
        clean_text = " ".join(filtered_words)

        # 2. KeyBERTで文脈を考慮した抽出を行う
        keywords = []
        if clean_text.strip():
            # ngram_range=(1,1)で単一単語のみ抽出
            extracted = kw_model.extract_keywords(
                clean_text, 
                keyphrase_ngram_range=(1, 1), 
                stop_words='english', 
                top_n=top_n_phrases
            )
            keywords = [kw[0] for kw in extracted]

        extracted_results[api_name] = keywords
        print(f"  - {api_name}: {keywords}")

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(extracted_results, f, indent=4, ensure_ascii=False)

    print(f"\n単一キーワードの抽出が完了し、'{output_filepath}' に保存しました。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="APIの説明文からKeyBERTでキーワードを抽出する。")
    parser.add_argument("--input", type=str, default="api_descriptions.json", help="入力JSONファイルのパス")
    parser.add_argument("--output", type=str, default="api_keywords_single.json", help="出力JSONファイルのパス")
    parser.add_argument("--top-n", type=int, default=3, help="抽出する上位N個のキーワード")
    parser.add_argument("--force", action="store_true", help="出力ファイルが既に存在しても上書きする")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        print(f"'{args.output}' は既に存在します。スキップします。上書きする場合は --force を付けてください。")
    else:
        extract_single_keywords(args.input, args.output, top_n_phrases=args.top_n)
