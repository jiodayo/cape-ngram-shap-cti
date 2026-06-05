"""
APIの説明文が記載されたJSONファイルから、TextRankアルゴリズムを用いて
キーワードを抽出し、別のJSONファイルに出力するプログラム。

■ 必要なライブラリのインストール:
pip install spacy==3.7.2 pytextrank==3.2.4
python -m spacy download en_core_web_sm
"""
import json
import spacy
import pytextrank  # TextRankをspaCyのパイプラインとして追加するために必要
from collections import Counter


def extract_single_keywords(input_filepath, output_filepath, top_n_phrases=5):
    """
    JSONファイルからAPI説明文を読み込み、単一のキーワードを抽出してJSONファイルに保存する。

    Args:
        input_filepath (str): 入力JSONファイルのパス。
        output_filepath (str): 出力JSONファイルのパス。
        top_n_phrases (int): キーワード候補とする元フレーズの上位N個。
    """
    print("spaCyモデルを読み込んでいます...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("spaCyの英語モデル 'en_core_web_sm' が見つかりません。")
        print("ダウンロードを開始します。完了まで数分かかることがあります...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    nlp.add_pipe("textrank")

    print(f"'{input_filepath}' から説明文を読み込んでいます...")
    with open(input_filepath, 'r', encoding='utf-8') as f:
        api_descriptions = json.load(f)

    extracted_results = {}

    print("単一キーワード抽出を開始します...")
    for api_name, description in api_descriptions.items():
        doc = nlp(description)

        # 貢献度の高い単語を格納するカウンター
        word_counter = Counter()

        # 上位のキーフレーズを候補としてループ
        for phrase in doc._.phrases[:top_n_phrases]:
            # フレーズ内の各単語（トークン）をチェック
            for token in phrase.chunks[0]:  # チャンク内のトークンを取得
                # ストップワードでなく、句読点でなく、名詞または固有名詞または動詞であるか
                if not token.is_stop and not token.is_punct and token.pos_ in ["NOUN", "PROPN", "VERB"]:
                    # 基本形（レンマ）を小文字で追加
                    word_counter[token.lemma_.lower()] += 1

        # 頻出する単語をキーワードとして採用
        keywords = [word for word, count in word_counter.most_common()]

        extracted_results[api_name] = keywords
        print(f"  - {api_name}: {keywords}")

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(extracted_results, f, indent=4, ensure_ascii=False)

    print(f"\n単一キーワードの抽出が完了し、'{output_filepath}' に保存しました。")


if __name__ == '__main__':
    INPUT_JSON_PATH = "api_descriptions.json"
    OUTPUT_JSON_PATH = "api_keywords_single.json"
    NUM_KEY_PHRASES = 3

    extract_single_keywords(
        INPUT_JSON_PATH, OUTPUT_JSON_PATH, top_n_phrases=NUM_KEY_PHRASES)
