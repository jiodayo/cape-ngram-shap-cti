#!/usr/bin/env python3
"""
MalEval論文の知見を取り入れた Evidence Attribution & Behavior Synthesis スクリプト。

SHAPの分析結果とMBCマッピングを基に、GPT-4oを用いて
「なぜそのキーワード/APIが根拠になるのか」を自然言語で説明するレポートを生成する。

参考論文:
  Zheng et al., "Beyond Classification: Evaluating LLMs for Fine-Grained
  Automatic Malware Behavior Auditing" (MalEval), arXiv:2509.14335, 2025.

使い方:
  export OPENAI_API_KEY="sk-..."
  python3 src/generate_evidence_report.py --model-type keyword
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("エラー: openai ライブラリがインストールされていません。")
    print("  pip install openai")
    sys.exit(1)


# ============================================================
# GPT プロンプト設計
# ============================================================
SYSTEM_PROMPT = """あなたはマルウェア動的解析の専門家です。
CAPEv2サンドボックスで収集されたWindows APIコールトレースのSHAP分析結果を基に、
マルウェアの挙動を因果的に説明するレポートを作成してください。

## あなたのタスク

### タスク1: Evidence Attribution（証拠帰属）
各キーワードについて、なぜそれがこのマルウェアラベル（機能）の根拠となるのかを、
具体的なAPIコールの文脈に基づいて説明してください。
セキュリティアナリストが読んで納得できる、技術的に正確な説明を心がけてください。

### タスク2: Behavior Synthesis（挙動要約）
与えられたエビデンスを統合し、このラベルが示すマルウェアの挙動を
簡潔かつ論理的に要約してください。

## 出力フォーマット
以下のJSON形式で返してください。すべて日本語で記述してください。
```json
{
  "evidence_attributions": [
    {
      "keyword": "キーワード名",
      "explanation": "このキーワードが根拠となる理由の説明",
      "causal_chain": "具体的なAPIコール → 抽出されたキーワード → MBCカテゴリ → マルウェア機能ラベル の因果連鎖",
      "confidence": "high/medium/low"
    }
  ],
  "behavior_synthesis": "このラベルが示すマルウェアの挙動を要約した説明文（3-5文程度）"
}
```"""

OVERALL_SYSTEM_PROMPT = """あなたはマルウェア動的解析の専門家です。
CAPEv2サンドボックスで収集されたWindows APIコールトレースのSHAP分析結果を基に、
マルウェアの全体的な攻撃シナリオを要約してください。

以下の情報が与えられます:
- マルウェアの各機能ラベルごとのSHAP分析で重要と判定されたキーワードとその説明
- 各ラベルの挙動要約

これらを統合し、マルウェアが全体としてどのような攻撃を行っているかを
セキュリティレポートとして要約してください。

## 出力フォーマット
以下のJSON形式で返してください。すべて日本語で記述してください。
```json
{
  "overall_narrative": "マルウェアの全体的な攻撃シナリオの要約（5-8文程度）",
  "key_findings": [
    "重要な発見1",
    "重要な発見2",
    "重要な発見3"
  ],
  "risk_assessment": "このマルウェアの危険度に関する評価（2-3文程度）"
}
```"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="MalEvalベースのEvidence Attribution & Behavior Synthesisレポート生成")
    parser.add_argument("--model-type", type=str, choices=["keyword", "pca"],
                        default="keyword", help="分析対象のモデル（default: keyword）")
    parser.add_argument("--shap-results", type=str, default=None,
                        help="SHAP構造化結果JSONのパス (default: 自動検出)")
    parser.add_argument("--mbc-mapping", type=str, default="features/mbc_keyword_mapping.json",
                        help="MBCマッピングJSONのパス")
    parser.add_argument("--api-keywords", type=str, default="api_keywords_single.json",
                        help="APIキーワードJSONのパス")
    parser.add_argument("--api-descriptions", type=str, default="api_descriptions.json",
                        help="API説明文JSONのパス")
    parser.add_argument("--gpt-model", type=str, default="gpt-4o",
                        help="使用するGPTモデル (default: gpt-4o)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="出力ディレクトリ (default: SHAP結果と同じディレクトリ)")
    parser.add_argument("--dry-run", action="store_true",
                        help="GPT APIを呼ばず、プロンプトのみを表示する")
    return parser.parse_args()


def load_shap_results(shap_results_path):
    """SHAP構造化結果JSONを読み込む"""
    with open(shap_results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mbc_mapping(mbc_mapping_path):
    """MBCマッピング辞書を読み込む"""
    if not os.path.exists(mbc_mapping_path):
        print(f"警告: MBCマッピング '{mbc_mapping_path}' が見つかりません。")
        return {}
    with open(mbc_mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_keyword_to_apis(api_keywords_path):
    """keyword → [API1, API2, ...] の逆引き辞書を構築"""
    keyword_to_apis = defaultdict(list)
    if not os.path.exists(api_keywords_path):
        print(f"警告: '{api_keywords_path}' が見つかりません。")
        return keyword_to_apis
    with open(api_keywords_path, "r", encoding="utf-8") as f:
        api_keywords = json.load(f)
    for api_name, keywords in api_keywords.items():
        for kw in keywords:
            keyword_to_apis[kw].append(api_name)
    return keyword_to_apis


def load_api_descriptions(api_descriptions_path):
    """API説明文を読み込む"""
    if not os.path.exists(api_descriptions_path):
        print(f"警告: '{api_descriptions_path}' が見つかりません。")
        return {}
    with open(api_descriptions_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_evidence_package(label, label_data, mbc_mapping, keyword_to_apis, api_descriptions):
    """ラベルごとの Evidence Package を構築する"""
    package = {
        "label": label,
        "category_ratio": {
            "keyword_ratio": label_data.get("main_ratio", 0),
            "api_ratio": label_data.get("api_ratio", 0)
        },
        "evidence_items": []
    }

    for kw_info in label_data.get("functional_keywords", []):
        keyword = kw_info["keyword"]
        source_apis = keyword_to_apis.get(keyword, [])

        # API説明文を収集（最大5件）
        api_descs = {}
        for api in source_apis[:5]:
            if api in api_descriptions:
                # 説明文が長い場合は先頭200文字に制限
                desc = api_descriptions[api]
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                api_descs[api] = desc

        item = {
            "keyword": keyword,
            "shap_importance": kw_info["shap_importance"],
            "mbc_category": kw_info.get("category", "Unknown"),
            "source_apis": source_apis[:8],  # 最大8件
            "api_descriptions": api_descs
        }
        package["evidence_items"].append(item)

    return package


def build_user_prompt_for_label(evidence_package):
    """ラベルごとのユーザープロンプトを構築"""
    label = evidence_package["label"]
    ratio = evidence_package["category_ratio"]

    lines = [
        f"## 分析対象ラベル: {label}",
        f"キーワード特徴量の寄与率: {ratio['keyword_ratio']:.1f}% / API頻度特徴量の寄与率: {ratio['api_ratio']:.1f}%",
        "",
        "## SHAPで重要と判定されたキーワード一覧:",
    ]

    for i, item in enumerate(evidence_package["evidence_items"]):
        lines.append(f"\n### {i+1}. キーワード: \"{item['keyword']}\" (SHAP重要度: {item['shap_importance']:.6f})")
        lines.append(f"   - MBCカテゴリ: {item['mbc_category']}")
        lines.append(f"   - 由来API: {', '.join(item['source_apis'])}")

        if item["api_descriptions"]:
            lines.append("   - API機能説明:")
            for api, desc in item["api_descriptions"].items():
                lines.append(f"     * {api}: {desc}")

    return "\n".join(lines)


def call_gpt(client, model, system_prompt, user_prompt, max_retries=3):
    """GPT APIを呼び出し、JSONレスポンスをパースして返す"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)

        except json.JSONDecodeError as e:
            print(f"  警告: JSONパースエラー (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                # 最後のリトライでも失敗した場合、raw textを返す
                return {"error": "JSONパース失敗", "raw_response": content}

        except Exception as e:
            print(f"  警告: API呼び出しエラー (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                return {"error": str(e)}


def generate_html_report(output_dir, all_label_results, overall_result, model_type):
    """Evidence Attribution & Behavior Synthesis のHTMLレポートを生成"""
    html_path = output_dir / "evidence_report.html"

    cat_css = {
        "File System": "cat-file", "Registry": "cat-registry",
        "Process/Thread": "cat-process", "Network/Communication": "cat-network",
        "Memory": "cat-memory", "Cryptography": "cat-crypto",
        "System Info/Discovery": "cat-sysinfo", "Service": "cat-service",
        "Synchronization": "cat-sync", "GUI/Input": "cat-gui", "COM": "cat-com"
    }
    confidence_icons = {"high": "🟢", "medium": "🟡", "low": "🔴"}

    lines = [
        '<!DOCTYPE html>',
        '<html lang="ja">',
        '<head>',
        '  <meta charset="utf-8">',
        f'  <title>Evidence Attribution レポート ({model_type}モデル)</title>',
        '  <style>',
        '    * { box-sizing: border-box; }',
        '    body { font-family: "Segoe UI", "Hiragino Sans", Arial, sans-serif; margin: 0; padding: 32px; background: #0f0f1a; color: #e0e0e8; line-height: 1.7; }',
        '    h1 { color: #a0c4ff; border-bottom: 3px solid #7b2ff7; padding-bottom: 12px; font-size: 28px; }',
        '    h2 { color: #c9b1ff; margin-top: 36px; font-size: 22px; border-left: 4px solid #7b2ff7; padding-left: 12px; }',
        '    h3 { color: #a0c4ff; margin-top: 20px; font-size: 16px; }',
        '    section { background: #1a1a2e; padding: 24px 28px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.3); margin-bottom: 28px; border: 1px solid #2a2a4a; }',
        '    .overview-section { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 1px solid #7b2ff7; }',
        '    .synthesis { background: #16213e; padding: 16px 20px; border-radius: 8px; border-left: 4px solid #00d4aa; font-size: 15px; margin: 12px 0; }',
        '    .narrative { background: #1e1e3a; padding: 16px 20px; border-radius: 8px; border-left: 4px solid #ff6b6b; font-size: 15px; margin: 12px 0; }',
        '    .finding { background: #16213e; padding: 10px 16px; border-radius: 6px; margin: 6px 0; border-left: 3px solid #ffd93d; }',
        '    table { border-collapse: collapse; width: 100%; margin: 16px 0; }',
        '    th, td { border: 1px solid #2a2a4a; padding: 10px 14px; text-align: left; font-size: 14px; }',
        '    th { background: #16213e; font-weight: 600; color: #a0c4ff; }',
        '    tr:nth-child(even) { background: #1e1e3a; }',
        '    tr:hover { background: #252550; }',
        '    .category-badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }',
        '    .cat-file { background: #1b4332; color: #95d5b2; }',
        '    .cat-registry { background: #1b3a5c; color: #90caf9; }',
        '    .cat-process { background: #4a1942; color: #f48fb1; }',
        '    .cat-network { background: #1a3c4a; color: #80deea; }',
        '    .cat-memory { background: #2a2a3a; color: #b0bec5; }',
        '    .cat-crypto { background: #4a3f1a; color: #ffd54f; }',
        '    .cat-sysinfo { background: #3a1a1a; color: #ef9a9a; }',
        '    .cat-service { background: #2a2a2a; color: #bdbdbd; }',
        '    .cat-sync { background: #1a2a3a; color: #81d4fa; }',
        '    .cat-gui { background: #2a1a3a; color: #ce93d8; }',
        '    .cat-com { background: #1a2a2a; color: #a5d6a7; }',
        '    .causal-chain { font-family: "Courier New", monospace; font-size: 13px; color: #80cbc4; background: #0d1117; padding: 8px 12px; border-radius: 6px; display: block; margin-top: 4px; white-space: pre-wrap; }',
        '    .confidence { font-size: 16px; }',
        '    nav { margin: 16px 0 24px 0; }',
        '    nav a { margin-right: 12px; color: #7b9ff7; text-decoration: none; font-size: 14px; }',
        '    nav a:hover { text-decoration: underline; color: #a0c4ff; }',
        '    .label-tag { display: inline-block; background: #7b2ff7; color: #fff; padding: 4px 12px; border-radius: 16px; font-size: 13px; font-weight: 600; margin-right: 8px; }',
        '    .subtitle { color: #888; font-size: 14px; margin-top: -8px; }',
        '  </style>',
        '</head>',
        '<body>',
        f'<h1>🔍 Evidence Attribution レポート — {model_type.upper()} モデル</h1>',
        '<p class="subtitle">MalEval (Zheng et al., 2025) の手法に基づく、SHAP分析結果の因果的説明</p>',
        '<nav>',
        '  <a href="#overview">📋 全体サマリ</a>',
    ]

    # ナビゲーション
    for label in all_label_results:
        lines.append(f'  <a href="#label-{label}">📌 {label}</a>')
    lines.append('</nav>')

    # === 全体サマリ ===
    lines.append('<section id="overview" class="overview-section">')
    lines.append('<h2>📋 全体サマリ（Overall Attack Narrative）</h2>')
    if overall_result and "overall_narrative" in overall_result:
        lines.append(f'<div class="synthesis">{overall_result["overall_narrative"]}</div>')

        if "key_findings" in overall_result:
            lines.append('<h3>🔑 重要な発見</h3>')
            for finding in overall_result["key_findings"]:
                lines.append(f'<div class="finding">{finding}</div>')

        if "risk_assessment" in overall_result:
            lines.append('<h3>⚠️ リスク評価</h3>')
            lines.append(f'<div class="narrative">{overall_result["risk_assessment"]}</div>')
    elif overall_result and "error" in overall_result:
        lines.append(f'<p>⚠️ 全体サマリの生成に失敗しました: {overall_result["error"]}</p>')
    lines.append('</section>')

    # === ラベル別セクション ===
    for label, result in all_label_results.items():
        lines.append(f'<section id="label-{label}">')
        lines.append(f'<h2><span class="label-tag">{label}</span> Evidence Attribution</h2>')

        if "error" in result:
            lines.append(f'<p>⚠️ このラベルの分析に失敗しました: {result["error"]}</p>')
            lines.append('</section>')
            continue

        # Behavior Synthesis
        if "behavior_synthesis" in result:
            lines.append('<h3>📝 挙動の要約（Behavior Synthesis）</h3>')
            lines.append(f'<div class="synthesis">{result["behavior_synthesis"]}</div>')

        # Evidence Attribution テーブル
        if "evidence_attributions" in result and result["evidence_attributions"]:
            lines.append('<h3>🔗 根拠の詳細（Evidence Attribution）</h3>')
            lines.append('<table>')
            lines.append('<tr><th>キーワード</th><th>カテゴリ</th><th>根拠の説明</th><th>確信度</th></tr>')

            for attr in result["evidence_attributions"]:
                kw = attr.get("keyword", "")
                explanation = attr.get("explanation", "")
                chain = attr.get("causal_chain", "")
                conf = attr.get("confidence", "medium")
                # カテゴリは evidence_package から取得できないので、explanation から推測
                conf_icon = confidence_icons.get(conf, "⚪")

                lines.append('<tr>')
                lines.append(f'  <td><strong>{kw}</strong></td>')
                lines.append(f'  <td>—</td>')
                lines.append(f'  <td>{explanation}<span class="causal-chain">{chain}</span></td>')
                lines.append(f'  <td class="confidence">{conf_icon} {conf}</td>')
                lines.append('</tr>')

            lines.append('</table>')

        lines.append('</section>')

    lines.extend([
        '<footer style="text-align: center; color: #555; margin-top: 40px; font-size: 12px;">',
        '  Generated by generate_evidence_report.py | Based on MalEval (Zheng et al., 2025)',
        '</footer>',
        '</body></html>'
    ])

    html_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"HTMLレポートを {html_path} に保存しました。")


def main():
    args = parse_args()

    # --- APIキーの確認 ---
    if not args.dry_run:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("エラー: 環境変数 OPENAI_API_KEY が設定されていません。")
            print("  export OPENAI_API_KEY='sk-...'")
            sys.exit(1)
        client = OpenAI()
        print(f"GPTモデル: {args.gpt_model}")
    else:
        client = None
        print("=== DRY RUN モード（GPT APIは呼び出しません） ===")

    # --- データの読み込み ---
    shap_results_path = args.shap_results
    if shap_results_path is None:
        shap_results_path = f"logs_shap_analysis/{args.model_type}/shap_structured_results.json"

    if not os.path.exists(shap_results_path):
        print(f"エラー: SHAP構造化結果が見つかりません: {shap_results_path}")
        print("先に analyze_shap_hybrid.py を実行してください。")
        sys.exit(1)

    print(f"SHAP結果を読み込み中: {shap_results_path}")
    shap_results = load_shap_results(shap_results_path)

    print(f"MBCマッピングを読み込み中: {args.mbc_mapping}")
    mbc_mapping = load_mbc_mapping(args.mbc_mapping)

    print(f"APIキーワード逆引き辞書を構築中: {args.api_keywords}")
    keyword_to_apis = build_keyword_to_apis(args.api_keywords)

    print(f"API説明文を読み込み中: {args.api_descriptions}")
    api_descriptions = load_api_descriptions(args.api_descriptions)

    # --- 出力ディレクトリ ---
    output_dir = Path(args.output_dir) if args.output_dir else Path(shap_results_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- ラベルごとの Evidence Attribution ---
    per_label = shap_results.get("per_label", {})
    all_label_results = {}
    label_syntheses = {}  # 全体サマリ用に各ラベルの要約を収集

    total_labels = len(per_label)
    print(f"\n{'='*60}")
    print(f"Evidence Attribution を {total_labels} ラベルについて実行します")
    print(f"{'='*60}")

    for i, (label, label_data) in enumerate(per_label.items()):
        print(f"\n[{i+1}/{total_labels}] ラベル: {label}")

        # キーワードが無いラベルはスキップ
        if not label_data.get("functional_keywords"):
            print(f"  → 機能直結キーワードが無いためスキップ")
            all_label_results[label] = {
                "behavior_synthesis": "このラベルに対して機能直結キーワードが検出されなかったため、分析をスキップしました。",
                "evidence_attributions": []
            }
            continue

        # Evidence Package の構築
        evidence_package = build_evidence_package(
            label, label_data, mbc_mapping, keyword_to_apis, api_descriptions
        )
        user_prompt = build_user_prompt_for_label(evidence_package)

        if args.dry_run:
            print(f"  --- プロンプト ---")
            print(user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt)
            all_label_results[label] = {"dry_run": True}
            continue

        # GPT API 呼び出し
        print(f"  GPT-4o に問い合わせ中...")
        result = call_gpt(client, args.gpt_model, SYSTEM_PROMPT, user_prompt)

        if "error" in result:
            print(f"  ⚠️ エラー: {result['error']}")
        else:
            n_attrs = len(result.get("evidence_attributions", []))
            print(f"  ✅ 完了: {n_attrs}件のEvidence Attribution を取得")
            if "behavior_synthesis" in result:
                label_syntheses[label] = result["behavior_synthesis"]

        all_label_results[label] = result

        # 中間結果を保存（万が一の途中失敗に備える）
        intermediate_path = output_dir / "evidence_results_partial.json"
        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump(all_label_results, f, indent=2, ensure_ascii=False)

        # レート制限対策
        time.sleep(1)

    # --- 全体サマリの生成 ---
    overall_result = {}
    if not args.dry_run and label_syntheses:
        print(f"\n{'='*60}")
        print(f"全体の攻撃シナリオサマリを生成中...")
        print(f"{'='*60}")

        overall_prompt_lines = ["以下は各マルウェア機能ラベルのSHAP分析に基づく挙動要約です:\n"]
        for label, synthesis in label_syntheses.items():
            overall_prompt_lines.append(f"### ラベル: {label}")
            overall_prompt_lines.append(f"{synthesis}\n")
        overall_prompt = "\n".join(overall_prompt_lines)
        overall_prompt += "\n上記の全ラベルの挙動を統合し、マルウェアの全体的な攻撃シナリオを要約してください。"

        overall_result = call_gpt(client, args.gpt_model, OVERALL_SYSTEM_PROMPT, overall_prompt)

        if "error" in overall_result:
            print(f"  ⚠️ 全体サマリのエラー: {overall_result['error']}")
        else:
            print(f"  ✅ 全体サマリの生成完了")

    # --- 結果の保存 ---
    # JSON
    final_results = {
        "model_type": shap_results.get("model_type", args.model_type),
        "gpt_model": args.gpt_model,
        "overall": overall_result,
        "per_label": all_label_results
    }
    json_path = output_dir / "evidence_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON結果を {json_path} に保存しました。")

    # 中間ファイルを削除
    intermediate_path = output_dir / "evidence_results_partial.json"
    if intermediate_path.exists():
        intermediate_path.unlink()

    # HTML
    if not args.dry_run:
        generate_html_report(output_dir, all_label_results, overall_result, args.model_type)

    print(f"\n{'='*60}")
    print(f"Evidence Attribution レポートの生成が完了しました！")
    print(f"{'='*60}")
    print(f"  - {json_path}   (構造化データ)")
    if not args.dry_run:
        print(f"  - {output_dir / 'evidence_report.html'}   (HTMLレポート)")


if __name__ == "__main__":
    main()
