#!/usr/bin/env python3
"""
MalEval論文の知見を取り入れた Evidence Attribution & Behavior Synthesis スクリプト。

2つのモードで動作する:
  (A) ラベル集約モード: 全検体を集約したSHAP結果から、ラベルごとの根拠を説明
  (B) 検体個別モード:   指定した検体のAPIコール列・SHAP値を基に、その検体の挙動を説明

参考論文:
  Zheng et al., "Beyond Classification: Evaluating LLMs for Fine-Grained
  Automatic Malware Behavior Auditing" (MalEval), arXiv:2509.14335, 2025.

使い方:
  # ラベル集約レポートのみ
  python3 src/generate_evidence_report.py --model-type keyword

  # ラベル集約 + 検体個別レポート（検体0,5,10を指定）
  python3 src/generate_evidence_report.py --model-type keyword --sample-indices 0,5,10

  # 検体個別レポートのみ
  python3 src/generate_evidence_report.py --model-type keyword --sample-indices 0,5,10 --skip-aggregate
"""

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

# scikit-learnのバージョン違い警告を非表示にする
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from openai import OpenAI
except ImportError:
    print("エラー: openai ライブラリがインストールされていません。")
    print("  pip install openai")
    sys.exit(1)


# ============================================================
# GPT プロンプト設計
# ============================================================

# --- ラベル集約分析用 ---
LABEL_SYSTEM_PROMPT = """あなたはマルウェア動的解析の専門家です。
CAPEv2サンドボックスで収集されたWindows APIコールトレースのSHAP分析結果を基に、
マルウェアの挙動を因果的に説明するレポートを作成してください。

### タスク1: Evidence Attribution（証拠帰属）
各キーワードについて、なぜそれがこのマルウェアラベル（機能）の根拠となるのかを、
具体的なAPIコールの文脈に基づいて説明してください。

### タスク2: Behavior Synthesis（挙動要約）
与えられたエビデンスを統合し、このラベルが示すマルウェアの挙動を
簡潔かつ論理的に要約してください。

出力は以下のJSON形式で返してください。すべて日本語で記述してください。
```json
{
  "evidence_attributions": [
    {
      "keyword": "キーワード名",
      "explanation": "このキーワードが根拠となる理由の説明",
      "causal_chain": "具体的なAPIコール → キーワード → MBCカテゴリ → マルウェア機能ラベル の因果連鎖",
      "confidence": "high/medium/low"
    }
  ],
  "behavior_synthesis": "このラベルが示すマルウェアの挙動を要約した説明文（3-5文程度）"
}
```"""

OVERALL_SYSTEM_PROMPT = """あなたはマルウェア動的解析の専門家です。
各機能ラベルごとの挙動要約を統合し、マルウェアの全体的な攻撃シナリオを要約してください。

出力は以下のJSON形式で返してください。すべて日本語で記述してください。
```json
{
  "overall_narrative": "マルウェアの全体的な攻撃シナリオの要約（5-8文程度）",
  "key_findings": ["重要な発見1", "重要な発見2", "重要な発見3"],
  "risk_assessment": "このマルウェアの危険度に関する評価（2-3文程度）"
}
```"""

# --- 検体個別分析用 ---
SAMPLE_SYSTEM_PROMPT = """あなたはマルウェア動的解析の専門家です。
CAPEv2サンドボックスで実行された特定のマルウェア検体について、
そのAPIコールトレースとSHAP分析結果を基に、この検体が何を行っているかを
因果的に説明してください。

### タスク
1. この検体のAPIコール列から、マルウェアとしての具体的な挙動を特定してください
2. SHAP値が高い特徴量について、なぜそれがこの検体の分類に重要だったのかを説明してください
3. この検体の全体的な攻撃シナリオを要約してください

出力は以下のJSON形式で返してください。すべて日本語で記述してください。
```json
{
  "sample_behavior": "この検体が行っている挙動の要約（3-5文）",
  "predicted_labels_explanation": [
    {
      "label": "ラベル名",
      "prediction_probability": 0.95,
      "explanation": "なぜこのラベルが予測されたかの説明",
      "key_evidence": ["根拠となるAPIコール/キーワード1", "根拠2"]
    }
  ],
  "attack_scenario": "この検体の攻撃シナリオ全体の要約（2-3文）",
  "risk_level": "high/medium/low",
  "risk_reason": "リスクレベルの根拠（1-2文）"
}
```"""


# ============================================================
# 引数パーサ
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="MalEvalベースのEvidence Attribution & Behavior Synthesisレポート生成")
    parser.add_argument("--model-type", type=str, choices=["keyword", "pca"],
                        default="keyword", help="分析対象のモデル (default: keyword)")
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
    # --- 検体個別分析用 ---
    parser.add_argument("--sample-indices", type=str, default=None,
                        help="分析対象の検体インデックス (カンマ区切り, 例: 0,5,10)")
    parser.add_argument("--test-dir", type=str, default="2024/Dataset_Extract/2017",
                        help="テストデータディレクトリ (default: 2024/Dataset_Extract/2017)")
    parser.add_argument("--skip-aggregate", action="store_true",
                        help="ラベル全体の集約分析をスキップする")
    parser.add_argument("--dry-run", action="store_true",
                        help="GPT APIを呼ばず、プロンプトのみを表示する")
    return parser.parse_args()


# ============================================================
# データ読み込み
# ============================================================
def load_shap_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_mbc_mapping(path):
    if not os.path.exists(path):
        print(f"警告: MBCマッピング '{path}' が見つかりません。")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_keyword_to_apis(path):
    kw2apis = defaultdict(list)
    if not os.path.exists(path):
        return kw2apis
    with open(path, "r", encoding="utf-8") as f:
        for api, keywords in json.load(f).items():
            for kw in keywords:
                kw2apis[kw].append(api)
    return kw2apis

def load_api_descriptions(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_models_and_test_data(model_type):
    """学習済みモデルとテストデータを読み込む"""
    features_dir = Path("features")
    label_set_path = features_dir / "label_set.json"
    with open(label_set_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if isinstance(labels, dict):
        label_names = [k for k, v in sorted(labels.items(), key=lambda item: item[1])]
    else:
        label_names = labels

    if model_type == "keyword":
        test_df = pd.read_csv(features_dir / "test_keyword_features.csv", index_col=0)
        feature_names = test_df.columns.tolist()
        X_test = test_df.values
        sample_names = test_df.index.tolist()
        models_dir = Path("logs_keyword/models_br_+freq_rf")
    else:
        raise ValueError(f"検体個別分析は現在 keyword モデルのみ対応しています")

    return X_test, feature_names, sample_names, label_names, models_dir

def load_sample_api_calls(test_dir, sample_name):
    """データセットJSONからサンプルの生APIコール列を読み込む"""
    sample_path = Path(test_dir) / f"{sample_name}.json"
    if not sample_path.exists():
        return None, None
    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("apicalls", []), data.get("functions", [])


# ============================================================
# ラベル集約分析（Mode A）
# ============================================================
def build_label_evidence_package(label, label_data, mbc_mapping, keyword_to_apis, api_descriptions):
    package = {"label": label, "evidence_items": []}
    for kw_info in label_data.get("functional_keywords", []):
        keyword = kw_info["keyword"]
        source_apis = keyword_to_apis.get(keyword, [])
        api_descs = {}
        for api in source_apis[:5]:
            if api in api_descriptions:
                desc = api_descriptions[api]
                api_descs[api] = desc[:200] + "..." if len(desc) > 200 else desc
        package["evidence_items"].append({
            "keyword": keyword,
            "shap_importance": kw_info["shap_importance"],
            "mbc_category": kw_info.get("category", "Unknown"),
            "source_apis": source_apis[:8],
            "api_descriptions": api_descs
        })
    return package

def build_user_prompt_for_label(pkg):
    lines = [
        f"## 分析対象ラベル: {pkg['label']}",
        "",
        "## SHAPで重要と判定されたキーワード一覧:",
    ]
    for i, item in enumerate(pkg["evidence_items"]):
        lines.append(f"\n### {i+1}. キーワード: \"{item['keyword']}\" (SHAP重要度: {item['shap_importance']:.6f})")
        lines.append(f"   - MBCカテゴリ: {item['mbc_category']}")
        lines.append(f"   - 由来API: {', '.join(item['source_apis'])}")
        if item["api_descriptions"]:
            lines.append("   - API機能説明:")
            for api, desc in item["api_descriptions"].items():
                lines.append(f"     * {api}: {desc}")
    return "\n".join(lines)


# ============================================================
# 検体個別分析（Mode B）
# ============================================================
def compute_sample_shap(sample_vector, models_dir, label_names, feature_names):
    """特定サンプルのSHAP値を全ラベルについて計算する"""
    results = {}
    sample_2d = sample_vector.reshape(1, -1)

    for label in label_names:
        model_path = models_dir / f"{label}.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        explainer = shap.TreeExplainer(model)
        shap_obj = explainer(sample_2d)

        if len(shap_obj.shape) == 3:
            shap_vals = shap_obj.values[0, :, 1]
        else:
            shap_vals = shap_obj.values[0]

        pred_proba = float(model.predict_proba(sample_2d)[0][1])

        # SHAP寄与量の絶対値上位10件
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:10]
        top_features = [
            {"feature": feature_names[j],
             "shap_value": round(float(shap_vals[j]), 6),
             "feature_value": round(float(sample_vector[j]), 4)}
            for j in top_idx
        ]
        results[label] = {"prediction_probability": round(pred_proba, 4),
                          "top_features": top_features}
    return results

def build_user_prompt_for_sample(sample_name, api_calls, ground_truth_labels,
                                  shap_results, feature_names, feature_vector,
                                  keyword_to_apis, mbc_mapping):
    """検体個別分析用のプロンプトを構築する"""
    lines = [f"## 検体名: {sample_name}", ""]

    # --- APIコール列サマリ ---
    if api_calls:
        api_counts = Counter(api_calls)
        lines.append("### この検体が呼び出したAPI（上位20件、呼び出し回数順）:")
        for rank, (api, cnt) in enumerate(api_counts.most_common(20), 1):
            lines.append(f"  {rank:2d}. {api}: {cnt}回")
        lines.append(f"\n  合計ユニークAPI数: {len(api_counts)}, 合計APIコール数: {len(api_calls)}")
    else:
        lines.append("### APIコール情報: 取得不可")

    # --- 正解ラベル ---
    if ground_truth_labels:
        lines.append(f"\n### 正解ラベル（データセット上の機能）: {', '.join(ground_truth_labels)}")

    # --- アクティブなキーワード特徴量 ---
    lines.append("\n### この検体で値が非ゼロのキーワード特徴量:")
    active_kws = []
    for i, fname in enumerate(feature_names):
        if not fname.startswith("api__") and feature_vector[i] > 0:
            mbc_cat = mbc_mapping.get(fname, {}).get("category", "N/A")
            active_kws.append((fname, feature_vector[i], mbc_cat))
    active_kws.sort(key=lambda x: x[1], reverse=True)
    for kw, val, cat in active_kws[:30]:
        lines.append(f"  - {kw}: {val:.0f} (MBC: {cat})")

    # --- 予測確率が高いラベルのSHAP ---
    lines.append("\n### ラベル予測確率とSHAP上位特徴量:")
    sorted_labels = sorted(shap_results.items(),
                           key=lambda x: x[1]["prediction_probability"], reverse=True)
    for label, ldata in sorted_labels:
        prob = ldata["prediction_probability"]
        if prob < 0.1:
            continue  # 予測確率が低いラベルは省略
        lines.append(f"\n  #### ラベル: {label} (予測確率: {prob:.3f})")
        for feat in ldata["top_features"][:5]:
            direction = "↑" if feat["shap_value"] > 0 else "↓"
            lines.append(f"    - {feat['feature']}: SHAP={feat['shap_value']:+.4f} {direction} (値={feat['feature_value']})")

    return "\n".join(lines)


# ============================================================
# GPT API 呼び出し
# ============================================================
def call_gpt(client, model, system_prompt, user_prompt, max_retries=3):
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
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            print(f"  警告: JSONパースエラー (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                return {"error": "JSONパース失敗", "raw": response.choices[0].message.content}
        except Exception as e:
            print(f"  警告: API呼び出しエラー (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                return {"error": str(e)}
        time.sleep(2 ** attempt)


# ============================================================
# HTML レポート生成
# ============================================================
def generate_html_report(output_dir, aggregate_results, overall_result,
                          sample_results, model_type):
    html_path = output_dir / "evidence_report.html"
    conf_icons = {"high": "🟢", "medium": "🟡", "low": "🔴"}

    lines = [
        '<!DOCTYPE html>', '<html lang="ja">', '<head>', '  <meta charset="utf-8">',
        f'  <title>Evidence Attribution レポート ({model_type}モデル)</title>',
        '  <style>',
        '    body { font-family: "ＭＳ Ｐゴシック", "MS PGothic", sans-serif; background-color: #ffffff; color: #333333; margin: 20px auto; max-width: 900px; padding: 10px; line-height: 1.6; }',
        '    h1 { font-size: 24px; border-bottom: 2px solid #666666; padding-bottom: 5px; margin-bottom: 20px; }',
        '    h2 { font-size: 20px; border-left: 5px solid #666666; padding-left: 10px; margin-top: 30px; background-color: #f0f0f0; padding-top: 5px; padding-bottom: 5px; }',
        '    h3 { font-size: 16px; border-bottom: 1px dotted #999999; margin-top: 20px; }',
        '    section { margin-bottom: 30px; }',
        '    .synthesis { background-color: #f9f9f9; border: 1px solid #cccccc; padding: 15px; margin: 10px 0; }',
        '    .narrative { background-color: #fff4f4; border: 1px solid #ffcccc; padding: 15px; margin: 10px 0; }',
        '    .finding { margin: 5px 0; }',
        '    table { border-collapse: collapse; width: 100%; margin: 15px 0; }',
        '    th, td { border: 1px solid #999999; padding: 8px; text-align: left; font-size: 14px; }',
        '    th { background-color: #eeeeee; }',
        '    .chain { font-family: monospace; font-size: 12px; color: #006600; display: block; margin-top: 5px; white-space: pre-wrap; }',
        '    .label-tag { font-weight: bold; }',
        '    .sample-tag { font-weight: bold; color: #000066; }',
        '    .prob-bar { height: 10px; display: inline-block; background-color: #666666; }',
        '    .risk-high { color: #cc0000; font-weight: bold; }',
        '    .risk-medium { color: #cc6600; font-weight: bold; }',
        '    .risk-low { color: #006600; font-weight: bold; }',
        '    nav { margin: 10px 0 20px 0; padding: 10px; border: 1px dashed #cccccc; background-color: #fafafa; }',
        '    nav a { margin-right: 15px; color: #0000ff; text-decoration: underline; font-size: 14px; }',
        '    nav a:hover { color: #ff0000; }',
        '    .subtitle { color: #666666; font-size: 14px; }',
        '    hr { border: none; border-top: 1px dashed #999; margin: 30px 0; }',
        '  </style>',
        '</head>', '<body>',
        f'<h1>🔍 Evidence Attribution レポート — {model_type.upper()} モデル</h1>',
        '<p class="subtitle">MalEval (Zheng et al., 2025) の手法に基づく、SHAP分析結果の因果的説明</p>',
        '<nav>',
    ]

    # ナビゲーション
    if overall_result:
        lines.append('  <a href="#overview">📋 全体サマリ</a>')
    if aggregate_results:
        for label in aggregate_results:
            lines.append(f'  <a href="#label-{label}">📌 {label}</a>')
    if sample_results:
        for sname in sample_results:
            lines.append(f'  <a href="#sample-{sname}">🔬 {sname}</a>')
    lines.append('</nav>')

    # === 全体サマリ ===
    if overall_result and "overall_narrative" in overall_result:
        lines.append('<section id="overview" class="overview">')
        lines.append('<h2>📋 全体サマリ</h2>')
        lines.append(f'<div class="synthesis">{overall_result["overall_narrative"]}</div>')
        if "key_findings" in overall_result:
            lines.append('<h3>🔑 重要な発見</h3>')
            for f in overall_result["key_findings"]:
                lines.append(f'<div class="finding">{f}</div>')
        if "risk_assessment" in overall_result:
            lines.append('<h3>⚠️ リスク評価</h3>')
            lines.append(f'<div class="narrative">{overall_result["risk_assessment"]}</div>')
        lines.append('</section>')

    # === ラベル別 ===
    for label, res in aggregate_results.items():
        lines.append(f'<section id="label-{label}">')
        lines.append(f'<h2><span class="label-tag">{label}</span></h2>')
        if "error" in res:
            lines.append(f'<p>⚠️ {res["error"]}</p>')
        else:
            if "behavior_synthesis" in res:
                lines.append('<h3>📝 挙動の要約</h3>')
                lines.append(f'<div class="synthesis">{res["behavior_synthesis"]}</div>')
            if "evidence_attributions" in res:
                lines.append('<h3>🔗 根拠の詳細</h3><table>')
                lines.append('<tr><th>キーワード</th><th>根拠の説明</th><th>因果連鎖</th><th>確信度</th></tr>')
                for a in res["evidence_attributions"]:
                    ci = conf_icons.get(a.get("confidence", "medium"), "⚪")
                    lines.append(f'<tr><td><strong>{a.get("keyword","")}</strong></td>')
                    lines.append(f'<td>{a.get("explanation","")}</td>')
                    lines.append(f'<td><span class="chain">{a.get("causal_chain","")}</span></td>')
                    lines.append(f'<td>{ci} {a.get("confidence","")}</td></tr>')
                lines.append('</table>')
        lines.append('</section>')

    # === 検体個別 ===
    for sname, sres in sample_results.items():
        lines.append(f'<section id="sample-{sname}" class="sample-section">')
        lines.append(f'<h2><span class="sample-tag">🔬 検体: {sname}</span></h2>')
        if "error" in sres:
            lines.append(f'<p>⚠️ {sres["error"]}</p>')
        else:
            if "sample_behavior" in sres:
                lines.append('<h3>📝 検体の挙動要約</h3>')
                lines.append(f'<div class="synthesis">{sres["sample_behavior"]}</div>')

            if "predicted_labels_explanation" in sres:
                lines.append('<h3>🏷️ 予測ラベルの根拠</h3><table>')
                lines.append('<tr><th>ラベル</th><th>予測確率</th><th>説明</th><th>根拠</th></tr>')
                for pl in sres["predicted_labels_explanation"]:
                    prob = pl.get("prediction_probability", 0)
                    pcls = "prob-high" if prob > 0.7 else ("prob-mid" if prob > 0.3 else "prob-low")
                    evidence = ", ".join(pl.get("key_evidence", []))
                    lines.append(f'<tr><td><strong>{pl.get("label","")}</strong></td>')
                    lines.append(f'<td><span class="prob-bar {pcls}" style="width:{int(prob*100)}px"></span> {prob:.1%}</td>')
                    lines.append(f'<td>{pl.get("explanation","")}</td>')
                    lines.append(f'<td>{evidence}</td></tr>')
                lines.append('</table>')

            if "attack_scenario" in sres:
                lines.append('<h3>🎯 攻撃シナリオ</h3>')
                lines.append(f'<div class="narrative">{sres["attack_scenario"]}</div>')

            risk = sres.get("risk_level", "")
            if risk:
                rcls = f"risk-{risk}"
                lines.append(f'<h3>⚠️ リスク評価: <span class="{rcls}">{risk.upper()}</span></h3>')
                if "risk_reason" in sres:
                    lines.append(f'<p>{sres["risk_reason"]}</p>')
        lines.append('</section>')

    lines.extend([
        '<footer style="text-align:center;color:#555;margin-top:40px;font-size:12px;">',
        '  Generated by generate_evidence_report.py | Based on MalEval (Zheng et al., 2025)',
        '</footer>', '</body></html>'
    ])
    html_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"HTMLレポートを {html_path} に保存しました。")


# ============================================================
# メイン
# ============================================================
def main():
    args = parse_args()

    # --- APIキー確認 ---
    if not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            print("エラー: 環境変数 OPENAI_API_KEY が設定されていません。")
            sys.exit(1)
        client = OpenAI()
        print(f"GPTモデル: {args.gpt_model}")
    else:
        client = None
        print("=== DRY RUN モード ===")

    # --- 共通データ読み込み ---
    print("共通データを読み込み中...")
    mbc_mapping = load_mbc_mapping(args.mbc_mapping)
    keyword_to_apis = build_keyword_to_apis(args.api_keywords)
    api_descriptions = load_api_descriptions(args.api_descriptions)

    # --- 出力ディレクトリ ---
    shap_results_path = args.shap_results or f"logs_shap_analysis/{args.model_type}/shap_structured_results.json"
    output_dir = Path(args.output_dir) if args.output_dir else Path(shap_results_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_results = {}
    overall_result = {}
    sample_results = {}

    # ==========================================
    # Mode A: ラベル集約分析
    # ==========================================
    if not args.skip_aggregate:
        if not os.path.exists(shap_results_path):
            print(f"警告: SHAP構造化結果が見つかりません: {shap_results_path}")
            print("ラベル集約分析をスキップします。")
        else:
            shap_results = load_shap_results(shap_results_path)
            per_label = shap_results.get("per_label", {})
            total = len(per_label)
            label_syntheses = {}

            print(f"\n{'='*60}")
            print(f"[Mode A] ラベル集約 Evidence Attribution ({total} ラベル)")
            print(f"{'='*60}")

            for i, (label, ldata) in enumerate(per_label.items()):
                print(f"\n[{i+1}/{total}] ラベル: {label}")
                if not ldata.get("functional_keywords"):
                    print("  → キーワード無し、スキップ")
                    aggregate_results[label] = {"behavior_synthesis": "機能直結キーワード無し", "evidence_attributions": []}
                    continue

                pkg = build_label_evidence_package(label, ldata, mbc_mapping, keyword_to_apis, api_descriptions)
                prompt = build_user_prompt_for_label(pkg)

                if args.dry_run:
                    print(prompt[:400] + "...")
                    aggregate_results[label] = {"dry_run": True}
                    continue

                print("  GPT に問い合わせ中...")
                result = call_gpt(client, args.gpt_model, LABEL_SYSTEM_PROMPT, prompt)
                if "error" not in result and "behavior_synthesis" in result:
                    label_syntheses[label] = result["behavior_synthesis"]
                    print(f"  ✅ 完了")
                else:
                    print(f"  ⚠️ {result.get('error', '不明なエラー')}")
                aggregate_results[label] = result
                time.sleep(1)

            # 全体サマリ
            if not args.dry_run and label_syntheses:
                print(f"\n全体攻撃シナリオを生成中...")
                summary_lines = [f"### {l}: {s}" for l, s in label_syntheses.items()]
                overall_prompt = "\n".join(summary_lines) + "\n\n上記を統合し、全体的な攻撃シナリオを要約してください。"
                overall_result = call_gpt(client, args.gpt_model, OVERALL_SYSTEM_PROMPT, overall_prompt)
                print("  ✅ 全体サマリ完了" if "error" not in overall_result else f"  ⚠️ {overall_result.get('error')}")

    # ==========================================
    # Mode B: 検体個別分析
    # ==========================================
    if args.sample_indices:
        indices = [int(x.strip()) for x in args.sample_indices.split(",")]

        print(f"\n{'='*60}")
        print(f"[Mode B] 検体個別 Evidence Attribution ({len(indices)} 検体)")
        print(f"{'='*60}")

        print("モデルとテストデータを読み込み中...")
        X_test, feature_names, sample_names, label_names, models_dir = \
            load_models_and_test_data(args.model_type)

        for si, idx in enumerate(indices):
            if idx >= len(sample_names):
                print(f"\n⚠️ インデックス {idx} は範囲外です (最大: {len(sample_names)-1})")
                continue

            sname = str(sample_names[idx])
            print(f"\n[{si+1}/{len(indices)}] 検体: {sname} (index={idx})")

            sample_vector = X_test[idx]

            # APIコール列を読み込み
            print("  APIコール列を読み込み中...")
            api_calls, gt_labels = load_sample_api_calls(args.test_dir, sname)
            if api_calls is None:
                print(f"  警告: {sname}.json が見つかりません。APIコール情報なしで続行します。")
                api_calls = []
                gt_labels = []

            # SHAP計算
            print(f"  SHAP値を計算中 ({len(label_names)} ラベル)...")
            shap_result = compute_sample_shap(sample_vector, models_dir, label_names, feature_names)

            # プロンプト構築
            prompt = build_user_prompt_for_sample(
                sname, api_calls, gt_labels, shap_result,
                feature_names, sample_vector, keyword_to_apis, mbc_mapping
            )

            if args.dry_run:
                print(prompt[:600] + "...")
                sample_results[sname] = {"dry_run": True}
                continue

            print("  GPT に問い合わせ中...")
            result = call_gpt(client, args.gpt_model, SAMPLE_SYSTEM_PROMPT, prompt)
            if "error" not in result:
                print(f"  ✅ 完了")
            else:
                print(f"  ⚠️ {result.get('error')}")
            sample_results[sname] = result
            time.sleep(1)

    # ==========================================
    # 結果保存
    # ==========================================
    final = {
        "model_type": args.model_type,
        "gpt_model": args.gpt_model,
        "overall": overall_result,
        "per_label": aggregate_results,
        "per_sample": sample_results
    }
    json_path = output_dir / "evidence_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {json_path}")

    if not args.dry_run:
        generate_html_report(output_dir, aggregate_results, overall_result, sample_results, args.model_type)

    print(f"\n{'='*60}")
    print("Evidence Attribution レポートの生成が完了しました！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
