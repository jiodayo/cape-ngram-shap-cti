"""
keywordモデルとpcaモデルの両方に対応したハイブリッド特徴量のSHAP分析スクリプト。

機能:
  1. カテゴリ別重要度の算出（Keyword/PCA vs API頻度）
  2. キーワードのAPI出自逆引き（方針A）
  3. 機能直結キーワードのフィルタリングと分類（方針B）
  4. ラベルごと + 全体のWaterfallプロット出力
  5. 学会向けの総合HTMLレポート生成
"""

import argparse
import json
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from tqdm import tqdm

# scikit-learnのバージョン違い警告をより確実に非表示にする
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 方針B: MBC準拠のキーワードマッピング (Zero-shot)
# ============================================================
MBC_KEYWORD_MAPPING = {}

def load_mbc_mapping(mapping_path="features/mbc_keyword_mapping.json"):
    global MBC_KEYWORD_MAPPING
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            MBC_KEYWORD_MAPPING = json.load(f)
        print(f"MBCマッピング辞書をロードしました: {len(MBC_KEYWORD_MAPPING)}件")
    else:
        print(f"警告: MBCマッピング辞書 '{mapping_path}' が見つかりません。すべての機能直結判定が失敗します。")


def parse_args():
    parser = argparse.ArgumentParser(description="ハイブリッドモデルのカテゴリ別SHAP分析 (API出自 + 機能フィルタ付き)")
    parser.add_argument("--model-type", type=str, choices=["keyword", "pca"], required=True,
                        help="分析対象のモデル（keyword または pca）")
    parser.add_argument("--sample-indices", type=str, default="0,1,2",
                        help="Waterfallプロットを出力するテストデータのインデックス(カンマ区切り) (default: 0,1,2)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="各カテゴリで表示する上位特徴量の数 (default: 20)")
    parser.add_argument("--api-keywords-path", type=str, default="api_keywords_single.json",
                        help="api_keywords_single.json のパス (default: api_keywords_single.json)")
    return parser.parse_args()


# ============================================================
# 方針A: キーワード → 由来API の逆引き辞書を構築
# ============================================================
def build_keyword_to_apis(api_keywords_path):
    """api_keywords_single.json を読み込み、keyword → [API1, API2, ...] の逆引き辞書を作る"""
    keyword_to_apis = defaultdict(list)
    
    if not os.path.exists(api_keywords_path):
        print(f"警告: {api_keywords_path} が見つかりません。API出自情報なしで実行します。")
        return keyword_to_apis
    
    with open(api_keywords_path, "r", encoding="utf-8") as f:
        api_keywords = json.load(f)
    
    for api_name, keywords in api_keywords.items():
        for kw in keywords:
            keyword_to_apis[kw].append(api_name)
    
    print(f"API出自逆引き辞書を構築: {len(keyword_to_apis)} キーワード ← {len(api_keywords)} API")
    return keyword_to_apis


def classify_keyword(keyword_name):
    """キーワードが機能直結語かどうかを判定し、カテゴリを返す"""
    kw_info = MBC_KEYWORD_MAPPING.get(keyword_name)
    if kw_info and kw_info.get("category") != "Uncategorized (Noise)":
        return kw_info.get("category")
    return None


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
        test_labels_path = features_dir / "test_labels.csv"
        test_df = pd.read_csv(test_features_path, index_col=0)
        test_labels_df = pd.read_csv(test_labels_path, index_col=0)
        
        feature_names = test_df.columns.tolist()
        X_test = test_df.values
        Y_test = test_labels_df.values
        sample_names = test_df.index.tolist()
        models_dir = Path("logs_keyword/models_br_+freq_rf")
    
    elif model_type == "pca":
        test_labels_path = "encoded_test_labels.npy"
        Y_test = np.load(test_labels_path)
        num_test_samples = Y_test.shape[0]
        
        pca_dim = 128
        X_test_pca = np.memmap("encoded_test_features_pca.mmap", dtype='float32',
                               mode='r', shape=(num_test_samples, pca_dim))
        
        test_csv_path = features_dir / "test_keyword_features.csv"
        test_df_api = pd.read_csv(test_csv_path, index_col=0)
        
        test_npz_files = sorted([f for f in os.listdir("encoded_test_npz") if f.endswith(".npz")])
        test_stems = [f.replace(".npz", "") for f in test_npz_files]
        test_df_api = test_df_api.reindex(test_stems).fillna(0)
        sample_names = test_df_api.index.tolist()
        
        api_cols = [col for col in test_df_api.columns if col.startswith("api__")]
        X_test_api = test_df_api[api_cols].values.astype('float32')
        X_test = np.concatenate([X_test_pca, X_test_api], axis=1)
        
        with open("logs/models_br_rf/hybrid_feature_names.json", "r") as f:
            feature_names = json.load(f)
        models_dir = Path("logs/models_br_rf")

    return X_test, Y_test, feature_names, sample_names, label_names, models_dir


def generate_html_report(output_dir, model_type, main_category_name,
                         main_ratio, api_ratio,
                         functional_results, abstract_results,
                         api_results, keyword_to_apis,
                         per_label_category_ratios,
                         per_label_functional_top):
    """学会向けの総合HTMLレポートを生成する"""
    html_path = output_dir / "shap_report.html"
    
    # --- 全ラベルで登場するMBCカテゴリの一覧を収集 ---
    all_mbc_categories = set()
    for label, top_kws in per_label_functional_top.items():
        for kw_name, kw_val, kw_cat in top_kws:
            all_mbc_categories.add(kw_cat)
    all_mbc_categories = sorted(all_mbc_categories)
    
    # --- ラベル×カテゴリのSHAP重要度マトリクスを構築 ---
    heatmap_data = {}
    for label, top_kws in per_label_functional_top.items():
        cat_sums = defaultdict(float)
        for kw_name, kw_val, kw_cat in top_kws:
            cat_sums[kw_cat] += kw_val
        heatmap_data[label] = cat_sums
    
    # ヒートマップの最大値（色の正規化用）
    heatmap_max = 0.0
    for cat_sums in heatmap_data.values():
        for v in cat_sums.values():
            if v > heatmap_max:
                heatmap_max = v
    if heatmap_max == 0:
        heatmap_max = 1.0

    # --- ラベル間で共通して重要なキーワードを抽出 ---
    keyword_label_count = defaultdict(lambda: {"count": 0, "labels": [], "total_shap": 0.0, "category": ""})
    for label, top_kws in per_label_functional_top.items():
        for kw_name, kw_val, kw_cat in top_kws:
            keyword_label_count[kw_name]["count"] += 1
            keyword_label_count[kw_name]["labels"].append(label)
            keyword_label_count[kw_name]["total_shap"] += kw_val
            keyword_label_count[kw_name]["category"] = kw_cat
    shared_keywords = [(k, v) for k, v in keyword_label_count.items() if v["count"] >= 2]
    shared_keywords.sort(key=lambda x: x[1]["count"], reverse=True)

    lines = [
        '<!DOCTYPE html>',
        '<html lang="ja">',
        '<head>',
        '  <meta charset="utf-8">',
        f'  <title>SHAP分析レポート ({model_type}モデル)</title>',
        '  <style>',
        '    body { font-family: "MS PGothic", "Segoe UI", Arial, sans-serif; margin: 20px auto; max-width: 1100px; background: #fff; color: #333; line-height: 1.6; padding: 10px; }',
        '    h1 { font-size: 22px; border-bottom: 2px solid #666; padding-bottom: 5px; }',
        '    h2 { font-size: 18px; border-left: 5px solid #666; padding-left: 10px; margin-top: 30px; background: #f0f0f0; padding: 5px 10px; }',
        '    h3 { font-size: 15px; border-bottom: 1px dotted #999; margin-top: 20px; }',
        '    table { border-collapse: collapse; width: 100%; margin: 12px 0; }',
        '    th, td { border: 1px solid #999; padding: 6px 10px; text-align: left; font-size: 13px; }',
        '    th { background: #eee; font-weight: bold; }',
        '    .functional { color: #006; font-weight: bold; }',
        '    .abstract { color: #666; font-style: italic; }',
        '    .api-origin { font-size: 11px; color: #555; }',
        '    .category-badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; font-weight: bold; background: #e8ecf1; color: #333; }',
        '    .shap-bar-container { display: inline-block; width: 120px; height: 14px; background: #eee; border: 1px solid #ccc; vertical-align: middle; }',
        '    .shap-bar { height: 100%; background: #336; display: block; }',
        '    .ratio-container { display: flex; width: 300px; height: 16px; border: 1px solid #999; overflow: hidden; }',
        '    .bar-main { background: #336; }',
        '    .bar-api { background: #c63; }',
        '    .waterfall-img { max-width: 100%; border: 1px solid #ccc; margin: 8px 0; }',
        '    nav { margin: 10px 0 20px 0; padding: 10px; border: 1px dashed #ccc; background: #fafafa; }',
        '    nav a { margin-right: 15px; color: #00f; text-decoration: underline; font-size: 13px; }',
        '    section { margin-bottom: 30px; }',
        '    .heatmap-cell { text-align: center; font-size: 11px; }',
        '    .chain-arrow { color: #999; font-weight: bold; }',
        '    .shared-labels { font-size: 11px; color: #555; }',
        '  </style>',
        '</head>',
        '<body>',
        f'<h1>SHAP分析レポート -- {model_type.upper()} モデル</h1>',
        '<nav>',
        '  <a href="#overview">カテゴリ重要度</a>',
        '  <a href="#functional">機能直結キーワード</a>',
        '  <a href="#heatmap">ラベル x MBCヒートマップ</a>',
        '  <a href="#causal">因果連鎖</a>',
        '  <a href="#shared">ラベル横断キーワード</a>',
        '  <a href="#api-freq">API頻度</a>',
        '  <a href="#per-label">ラベル別分析</a>',
        '</nav>',
    ]
    
    # === 総合カテゴリ重要度 ===
    lines.append('<section id="overview">')
    lines.append('<h2>総合カテゴリ重要度割合（全ラベル平均）</h2>')
    lines.append(f'<p><strong>{main_category_name} Features:</strong> {main_ratio:.1f}% &nbsp; | &nbsp; ')
    lines.append(f'<strong>API Frequency Features:</strong> {api_ratio:.1f}%</p>')
    lines.append(f'<div class="ratio-container">')
    lines.append(f'  <div class="bar-main" style="width: {main_ratio}%;" title="{main_category_name} {main_ratio:.1f}%"></div>')
    lines.append(f'  <div class="bar-api" style="width: {api_ratio}%;" title="API {api_ratio:.1f}%"></div>')
    lines.append(f'</div>')
    lines.append('</section>')
    
    # === 機能直結キーワード Top（インラインバーチャート付き）===
    lines.append('<section id="functional">')
    lines.append('<h2>機能直結キーワード Top（インラインバーチャート付き）</h2>')
    lines.append('<table>')
    lines.append('<tr><th>#</th><th>キーワード</th><th>MBCカテゴリ</th><th>平均|SHAP|</th><th>重要度</th><th>由来API</th></tr>')
    
    # バーチャートの最大値
    max_shap = max((val for _, val, _ in functional_results), default=1.0)
    if max_shap == 0:
        max_shap = 1.0
    
    for i, (name, val, cat) in enumerate(functional_results):
        bar_width = val / max_shap * 100
        apis = keyword_to_apis.get(name, [])
        api_str = ", ".join(apis[:5])
        if len(apis) > 5:
            api_str += f" ...他{len(apis)-5}件"
        lines.append(f'<tr>')
        lines.append(f'  <td>{i+1}</td>')
        lines.append(f'  <td class="functional">{name}</td>')
        lines.append(f'  <td><span class="category-badge">{cat}</span></td>')
        lines.append(f'  <td>{val:.6f}</td>')
        lines.append(f'  <td><span class="shap-bar-container"><span class="shap-bar" style="width:{bar_width:.1f}%"></span></span></td>')
        lines.append(f'  <td class="api-origin">{api_str if api_str else "---"}</td>')
        lines.append(f'</tr>')
    lines.append('</table>')
    lines.append('</section>')
    
    # === ラベル x MBCカテゴリ ヒートマップ ===
    lines.append('<section id="heatmap">')
    lines.append('<h2>ラベル x MBCカテゴリ SHAP重要度ヒートマップ</h2>')
    lines.append('<p>各セルの色の濃さは、そのラベルにおける当該MBCカテゴリの累積SHAP重要度を表す。</p>')
    lines.append('<table>')
    # ヘッダー行
    lines.append('<tr><th>ラベル</th>')
    for cat in all_mbc_categories:
        # カテゴリ名が長い場合は短縮
        short_cat = cat[:12] + ".." if len(cat) > 14 else cat
        lines.append(f'<th style="font-size:10px;writing-mode:vertical-rl;text-orientation:mixed;height:100px;">{short_cat}</th>')
    lines.append('</tr>')
    # データ行
    for label in sorted(heatmap_data.keys()):
        cat_sums = heatmap_data[label]
        lines.append(f'<tr><td style="font-size:12px;white-space:nowrap;">{label}</td>')
        for cat in all_mbc_categories:
            v = cat_sums.get(cat, 0.0)
            if v > 0:
                intensity = min(v / heatmap_max, 1.0)
                # 青系のグラデーション: 白(0) → 濃い青(1)
                r = int(255 * (1 - intensity * 0.8))
                g = int(255 * (1 - intensity * 0.8))
                b = int(255 * (1 - intensity * 0.3))
                lines.append(f'<td class="heatmap-cell" style="background:rgb({r},{g},{b});">{v:.4f}</td>')
            else:
                lines.append(f'<td class="heatmap-cell" style="background:#f8f8f8;">-</td>')
        lines.append('</tr>')
    lines.append('</table>')
    lines.append('</section>')
    
    # === 因果連鎖テーブル ===
    lines.append('<section id="causal">')
    lines.append('<h2>因果連鎖: API → キーワード → MBCカテゴリ → 予測ラベル</h2>')
    lines.append('<p>SHAP重要度上位のキーワードについて、データの流れ（根拠の追跡可能性）を示す。</p>')
    lines.append('<table>')
    lines.append('<tr><th>#</th><th>由来API</th><th></th><th>抽出キーワード</th><th></th><th>MBCカテゴリ</th><th></th><th>寄与ラベル</th><th>|SHAP|</th></tr>')
    
    for i, (name, val, cat) in enumerate(functional_results[:15]):
        apis = keyword_to_apis.get(name, [])
        api_str = ", ".join(apis[:3])
        if len(apis) > 3:
            api_str += f" (+{len(apis)-3})"
        # このキーワードが寄与しているラベルを検索
        contributing_labels = []
        for label, top_kws in per_label_functional_top.items():
            for kw_name, kw_val, kw_cat in top_kws:
                if kw_name == name:
                    contributing_labels.append(label)
                    break
        label_str = ", ".join(contributing_labels[:4])
        if len(contributing_labels) > 4:
            label_str += f" (+{len(contributing_labels)-4})"
        
        lines.append(f'<tr>')
        lines.append(f'  <td>{i+1}</td>')
        lines.append(f'  <td class="api-origin">{api_str if api_str else "---"}</td>')
        lines.append(f'  <td class="chain-arrow">→</td>')
        lines.append(f'  <td class="functional">{name}</td>')
        lines.append(f'  <td class="chain-arrow">→</td>')
        lines.append(f'  <td><span class="category-badge">{cat}</span></td>')
        lines.append(f'  <td class="chain-arrow">→</td>')
        lines.append(f'  <td style="font-size:12px;">{label_str}</td>')
        lines.append(f'  <td>{val:.6f}</td>')
        lines.append(f'</tr>')
    lines.append('</table>')
    lines.append('</section>')
    
    # === ラベル横断キーワード ===
    if shared_keywords:
        lines.append('<section id="shared">')
        lines.append('<h2>複数ラベルに共通する重要キーワード</h2>')
        lines.append('<p>2つ以上のラベルで上位に出現したキーワード。攻撃の複数フェーズに跨る重要な根拠を示す。</p>')
        lines.append('<table>')
        lines.append('<tr><th>キーワード</th><th>MBCカテゴリ</th><th>出現ラベル数</th><th>関連ラベル</th><th>累積|SHAP|</th></tr>')
        for kw_name, info in shared_keywords[:20]:
            lines.append(f'<tr>')
            lines.append(f'  <td class="functional">{kw_name}</td>')
            lines.append(f'  <td><span class="category-badge">{info["category"]}</span></td>')
            lines.append(f'  <td>{info["count"]}</td>')
            lines.append(f'  <td class="shared-labels">{", ".join(info["labels"])}</td>')
            lines.append(f'  <td>{info["total_shap"]:.6f}</td>')
            lines.append(f'</tr>')
        lines.append('</table>')
        lines.append('</section>')
    
    # === 除外された抽象キーワード ===
    lines.append('<section id="abstract">')
    lines.append('<h2>除外された抽象キーワード（参考）</h2>')
    lines.append('<table>')
    lines.append('<tr><th>#</th><th>キーワード</th><th>平均|SHAP|</th><th>由来API</th></tr>')
    for i, (name, val) in enumerate(abstract_results[:20]):
        apis = keyword_to_apis.get(name, [])
        api_str = ", ".join(apis[:3])
        if len(apis) > 3:
            api_str += f" ...他{len(apis)-3}件"
        lines.append(f'<tr><td>{i+1}</td><td class="abstract">{name}</td><td>{val:.6f}</td><td class="api-origin">{api_str}</td></tr>')
    lines.append('</table>')
    lines.append('</section>')
    
    # === API頻度 Top ===
    lines.append('<section id="api-freq">')
    lines.append('<h2>API頻度 Top 特徴量</h2>')
    lines.append('<table>')
    lines.append('<tr><th>#</th><th>API名</th><th>平均|SHAP|</th><th>重要度</th></tr>')
    max_api_shap = max((val for _, val in api_results), default=1.0) if api_results else 1.0
    if max_api_shap == 0:
        max_api_shap = 1.0
    for i, (name, val) in enumerate(api_results):
        clean = name.replace("api__", "")
        bar_w = val / max_api_shap * 100
        lines.append(f'<tr><td>{i+1}</td><td>{clean}</td><td>{val:.6f}</td>')
        lines.append(f'<td><span class="shap-bar-container"><span class="shap-bar" style="width:{bar_w:.1f}%;background:#c63;"></span></span></td></tr>')
    lines.append('</table>')
    lines.append('</section>')
    
    # === ラベル別分析 ===
    lines.append('<section id="per-label">')
    lines.append('<h2>ラベル別分析</h2>')
    
    for label in sorted(per_label_category_ratios.keys()):
        mr, ar = per_label_category_ratios[label]
        lines.append(f'<h3>{label}</h3>')
        lines.append(f'<p>カテゴリ比率: {main_category_name} {mr:.1f}% / API {ar:.1f}%</p>')
        
        # Waterfall画像
        wf_path = f"waterfall_{label}.png"
        if (output_dir / wf_path).exists():
            lines.append(f'<img src="{wf_path}" class="waterfall-img" alt="Waterfall {label}">')
        
        # Top functional keywords for this label（バーチャート付き）
        if label in per_label_functional_top:
            top_kws = per_label_functional_top[label]
            if top_kws:
                label_max = max((kw_val for _, kw_val, _ in top_kws), default=1.0)
                if label_max == 0:
                    label_max = 1.0
                lines.append('<table>')
                lines.append('<tr><th>#</th><th>キーワード</th><th>カテゴリ</th><th>|SHAP|</th><th>重要度</th><th>由来API</th></tr>')
                for j, (kw_name, kw_val, kw_cat) in enumerate(top_kws[:10]):
                    kw_apis = keyword_to_apis.get(kw_name, [])
                    kw_api_str = ", ".join(kw_apis[:3])
                    kw_bar_w = kw_val / label_max * 100
                    lines.append(f'<tr><td>{j+1}</td><td class="functional">{kw_name}</td>'
                                 f'<td><span class="category-badge">{kw_cat}</span></td>'
                                 f'<td>{kw_val:.6f}</td>'
                                 f'<td><span class="shap-bar-container"><span class="shap-bar" style="width:{kw_bar_w:.1f}%"></span></span></td>'
                                 f'<td class="api-origin">{kw_api_str}</td></tr>')
                lines.append('</table>')
    
    lines.append('</section>')
    lines.append('</body></html>')
    
    html_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"HTMLレポートを {html_path} に保存しました。")


def generate_sample_html_report(output_dir, model_type, sample_name, sample_analysis_results, keyword_to_apis, all_mbc_categories):
    """特定の検体にフォーカスしたSHAP分析HTMLレポートを生成する"""
    html_path = output_dir / f"sample_{sample_name}_report.html"
    
    # --- サンプル用 ヒートマップデータ構築 ---
    heatmap_data = {}
    heatmap_max = 0.0
    for label, res in sample_analysis_results.items():
        cat_sums = defaultdict(float)
        for kw_name, kw_val, kw_cat in res["top_keywords"]:
            cat_sums[kw_cat] += kw_val
            if cat_sums[kw_cat] > heatmap_max:
                heatmap_max = cat_sums[kw_cat]
        heatmap_data[label] = cat_sums
    if heatmap_max == 0:
        heatmap_max = 1.0

    lines = [
        '<!DOCTYPE html>',
        '<html lang="ja">',
        '<head>',
        '  <meta charset="utf-8">',
        f'  <title>検体別SHAP分析 ({sample_name})</title>',
        '  <style>',
        '    body { font-family: "MS PGothic", "Segoe UI", Arial, sans-serif; margin: 20px auto; max-width: 1100px; background: #fff; color: #333; line-height: 1.6; padding: 10px; }',
        '    h1 { font-size: 22px; border-bottom: 2px solid #666; padding-bottom: 5px; }',
        '    h2 { font-size: 18px; border-left: 5px solid #666; padding-left: 10px; margin-top: 30px; background: #f0f0f0; padding: 5px 10px; }',
        '    h3 { font-size: 15px; border-bottom: 1px dotted #999; margin-top: 20px; }',
        '    table { border-collapse: collapse; width: 100%; margin: 12px 0; }',
        '    th, td { border: 1px solid #999; padding: 6px 10px; text-align: left; font-size: 13px; }',
        '    th { background: #eee; font-weight: bold; }',
        '    .functional { color: #006; font-weight: bold; }',
        '    .api-origin { font-size: 11px; color: #555; }',
        '    .category-badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; font-weight: bold; background: #e8ecf1; color: #333; }',
        '    .shap-bar-container { display: inline-block; width: 120px; height: 14px; background: #eee; border: 1px solid #ccc; vertical-align: middle; }',
        '    .shap-bar-pos { height: 100%; background: #c33; display: block; }',
        '    .shap-bar-neg { height: 100%; background: #33c; display: block; }',
        '    .waterfall-img { max-width: 100%; border: 1px solid #ccc; margin: 8px 0; }',
        '    nav { margin: 10px 0 20px 0; padding: 10px; border: 1px dashed #ccc; background: #fafafa; }',
        '    nav a { margin-right: 15px; color: #00f; text-decoration: underline; font-size: 13px; }',
        '    section { margin-bottom: 30px; }',
        '    .heatmap-cell { text-align: center; font-size: 11px; }',
        '    .chain-arrow { color: #999; font-weight: bold; }',
        '    .true-pos { background-color: #d4edda; color: #155724; font-weight: bold; text-align: center; }',
        '    .true-neg { background-color: #e2e3e5; color: #383d41; text-align: center; }',
        '    .false-pos { background-color: #f8d7da; color: #721c24; font-weight: bold; text-align: center; }',
        '    .false-neg { background-color: #fff3cd; color: #856404; font-weight: bold; text-align: center; }',
        '  </style>',
        '</head>',
        '<body>',
        f'<h1>検体別SHAP分析レポート ({model_type.upper()} モデル)</h1>',
        f'<p><strong>対象検体:</strong> {sample_name}</p>',
        '<nav>',
        '  <a href="#summary">予測 vs 正解 サマリー</a>',
        '  <a href="#heatmap">ラベル x MBCヒートマップ (当検体)</a>',
        '  <a href="#details">検出根拠の詳細 (全ラベル)</a>',
        '</nav>',
    ]
    
    # === 予測 vs 正解 サマリー ===
    lines.append('<section id="summary">')
    lines.append('<h2>予測 vs 正解 サマリー</h2>')
    lines.append('<table>')
    lines.append('<tr><th>フェーズ (ラベル)</th><th>正解 (Ground Truth)</th><th>予測確率 (Prob)</th><th>判定結果</th></tr>')
    
    for label in sorted(sample_analysis_results.keys()):
        res = sample_analysis_results[label]
        true_label = res["true_label"]
        pred_prob = res["pred_prob"]
        pred_label = 1 if pred_prob >= 0.5 else 0
        
        if true_label == 1 and pred_label == 1:
            decision = '<td class="true-pos">True Positive (正解)</td>'
        elif true_label == 0 and pred_label == 0:
            decision = '<td class="true-neg">True Negative (正解)</td>'
        elif true_label == 0 and pred_label == 1:
            decision = '<td class="false-pos">False Positive (過剰検知)</td>'
        else:
            decision = '<td class="false-neg">False Negative (見逃し)</td>'
            
        lines.append(f'<tr><td>{label}</td><td style="text-align:center;">{true_label}</td>')
        lines.append(f'<td style="text-align:center;">{pred_prob:.4f}</td>{decision}</tr>')
    lines.append('</table>')
    lines.append('</section>')
    
    # === ヒートマップ ===
    lines.append('<section id="heatmap">')
    lines.append('<h2>ラベル x MBCカテゴリ SHAP重要度ヒートマップ (当検体のみ)</h2>')
    lines.append('<table>')
    lines.append('<tr><th>ラベル</th>')
    for cat in all_mbc_categories:
        short_cat = cat[:12] + ".." if len(cat) > 14 else cat
        lines.append(f'<th style="font-size:10px;writing-mode:vertical-rl;text-orientation:mixed;height:100px;">{short_cat}</th>')
    lines.append('</tr>')
    for label in sorted(heatmap_data.keys()):
        cat_sums = heatmap_data[label]
        lines.append(f'<tr><td style="font-size:12px;white-space:nowrap;">{label}</td>')
        for cat in all_mbc_categories:
            v = cat_sums.get(cat, 0.0)
            if v > 0:
                intensity = min(v / heatmap_max, 1.0)
                # 赤系のグラデーション (当検体のポジティブな寄与)
                r = 255
                g = int(255 * (1 - intensity * 0.8))
                b = int(255 * (1 - intensity * 0.8))
                lines.append(f'<td class="heatmap-cell" style="background:rgb({r},{g},{b});">{v:.4f}</td>')
            else:
                lines.append(f'<td class="heatmap-cell" style="background:#f8f8f8;">-</td>')
        lines.append('</tr>')
    lines.append('</table>')
    lines.append('</section>')
    
    # === 検出根拠の詳細 ===
    lines.append('<section id="details">')
    lines.append('<h2>検出根拠の詳細 (全ラベル)</h2>')
    
    for label in sorted(sample_analysis_results.keys()):
        res = sample_analysis_results[label]
        
        lines.append(f'<h3>{label}</h3>')
        lines.append(f'<p>正解: {res["true_label"]} / 予測: {res["pred_prob"]:.4f}</p>')
        
        top_kws = res["top_keywords"]
        if top_kws:
            label_max = max((abs(kw_val) for _, kw_val, _ in top_kws), default=1.0)
            if label_max == 0: label_max = 1.0
            
            lines.append('<table>')
            lines.append('<tr><th>#</th><th>キーワード</th><th>MBCカテゴリ</th><th>SHAP (当検体)</th><th>重要度</th><th>由来API</th></tr>')
            for j, (kw_name, kw_val, kw_cat) in enumerate(top_kws):
                kw_apis = keyword_to_apis.get(kw_name, [])
                kw_api_str = ", ".join(kw_apis[:3])
                kw_bar_w = abs(kw_val) / label_max * 100
                bar_class = "shap-bar-pos" if kw_val > 0 else "shap-bar-neg"
                lines.append(f'<tr><td>{j+1}</td><td class="functional">{kw_name}</td>'
                             f'<td><span class="category-badge">{kw_cat}</span></td>'
                             f'<td>{kw_val:.6f}</td>'
                             f'<td><span class="shap-bar-container"><span class="{bar_class}" style="width:{kw_bar_w:.1f}%"></span></span></td>'
                             f'<td class="api-origin">{kw_api_str}</td></tr>')
            lines.append('</table>')
        
        # Waterfall画像
        wf_path = f"waterfall_{label}_{sample_name}.png"
        if (output_dir / wf_path).exists():
            lines.append(f'<img src="{wf_path}" class="waterfall-img" alt="Waterfall {label}">')
                
    lines.append('</section>')
    lines.append('</body></html>')
    
    html_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"検体専用のHTMLレポートを {html_path} に保存しました。")


def analyze_shap(model_type, X_test, Y_test, feature_names, sample_names, label_names, 
                 models_dir, sample_indices, top_n, keyword_to_apis):
    output_dir = Path(f"logs_shap_analysis/{model_type}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # カテゴリのインデックスを特定
    api_indices = [i for i, name in enumerate(feature_names) if name.startswith("api__")]
    main_indices = [i for i, name in enumerate(feature_names) if not name.startswith("api__")]
    
    main_category_name = "Keyword" if model_type == "keyword" else "PCA(Mean Pool)"
    print(f"\n[{model_type.upper()} Model] 特徴量の構成:")
    print(f"  - {main_category_name} 次元数: {len(main_indices)}")
    print(f"  - API Frequency 次元数: {len(api_indices)}")
    
    # 総合的な重要度
    overall_importances = {
        "main_sum": 0.0,
        "api_sum": 0.0,
        "feature_importances": np.zeros(len(feature_names))
    }
    
    # 当該サンプルの分析結果
    samples_analysis_results = {idx: {} for idx in sample_indices}
    for s_idx in sample_indices:
        print(f"\n対象検体 (sample_index={s_idx}): {sample_names[s_idx]}")
    
    for label_idx, label in enumerate(tqdm(label_names, desc="各ラベルのSHAPを計算中")):
        model_path = models_dir / f"{label}.joblib"
        if not model_path.exists():
            print(f"モデルが見つかりません: {model_path} (スキップ)")
            continue
            
        model = joblib.load(model_path)
        explainer = shap.TreeExplainer(model)
        
        # サンプリング評価
        eval_size = min(500, X_test.shape[0])
        np.random.seed(42)
        idx = np.random.choice(X_test.shape[0], eval_size, replace=False)
        X_eval = X_test[idx]
        
        shap_values_obj = explainer(X_eval)
        
        if len(shap_values_obj.shape) == 3:
            shap_values_matrix = shap_values_obj.values[:, :, 1]
        else:
            shap_values_matrix = shap_values_obj.values
            
        mean_abs_shap = np.mean(np.abs(shap_values_matrix), axis=0)
        overall_importances["feature_importances"] += mean_abs_shap
        
        main_importance = np.sum(mean_abs_shap[main_indices])
        api_importance = np.sum(mean_abs_shap[api_indices])
        overall_importances["main_sum"] += main_importance
        overall_importances["api_sum"] += api_importance
        
        # ラベル別カテゴリ比率
        label_total = main_importance + api_importance
        if label_total > 0:
            per_label_category_ratios[label] = (
                main_importance / label_total * 100,
                api_importance / label_total * 100
            )
        else:
            per_label_category_ratios[label] = (0, 0)
        
        # ラベル別: 機能直結キーワードのTop (全体平均)
        label_functional = []
        for i in main_indices:
            cat = classify_keyword(feature_names[i])
            if cat:
                label_functional.append((feature_names[i], mean_abs_shap[i], cat))
        label_functional.sort(key=lambda x: x[1], reverse=True)
        per_label_functional_top[label] = label_functional[:10]
        
        # Waterfallプロット (特定検体)
        for s_idx in sample_indices:
            sample_vector = X_test[s_idx].reshape(1, -1)
            sample_name = sample_names[s_idx]
            
            shap_values_sample = explainer(sample_vector)
            if len(shap_values_sample.shape) == 3:
                shap_val = shap_values_sample[:, :, 1]
            else:
                shap_val = shap_values_sample
            shap_val.feature_names = feature_names
            
            plt.figure(figsize=(10, 8))
            shap.plots.waterfall(shap_val[0], max_display=20, show=False)
            plt.title(f"Waterfall — {label} (sample: {sample_name})", fontsize=12)
            plt.tight_layout()
            plt.savefig(output_dir / f"waterfall_{label}_{sample_name}.png", dpi=150)
            plt.close()
            
            # ===== サンプル専用の予測と根拠の記録 =====
            true_label = int(Y_test[s_idx, label_idx])
            if hasattr(model, "predict_proba"):
                pred_prob = model.predict_proba(sample_vector)[0, 1]
            else:
                pred_prob = float(model.predict(sample_vector)[0])
                
            sample_label_functional = []
            s_vals = shap_val[0].values
            for i in main_indices:
                cat = classify_keyword(feature_names[i])
                if cat:
                    v = s_vals[i]
                    if abs(v) > 0:  # 全ての寄与を記録
                        sample_label_functional.append((feature_names[i], v, cat))
            sample_label_functional.sort(key=lambda x: abs(x[1]), reverse=True)
            
            samples_analysis_results[s_idx][label] = {
                "true_label": true_label,
                "pred_prob": pred_prob,
                "top_keywords": sample_label_functional[:15]
            }
        
    # === 全ラベルの総合レポート ===
    total_importance = overall_importances["main_sum"] + overall_importances["api_sum"]
    if total_importance > 0:
        main_ratio = overall_importances["main_sum"] / total_importance * 100
        api_ratio = overall_importances["api_sum"] / total_importance * 100
    else:
        main_ratio = api_ratio = 0
        
    avg_fi = overall_importances["feature_importances"] / len(label_names)
    
    # 方針B: 機能直結キーワードと抽象キーワードに分離
    functional_results = []
    abstract_results = []
    
    for i in main_indices:
        name = feature_names[i]
        val = avg_fi[i]
        cat = classify_keyword(name)
        if cat:
            functional_results.append((name, val, cat))
        else:
            abstract_results.append((name, val))
    
    functional_results.sort(key=lambda x: x[1], reverse=True)
    abstract_results.sort(key=lambda x: x[1], reverse=True)
    
    api_results = [(feature_names[i], avg_fi[i]) for i in api_indices]
    api_results.sort(key=lambda x: x[1], reverse=True)
    
    # --- 全ラベルで登場するMBCカテゴリの一覧を収集 ---
    all_mbc_categories = set()
    for label, top_kws in per_label_functional_top.items():
        for kw_name, kw_val, kw_cat in top_kws:
            all_mbc_categories.add(kw_cat)
    all_mbc_categories = sorted(all_mbc_categories)
    
    # --- コンソール出力 ---
    print("\n" + "=" * 60)
    print("総合カテゴリ重要度割合 (全ラベル平均)")
    print("=" * 60)
    print(f"{main_category_name} Features: {main_ratio:.1f}%")
    print(f"API Frequency Features: {api_ratio:.1f}%")
    
    print(f"\n{'=' * 60}")
    print(f"Top {top_n} 機能直結キーワード (方針B フィルタ後)")
    print(f"{'=' * 60}")
    for i, (name, val, cat) in enumerate(functional_results[:top_n]):
        apis = keyword_to_apis.get(name, [])
        api_str = ", ".join(apis[:3])
        if len(apis) > 3:
            api_str += f" ...他{len(apis)-3}"
        print(f"{i+1:2d}. {name:20s} [{cat:15s}] : {val:.6f}  ← {api_str}")
    
    print(f"\n{'=' * 60}")
    print(f"除外された抽象キーワード上位10 (参考)")
    print(f"{'=' * 60}")
    for i, (name, val) in enumerate(abstract_results[:10]):
        apis = keyword_to_apis.get(name, [])
        api_str = ", ".join(apis[:3])
        print(f"{i+1:2d}. {name:20s} : {val:.6f}  ← {api_str}")
    
    print(f"\n{'=' * 60}")
    print(f"Top {top_n} API Frequency Features")
    print(f"{'=' * 60}")
    for i, (name, val) in enumerate(api_results[:top_n]):
        print(f"{i+1:2d}. {name.replace('api__', ''):40s} : {val:.6f}")
    
    # --- テキストレポート ---
    with open(output_dir / "overall_report.txt", "w", encoding="utf-8") as f:
        f.write(f"=== {model_type.upper()} Model SHAP Analysis Report ===\n\n")
        f.write(f"Category Ratio: {main_category_name} {main_ratio:.1f}% / API {api_ratio:.1f}%\n\n")
        
        f.write(f"=== Top {top_n} Functional Keywords ===\n")
        for i, (name, val, cat) in enumerate(functional_results[:top_n]):
            apis = keyword_to_apis.get(name, [])
            f.write(f"{i+1:2d}. {name:20s} [{cat:15s}] : {val:.6f}  ← {', '.join(apis)}\n")
        
        f.write(f"\n=== Excluded Abstract Keywords (Top 20) ===\n")
        for i, (name, val) in enumerate(abstract_results[:20]):
            apis = keyword_to_apis.get(name, [])
            f.write(f"{i+1:2d}. {name:20s} : {val:.6f}  ← {', '.join(apis)}\n")
        
        f.write(f"\n=== Top {top_n} API Frequency Features ===\n")
        for i, (name, val) in enumerate(api_results[:top_n]):
            f.write(f"{i+1:2d}. {name.replace('api__', ''):40s} : {val:.6f}\n")
        
        f.write(f"\n=== Per-Label Category Ratios ===\n")
        for label in label_names:
            if label in per_label_category_ratios:
                mr, ar = per_label_category_ratios[label]
                f.write(f"{label:25s} : {main_category_name} {mr:.1f}% / API {ar:.1f}%\n")

    # --- HTMLレポート ---
    generate_html_report(
        output_dir, model_type, main_category_name,
        main_ratio, api_ratio,
        functional_results[:top_n], abstract_results,
        api_results[:top_n], keyword_to_apis,
        per_label_category_ratios,
        per_label_functional_top
    )
    
    # --- 検体専用HTMLレポート ---
    for s_idx in sample_indices:
        generate_sample_html_report(
            output_dir, model_type, sample_names[s_idx],
            samples_analysis_results[s_idx], keyword_to_apis,
            all_mbc_categories
        )
    
    # --- 構造化JSONレポート（Evidence Report用） ---
    structured_results = {
        "model_type": model_type,
        "overall": {
            "main_category_name": main_category_name,
            "main_ratio": round(main_ratio, 2),
            "api_ratio": round(api_ratio, 2),
            "functional_keywords": [
                {"keyword": name, "shap_importance": round(float(val), 6), "category": cat}
                for name, val, cat in functional_results[:top_n]
            ],
            "abstract_keywords": [
                {"keyword": name, "shap_importance": round(float(val), 6)}
                for name, val in abstract_results[:20]
            ],
            "api_features": [
                {"api": name.replace("api__", ""), "shap_importance": round(float(val), 6)}
                for name, val in api_results[:top_n]
            ]
        },
        "per_label": {}
    }
    for label in label_names:
        if label in per_label_category_ratios:
            mr, ar = per_label_category_ratios[label]
            label_data = {
                "main_ratio": round(mr, 2),
                "api_ratio": round(ar, 2),
                "functional_keywords": []
            }
            if label in per_label_functional_top:
                label_data["functional_keywords"] = [
                    {"keyword": kw_name, "shap_importance": round(float(kw_val), 6), "category": kw_cat}
                    for kw_name, kw_val, kw_cat in per_label_functional_top[label]
                ]
            structured_results["per_label"][label] = label_data

    with open(output_dir / "shap_structured_results.json", "w", encoding="utf-8") as f:
        json.dump(structured_results, f, indent=2, ensure_ascii=False)

    print(f"\n結果を {output_dir} に保存しました。")
    print(f"  - overall_report.txt             (テキストレポート)")
    print(f"  - shap_report.html               (学会向けHTMLレポート)")
    print(f"  - sample_*_report.html           (検体別レポート)")
    print(f"  - shap_structured_results.json   (構造化データ)")
    print(f"  - waterfall_*.png                (各ラベルのWaterfall)")


if __name__ == "__main__":
    args = parse_args()
    
    print("MBCキーワードマッピングをロード中...")
    load_mbc_mapping()
    
    print("データの読み込み中...")
    X_test, Y_test, feature_names, sample_names, label_names, models_dir = load_data(args.model_type)
    
    print("API出自逆引き辞書を構築中...")
    keyword_to_apis = build_keyword_to_apis(args.api_keywords_path)
    
    sample_indices = [int(x.strip()) for x in args.sample_indices.split(",")]
    
    print(f"\nSHAP分析の実行 ({args.model_type}モデル)...")
    analyze_shap(args.model_type, X_test, Y_test, feature_names, sample_names, label_names,
                 models_dir, sample_indices, args.top_n, keyword_to_apis)
