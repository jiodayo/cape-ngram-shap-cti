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

# ============================================================
# 方針B: 機能直結キーワードの分類辞書
# ============================================================
# キーワード → カテゴリ のマッピング。ここに載っているキーワードだけが
# 「機能直結キーワード」として採用される。それ以外は「抽象語」としてフィルタされる。
FUNCTIONAL_KEYWORD_CATEGORIES = {
    # --- File 操作 ---
    "file":         "File Operation",
    "read":         "File Operation",
    "write":        "File Operation",
    "delete":       "File Operation",
    "copy":         "File Operation",
    "move":         "File Operation",
    "create":       "File Operation",
    "open":         "File Operation",
    "directory":    "File Operation",
    "path":         "File Operation",
    "attribute":    "File Operation",
    "search":       "File Operation",
    "rename":       "File Operation",
    "size":         "File Operation",
    "pointer":      "File Operation",
    "temporary":    "File Operation",
    "save":         "File Operation",
    "download":     "File Operation",
    "cache":        "File Operation",
    # --- Registry ---
    "registry":     "Registry",
    "key":          "Registry",
    "value":        "Registry",
    "subkey":       "Registry",
    "enumerate":    "Registry",
    # --- Process / Thread ---
    "process":      "Process/Thread",
    "thread":       "Process/Thread",
    "execute":      "Process/Thread",
    "terminate":    "Process/Thread",
    "suspend":      "Process/Thread",
    "resume":       "Process/Thread",
    "inject":       "Process/Thread",
    "snapshot":     "Process/Thread",
    "module":       "Process/Thread",
    "load":         "Process/Thread",
    "library":      "Process/Thread",
    "dll":          "Process/Thread",
    "function":     "Process/Thread",
    "address":      "Process/Thread",
    # --- Network ---
    "connect":      "Network",
    "send":         "Network",
    "receive":      "Network",
    "socket":       "Network",
    "server":       "Network",
    "host":         "Network",
    "url":          "Network",
    "http":         "Network",
    "dns":          "Network",
    "resolve":      "Network",
    "query":        "Network",
    "request":      "Network",
    "response":     "Network",
    "connection":   "Network",
    "internet":     "Network",
    "network":      "Network",
    "adapter":      "Network",
    "port":         "Network",
    # --- Memory ---
    "memory":       "Memory",
    "allocate":     "Memory",
    "protect":      "Memory",
    "virtual":      "Memory",
    "heap":         "Memory",
    "map":          "Memory",
    "section":      "Memory",
    "page":         "Memory",
    "region":       "Memory",
    # --- Service ---
    "service":      "Service",
    "driver":       "Service",
    "install":      "Service",
    "start":        "Service",
    "control":      "Service",
    "configuration": "Service",
    # --- Crypto ---
    "encrypt":      "Crypto",
    "decrypt":      "Crypto",
    "hash":         "Crypto",
    "certificate":  "Crypto",
    "cryptographic": "Crypto",
    "cipher":       "Crypto",
    "provider":     "Crypto",
    # --- Security / Credential ---
    "token":        "Security",
    "privilege":    "Security",
    "credential":   "Security",
    "impersonate":  "Security",
    "account":      "Security",
    "security":     "Security",
    # --- System Info ---
    "system":       "System Info",
    "computer":     "System Info",
    "user":         "System Info",
    "version":      "System Info",
    "environment":  "System Info",
    "volume":       "System Info",
    "disk":         "System Info",
    # --- Screen / Input ---
    "screen":       "Screen/Input",
    "capture":      "Screen/Input",
    "keyboard":     "Screen/Input",
    "hook":         "Screen/Input",
    "input":        "Screen/Input",
    "clipboard":    "Screen/Input",
    "window":       "Screen/Input",
    # --- Anti-Analysis ---
    "debugger":     "Anti-Analysis",
    "debug":        "Anti-Analysis",
    "evasion":      "Anti-Analysis",
    "detect":       "Anti-Analysis",
}


def parse_args():
    parser = argparse.ArgumentParser(description="ハイブリッドモデルのカテゴリ別SHAP分析 (API出自 + 機能フィルタ付き)")
    parser.add_argument("--model-type", type=str, choices=["keyword", "pca"], required=True,
                        help="分析対象のモデル（keyword または pca）")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Waterfallプロットを出力するテストデータのインデックス (default: 0)")
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
    kw_lower = keyword_name.lower()
    return FUNCTIONAL_KEYWORD_CATEGORIES.get(kw_lower, None)


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

    return X_test, feature_names, sample_names, label_names, models_dir


def generate_html_report(output_dir, model_type, main_category_name,
                         main_ratio, api_ratio,
                         functional_results, abstract_results,
                         api_results, keyword_to_apis,
                         per_label_category_ratios,
                         per_label_functional_top):
    """学会向けの総合HTMLレポートを生成する"""
    html_path = output_dir / "shap_report.html"
    
    lines = [
        '<!DOCTYPE html>',
        '<html lang="ja">',
        '<head>',
        '  <meta charset="utf-8">',
        f'  <title>SHAP分析レポート ({model_type}モデル)</title>',
        '  <style>',
        '    body { font-family: "Segoe UI", Arial, sans-serif; margin: 32px; background: #f8f9fa; color: #1a1a2e; }',
        '    h1 { color: #0f3460; border-bottom: 3px solid #e94560; padding-bottom: 8px; }',
        '    h2 { color: #16213e; margin-top: 32px; }',
        '    h3 { color: #0f3460; }',
        '    table { border-collapse: collapse; width: 100%; margin: 12px 0; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }',
        '    th, td { border: 1px solid #dee2e6; padding: 8px 12px; text-align: left; font-size: 14px; }',
        '    th { background: #e8ecf1; font-weight: 600; }',
        '    tr:nth-child(even) { background: #f8f9fa; }',
        '    .highlight { background: #fff3cd !important; }',
        '    .functional { color: #0d6efd; font-weight: 600; }',
        '    .abstract { color: #6c757d; font-style: italic; }',
        '    .api-origin { font-size: 12px; color: #495057; }',
        '    .category-badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }',
        '    .cat-file { background: #d4edda; color: #155724; }',
        '    .cat-registry { background: #cce5ff; color: #004085; }',
        '    .cat-process { background: #f8d7da; color: #721c24; }',
        '    .cat-network { background: #d1ecf1; color: #0c5460; }',
        '    .cat-memory { background: #e2e3e5; color: #383d41; }',
        '    .cat-crypto { background: #fff3cd; color: #856404; }',
        '    .cat-security { background: #f5c6cb; color: #721c24; }',
        '    .cat-service { background: #d6d8db; color: #1b1e21; }',
        '    .cat-other { background: #e8ecf1; color: #495057; }',
        '    .bar { height: 20px; display: inline-block; border-radius: 4px; }',
        '    .bar-main { background: #0d6efd; }',
        '    .bar-api { background: #fd7e14; }',
        '    .ratio-container { display: flex; width: 300px; border-radius: 4px; overflow: hidden; }',
        '    .waterfall-img { max-width: 100%; border: 1px solid #dee2e6; border-radius: 4px; margin: 8px 0; }',
        '    nav a { margin-right: 12px; color: #0d6efd; text-decoration: none; }',
        '    nav a:hover { text-decoration: underline; }',
        '    section { background: #fff; padding: 20px 24px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 24px; }',
        '  </style>',
        '</head>',
        '<body>',
        f'<h1>SHAP分析レポート — {model_type.upper()} モデル</h1>',
        '<nav>',
        '  <a href="#overview">総合カテゴリ重要度</a>',
        '  <a href="#functional">機能直結キーワード Top</a>',
        '  <a href="#abstract">除外された抽象キーワード</a>',
        '  <a href="#api-freq">API頻度 Top</a>',
        '  <a href="#per-label">ラベル別分析</a>',
        '</nav>',
    ]
    
    # --- 総合カテゴリ重要度 ---
    lines.append('<section id="overview">')
    lines.append('<h2>総合カテゴリ重要度割合（全ラベル平均）</h2>')
    lines.append(f'<p><strong>{main_category_name} Features:</strong> {main_ratio:.1f}% &nbsp; | &nbsp; ')
    lines.append(f'<strong>API Frequency Features:</strong> {api_ratio:.1f}%</p>')
    lines.append(f'<div class="ratio-container">')
    lines.append(f'  <div class="bar bar-main" style="width: {main_ratio}%;" title="{main_category_name} {main_ratio:.1f}%"></div>')
    lines.append(f'  <div class="bar bar-api" style="width: {api_ratio}%;" title="API {api_ratio:.1f}%"></div>')
    lines.append(f'</div>')
    lines.append('</section>')
    
    # --- 機能直結キーワード Top ---
    lines.append('<section id="functional">')
    lines.append('<h2>機能直結キーワード Top（方針B フィルタ後）</h2>')
    lines.append('<p>マルウェアの動作を直接表すキーワードのみを抽出しています。各キーワードの由来APIも併記しています。</p>')
    lines.append('<table>')
    lines.append('<tr><th>#</th><th>キーワード</th><th>機能カテゴリ</th><th>平均|SHAP|</th><th>由来API（方針A）</th></tr>')
    
    cat_css = {
        "File Operation": "cat-file", "Registry": "cat-registry",
        "Process/Thread": "cat-process", "Network": "cat-network",
        "Memory": "cat-memory", "Crypto": "cat-crypto",
        "Security": "cat-security", "Service": "cat-service",
    }
    
    for i, (name, val, cat) in enumerate(functional_results):
        css_class = cat_css.get(cat, "cat-other")
        apis = keyword_to_apis.get(name, [])
        api_str = ", ".join(apis[:5])
        if len(apis) > 5:
            api_str += f" ...他{len(apis)-5}件"
        lines.append(f'<tr>')
        lines.append(f'  <td>{i+1}</td>')
        lines.append(f'  <td class="functional">{name}</td>')
        lines.append(f'  <td><span class="category-badge {css_class}">{cat}</span></td>')
        lines.append(f'  <td>{val:.6f}</td>')
        lines.append(f'  <td class="api-origin">{api_str if api_str else "—"}</td>')
        lines.append(f'</tr>')
    lines.append('</table>')
    lines.append('</section>')
    
    # --- 除外された抽象キーワード ---
    lines.append('<section id="abstract">')
    lines.append('<h2>除外された抽象キーワード（参考）</h2>')
    lines.append('<p>機能カテゴリに該当しなかったキーワードです。「life_cycle」「option」「handle」などが含まれます。</p>')
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
    
    # --- API頻度 Top ---
    lines.append('<section id="api-freq">')
    lines.append('<h2>API頻度 Top 特徴量</h2>')
    lines.append('<table>')
    lines.append('<tr><th>#</th><th>API名</th><th>平均|SHAP|</th></tr>')
    for i, (name, val) in enumerate(api_results):
        clean = name.replace("api__", "")
        lines.append(f'<tr><td>{i+1}</td><td>{clean}</td><td>{val:.6f}</td></tr>')
    lines.append('</table>')
    lines.append('</section>')
    
    # --- ラベル別分析 ---
    lines.append('<section id="per-label">')
    lines.append('<h2>ラベル別分析</h2>')
    
    for label in per_label_category_ratios:
        mr, ar = per_label_category_ratios[label]
        lines.append(f'<h3>{label}</h3>')
        lines.append(f'<p>カテゴリ比率: {main_category_name} {mr:.1f}% / API {ar:.1f}%</p>')
        
        # Waterfall画像
        wf_path = f"waterfall_{label}.png"
        if (output_dir / wf_path).exists():
            lines.append(f'<img src="{wf_path}" class="waterfall-img" alt="Waterfall {label}">')
        
        # Top functional keywords for this label
        if label in per_label_functional_top:
            top_kws = per_label_functional_top[label]
            if top_kws:
                lines.append('<table>')
                lines.append('<tr><th>#</th><th>キーワード</th><th>カテゴリ</th><th>|SHAP|</th><th>由来API</th></tr>')
                for j, (kw_name, kw_val, kw_cat) in enumerate(top_kws[:10]):
                    css_c = cat_css.get(kw_cat, "cat-other")
                    kw_apis = keyword_to_apis.get(kw_name, [])
                    kw_api_str = ", ".join(kw_apis[:3])
                    lines.append(f'<tr><td>{j+1}</td><td class="functional">{kw_name}</td>'
                                 f'<td><span class="category-badge {css_c}">{kw_cat}</span></td>'
                                 f'<td>{kw_val:.6f}</td><td class="api-origin">{kw_api_str}</td></tr>')
                lines.append('</table>')
    
    lines.append('</section>')
    lines.append('</body></html>')
    
    html_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"HTMLレポートを {html_path} に保存しました。")


def analyze_shap(model_type, X_test, feature_names, sample_names, label_names, 
                 models_dir, sample_index, top_n, keyword_to_apis):
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
    
    sample_vector = X_test[sample_index].reshape(1, -1)
    sample_name = sample_names[sample_index]
    
    # ラベル別の結果を記録
    per_label_category_ratios = {}
    per_label_functional_top = {}
    
    for label in tqdm(label_names, desc="各ラベルのSHAPを計算中"):
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
        
        # ラベル別: 機能直結キーワードのTop
        label_functional = []
        for i in main_indices:
            cat = classify_keyword(feature_names[i])
            if cat:
                label_functional.append((feature_names[i], mean_abs_shap[i], cat))
        label_functional.sort(key=lambda x: x[1], reverse=True)
        per_label_functional_top[label] = label_functional[:10]
        
        # Waterfallプロット
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
        plt.savefig(output_dir / f"waterfall_{label}.png", dpi=150)
        plt.close()
        
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
    
    print(f"\n結果を {output_dir} に保存しました。")
    print(f"  - overall_report.txt  (テキストレポート)")
    print(f"  - shap_report.html    (学会向けHTMLレポート)")
    print(f"  - waterfall_*.png     (各ラベルのWaterfall)")


if __name__ == "__main__":
    args = parse_args()
    
    print("データの読み込み中...")
    X_test, feature_names, sample_names, label_names, models_dir = load_data(args.model_type)
    
    print("API出自逆引き辞書を構築中...")
    keyword_to_apis = build_keyword_to_apis(args.api_keywords_path)
    
    print(f"\nSHAP分析の実行 ({args.model_type}モデル)...")
    analyze_shap(args.model_type, X_test, feature_names, sample_names, label_names,
                 models_dir, args.sample_index, args.top_n, keyword_to_apis)
