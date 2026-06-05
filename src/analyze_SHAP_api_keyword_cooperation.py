"""Quantify how keyword and API presence features cooperate via SHAP interaction values.

This script is based on `analyze_SHAP.py`, but focuses on aggregating the
interaction strengths between Bag-of-Words keyword features and API presence
features for a chosen sample across all labels.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

try:
    from jinja2 import Environment, select_autoescape
except ImportError:  # pragma: no cover - optional dependency
    Environment = None
    select_autoescape = None

# --- Paths ---
FEATURES_DIR = Path("features")
LOGS_DIR = Path("logs_keyword")
MODEL_SUBDIRS = [
    "models_br_+exist_rf",
    "models_br_rf",
]
ANALYSIS_DIR = Path("shap_analysis")
COOP_DIR = ANALYSIS_DIR / "api_keyword_cooperation"
REPORT_PATH = COOP_DIR / "api_keyword_cooperation_report.html"

TEST_FEATURES_PATH = FEATURES_DIR / "test_keyword_features.csv"
TEST_LABELS_PATH = FEATURES_DIR / "test_labels.csv"
VOCABULARY_PATH = FEATURES_DIR / "keyword_vocabulary.json"
API_FEATURES_PATH = FEATURES_DIR / "api_presence_features.json"
FEATURE_COLUMNS_PATH = FEATURES_DIR / "feature_columns.json"
LABEL_SET_PATH_PRIMARY = FEATURES_DIR / "label_set.json"
LABEL_SET_PATH_FALLBACK = Path("label_set.json")
API_FEATURE_PREFIX = "api__"

LABELS_TO_ANALYZE: Optional[Iterable[str]] = None
SAMPLE_INDEX_TO_ANALYZE: int = 0
TOP_PAIR_LIMIT: int = 30
TOP_FEATURE_LIMIT: int = 30
HEATMAP_KEYWORD_LIMIT: int = 30
HEATMAP_API_LIMIT: int = 30


def resolve_model_path(label_name: str) -> Optional[Path]:
    for subdir in MODEL_SUBDIRS:
        candidate = LOGS_DIR / subdir / f"{label_name}.joblib"
        if candidate.exists():
            return candidate
    return None


def load_label_names() -> List[str]:
    for path in (LABEL_SET_PATH_PRIMARY, LABEL_SET_PATH_FALLBACK):
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return [name for name, _ in sorted(data.items(), key=lambda item: item[1])]
            if isinstance(data, list):
                return data
    raise FileNotFoundError(
        "label_set.json が見つかりません。features/ 配下またはカレントディレクトリに配置してください。"
    )


def load_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    X_test_df = pd.read_csv(TEST_FEATURES_PATH, index_col=0)
    Y_test_df = pd.read_csv(TEST_LABELS_PATH, index_col=0)

    if FEATURE_COLUMNS_PATH.exists():
        with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as fh:
            feature_columns = json.load(fh)
    else:
        with open(VOCABULARY_PATH, "r", encoding="utf-8") as fh:
            keyword_vocabulary = json.load(fh)
        api_features: List[str] = []
        if API_FEATURES_PATH.exists():
            with open(API_FEATURES_PATH, "r", encoding="utf-8") as fh:
                api_features = json.load(fh)
        feature_columns = keyword_vocabulary + api_features

    X_test_df = X_test_df.reindex(columns=feature_columns).fillna(0.0)
    return X_test_df, Y_test_df, feature_columns


def align_features_to_model(
    model,
    sample_vector_df: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    expected_feature_count = getattr(
        model, "n_features_in_", sample_vector_df.shape[1])
    active_feature_names = list(feature_columns)

    if sample_vector_df.shape[1] == expected_feature_count:
        return sample_vector_df, active_feature_names

    if hasattr(model, "feature_names_in_"):
        model_feature_names = list(model.feature_names_in_)
        aligned = sample_vector_df.reindex(
            columns=model_feature_names).fillna(0.0)
        return aligned, model_feature_names

    if expected_feature_count <= sample_vector_df.shape[1]:
        active_feature_names = active_feature_names[:expected_feature_count]
        aligned = sample_vector_df.loc[:, active_feature_names]
        return aligned, active_feature_names

    raise ValueError("モデルが期待する特徴数が入力より多いため整列できません。")


def compute_interaction_matrix(
    model,
    sample_frame: pd.DataFrame,
) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    try:
        interaction_values = explainer.shap_interaction_values(sample_frame)
    except (AttributeError, NotImplementedError, TypeError) as err:
        raise RuntimeError(f"SHAP interactionの計算に失敗しました: {err}") from err

    if isinstance(interaction_values, list):
        target_index = 1 if len(interaction_values) > 1 else 0
        matrix = np.asarray(interaction_values[target_index])
    else:
        matrix = np.asarray(interaction_values)

    if matrix.ndim == 4:
        target_index = 1 if matrix.shape[-1] > 1 else 0
        matrix = matrix[:, :, :, target_index]
    if matrix.ndim == 3:
        matrix = matrix[0]
    if matrix.ndim == 2 and matrix.shape[0] == 1:
        matrix = matrix[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError("相互作用行列が正方行列ではありません。")
    return matrix


def extract_api_keyword_indices(feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    keyword_idx = np.array(
        [idx for idx, name in enumerate(
            feature_names) if not name.startswith(API_FEATURE_PREFIX)],
        dtype=int,
    )
    api_idx = np.array(
        [idx for idx, name in enumerate(
            feature_names) if name.startswith(API_FEATURE_PREFIX)],
        dtype=int,
    )
    return keyword_idx, api_idx


def format_feature_name(name: str) -> str:
    if name.startswith(API_FEATURE_PREFIX):
        return name[len(API_FEATURE_PREFIX):]
    return name


def build_pair_dataframe(
    interaction_matrix: np.ndarray,
    keyword_idx: np.ndarray,
    api_idx: np.ndarray,
    feature_names: List[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    if keyword_idx.size == 0 or api_idx.size == 0:
        return pd.DataFrame(), np.zeros((0, 0))

    sub_matrix = interaction_matrix[np.ix_(keyword_idx, api_idx)]
    records: List[Dict[str, object]] = []
    for kw_pos, kw_idx in enumerate(keyword_idx):
        for api_pos, api_idx_single in enumerate(api_idx):
            value = float(sub_matrix[kw_pos, api_pos])
            records.append(
                {
                    "keyword_feature": feature_names[kw_idx],
                    "api_feature": feature_names[api_idx_single],
                    "keyword_display": format_feature_name(feature_names[kw_idx]),
                    "api_display": format_feature_name(feature_names[api_idx_single]),
                    "interaction_value": value,
                    "abs_interaction_value": abs(value),
                }
            )

    pairs_df = pd.DataFrame(records)
    if not pairs_df.empty:
        pairs_df.sort_values("abs_interaction_value",
                             ascending=False, inplace=True)
    return pairs_df, sub_matrix


def summarize_pairs(pairs_df: pd.DataFrame) -> Dict[str, object]:
    if pairs_df.empty:
        return {
            "positive_sum": 0.0,
            "negative_sum": 0.0,
            "abs_sum": 0.0,
            "positive_count": 0,
            "negative_count": 0,
        }

    positive_mask = pairs_df["interaction_value"] > 0
    negative_mask = pairs_df["interaction_value"] < 0
    positive_sum = float(
        pairs_df.loc[positive_mask, "interaction_value"].sum())
    negative_sum = float(
        pairs_df.loc[negative_mask, "interaction_value"].sum())
    abs_sum = float(pairs_df["interaction_value"].abs().sum())

    return {
        "positive_sum": positive_sum,
        "negative_sum": negative_sum,
        "abs_sum": abs_sum,
        "positive_count": int(positive_mask.sum()),
        "negative_count": int(negative_mask.sum()),
    }


def aggregate_per_feature(sub_matrix: np.ndarray, features: List[str], axis: int) -> pd.DataFrame:
    if sub_matrix.size == 0:
        return pd.DataFrame()

    positive = np.sum(np.where(sub_matrix > 0, sub_matrix, 0.0), axis=axis)
    negative = np.sum(np.where(sub_matrix < 0, sub_matrix, 0.0), axis=axis)
    abs_sum = np.sum(np.abs(sub_matrix), axis=axis)
    positive_count = np.sum(sub_matrix > 0, axis=axis)
    negative_count = np.sum(sub_matrix < 0, axis=axis)

    df = pd.DataFrame(
        {
            "feature": features,
            "display_feature": [format_feature_name(name) for name in features],
            "positive_sum": positive.astype(float),
            "negative_sum": negative.astype(float),
            "abs_sum": abs_sum.astype(float),
            "positive_count": positive_count.astype(int),
            "negative_count": negative_count.astype(int),
        }
    )
    df.sort_values("abs_sum", ascending=False, inplace=True)
    return df


def build_heatmap(
    sub_matrix: np.ndarray,
    keyword_features: List[str],
    api_features: List[str],
    base_name: str,
) -> None:
    if sub_matrix.size == 0:
        return
    keyword_strength = np.abs(sub_matrix).sum(axis=1)
    api_strength = np.abs(sub_matrix).sum(axis=0)
    keyword_limit = min(HEATMAP_KEYWORD_LIMIT, len(keyword_features))
    api_limit = min(HEATMAP_API_LIMIT, len(api_features))
    keyword_order = np.argsort(keyword_strength)[::-1][:keyword_limit]
    api_order = np.argsort(api_strength)[::-1][:api_limit]
    heat_matrix = sub_matrix[np.ix_(keyword_order, api_order)]
    if heat_matrix.size == 0:
        return

    plt.figure(figsize=(max(6, api_limit * 0.4), max(6, keyword_limit * 0.4)))
    im = plt.imshow(heat_matrix, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(im, label="SHAP interaction value")
    plt.xticks(
        range(api_limit),
        [format_feature_name(api_features[idx]) for idx in api_order],
        rotation=90,
        fontsize=8,
    )
    plt.yticks(
        range(keyword_limit),
        [format_feature_name(keyword_features[idx]) for idx in keyword_order],
        fontsize=8,
    )
    plt.tight_layout()
    heatmap_path = COOP_DIR / f"api_keyword_heatmap_{base_name}.png"
    plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_outputs(
    label_name: str,
    sample_name: str,
    sub_matrix: np.ndarray,
    keyword_idx: np.ndarray,
    api_idx: np.ndarray,
    feature_names: List[str],
    pairs_df: pd.DataFrame,
    positive_df: pd.DataFrame,
    negative_df: pd.DataFrame,
    keyword_agg: pd.DataFrame,
    api_agg: pd.DataFrame,
    summary_metrics: Dict[str, object],
) -> None:
    COOP_DIR.mkdir(parents=True, exist_ok=True)
    base_name = f"{label_name}_{sample_name}"

    keyword_features = [feature_names[i] for i in keyword_idx]
    api_features = [feature_names[j] for j in api_idx]
    keyword_display = [format_feature_name(name) for name in keyword_features]
    api_display = [format_feature_name(name) for name in api_features]

    if sub_matrix.size:
        matrix_df = pd.DataFrame(
            sub_matrix, index=keyword_display, columns=api_display)
        matrix_csv = COOP_DIR / f"api_keyword_matrix_{base_name}.csv"
        matrix_df.to_csv(matrix_csv)

    if not positive_df.empty:
        pos_csv = COOP_DIR / f"api_keyword_top_positive_{base_name}.csv"
        pos_html = COOP_DIR / f"api_keyword_top_positive_{base_name}.html"
        positive_df.to_csv(pos_csv, index=False)
        positive_df.to_html(pos_html, index=False,
                            float_format="{:.6g}".format)

    if not negative_df.empty:
        neg_csv = COOP_DIR / f"api_keyword_top_negative_{base_name}.csv"
        neg_html = COOP_DIR / f"api_keyword_top_negative_{base_name}.html"
        negative_df.to_csv(neg_csv, index=False)
        negative_df.to_html(neg_html, index=False,
                            float_format="{:.6g}".format)

    if not keyword_agg.empty:
        keyword_csv = COOP_DIR / f"api_keyword_keyword_summary_{base_name}.csv"
        keyword_agg.to_csv(keyword_csv, index=False)

    if not api_agg.empty:
        api_csv = COOP_DIR / f"api_keyword_api_summary_{base_name}.csv"
        api_agg.to_csv(api_csv, index=False)

    summary_df = pd.DataFrame([summary_metrics])
    summary_csv = COOP_DIR / f"api_keyword_summary_{base_name}.csv"
    summary_df.to_csv(summary_csv, index=False)

    build_heatmap(sub_matrix, keyword_features, api_features, base_name)

    if pairs_df.empty:
        print("  -> API×キーワードの相互作用ペアが存在しませんでした。")
    else:
        print(
            "  -> 上位の協力ペアを 'api_keyword_top_positive_*.csv' 等に保存しました。"
        )


def render_html_report(summary_df: pd.DataFrame, sample_name: str) -> None:
    COOP_DIR.mkdir(parents=True, exist_ok=True)

    def fmt(value: Optional[float], digits: int = 6) -> str:
        if value is None:
            return "-"
        if isinstance(value, (np.floating, float, int, np.integer)):
            numeric = float(value)
            if np.isnan(numeric):
                return "-"
            return f"{numeric:.{digits}g}"
        return str(value)

    label_entries = []
    for row in summary_df.to_dict(orient="records"):
        label = row["label"]
        sample = row["sample_name"]
        base_name = f"{label}_{sample}"

        entry = {
            "label": label,
            "true_label": row["true_label"],
            "predicted_label": row["predicted_label"],
            "predicted_probability": fmt(row.get("predicted_probability"), 4),
            "abs_sum": fmt(row.get("abs_sum")),
            "positive_sum": fmt(row.get("positive_sum")),
            "negative_sum": fmt(row.get("negative_sum")),
            "positive_ratio": fmt(row.get("positive_ratio")),
            "negative_ratio": fmt(row.get("negative_ratio")),
            "top_positive": None,
            "top_negative": None,
            "top_positive_path": None,
            "top_negative_path": None,
            "keyword_summary_path": None,
            "api_summary_path": None,
            "heatmap_path": None,
        }

        top_pos_kw = row.get("top_positive_keyword")
        top_pos_api = row.get("top_positive_api")
        top_pos_val = row.get("top_positive_value")
        if top_pos_kw and top_pos_api and top_pos_val is not None:
            entry["top_positive"] = f"{top_pos_kw} × {top_pos_api} ({fmt(top_pos_val)})"
            entry["top_positive_path"] = f"api_keyword_top_positive_{base_name}.html"

        top_neg_kw = row.get("top_negative_keyword")
        top_neg_api = row.get("top_negative_api")
        top_neg_val = row.get("top_negative_value")
        if top_neg_kw and top_neg_api and top_neg_val is not None:
            entry["top_negative"] = f"{top_neg_kw} × {top_neg_api} ({fmt(top_neg_val)})"
            entry["top_negative_path"] = f"api_keyword_top_negative_{base_name}.html"

        keyword_summary = COOP_DIR / \
            f"api_keyword_keyword_summary_{base_name}.csv"
        if keyword_summary.exists():
            entry["keyword_summary_path"] = keyword_summary.name

        api_summary = COOP_DIR / f"api_keyword_api_summary_{base_name}.csv"
        if api_summary.exists():
            entry["api_summary_path"] = api_summary.name

        heatmap = COOP_DIR / f"api_keyword_heatmap_{base_name}.png"
        if heatmap.exists():
            entry["heatmap_path"] = heatmap.name

        label_entries.append(entry)

    if Environment is None or select_autoescape is None:
        lines = [
            "<!DOCTYPE html>",
            "<html lang=\"ja\">",
            "<head>",
            "    <meta charset=\"utf-8\">",
            f"    <title>API × キーワード協力分析 ({sample_name})</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 24px; background: #f6f7fb; color: #202124; }",
            "        header { margin-bottom: 24px; }",
            "        h1 { margin: 0 0 8px; font-size: 28px; }",
            "        h2 { margin-top: 32px; font-size: 22px; }",
            "        h3 { margin: 16px 0 8px; font-size: 18px; }",
            "        table { border-collapse: collapse; width: 100%; margin-top: 12px; background: #fff; }",
            "        th, td { border: 1px solid #dce0e6; padding: 6px 8px; text-align: left; font-size: 13px; }",
            "        th { background: #eef2fa; font-weight: 600; }",
            "        nav { margin: 16px 0; }",
            "        nav a { margin-right: 12px; text-decoration: none; color: #1a73e8; }",
            "        nav a:hover { text-decoration: underline; }",
            "        section { padding: 16px 20px; background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 28px; }",
            "        .metric { font-weight: 600; }",
            "        .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; background: #e8f0fe; color: #1a73e8; }",
            "        .heatmap { margin-top: 12px; }",
            "        footer { margin-top: 32px; font-size: 12px; color: #5f6368; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <header>",
            "        <h1>API × キーワード協力分析</h1>",
            f"        <p>対象サンプル: <span class=\"metric\">{sample_name}</span></p>",
            "        <p>キーワード特徴とAPI存在特徴のSHAP相互作用値から、協力関係を定量化しました。</p>",
            "    </header>",
            "    <nav>",
        ]

        for record in label_entries:
            lines.append(
                f"        <a href=\"#label-{record['label']}\">{record['label']}</a>"
            )

        lines.extend(
            [
                "    </nav>",
                "    <section>",
                "        <h2>ラベル別 集計サマリー</h2>",
                "        <table>",
                "            <thead>",
                "                <tr>",
                "                    <th>ラベル</th>",
                "                    <th>真値</th>",
                "                    <th>予測</th>",
                "                    <th>予測確率</th>",
                "                    <th>|相互作用|合計</th>",
                "                    <th>正方向合計</th>",
                "                    <th>負方向合計</th>",
                "                    <th>正割合</th>",
                "                    <th>負割合</th>",
                "                    <th>最大正ペア</th>",
                "                    <th>最大負ペア</th>",
                "                </tr>",
                "            </thead>",
                "            <tbody>",
            ]
        )

        for record in label_entries:
            lines.append("            <tr>")
            lines.append(f"                <td>{record['label']}</td>")
            lines.append(f"                <td>{record['true_label']}</td>")
            lines.append(
                f"                <td>{record['predicted_label']}</td>")
            lines.append(
                f"                <td>{record['predicted_probability']}</td>")
            lines.append(f"                <td>{record['abs_sum']}</td>")
            lines.append(f"                <td>{record['positive_sum']}</td>")
            lines.append(f"                <td>{record['negative_sum']}</td>")
            lines.append(
                f"                <td>{record['positive_ratio']}</td>")
            lines.append(
                f"                <td>{record['negative_ratio']}</td>")
            lines.append(
                f"                <td>{record['top_positive'] if record['top_positive'] else '-'}</td>"
            )
            lines.append(
                f"                <td>{record['top_negative'] if record['top_negative'] else '-'}</td>"
            )
            lines.append("            </tr>")

        lines.extend(
            [
                "            </tbody>",
                "        </table>",
                "    </section>",
            ]
        )

        for record in label_entries:
            lines.append(f"    <section id=\"label-{record['label']}\">")
            lines.append(f"        <h2>{record['label']}</h2>")
            lines.append(
                "        <p>真値: <span class=\"metric\">{true}</span> / 予測: <span class=\"metric\">{pred}</span> / 予測確率: <span class=\"metric\">{prob}</span></p>".format(
                    true=record["true_label"],
                    pred=record["predicted_label"],
                    prob=record["predicted_probability"],
                )
            )
            lines.append(
                "        <p>|相互作用|合計: <span class=\"metric\">{abs_sum}</span>, 正方向: <span class=\"metric\">{pos}</span>, 負方向: <span class=\"metric\">{neg}</span></p>".format(
                    abs_sum=record["abs_sum"],
                    pos=record["positive_sum"],
                    neg=record["negative_sum"],
                )
            )

            if record["top_positive_path"]:
                lines.append("        <h3>正方向 協力ペア上位</h3>")
                lines.append(
                    f"        <p><span class=\"badge\">CSV/HTML: {record['top_positive_path']}</span></p>"
                )

            if record["top_negative_path"]:
                lines.append("        <h3>負方向 協力ペア上位</h3>")
                lines.append(
                    f"        <p><span class=\"badge\">CSV/HTML: {record['top_negative_path']}</span></p>"
                )

            if record["keyword_summary_path"]:
                lines.append("        <h3>キーワード別集計</h3>")
                lines.append(
                    f"        <p><span class=\"badge\">CSV: {record['keyword_summary_path']}</span></p>"
                )

            if record["api_summary_path"]:
                lines.append("        <h3>API別集計</h3>")
                lines.append(
                    f"        <p><span class=\"badge\">CSV: {record['api_summary_path']}</span></p>"
                )

            if record["heatmap_path"]:
                lines.append("        <div class=\"heatmap\">")
                lines.append("            <h3>ヒートマップ (相互作用強度が大きい順)</h3>")
                lines.append(
                    f"            <img src=\"{record['heatmap_path']}\" alt=\"heatmap {record['label']}\" style=\"max-width: 100%; height: auto; border: 1px solid #dce0e6;\" />"
                )
                lines.append("        </div>")

            lines.append("    </section>")

        lines.extend(
            [
                "    <footer>",
                "        <p>本ページは <code>analyze_SHAP_api_keyword_cooperation.py</code> により生成されました。</p>",
                "    </footer>",
                "</body>",
                "</html>",
            ]
        )

        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        return

    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    template = env.from_string(
        """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>API × キーワード協力分析 ({{ sample_name }})</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 24px; background: #f6f7fb; color: #202124; }
            header { margin-bottom: 24px; }
            h1 { margin: 0 0 8px; font-size: 28px; }
            h2 { margin-top: 32px; font-size: 22px; }
            h3 { margin: 16px 0 8px; font-size: 18px; }
            table { border-collapse: collapse; width: 100%; margin-top: 12px; background: #fff; }
            th, td { border: 1px solid #dce0e6; padding: 6px 8px; text-align: left; font-size: 13px; }
            th { background: #eef2fa; font-weight: 600; }
            nav { margin: 16px 0; }
            nav a { margin-right: 12px; text-decoration: none; color: #1a73e8; }
            nav a:hover { text-decoration: underline; }
            section { padding: 16px 20px; background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 28px; }
            .metric { font-weight: 600; }
            .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; background: #e8f0fe; color: #1a73e8; }
            .heatmap { margin-top: 12px; }
            footer { margin-top: 32px; font-size: 12px; color: #5f6368; }
        </style>
    </head>
    <body>
        <header>
            <h1>API × キーワード協力分析</h1>
            <p>対象サンプル: <span class="metric">{{ sample_name }}</span></p>
            <p>キーワード特徴とAPI存在特徴のSHAP相互作用値から、協力関係を定量化しました。</p>
        </header>

        <nav>
        {% for record in label_entries %}
            <a href="#label-{{ record.label }}">{{ record.label }}</a>
        {% endfor %}
        </nav>

        <section>
            <h2>ラベル別 集計サマリー</h2>
            <table>
                <thead>
                    <tr>
                        <th>ラベル</th>
                        <th>真値</th>
                        <th>予測</th>
                        <th>予測確率</th>
                        <th>|相互作用|合計</th>
                        <th>正方向合計</th>
                        <th>負方向合計</th>
                        <th>正割合</th>
                        <th>負割合</th>
                        <th>最大正ペア</th>
                        <th>最大負ペア</th>
                    </tr>
                </thead>
                <tbody>
                {% for record in label_entries %}
                    <tr>
                        <td>{{ record.label }}</td>
                        <td>{{ record.true_label }}</td>
                        <td>{{ record.predicted_label }}</td>
                        <td>{{ record.predicted_probability }}</td>
                        <td>{{ record.abs_sum }}</td>
                        <td>{{ record.positive_sum }}</td>
                        <td>{{ record.negative_sum }}</td>
                        <td>{{ record.positive_ratio }}</td>
                        <td>{{ record.negative_ratio }}</td>
                        <td>{% if record.top_positive %}{{ record.top_positive }}{% else %}-{% endif %}</td>
                        <td>{% if record.top_negative %}{{ record.top_negative }}{% else %}-{% endif %}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </section>

        {% for record in label_entries %}
        <section id="label-{{ record.label }}">
            <h2>{{ record.label }}</h2>
            <p>真値: <span class="metric">{{ record.true_label }}</span> / 予測: <span class="metric">{{ record.predicted_label }}</span> / 予測確率: <span class="metric">{{ record.predicted_probability }}</span></p>
            <p>|相互作用|合計: <span class="metric">{{ record.abs_sum }}</span>, 正方向: <span class="metric">{{ record.positive_sum }}</span>, 負方向: <span class="metric">{{ record.negative_sum }}</span></p>

            {% if record.top_positive_path %}
            <h3>正方向 協力ペア上位</h3>
            <p><span class="badge">CSV/HTML: {{ record.top_positive_path }}</span></p>
            {% endif %}

            {% if record.top_negative_path %}
            <h3>負方向 協力ペア上位</h3>
            <p><span class="badge">CSV/HTML: {{ record.top_negative_path }}</span></p>
            {% endif %}

            {% if record.keyword_summary_path %}
            <h3>キーワード別集計</h3>
            <p><span class="badge">CSV: {{ record.keyword_summary_path }}</span></p>
            {% endif %}

            {% if record.api_summary_path %}
            <h3>API別集計</h3>
            <p><span class="badge">CSV: {{ record.api_summary_path }}</span></p>
            {% endif %}

            {% if record.heatmap_path %}
            <div class="heatmap">
                <h3>ヒートマップ (相互作用強度が大きい順)</h3>
                <img src="{{ record.heatmap_path }}" alt="heatmap {{ record.label }}" style="max-width: 100%; height: auto; border: 1px solid #dce0e6;" />
            </div>
            {% endif %}
        </section>
        {% endfor %}

        <footer>
            <p>本ページは <code>analyze_SHAP_api_keyword_cooperation.py</code> により生成されました。</p>
        </footer>
    </body>
    </html>
            """
    )

    output = template.render(
        sample_name=sample_name,
        label_entries=label_entries,
    )

    REPORT_PATH.write_text(output, encoding="utf-8")


def analyze_label(
    label_name: str,
    sample_index: int,
    X_test_df: pd.DataFrame,
    Y_test_df: pd.DataFrame,
    feature_columns: List[str],
) -> Optional[Dict[str, object]]:
    if label_name not in Y_test_df.columns:
        print(f"警告: テストラベルに '{label_name}' が含まれていないためスキップします。")
        return None

    model_path = resolve_model_path(label_name)
    if not model_path:
        searched = [
            str(LOGS_DIR / subdir / f"{label_name}.joblib") for subdir in MODEL_SUBDIRS]
        print("警告: モデルファイルが見つかりませんでした。探索したパス:\n  " + "\n  ".join(searched))
        return None

    model = joblib.load(model_path)
    sample_vector_df = X_test_df.iloc[[sample_index]].copy()
    sample_name = sample_vector_df.index[0]

    try:
        sample_vector_df, active_feature_names = align_features_to_model(
            model,
            sample_vector_df,
            feature_columns,
        )
    except ValueError as err:
        print(f"警告: 特徴量の整列に失敗しました ({err})。")
        return None

    try:
        interaction_matrix = compute_interaction_matrix(
            model, sample_vector_df)
    except RuntimeError as err:
        print(f"警告: ラベル '{label_name}' の相互作用計算で問題が発生しました: {err}")
        return None

    keyword_idx, api_idx = extract_api_keyword_indices(active_feature_names)
    pairs_df, sub_matrix = build_pair_dataframe(
        interaction_matrix,
        keyword_idx,
        api_idx,
        active_feature_names,
    )

    positive_df = pairs_df[pairs_df["interaction_value"] > 0]
    positive_df = positive_df.nlargest(TOP_PAIR_LIMIT, "interaction_value")

    negative_df = pairs_df[pairs_df["interaction_value"] < 0]
    negative_df = negative_df.nsmallest(TOP_PAIR_LIMIT, "interaction_value")

    summary_metrics = summarize_pairs(pairs_df)

    keyword_features = [active_feature_names[i] for i in keyword_idx]
    api_features = [active_feature_names[j] for j in api_idx]

    keyword_agg = aggregate_per_feature(sub_matrix, keyword_features, axis=1)
    api_agg = aggregate_per_feature(sub_matrix, api_features, axis=0)
    if not keyword_agg.empty:
        keyword_agg = keyword_agg.head(TOP_FEATURE_LIMIT)
    if not api_agg.empty:
        api_agg = api_agg.head(TOP_FEATURE_LIMIT)

    true_label = int(Y_test_df.loc[sample_name, label_name])
    predicted_label = int(model.predict(sample_vector_df)[0])
    predicted_prob: Optional[float] = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(sample_vector_df)
        if isinstance(proba, list):
            proba = proba[0]
        if isinstance(proba, np.ndarray):
            if proba.ndim == 2 and proba.shape[1] > 1:
                predicted_prob = float(proba[0, 1])
            elif proba.ndim == 1 and len(proba) > 1:
                predicted_prob = float(proba[1])

    if not positive_df.empty:
        top_positive = positive_df.iloc[0]
        summary_metrics["top_positive_keyword"] = top_positive["keyword_display"]
        summary_metrics["top_positive_api"] = top_positive["api_display"]
        summary_metrics["top_positive_value"] = float(
            top_positive["interaction_value"])
    else:
        summary_metrics["top_positive_keyword"] = None
        summary_metrics["top_positive_api"] = None
        summary_metrics["top_positive_value"] = None

    if not negative_df.empty:
        top_negative = negative_df.iloc[0]
        summary_metrics["top_negative_keyword"] = top_negative["keyword_display"]
        summary_metrics["top_negative_api"] = top_negative["api_display"]
        summary_metrics["top_negative_value"] = float(
            top_negative["interaction_value"])
    else:
        summary_metrics["top_negative_keyword"] = None
        summary_metrics["top_negative_api"] = None
        summary_metrics["top_negative_value"] = None

    abs_sum = summary_metrics.get("abs_sum", 0.0)
    if abs_sum:
        summary_metrics["positive_ratio"] = summary_metrics["positive_sum"] / abs_sum
        summary_metrics["negative_ratio"] = summary_metrics["negative_sum"] / abs_sum
    else:
        summary_metrics["positive_ratio"] = 0.0
        summary_metrics["negative_ratio"] = 0.0

    summary_metrics.update(
        {
            "sample_name": sample_name,
            "label": label_name,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "predicted_probability": predicted_prob,
        }
    )

    save_outputs(
        label_name,
        sample_name,
        sub_matrix,
        keyword_idx,
        api_idx,
        active_feature_names,
        pairs_df,
        positive_df,
        negative_df,
        keyword_agg,
        api_agg,
        summary_metrics,
    )

    return summary_metrics


def main() -> None:
    try:
        label_names_all = load_label_names()
    except FileNotFoundError as err:
        print(f"エラー: {err}")
        return

    try:
        X_test_df, Y_test_df, feature_columns = load_test_data()
    except FileNotFoundError as err:
        print(f"エラー: 必要なファイルが見つかりません ({err.filename})")
        print("先に 'make_bagofwords.py' や モデル学習を実行してください。")
        return

    if SAMPLE_INDEX_TO_ANALYZE < 0 or SAMPLE_INDEX_TO_ANALYZE >= len(X_test_df):
        print(
            f"エラー: SAMPLE_INDEX_TO_ANALYZE={SAMPLE_INDEX_TO_ANALYZE} はテストデータ範囲外です (0〜{len(X_test_df) - 1})。"
        )
        return

    labels_to_process = list(
        LABELS_TO_ANALYZE) if LABELS_TO_ANALYZE else label_names_all

    summary_records: List[Dict[str, object]] = []
    for label in labels_to_process:
        print(f"=== API×キーワード協力分析: {label} ===")
        result = analyze_label(
            label,
            SAMPLE_INDEX_TO_ANALYZE,
            X_test_df,
            Y_test_df,
            feature_columns,
        )
        if result:
            summary_records.append(result)

    if summary_records:
        COOP_DIR.mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame(summary_records)
        sample_name = summary_records[0]["sample_name"]
        summary_csv = COOP_DIR / \
            f"api_keyword_summary_all_labels_{sample_name}.csv"
        summary_df.to_csv(summary_csv, index=False)
        display_df = summary_df.copy()
        for col in ("positive_sum", "negative_sum", "abs_sum"):
            display_df[col] = display_df[col].map(
                lambda x: f"{x:.6g}" if isinstance(x, (float, int)) else x)
        display_df["predicted_probability"] = display_df["predicted_probability"].map(
            lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else "-"
        )
        print("\n--- ラベル別 API×キーワード協力サマリー ---")
        print(display_df[[
            "label",
            "true_label",
            "predicted_label",
            "predicted_probability",
            "positive_sum",
            "negative_sum",
            "abs_sum",
            "positive_ratio",
            "negative_ratio",
            "top_positive_keyword",
            "top_positive_api",
            "top_positive_value",
        ]].to_string(index=False))
        print(f"サマリーを '{summary_csv}' に保存しました。")

        try:
            render_html_report(summary_df, sample_name)
            print(f"HTMLレポートを '{REPORT_PATH}' に保存しました。")
        except Exception as err:  # noqa: BLE001
            print(f"警告: HTMLレポートの生成に失敗しました: {err}")
    else:
        print("有効な協力分析結果がありませんでした。")


if __name__ == "__main__":
    main()
