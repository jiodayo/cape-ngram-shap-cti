"""Analyze SHAP interaction strengths between the top keyword and API features.

This script answers the question: "Does this top-ranked keyword really work hand-in-hand
with the APIs highlighted by SHAP?" For a chosen test sample and a set of labels, it:

1. Computes SHAP values to identify the top keyword features and API features.
2. Extracts the SHAP interaction matrix and focuses on those top features only.
3. Saves CSV summaries, heatmaps, and an aggregated HTML report for easy review.
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
OUTPUT_DIR = ANALYSIS_DIR / "top_keyword_api_interactions"
REPORT_PATH = OUTPUT_DIR / "top_keyword_api_report.html"

TEST_FEATURES_PATH = FEATURES_DIR / "test_keyword_features.csv"
TEST_LABELS_PATH = FEATURES_DIR / "test_labels.csv"
VOCABULARY_PATH = FEATURES_DIR / "keyword_vocabulary.json"
API_FEATURES_PATH = FEATURES_DIR / "api_presence_features.json"
FEATURE_COLUMNS_PATH = FEATURES_DIR / "feature_columns.json"
LABEL_SET_PATH_PRIMARY = FEATURES_DIR / "label_set.json"
LABEL_SET_PATH_FALLBACK = Path("label_set.json")
API_FEATURE_PREFIX = "api__"

# --- Configuration ---
LABELS_TO_ANALYZE: Optional[Iterable[str]] = None
SAMPLE_INDEX_TO_ANALYZE: int = 0
TOP_KEYWORD_COUNT: int = 12
TOP_API_COUNT: int = 8
TOP_PAIR_LIMIT: int = 20
HEATMAP_MAX_KEYWORDS: int = 12
HEATMAP_MAX_APIS: int = 10


# --- Utility functions ---
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
        "label_set.json が見つかりません。features/ 配下またはカレントディレクトリに配置してください。")


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


def compute_shap_outputs(
    model,
    sample_frame: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_frame)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(
            shap_values) > 1 else shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        target_index = 1 if shap_values.shape[-1] > 1 else 0
        shap_values = shap_values[:, :, target_index]
    if shap_values.ndim == 2:
        shap_values = shap_values[0]
    if shap_values.ndim != 1:
        raise RuntimeError(
            f"想定外のSHAP値の形状です: {shap_values.shape}")

    interaction_values = explainer.shap_interaction_values(sample_frame)
    if isinstance(interaction_values, list):
        interaction_values = interaction_values[1] if len(
            interaction_values) > 1 else interaction_values[0]
    interaction_values = np.asarray(interaction_values)
    if interaction_values.ndim == 4:
        target_index = 1 if interaction_values.shape[-1] > 1 else 0
        interaction_values = interaction_values[:, :, :, target_index]
    if interaction_values.ndim == 3:
        interaction_values = interaction_values[0]
    if interaction_values.ndim == 2 and interaction_values.shape[0] == 1:
        interaction_values = interaction_values[0]
    if interaction_values.shape[0] != interaction_values.shape[1]:
        raise RuntimeError("相互作用行列が正方行列ではありません。")

    return shap_values, interaction_values


def split_feature_indices(feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
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


def select_top_indices(
    shap_values: np.ndarray,
    candidate_indices: np.ndarray,
    top_n: int,
) -> np.ndarray:
    if candidate_indices.size == 0:
        return np.zeros(0, dtype=int)
    subset_values = np.abs(shap_values[candidate_indices])
    order = np.argsort(subset_values)[::-1]
    top_order = order[:top_n]
    return candidate_indices[top_order]


def format_feature_name(name: str) -> str:
    if name.startswith(API_FEATURE_PREFIX):
        return name[len(API_FEATURE_PREFIX):]
    return name


def build_pair_dataframe(
    interaction_matrix: np.ndarray,
    keyword_indices: np.ndarray,
    api_indices: np.ndarray,
    feature_names: List[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    if keyword_indices.size == 0 or api_indices.size == 0:
        return pd.DataFrame(), np.zeros((0, 0))

    sub_matrix = interaction_matrix[np.ix_(keyword_indices, api_indices)]
    records: List[Dict[str, object]] = []
    for kw_pos, kw_idx in enumerate(keyword_indices):
        for api_pos, api_idx in enumerate(api_indices):
            value = float(sub_matrix[kw_pos, api_pos])
            records.append(
                {
                    "keyword_feature": feature_names[kw_idx],
                    "keyword_display": format_feature_name(feature_names[kw_idx]),
                    "api_feature": feature_names[api_idx],
                    "api_display": format_feature_name(feature_names[api_idx]),
                    "interaction_value": value,
                    "abs_interaction": abs(value),
                }
            )

    df = pd.DataFrame(records)
    if not df.empty:
        df.sort_values("abs_interaction", ascending=False, inplace=True)
    return df, sub_matrix


def aggregate_feature_stats(sub_matrix: np.ndarray, labels: List[str], axis: int) -> pd.DataFrame:
    if sub_matrix.size == 0:
        return pd.DataFrame()

    positive = np.sum(np.where(sub_matrix > 0, sub_matrix, 0.0), axis=axis)
    negative = np.sum(np.where(sub_matrix < 0, sub_matrix, 0.0), axis=axis)
    abs_sum = np.sum(np.abs(sub_matrix), axis=axis)
    df = pd.DataFrame(
        {
            "display_feature": labels,
            "positive_sum": positive.astype(float),
            "negative_sum": negative.astype(float),
            "abs_sum": abs_sum.astype(float),
        }
    )
    df.sort_values("abs_sum", ascending=False, inplace=True)
    return df


def save_heatmap(
    sub_matrix: np.ndarray,
    keyword_labels: List[str],
    api_labels: List[str],
    base_name: str,
) -> Optional[Path]:
    if sub_matrix.size == 0:
        return None

    keyword_limit = min(len(keyword_labels), HEATMAP_MAX_KEYWORDS)
    api_limit = min(len(api_labels), HEATMAP_MAX_APIS)
    heat_matrix = sub_matrix[:keyword_limit, :api_limit]
    if heat_matrix.size == 0:
        return None

    plt.figure(figsize=(max(5, api_limit * 0.5), max(5, keyword_limit * 0.5)))
    im = plt.imshow(heat_matrix, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(im, label="SHAP interaction value")
    plt.xticks(range(api_limit),
               api_labels[:api_limit], rotation=90, fontsize=8)
    plt.yticks(range(keyword_limit),
               keyword_labels[:keyword_limit], fontsize=8)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    heatmap_path = OUTPUT_DIR / f"top_keyword_api_heatmap_{base_name}.png"
    plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
    plt.close()
    return heatmap_path


def fmt_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, (np.floating, float, int, np.integer)):
        numeric = float(value)
        if np.isnan(numeric):
            return "-"
        return f"{numeric:.{digits}g}"
    return str(value)


def render_html_report(summary_records: List[Dict[str, object]], sample_name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if Environment is None or select_autoescape is None:
        lines = [
            "<!DOCTYPE html>",
            "<html lang=\"ja\">",
            "<head>",
            "  <meta charset=\"utf-8\">",
            f"  <title>トップ単語とAPIの協調分析 ({sample_name})</title>",
            "  <style>",
            "    body { font-family: Arial, sans-serif; margin: 24px; background: #f6f7fb; color: #202124; }",
            "    section { background: #fff; padding: 16px 20px; border-radius: 10px; margin-bottom: 28px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
            "    table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }",
            "    th, td { border: 1px solid #dce0e6; padding: 6px 8px; text-align: left; }",
            "    th { background: #eef2fa; }",
            "    h1 { font-size: 28px; margin-bottom: 8px; }",
            "    h2 { margin-top: 0; }",
            "    .metric { font-weight: 600; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <header>",
            "    <h1>トップ単語×API 協調解析</h1>",
            f"    <p>対象サンプル: <span class=\"metric\">{sample_name}</span></p>",
            "  </header>",
        ]

        for record in summary_records:
            lines.append(f"  <section id=\"label-{record['label']}\'>")
            lines.append(f"    <h2>{record['label']}</h2>")
            lines.append(
                "    <p>真値: <span class=\"metric\">{true}</span> / 予測: <span class=\"metric\">{pred}</span> / 予測確率: <span class=\"metric\">{prob}</span></p>".format(
                    true=record["true_label"],
                    pred=record["predicted_label"],
                    prob=fmt_float(record.get("predicted_probability"), 4),
                )
            )
            lines.append(
                "    <p>|相互作用|合計: <span class=\"metric\">{total}</span>, 正方向: <span class=\"metric\">{pos}</span>, 負方向: <span class=\"metric\">{neg}</span></p>".format(
                    total=fmt_float(record.get("abs_sum")),
                    pos=fmt_float(record.get("positive_sum")),
                    neg=fmt_float(record.get("negative_sum")),
                )
            )
            lines.append("    <h3>上位 キーワード×API ペア</h3>")
            lines.append(record["pairs_html"])
            lines.append("    <h3>キーワード別 集計</h3>")
            lines.append(record["keyword_html"])
            lines.append("    <h3>API別 集計</h3>")
            lines.append(record["api_html"])
            if record.get("heatmap_name"):
                lines.append("    <h3>ヒートマップ</h3>")
                lines.append(
                    f"    <img src=\"{record['heatmap_name']}\" alt=\"heatmap {record['label']}\" style=\"max-width: 100%; border: 1px solid #dce0e6;\" />"
                )
            lines.append("  </section>")

        lines.extend(["</body>", "</html>"])
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        return

    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    template = env.from_string(
        """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>トップ単語とAPIの協調分析 ({{ sample_name }})</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 24px; background: #f6f7fb; color: #202124; }
            header { margin-bottom: 24px; }
            section { background: #fff; padding: 16px 20px; border-radius: 10px; margin-bottom: 28px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }
            th, td { border: 1px solid #dce0e6; padding: 6px 8px; text-align: left; }
            th { background: #eef2fa; }
            h1 { font-size: 28px; margin: 0 0 8px; }
            .metric { font-weight: 600; }
            img { max-width: 100%; border: 1px solid #dce0e6; }
        </style>
    </head>
    <body>
        <header>
            <h1>トップ単語×API 協調解析</h1>
            <p>対象サンプル: <span class="metric">{{ sample_name }}</span></p>
        </header>

        {% for record in summary_records %}
        <section id="label-{{ record.label }}">
            <h2>{{ record.label }}</h2>
            <p>真値: <span class="metric">{{ record.true_label }}</span> / 予測: <span class="metric">{{ record.predicted_label }}</span> / 予測確率: <span class="metric">{{ record.predicted_probability }}</span></p>
            <p>|相互作用|合計: <span class="metric">{{ record.abs_sum }}</span>, 正方向: <span class="metric">{{ record.positive_sum }}</span>, 負方向: <span class="metric">{{ record.negative_sum }}</span></p>

            <h3>上位 キーワード×API ペア</h3>
            {{ record.pairs_html | safe }}

            <h3>キーワード別 集計</h3>
            {{ record.keyword_html | safe }}

            <h3>API別 集計</h3>
            {{ record.api_html | safe }}

            {% if record.heatmap_name %}
            <h3>ヒートマップ</h3>
            <img src="{{ record.heatmap_name }}" alt="heatmap {{ record.label }}" />
            {% endif %}
        </section>
        {% endfor %}
    </body>
    </html>
        """
    )

    html = template.render(sample_name=sample_name,
                           summary_records=summary_records)
    REPORT_PATH.write_text(html, encoding="utf-8")


# --- Core analysis ---
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
        print(f"警告: モデル '{label_name}' のファイルが見つかりませんでした。")
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
        shap_values, interaction_matrix = compute_shap_outputs(
            model, sample_vector_df)
    except RuntimeError as err:
        print(f"警告: SHAP計算で問題が発生しました ({err})。")
        return None

    keyword_idx, api_idx = split_feature_indices(active_feature_names)
    top_keyword_idx = select_top_indices(
        shap_values, keyword_idx, TOP_KEYWORD_COUNT)
    top_api_idx = select_top_indices(shap_values, api_idx, TOP_API_COUNT)

    pairs_df, sub_matrix = build_pair_dataframe(
        interaction_matrix,
        top_keyword_idx,
        top_api_idx,
        active_feature_names,
    )

    keyword_labels = [format_feature_name(
        active_feature_names[i]) for i in top_keyword_idx]
    api_labels = [format_feature_name(
        active_feature_names[i]) for i in top_api_idx]

    keyword_stats = aggregate_feature_stats(sub_matrix, keyword_labels, axis=1)
    api_stats = aggregate_feature_stats(sub_matrix, api_labels, axis=0)

    if not pairs_df.empty:
        pairs_df = pairs_df.head(TOP_PAIR_LIMIT)

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

    summary_metrics = {
        "positive_sum": float(np.sum(sub_matrix[sub_matrix > 0])) if sub_matrix.size else 0.0,
        "negative_sum": float(np.sum(sub_matrix[sub_matrix < 0])) if sub_matrix.size else 0.0,
        "abs_sum": float(np.sum(np.abs(sub_matrix))) if sub_matrix.size else 0.0,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = f"{label_name}_{sample_name}"
    pairs_csv = OUTPUT_DIR / f"top_keyword_api_pairs_{base_name}.csv"
    keywords_csv = OUTPUT_DIR / f"top_keywords_{base_name}.csv"
    apis_csv = OUTPUT_DIR / f"top_apis_{base_name}.csv"
    pairs_df.to_csv(pairs_csv, index=False)
    keyword_stats.to_csv(keywords_csv, index=False)
    api_stats.to_csv(apis_csv, index=False)

    heatmap_path = save_heatmap(
        sub_matrix, keyword_labels, api_labels, base_name)

    pairs_html = pairs_df.to_html(
        index=False, float_format="{:.6g}".format) if not pairs_df.empty else "<p>データなし</p>"
    keyword_html = keyword_stats.to_html(
        index=False, float_format="{:.6g}".format) if not keyword_stats.empty else "<p>データなし</p>"
    api_html = api_stats.to_html(
        index=False, float_format="{:.6g}".format) if not api_stats.empty else "<p>データなし</p>"

    return {
        "sample_name": sample_name,
        "label": label_name,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "predicted_probability": fmt_float(predicted_prob, 4),
        "positive_sum": fmt_float(summary_metrics["positive_sum"]),
        "negative_sum": fmt_float(summary_metrics["negative_sum"]),
        "abs_sum": fmt_float(summary_metrics["abs_sum"]),
        "pairs_html": pairs_html,
        "keyword_html": keyword_html,
        "api_html": api_html,
        "heatmap_name": heatmap_path.name if heatmap_path else None,
    }


def main() -> None:
    try:
        label_names_all = load_label_names()
    except FileNotFoundError as err:
        print(f"エラー: {err}")
        return

    try:
        X_test_df, Y_test_df, feature_columns = load_test_data()
    except FileNotFoundError as err:
        print(f"エラー: 必要なファイルが見つかりません ({err.filename})。先にデータ生成を行ってください。")
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
        print(f"=== トップ単語×API 協調分析: {label} ===")
        result = analyze_label(
            label,
            SAMPLE_INDEX_TO_ANALYZE,
            X_test_df,
            Y_test_df,
            feature_columns,
        )
        if result:
            summary_records.append(result)

    if not summary_records:
        print("有効な分析結果が得られませんでした。")
        return

    sample_name = summary_records[0]["sample_name"]
    summary_df = pd.DataFrame(summary_records)
    summary_csv = OUTPUT_DIR / f"top_keyword_api_summary_{sample_name}.csv"
    summary_df.to_csv(summary_csv, index=False)

    render_html_report(summary_records, sample_name)
    print(f"サマリーを '{summary_csv}' に保存しました。")
    print(f"HTMLレポートを '{REPORT_PATH}' に保存しました。")


if __name__ == "__main__":
    main()
