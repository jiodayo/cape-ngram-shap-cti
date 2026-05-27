#!/usr/bin/env python3
"""Generate SHAP + CTI integrated visualizations.

Outputs:
  1. Label × ATT&CK Technique Heatmap  — which techniques dominate per family
  2. Kill Chain (Tactic) Flow           — technique distribution across MITRE tactics
  3. SHAP Feature → Technique Sankey    — feature groups flow into techniques
  4. Per-label Technique Radar          — radar chart comparing technique profiles

Usage:
  python3 src/visualize_shap_cti_integrated.py \
      --cti-db shap_analysis/.../cti/cti_results.sqlite \
      --attack-db reference/mitre/attack.sqlite \
      --group-csv shap_analysis/.../shap_group_contributions.csv \
      --output-dir figures/cti_integrated
"""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

STYLE = {
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": "#fafafa",
    "axes.facecolor": "#fafafa",
}

# Kill chain order (MITRE ATT&CK tactics)
TACTIC_ORDER = [
    "reconnaissance",
    "resource-development",
    "initial-access",
    "execution",
    "persistence",
    "privilege-escalation",
    "defense-evasion",
    "credential-access",
    "discovery",
    "lateral-movement",
    "collection",
    "command-and-control",
    "exfiltration",
    "impact",
]

TACTIC_DISPLAY = {
    "reconnaissance": "偵察",
    "resource-development": "リソース開発",
    "initial-access": "初期アクセス",
    "execution": "実行",
    "persistence": "永続化",
    "privilege-escalation": "権限昇格",
    "defense-evasion": "防御回避",
    "credential-access": "認証情報アクセス",
    "discovery": "探索",
    "lateral-movement": "横展開",
    "collection": "収集",
    "command-and-control": "C2",
    "exfiltration": "持ち出し",
    "impact": "影響",
}

TACTIC_COLORS = {
    "reconnaissance": "#90caf9",
    "resource-development": "#80cbc4",
    "initial-access": "#a5d6a7",
    "execution": "#ef9a9a",
    "persistence": "#ce93d8",
    "privilege-escalation": "#f48fb1",
    "defense-evasion": "#ffcc80",
    "credential-access": "#ffab91",
    "discovery": "#81d4fa",
    "lateral-movement": "#80deea",
    "collection": "#c5e1a5",
    "command-and-control": "#b39ddb",
    "exfiltration": "#ffe082",
    "impact": "#ef5350",
}

CATEGORY_COLORS = {
    "injection": "#e53935",
    "discovery": "#1e88e5",
    "persistence": "#8e24aa",
    "defense_evasion": "#fb8c00",
    "c2": "#43a047",
    "collection": "#00897b",
    "credential_access": "#d81b60",
    "execution": "#f4511e",
    "privilege_escalation": "#5e35b1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SHAP + CTI integrated visualizations."
    )
    parser.add_argument(
        "--cti-db",
        type=Path,
        default=Path("shap_analysis/ngram_desc_lgbm_group_nometa_106/cti/cti_results.sqlite"),
    )
    parser.add_argument(
        "--attack-db",
        type=Path,
        default=Path("reference/mitre/attack.sqlite"),
    )
    parser.add_argument(
        "--group-csv",
        type=Path,
        default=Path("shap_analysis/ngram_desc_lgbm_group/shap_group_contributions.csv"),
    )
    parser.add_argument(
        "--top-csv",
        type=Path,
        default=Path("shap_analysis/ngram_desc_lgbm_group/shap_top_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/cti_integrated"),
    )
    parser.add_argument("--top-n-techniques", type=int, default=15)
    parser.add_argument("--top-n-labels", type=int, default=20)
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="Comma-separated labels (empty = all).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_technique_scores(
    db_path: Path,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str], Dict[str, List[str]]]:
    """Load label → technique → score from CTI DB.

    Returns:
        scores: {label: {technique_id: total_score}}
        tech_names: {technique_id: name}
        tech_categories: {technique_id: category}
    """
    conn = sqlite3.connect(str(db_path))
    scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    tech_names: Dict[str, str] = {}
    tech_categories: Dict[str, str] = {}

    try:
        rows = conn.execute(
            "SELECT label, technique_id, technique_name, SUM(score), category "
            "FROM sample_techniques GROUP BY label, technique_id"
        ).fetchall()
        for label, tid, name, score, cat in rows:
            scores[label][tid] = score
            tech_names[tid] = name or tid
            if cat:
                tech_categories[tid] = cat
    except sqlite3.OperationalError:
        # Fallback to sample_summary
        rows = conn.execute(
            "SELECT label, technique_ids, technique_names, technique_scores "
            "FROM sample_summary"
        ).fetchall()
        for label, tids_str, names_str, scores_str in rows:
            if not tids_str:
                continue
            tids = tids_str.split("|")
            names = (names_str or "").split("|")
            for i, tid in enumerate(tids):
                scores[label][tid] = scores[label].get(tid, 0) + 1.0
                if i < len(names) and names[i]:
                    tech_names[tid] = names[i]

    conn.close()
    return dict(scores), tech_names, tech_categories


def load_tactic_mapping(db_path: Path) -> Dict[str, List[str]]:
    """Load technique_id → tactics from ATT&CK DB."""
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    mapping: Dict[str, List[str]] = defaultdict(list)
    try:
        for tid, tactic in conn.execute(
            "SELECT technique_id, tactic FROM technique_tactics"
        ):
            mapping[tid].append(tactic)
    except sqlite3.OperationalError:
        pass
    conn.close()
    return dict(mapping)


def load_group_contributions(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Load label → group → share."""
    data: Dict[str, Dict[str, float]] = {}
    if not csv_path.exists():
        return data
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("label", "")
            group = row.get("group", "")
            share = float(row.get("share", 0))
            data.setdefault(label, {})[group] = share
    return data


# ---------------------------------------------------------------------------
# 1. Label × Technique Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    scores: Dict[str, Dict[str, float]],
    tech_names: Dict[str, str],
    output_path: Path,
    top_n_techniques: int = 15,
    top_n_labels: int = 20,
    filter_labels: Optional[List[str]] = None,
) -> None:
    plt.rcParams.update(STYLE)

    # Select labels
    labels = sorted(scores.keys(), key=lambda l: sum(scores[l].values()), reverse=True)
    if filter_labels:
        labels = [l for l in labels if l in filter_labels]
    labels = labels[:top_n_labels]

    # Find top techniques across selected labels
    tech_totals: Dict[str, float] = defaultdict(float)
    for label in labels:
        for tid, score in scores[label].items():
            tech_totals[tid] += score

    top_techs = sorted(tech_totals.keys(), key=lambda t: tech_totals[t], reverse=True)[:top_n_techniques]

    if not labels or not top_techs:
        print("  [SKIP] Heatmap: insufficient data")
        return

    # Build matrix
    matrix = np.zeros((len(labels), len(top_techs)))
    for i, label in enumerate(labels):
        row_total = sum(scores[label].values()) or 1
        for j, tid in enumerate(top_techs):
            matrix[i, j] = scores[label].get(tid, 0) / row_total

    # Plot
    fig_h = max(6, len(labels) * 0.45)
    fig_w = max(10, len(top_techs) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", interpolation="nearest")

    # Labels
    tech_labels = [f"{tid}\n{tech_names.get(tid, '')[:20]}" for tid in top_techs]
    ax.set_xticks(range(len(top_techs)))
    ax.set_xticklabels(tech_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(top_techs)):
            val = matrix[i, j]
            if val > 0.01:
                color = "white" if val > 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    ax.set_title("Label × ATT&CK Technique Heatmap\n(normalized per label)", fontweight="bold", fontsize=14)
    fig.colorbar(im, ax=ax, label="Relative Score", shrink=0.8)

    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    print(f"  [1] Heatmap: {output_path}")


# ---------------------------------------------------------------------------
# 2. Kill Chain (Tactic Flow)
# ---------------------------------------------------------------------------

def plot_kill_chain(
    scores: Dict[str, Dict[str, float]],
    tactic_mapping: Dict[str, List[str]],
    output_path: Path,
    filter_labels: Optional[List[str]] = None,
) -> None:
    plt.rcParams.update(STYLE)

    # Aggregate by tactic across all labels
    tactic_scores: Dict[str, float] = defaultdict(float)
    tactic_technique_counts: Dict[str, set] = defaultdict(set)

    labels = list(scores.keys())
    if filter_labels:
        labels = [l for l in labels if l in filter_labels]

    for label in labels:
        for tid, score in scores[label].items():
            tactics = tactic_mapping.get(tid, [])
            for tactic in tactics:
                tactic_scores[tactic] += score
                tactic_technique_counts[tactic].add(tid)

    # Filter to known tactics in order
    active_tactics = [t for t in TACTIC_ORDER if t in tactic_scores]
    if not active_tactics:
        print("  [SKIP] Kill chain: no tactic data")
        return

    values = [tactic_scores[t] for t in active_tactics]
    max_val = max(values) if values else 1
    normalized = [v / max_val for v in values]
    colors = [TACTIC_COLORS.get(t, "#cccccc") for t in active_tactics]
    display_names = [TACTIC_DISPLAY.get(t, t) for t in active_tactics]
    tech_counts = [len(tactic_technique_counts[t]) for t in active_tactics]

    fig, ax = plt.subplots(figsize=(max(12, len(active_tactics) * 1.2), 6))

    # Draw connected bars with arrows
    bars = ax.bar(range(len(active_tactics)), normalized, color=colors,
                  edgecolor="white", linewidth=1.5, width=0.7, zorder=3)

    # Connect bars with arrows
    for i in range(len(active_tactics) - 1):
        ax.annotate(
            "", xy=(i + 1, normalized[i + 1] * 0.5),
            xytext=(i, normalized[i] * 0.5),
            arrowprops=dict(arrowstyle="->", color="#666666", lw=1.5),
            zorder=2,
        )

    # Annotations
    for i, (bar, val, n_tech) in enumerate(zip(bars, values, tech_counts)):
        ax.text(i, bar.get_height() + 0.02, f"{val:.1f}\n({n_tech}技術)",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(active_tactics)))
    ax.set_xticklabels(display_names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Relative Score (normalized)")
    ax.set_title("ATT&CK Kill Chain — Tactic Distribution\n(全ラベル集計)", fontweight="bold", fontsize=14)

    ax.set_ylim(0, max(normalized) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    print(f"  [2] Kill chain: {output_path}")


# ---------------------------------------------------------------------------
# 3. Per-Label Kill Chain Comparison
# ---------------------------------------------------------------------------

def plot_label_tactic_comparison(
    scores: Dict[str, Dict[str, float]],
    tactic_mapping: Dict[str, List[str]],
    output_path: Path,
    top_n_labels: int = 8,
    filter_labels: Optional[List[str]] = None,
) -> None:
    plt.rcParams.update(STYLE)

    labels = sorted(scores.keys(), key=lambda l: sum(scores[l].values()), reverse=True)
    if filter_labels:
        labels = [l for l in labels if l in filter_labels]
    labels = labels[:top_n_labels]

    if not labels:
        print("  [SKIP] Label tactic comparison: no data")
        return

    # Compute per-label tactic scores
    label_tactic_scores: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tactic_score: Dict[str, float] = defaultdict(float)
        for tid, score in scores[label].items():
            for tactic in tactic_mapping.get(tid, []):
                tactic_score[tactic] += score
        total = sum(tactic_score.values()) or 1
        label_tactic_scores[label] = {t: v / total for t, v in tactic_score.items()}

    # Get active tactics
    all_tactics = set()
    for ts in label_tactic_scores.values():
        all_tactics.update(ts.keys())
    active = [t for t in TACTIC_ORDER if t in all_tactics]

    if len(active) < 3:
        print("  [SKIP] Radar: need at least 3 tactics")
        return

    # Radar chart
    n_tactics = len(active)
    angles = np.linspace(0, 2 * np.pi, n_tactics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    cmap = plt.cm.Set2
    for i, label in enumerate(labels):
        values = [label_tactic_scores[label].get(t, 0) for t in active]
        values += values[:1]
        color = cmap(i / max(len(labels) - 1, 1))
        ax.plot(angles, values, linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    tactic_labels = [TACTIC_DISPLAY.get(t, t) for t in active]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tactic_labels, fontsize=9)
    ax.set_title("Label × Tactic Profile (Radar)\nラベルごとの攻撃パターン比較",
                  fontweight="bold", fontsize=14, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    print(f"  [3] Radar: {output_path}")


# ---------------------------------------------------------------------------
# 4. Category Distribution (Stacked Bar)
# ---------------------------------------------------------------------------

def plot_category_distribution(
    scores: Dict[str, Dict[str, float]],
    tech_categories: Dict[str, str],
    output_path: Path,
    top_n_labels: int = 15,
    filter_labels: Optional[List[str]] = None,
) -> None:
    plt.rcParams.update(STYLE)

    labels = sorted(scores.keys(), key=lambda l: sum(scores[l].values()), reverse=True)
    if filter_labels:
        labels = [l for l in labels if l in filter_labels]
    labels = labels[:top_n_labels]

    if not labels:
        print("  [SKIP] Category distribution: no data")
        return

    # Compute per-label category shares
    all_categories = set()
    label_cat_scores: Dict[str, Dict[str, float]] = {}
    for label in labels:
        cat_score: Dict[str, float] = defaultdict(float)
        for tid, score in scores[label].items():
            cat = tech_categories.get(tid, "other")
            cat_score[cat] += score
        total = sum(cat_score.values()) or 1
        label_cat_scores[label] = {c: v / total for c, v in cat_score.items()}
        all_categories.update(cat_score.keys())

    categories = sorted(all_categories)

    # Stacked bar
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 7))

    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels))

    for cat in categories:
        values = [label_cat_scores[l].get(cat, 0) for l in labels]
        color = CATEGORY_COLORS.get(cat, "#999999")
        ax.bar(x, values, bottom=bottoms, label=cat, color=color, width=0.7, edgecolor="white", linewidth=0.5)
        bottoms += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Share")
    ax.set_title("Attack Category Distribution per Label\nラベルごとの攻撃カテゴリ構成", fontweight="bold", fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    print(f"  [4] Category dist: {output_path}")


# ---------------------------------------------------------------------------
# 5. Feature Group → Tactic Alluvial/Flow
# ---------------------------------------------------------------------------

def plot_group_to_tactic_flow(
    group_data: Dict[str, Dict[str, float]],
    scores: Dict[str, Dict[str, float]],
    tactic_mapping: Dict[str, List[str]],
    output_path: Path,
    filter_labels: Optional[List[str]] = None,
) -> None:
    """Simple grouped bar showing feature group contribution vs tactic distribution side-by-side."""
    plt.rcParams.update(STYLE)

    labels = sorted(group_data.keys(), key=lambda l: sum(scores.get(l, {}).values()), reverse=True)
    if filter_labels:
        labels = [l for l in labels if l in filter_labels]
    labels = labels[:10]

    if not labels:
        print("  [SKIP] Group-to-tactic flow: no data")
        return

    # Aggregate across labels
    group_totals = defaultdict(float)
    tactic_totals = defaultdict(float)

    for label in labels:
        for group, share in group_data.get(label, {}).items():
            group_totals[group] += share

        for tid, score in scores.get(label, {}).items():
            for tactic in tactic_mapping.get(tid, []):
                tactic_totals[tactic] += score

    # Normalize
    g_total = sum(group_totals.values()) or 1
    t_total = sum(tactic_totals.values()) or 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Feature groups
    groups = sorted(group_totals.keys())
    group_colors = {"seq_ngram": "#339af0", "skipgram": "#ffa94d", "desc_tfidf": "#51cf66"}
    group_labels_display = {"seq_ngram": "Seq N-gram", "skipgram": "Skip-gram", "desc_tfidf": "Desc TF-IDF"}

    g_values = [group_totals[g] / g_total for g in groups]
    g_colors = [group_colors.get(g, "#999") for g in groups]
    g_names = [group_labels_display.get(g, g) for g in groups]

    bars1 = ax1.barh(range(len(groups)), g_values, color=g_colors, edgecolor="white")
    ax1.set_yticks(range(len(groups)))
    ax1.set_yticklabels(g_names, fontsize=11)
    ax1.set_xlabel("Average Share")
    ax1.set_title("Feature Group\nContribution", fontweight="bold")
    ax1.invert_yaxis()
    for bar, val in zip(bars1, g_values):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1%}", va="center", fontsize=10, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: Tactics
    active_tactics = [t for t in TACTIC_ORDER if t in tactic_totals][:8]
    t_values = [tactic_totals[t] / t_total for t in active_tactics]
    t_colors = [TACTIC_COLORS.get(t, "#ccc") for t in active_tactics]
    t_names = [TACTIC_DISPLAY.get(t, t) for t in active_tactics]

    bars2 = ax2.barh(range(len(active_tactics)), t_values, color=t_colors, edgecolor="white")
    ax2.set_yticks(range(len(active_tactics)))
    ax2.set_yticklabels(t_names, fontsize=11)
    ax2.set_xlabel("Relative Score")
    ax2.set_title("ATT&CK Tactic\nDistribution", fontweight="bold")
    ax2.invert_yaxis()
    for bar, val in zip(bars2, t_values):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1%}", va="center", fontsize=10, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("SHAP Feature Groups → ATT&CK Tactics\n特徴量グループと攻撃戦術の対応",
                 fontweight="bold", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    print(f"  [5] Group→Tactic: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    filter_labels = [l.strip() for l in args.labels.split(",") if l.strip()] if args.labels else None

    # Load data
    if not args.cti_db.exists():
        print(f"[ERROR] CTI DB not found: {args.cti_db}")
        print("  Run cti_attach_shap_explanations.py first.")
        return

    scores, tech_names, tech_categories = load_technique_scores(args.cti_db)
    tactic_mapping = load_tactic_mapping(args.attack_db)
    group_data = load_group_contributions(args.group_csv)

    n_labels = len(scores)
    n_techniques = len(tech_names)
    print(f"[INFO] Labels: {n_labels}, Techniques: {n_techniques}")
    print(f"[INFO] Tactic mappings: {len(tactic_mapping)}")
    print(f"[INFO] Group data: {len(group_data)} labels")
    print()

    # Generate visualizations
    print("Generating visualizations...")

    # 1. Heatmap
    plot_heatmap(
        scores, tech_names,
        args.output_dir / "label_technique_heatmap.png",
        top_n_techniques=args.top_n_techniques,
        top_n_labels=args.top_n_labels,
        filter_labels=filter_labels,
    )

    # 2. Kill chain
    plot_kill_chain(
        scores, tactic_mapping,
        args.output_dir / "kill_chain_flow.png",
        filter_labels=filter_labels,
    )

    # 3. Radar
    plot_label_tactic_comparison(
        scores, tactic_mapping,
        args.output_dir / "label_tactic_radar.png",
        top_n_labels=8,
        filter_labels=filter_labels,
    )

    # 4. Category distribution
    if tech_categories:
        plot_category_distribution(
            scores, tech_categories,
            args.output_dir / "category_distribution.png",
            top_n_labels=args.top_n_labels,
            filter_labels=filter_labels,
        )

    # 5. Group → Tactic flow
    if group_data:
        plot_group_to_tactic_flow(
            group_data, scores, tactic_mapping,
            args.output_dir / "group_tactic_flow.png",
            filter_labels=filter_labels,
        )

    print(f"\n[INFO] All outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
