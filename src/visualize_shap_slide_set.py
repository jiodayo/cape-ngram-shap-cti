#!/usr/bin/env python3
"""Generate slide-ready SHAP visualization 3-piece set per label.

Outputs for each label:
  1. Summary Bar Chart   – top-20 features by mean |SHAP|
  2. Group Share Donut   – seq_ngram / skipgram / desc contribution shares
  3. Representative Waterfall – waterfall plot for the most "typical" sample

Usage:
  python3 src/visualize_shap_slide_set.py \
      --group-csv  shap_analysis/.../shap_group_contributions.csv \
      --top-csv    shap_analysis/.../shap_top_features.csv \
      --per-sample-dir shap_analysis/.../per_sample \
      --output-dir figures/slide_set
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

STYLE_CONFIG = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
}

# Color palette
COLORS = {
    "seq_ngram": "#2196F3",   # blue
    "skipgram": "#FF9800",    # orange
    "desc_tfidf": "#4CAF50",  # green
    "bar_positive": "#E53935",
    "bar_negative": "#1E88E5",
    "bar_default": "#5C6BC0",
}

GROUP_LABELS = {
    "seq_ngram": "Sequence N-gram",
    "skipgram": "Skip-gram",
    "desc_tfidf": "Description TF-IDF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate slide-ready SHAP 3-piece visualization set."
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
        "--per-sample-dir",
        type=Path,
        default=None,
        help="Directory with per-sample SHAP CSVs for waterfall generation.",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=None,
        help="Feature root for loading vectorizers (needed for waterfall).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Model directory for loading models (needed for waterfall).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/slide_set"),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="Comma-separated labels to plot. Empty = all available.",
    )
    parser.add_argument(
        "--waterfall-top-k",
        type=int,
        default=15,
        help="Number of features to show in waterfall plot.",
    )
    parser.add_argument(
        "--figsize-bar",
        type=float,
        nargs=2,
        default=[10, 8],
    )
    parser.add_argument(
        "--figsize-donut",
        type=float,
        nargs=2,
        default=[8, 8],
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_group_data(path: Path) -> Dict[str, List[Dict[str, object]]]:
    """Load group contribution CSV, grouped by label."""
    data: Dict[str, List[Dict[str, object]]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "")
            data.setdefault(label, []).append({
                "group": row.get("group", ""),
                "abs_shap_sum": float(row.get("abs_shap_sum", 0)),
                "share": float(row.get("share", 0)),
                "num_samples": int(row.get("num_samples", 0)),
            })
    return data


def load_top_features(path: Path) -> Dict[str, List[Dict[str, object]]]:
    """Load top features CSV, grouped by label."""
    data: Dict[str, List[Dict[str, object]]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "")
            data.setdefault(label, []).append({
                "rank": int(row.get("rank", 0)),
                "feature": row.get("feature", ""),
                "mean_abs_shap": float(row.get("mean_abs_shap", 0)),
            })
    return data


def load_per_sample_data(
    per_sample_dir: Path,
    label: str,
) -> Optional[List[Dict[str, object]]]:
    """Load per-sample SHAP data for a specific label."""
    csv_path = per_sample_dir / f"shap_per_sample_topk_{label}.csv"
    if not csv_path.exists():
        return None

    samples: Dict[str, List[Dict[str, object]]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = row.get("sample", "")
            samples.setdefault(sample, []).append({
                "feature": row.get("feature", ""),
                "shap_value": float(row.get("shap_value", 0)),
                "abs_shap": float(row.get("abs_shap", 0)),
                "feature_value": float(row.get("feature_value", 0)),
                "rank": int(row.get("rank", 0)),
            })
    return samples


def find_representative_sample(
    samples: Dict[str, List[Dict[str, object]]],
) -> Optional[Tuple[str, List[Dict[str, object]]]]:
    """Find the most 'typical' sample (closest to mean SHAP profile)."""
    if not samples:
        return None

    # Compute mean abs_shap across all samples
    all_totals = []
    sample_names = []
    for name, features in samples.items():
        total = sum(f["abs_shap"] for f in features)
        all_totals.append(total)
        sample_names.append(name)

    if not all_totals:
        return None

    mean_total = np.mean(all_totals)
    # Find sample closest to mean
    best_idx = int(np.argmin(np.abs(np.array(all_totals) - mean_total)))
    best_name = sample_names[best_idx]
    return best_name, samples[best_name]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_summary_bar(
    features: List[Dict[str, object]],
    label: str,
    top_k: int,
    figsize: Tuple[float, float],
    output_path: Path,
) -> None:
    """Create a horizontal bar chart of top-k features by mean |SHAP|."""
    plt.rcParams.update(STYLE_CONFIG)

    top = sorted(features, key=lambda x: x["mean_abs_shap"], reverse=True)[:top_k]
    top.reverse()  # Bottom to top

    names = []
    for f in top:
        name = str(f["feature"])
        if len(name) > 40:
            name = name[:18] + "..." + name[-18:]
        names.append(name)
    values = [f["mean_abs_shap"] for f in top]

    # Color by group
    colors = []
    for f in top:
        feat = str(f["feature"])
        if feat.startswith("seq:"):
            colors.append(COLORS["seq_ngram"])
        elif feat.startswith("skip:"):
            colors.append(COLORS["skipgram"])
        elif feat.startswith("desc:"):
            colors.append(COLORS["desc_tfidf"])
        else:
            colors.append(COLORS["bar_default"])

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top-{top_k} Features – {label}", fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = []
    for group, color in [("seq_ngram", COLORS["seq_ngram"]),
                          ("skipgram", COLORS["skipgram"]),
                          ("desc_tfidf", COLORS["desc_tfidf"])]:
        legend_elements.append(Patch(facecolor=color, label=GROUP_LABELS.get(group, group)))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    print(f"  [1/3] Summary bar: {output_path}")


def plot_group_donut(
    groups: List[Dict[str, object]],
    label: str,
    figsize: Tuple[float, float],
    output_path: Path,
) -> None:
    """Create a donut chart of feature group contribution shares."""
    plt.rcParams.update(STYLE_CONFIG)

    # Filter out zero-share groups
    active = [g for g in groups if g["share"] > 0]
    if not active:
        print(f"  [2/3] Skipped (no data)")
        return

    names = [GROUP_LABELS.get(str(g["group"]), str(g["group"])) for g in active]
    shares = [g["share"] for g in active]
    colors = [COLORS.get(str(g["group"]), "#999999") for g in active]

    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(
        shares,
        labels=names,
        autopct=lambda pct: f"{pct:.1f}%",
        colors=colors,
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        textprops={"fontsize": 11},
    )

    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")

    ax.set_title(f"Feature Group Contribution – {label}", fontweight="bold", fontsize=14)

    # Center text
    num_samples = active[0].get("num_samples", "?")
    ax.text(0, 0, f"n={num_samples}", ha="center", va="center", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    print(f"  [2/3] Group donut: {output_path}")


def plot_representative_waterfall(
    sample_name: str,
    features: List[Dict[str, object]],
    label: str,
    top_k: int,
    output_path: Path,
) -> None:
    """Create a waterfall-style bar chart for a representative sample."""
    plt.rcParams.update(STYLE_CONFIG)

    top = sorted(features, key=lambda x: x["abs_shap"], reverse=True)[:top_k]
    top.reverse()

    names = []
    for f in top:
        name = str(f["feature"])
        if len(name) > 35:
            name = name[:15] + "..." + name[-15:]
        names.append(name)

    values = [f["shap_value"] for f in top]
    colors = [COLORS["bar_positive"] if v > 0 else COLORS["bar_negative"] for v in values]

    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.4)))
    ax.barh(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("SHAP value (positive → predicted)")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_title(
        f"Representative Sample – {label}\n({sample_name})",
        fontweight="bold",
        fontsize=13,
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["bar_positive"], label="Positive (→ predicted)"),
        Patch(facecolor=COLORS["bar_negative"], label="Negative (→ not predicted)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    print(f"  [3/3] Waterfall:   {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.group_csv.exists():
        raise FileNotFoundError(f"Group CSV not found: {args.group_csv}")
    if not args.top_csv.exists():
        raise FileNotFoundError(f"Top features CSV not found: {args.top_csv}")

    group_data = load_group_data(args.group_csv)
    top_data = load_top_features(args.top_csv)

    all_labels = sorted(set(group_data.keys()) | set(top_data.keys()))
    if args.labels.strip():
        requested = [l.strip() for l in args.labels.split(",") if l.strip()]
        all_labels = [l for l in requested if l in all_labels]

    if not all_labels:
        raise RuntimeError("No labels found in input CSVs.")

    print(f"[INFO] Labels to plot: {all_labels}")

    for label in all_labels:
        label_dir = args.output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {label} ---")

        # 1. Summary bar
        if label in top_data:
            plot_summary_bar(
                top_data[label],
                label,
                top_k=args.top_k,
                figsize=tuple(args.figsize_bar),
                output_path=label_dir / "summary_bar.png",
            )
        else:
            print(f"  [1/3] Skipped (no top features data)")

        # 2. Group donut
        if label in group_data:
            plot_group_donut(
                group_data[label],
                label,
                figsize=tuple(args.figsize_donut),
                output_path=label_dir / "group_donut.png",
            )
        else:
            print(f"  [2/3] Skipped (no group data)")

        # 3. Representative waterfall
        if args.per_sample_dir and args.per_sample_dir.exists():
            samples = load_per_sample_data(args.per_sample_dir, label)
            if samples:
                result = find_representative_sample(samples)
                if result:
                    sample_name, sample_features = result
                    plot_representative_waterfall(
                        sample_name,
                        sample_features,
                        label,
                        top_k=args.waterfall_top_k,
                        output_path=label_dir / "representative_waterfall.png",
                    )
                else:
                    print(f"  [3/3] Skipped (could not find representative)")
            else:
                print(f"  [3/3] Skipped (no per-sample data for {label})")
        else:
            print(f"  [3/3] Skipped (--per-sample-dir not provided or not found)")

    # Summary index
    index_path = args.output_dir / "index.json"
    index = {
        "labels": all_labels,
        "outputs": {
            label: {
                "summary_bar": str(args.output_dir / label / "summary_bar.png"),
                "group_donut": str(args.output_dir / label / "group_donut.png"),
                "representative_waterfall": str(
                    args.output_dir / label / "representative_waterfall.png"
                ),
            }
            for label in all_labels
        },
    }
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"\n[INFO] All outputs: {args.output_dir}")
    print(f"[INFO] Index: {index_path}")


if __name__ == "__main__":
    main()
