#!/usr/bin/env python3
"""Visualize advanced CTI analysis results.

Reads outputs from cti_advanced_analysis.py and generates:
  1. Technique co-occurrence heatmap
  2. Threat level distribution (per label)
  3. Kill chain coverage comparison
  4. Label technique similarity heatmap
  5. Top co-occurring technique pairs (network-style)

Usage:
  python3 src/visualize_cti_analysis.py \
      --input-dir reports/cti_analysis \
      --output-dir figures/cti_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.use("Agg")

STYLE = {
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": "#fafafa",
}

THREAT_COLORS = {
    "critical": "#d32f2f",
    "high": "#f57c00",
    "medium": "#fbc02d",
    "low": "#66bb6a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CTI analysis results.")
    parser.add_argument("--input-dir", type=Path, default=Path("reports/cti_analysis"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/cti_analysis"))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 1. Co-occurrence Heatmap
# ---------------------------------------------------------------------------

def plot_cooccurrence_heatmap(input_dir: Path, output_dir: Path) -> None:
    """Plot top technique co-occurrences as a heatmap."""
    plt.rcParams.update(STYLE)

    csv_path = input_dir / "technique_cooccurrence.csv"
    if not csv_path.exists():
        print("  [SKIP] Co-occurrence heatmap: no data")
        return

    # Load pairs
    pairs: List[Dict[str, Any]] = []
    all_techs: set = set()
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t1, t2 = row["technique_1"], row["technique_2"]
            count = int(row["co_count"])
            pairs.append({"t1": t1, "t2": t2, "count": count,
                          "n1": row.get("name_1", ""), "n2": row.get("name_2", "")})
            all_techs.add(t1)
            all_techs.add(t2)

    if not pairs:
        return

    # Take top 15 techniques by total co-occurrence
    tech_totals: Dict[str, int] = defaultdict(int)
    for p in pairs:
        tech_totals[p["t1"]] += p["count"]
        tech_totals[p["t2"]] += p["count"]

    top_techs = sorted(tech_totals.keys(), key=lambda t: tech_totals[t], reverse=True)[:15]
    tech_idx = {t: i for i, t in enumerate(top_techs)}
    n = len(top_techs)

    matrix = np.zeros((n, n))
    for p in pairs:
        if p["t1"] in tech_idx and p["t2"] in tech_idx:
            i, j = tech_idx[p["t1"]], tech_idx[p["t2"]]
            matrix[i, j] = p["count"]
            matrix[j, i] = p["count"]

    # Get display names
    name_lookup: Dict[str, str] = {}
    for p in pairs:
        name_lookup[p["t1"]] = p.get("n1", "")
        name_lookup[p["t2"]] = p.get("n2", "")

    labels = [f"{t}\n{name_lookup.get(t, '')[:15]}" for t in top_techs]

    fig, ax = plt.subplots(figsize=(12, 10))

    mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)
    masked = np.ma.array(matrix, mask=mask)

    im = ax.imshow(masked, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate
    for i in range(n):
        for j in range(i):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(j, i, f"{int(val)}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    ax.set_title("ATT&CK Technique Co-occurrence\n(同一検体で共起した技術ペア)", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Co-occurrence Count", shrink=0.8)

    plt.tight_layout()
    plt.savefig(str(output_dir / "cooccurrence_heatmap.png"))
    plt.close()
    print(f"  [1] Co-occurrence heatmap saved")


# ---------------------------------------------------------------------------
# 2. Threat Level Distribution
# ---------------------------------------------------------------------------

def plot_threat_levels(input_dir: Path, output_dir: Path) -> None:
    """Stacked bar chart of threat levels per label."""
    plt.rcParams.update(STYLE)

    csv_path = input_dir / "threat_level_summary.csv"
    if not csv_path.exists():
        print("  [SKIP] Threat levels: no data")
        return

    labels = []
    data: Dict[str, List[int]] = {"critical": [], "high": [], "medium": [], "low": []}

    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(row["label"])
            for level in ["critical", "high", "medium", "low"]:
                data[level].append(int(row.get(level, 0)))

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7), 6))
    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels))

    for level in ["critical", "high", "medium", "low"]:
        values = np.array(data[level])
        ax.bar(x, values, bottom=bottoms, label=level.capitalize(),
               color=THREAT_COLORS[level], width=0.7, edgecolor="white")
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Sample Count")
    ax.set_title("Threat Level Distribution per Label\nラベルごとの脅威レベル分布", fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(output_dir / "threat_level_distribution.png"))
    plt.close()
    print(f"  [2] Threat levels saved")


# ---------------------------------------------------------------------------
# 3. Kill Chain Coverage Comparison
# ---------------------------------------------------------------------------

def plot_kill_chain_coverage(input_dir: Path, output_dir: Path) -> None:
    """Horizontal bar chart comparing kill chain coverage across labels."""
    plt.rcParams.update(STYLE)

    json_path = input_dir / "threat_profiles.json"
    if not json_path.exists():
        print("  [SKIP] Kill chain coverage: no data")
        return

    with json_path.open("r", encoding="utf-8") as f:
        profiles = json.load(f)

    if not profiles:
        return

    items = sorted(profiles.items(), key=lambda x: x[1]["kill_chain_coverage"], reverse=True)
    labels = [label for label, _ in items]
    coverages = [p["kill_chain_coverage"] for _, p in items]
    n_techs = [p["n_unique_techniques"] for _, p in items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.4)))

    colors = []
    for cov in coverages:
        if cov >= 0.5:
            colors.append("#e53935")
        elif cov >= 0.3:
            colors.append("#f57c00")
        elif cov >= 0.15:
            colors.append("#fbc02d")
        else:
            colors.append("#66bb6a")

    bars = ax.barh(range(len(labels)), coverages, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Kill Chain Coverage (0 = none, 1 = full)")
    ax.set_title("ATT&CK Kill Chain Coverage per Label\nラベルごとのキルチェーンカバレッジ", fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.axvline(x=0.5, color="#999", linestyle="--", alpha=0.5, label="50%")

    for bar, cov, n in zip(bars, coverages, n_techs):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{cov:.0%} ({n}技術)", va="center", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(output_dir / "kill_chain_coverage.png"))
    plt.close()
    print(f"  [3] Kill chain coverage saved")


# ---------------------------------------------------------------------------
# 4. Label Similarity Heatmap
# ---------------------------------------------------------------------------

def plot_label_similarity(input_dir: Path, output_dir: Path) -> None:
    """Heatmap of technique-based label similarity (Jaccard)."""
    plt.rcParams.update(STYLE)

    csv_path = input_dir / "label_technique_similarity.csv"
    if not csv_path.exists():
        print("  [SKIP] Label similarity: no data")
        return

    labels = []
    matrix_rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_labels = header[1:]
        for row in reader:
            labels.append(row[0])
            matrix_rows.append([float(x) for x in row[1:]])

    if len(labels) < 2:
        return

    matrix = np.array(matrix_rows)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(7, n * 0.7)))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title("Label Similarity (Jaccard on ATT&CK Techniques)\nATT&CK技術に基づくラベル間の類似度",
                 fontweight="bold")
    fig.colorbar(im, ax=ax, label="Jaccard Similarity", shrink=0.8)

    plt.tight_layout()
    plt.savefig(str(output_dir / "label_similarity_heatmap.png"))
    plt.close()
    print(f"  [4] Label similarity heatmap saved")


# ---------------------------------------------------------------------------
# 5. Top Co-occurring Pairs (Bubble Chart)
# ---------------------------------------------------------------------------

def plot_top_pairs(input_dir: Path, output_dir: Path, top_n: int = 20) -> None:
    """Bubble chart of top co-occurring technique pairs."""
    plt.rcParams.update(STYLE)

    csv_path = input_dir / "technique_cooccurrence.csv"
    if not csv_path.exists():
        return

    pairs = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append({
                "label": f"{row['technique_1']}—{row['technique_2']}",
                "names": f"{row.get('name_1', '')[:12]} × {row.get('name_2', '')[:12]}",
                "count": int(row["co_count"]),
                "jaccard": float(row["jaccard"]),
            })

    if not pairs:
        return

    pairs = pairs[:top_n]
    pairs.reverse()

    fig, ax = plt.subplots(figsize=(12, max(6, len(pairs) * 0.4)))

    y = range(len(pairs))
    counts = [p["count"] for p in pairs]
    jaccards = [p["jaccard"] for p in pairs]
    max_count = max(counts) if counts else 1

    colors = plt.cm.RdYlBu_r([j for j in jaccards])
    sizes = [max(20, (c / max_count) * 300) for c in counts]

    scatter = ax.scatter(counts, y, s=sizes, c=jaccards, cmap="RdYlBu_r",
                         edgecolors="white", linewidth=1, vmin=0, vmax=max(jaccards) * 1.2)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{p['label']}\n{p['names']}" for p in pairs], fontsize=8)
    ax.set_xlabel("Co-occurrence Count")
    ax.set_title(f"Top-{top_n} Co-occurring ATT&CK Technique Pairs\n頻出する技術ペア", fontweight="bold")

    fig.colorbar(scatter, ax=ax, label="Jaccard Similarity", shrink=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(output_dir / "top_cooccurring_pairs.png"))
    plt.close()
    print(f"  [5] Top pairs chart saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating CTI analysis visualizations...\n")

    plot_cooccurrence_heatmap(args.input_dir, args.output_dir)
    plot_threat_levels(args.input_dir, args.output_dir)
    plot_kill_chain_coverage(args.input_dir, args.output_dir)
    plot_label_similarity(args.input_dir, args.output_dir)
    plot_top_pairs(args.input_dir, args.output_dir)

    print(f"\n[INFO] All outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
