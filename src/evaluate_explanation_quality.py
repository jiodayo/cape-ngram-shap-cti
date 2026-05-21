#!/usr/bin/env python3
"""Evaluate explanation quality beyond model performance.

Metrics:
  1. rule_match_rate     – fraction of top-k features matching ATT&CK rules
  2. technique_diversity – unique technique_ids per label
  3. technique_concentration – top-3 technique score share (Gini-like)
  4. group_entropy       – entropy of seq/skip/desc contribution shares
  5. explanation_coverage – fraction of samples with template-based explanations

Usage:
  python3 src/evaluate_explanation_quality.py \
      --cti-db shap_analysis/.../cti/cti_results.sqlite \
      --group-csv shap_analysis/.../shap_group_contributions.csv \
      --rules reference/mitre/api_to_attack_rules.csv \
      --templates reference/mitre/explanation_templates.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from common import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate explanation quality metrics."
    )
    parser.add_argument(
        "--cti-db",
        type=Path,
        default=Path("shap_analysis/ngram_desc_lgbm_group_nometa_106/cti/cti_results.sqlite"),
    )
    parser.add_argument(
        "--group-csv",
        type=Path,
        default=Path("shap_analysis/ngram_desc_lgbm_group/shap_group_contributions.csv"),
    )
    parser.add_argument(
        "--per-sample-dir",
        type=Path,
        default=Path("shap_analysis/ngram_desc_lgbm_group_nometa_106/per_sample"),
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path("reference/mitre/api_to_attack_rules.csv"),
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path("reference/mitre/explanation_templates.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation"),
    )
    return parser.parse_args()


def compute_entropy(shares: List[float]) -> float:
    """Compute Shannon entropy of a distribution."""
    total = sum(shares)
    if total <= 0:
        return 0.0
    probs = [s / total for s in shares if s > 0]
    return -sum(p * math.log2(p) for p in probs)


def compute_rule_match_rate(
    per_sample_dir: Path,
    rules_path: Path,
) -> Dict[str, float]:
    """Compute fraction of per-sample top-k features that match any rule."""
    import re

    # Load rules
    patterns = []
    with rules_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pattern = (row.get("pattern") or "").strip()
            if not pattern or pattern.startswith("#"):
                continue
            try:
                patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                continue

    if not patterns:
        return {}

    results = {}
    for csv_path in sorted(per_sample_dir.glob("shap_per_sample_topk_*.csv")):
        label = csv_path.stem.replace("shap_per_sample_topk_", "")
        total_features = 0
        matched_features = 0

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feature = row.get("feature", "")
                total_features += 1

                # Extract tokens from feature
                if feature.startswith("seq:"):
                    tokens = feature[4:].split()
                elif feature.startswith("skip:"):
                    tokens = [feature[5:]]
                elif feature.startswith("desc:"):
                    tokens = [feature[5:]]
                else:
                    tokens = [feature]

                # Check if any token matches any rule
                found = False
                for token in tokens + [feature]:
                    for regex in patterns:
                        if regex.search(token):
                            found = True
                            break
                    if found:
                        break
                if found:
                    matched_features += 1

        if total_features > 0:
            results[label] = matched_features / total_features

    return results


def compute_technique_diversity(conn: sqlite3.Connection) -> Dict[str, int]:
    """Count unique technique_ids per label."""
    try:
        rows = conn.execute(
            "SELECT label, COUNT(DISTINCT technique_id) FROM sample_techniques GROUP BY label"
        ).fetchall()
        return {label: count for label, count in rows}
    except sqlite3.OperationalError:
        # Fallback to sample_summary
        rows = conn.execute(
            "SELECT label, technique_ids FROM sample_summary"
        ).fetchall()
        label_tech: Dict[str, set] = {}
        for label, tech_str in rows:
            if tech_str:
                for tid in tech_str.split("|"):
                    if tid.strip():
                        label_tech.setdefault(label, set()).add(tid.strip())
        return {label: len(tids) for label, tids in label_tech.items()}


def compute_technique_concentration(conn: sqlite3.Connection) -> Dict[str, float]:
    """Compute top-3 technique score share per label."""
    try:
        labels = [row[0] for row in conn.execute(
            "SELECT DISTINCT label FROM sample_techniques"
        ).fetchall()]
    except sqlite3.OperationalError:
        return {}

    results = {}
    for label in labels:
        rows = conn.execute(
            "SELECT technique_id, SUM(score) as total FROM sample_techniques "
            "WHERE label = ? GROUP BY technique_id ORDER BY total DESC",
            (label,),
        ).fetchall()

        if not rows:
            continue

        total = sum(score for _, score in rows)
        if total <= 0:
            continue

        top3 = sum(score for _, score in rows[:3])
        results[label] = top3 / total

    return results


def compute_group_entropy(group_csv: Path) -> Dict[str, float]:
    """Compute entropy of feature group shares per label."""
    label_shares: Dict[str, List[float]] = {}

    if not group_csv.exists():
        return {}

    with group_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "")
            share = float(row.get("share", 0))
            label_shares.setdefault(label, []).append(share)

    return {label: compute_entropy(shares) for label, shares in label_shares.items()}


def compute_explanation_coverage(
    conn: sqlite3.Connection,
    templates: Dict,
) -> Dict[str, float]:
    """Compute fraction of samples with at least one template-matched technique."""
    template_tids = set(templates.keys())
    if not template_tids:
        return {}

    try:
        rows = conn.execute(
            "SELECT label, sample, technique_ids FROM sample_summary"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}

    label_total: Dict[str, int] = {}
    label_covered: Dict[str, int] = {}

    for label, sample, tech_str in rows:
        label_total[label] = label_total.get(label, 0) + 1
        if tech_str:
            tids = {t.strip() for t in tech_str.split("|") if t.strip()}
            if tids & template_tids:
                label_covered[label] = label_covered.get(label, 0) + 1

    return {
        label: label_covered.get(label, 0) / total
        for label, total in label_total.items()
        if total > 0
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    conn = None
    if args.cti_db.exists():
        conn = sqlite3.connect(str(args.cti_db))
        print(f"[INFO] CTI DB: {args.cti_db}")
    else:
        print(f"[WARN] CTI DB not found: {args.cti_db}")

    templates = {}
    if args.templates.exists():
        templates = load_json(args.templates)
        print(f"[INFO] Templates: {len(templates)} techniques")

    # Compute metrics
    all_metrics: Dict[str, Dict[str, float]] = {}

    # 1. Rule match rate
    if args.per_sample_dir.exists() and args.rules.exists():
        print("[INFO] Computing rule match rate...")
        rule_match = compute_rule_match_rate(args.per_sample_dir, args.rules)
        for label, rate in rule_match.items():
            all_metrics.setdefault(label, {})["rule_match_rate"] = rate
    else:
        print("[WARN] Skipping rule match rate (missing per-sample dir or rules)")

    if conn:
        # 2. Technique diversity
        print("[INFO] Computing technique diversity...")
        diversity = compute_technique_diversity(conn)
        for label, count in diversity.items():
            all_metrics.setdefault(label, {})["technique_diversity"] = count

        # 3. Technique concentration
        print("[INFO] Computing technique concentration...")
        concentration = compute_technique_concentration(conn)
        for label, conc in concentration.items():
            all_metrics.setdefault(label, {})["technique_concentration"] = conc

        # 5. Explanation coverage
        if templates:
            print("[INFO] Computing explanation coverage...")
            coverage = compute_explanation_coverage(conn, templates)
            for label, cov in coverage.items():
                all_metrics.setdefault(label, {})["explanation_coverage"] = cov

        conn.close()

    # 4. Group entropy
    if args.group_csv.exists():
        print("[INFO] Computing group entropy...")
        entropy = compute_group_entropy(args.group_csv)
        for label, ent in entropy.items():
            all_metrics.setdefault(label, {})["group_entropy"] = ent

    # Output
    if not all_metrics:
        print("[WARN] No metrics computed. Check input paths.")
        return

    # JSON output
    json_path = args.output_dir / "explanation_quality.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] JSON: {json_path}")

    # CSV output
    csv_path = args.output_dir / "explanation_quality.csv"
    all_metric_names = sorted(set(
        k for metrics in all_metrics.values() for k in metrics.keys()
    ))

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + all_metric_names)
        for label in sorted(all_metrics.keys()):
            row = [label]
            for metric in all_metric_names:
                val = all_metrics[label].get(metric, "")
                if isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(val)
            writer.writerow(row)
    print(f"[INFO] CSV: {csv_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPLANATION QUALITY SUMMARY")
    print("=" * 70)

    # Aggregate means
    for metric in all_metric_names:
        values = [m[metric] for m in all_metrics.values() if metric in m and isinstance(m[metric], (int, float))]
        if values:
            mean_val = sum(values) / len(values)
            print(f"  {metric:30s}  mean={mean_val:.4f}  (n={len(values)} labels)")

    print()


if __name__ == "__main__":
    main()
