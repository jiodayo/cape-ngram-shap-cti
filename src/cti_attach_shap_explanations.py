#!/usr/bin/env python3
"""Attach ATT&CK technique hints to per-sample SHAP results.

Supports confidence-weighted scoring and template-based explanations.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach ATT&CK techniques to SHAP per-sample top-k outputs."
    )
    parser.add_argument(
        "--shap-per-sample-dir",
        type=Path,
        default=Path(
            "shap_analysis/ngram_desc_lgbm_group_nometa_106/per_sample"),
    )
    parser.add_argument(
        "--attack-db",
        type=Path,
        default=Path("reference/mitre/attack.sqlite"),
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
        help="Optional JSON with technique_id -> explanation templates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("shap_analysis/ngram_desc_lgbm_group_nometa_106/cti"),
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=Path(
            "shap_analysis/ngram_desc_lgbm_group_nometa_106/cti/cti_results.sqlite"),
    )
    parser.add_argument("--top-techniques", type=int, default=3)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence_score to include a rule match (0.0-1.0).",
    )
    parser.add_argument(
        "--use-confidence-weight",
        action="store_true",
        default=True,
        help="Weight technique scores by confidence_score.",
    )
    return parser.parse_args()


def load_attack_db(path: Path) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    conn = sqlite3.connect(str(path))
    techniques: Dict[str, str] = {}
    tactics: Dict[str, List[str]] = {}

    for technique_id, name in conn.execute("SELECT technique_id, name FROM techniques"):
        techniques[str(technique_id)] = str(name)

    for technique_id, tactic in conn.execute(
        "SELECT technique_id, tactic FROM technique_tactics"
    ):
        tid = str(technique_id)
        tactics.setdefault(tid, []).append(str(tactic))

    conn.close()
    for tid in tactics:
        tactics[tid] = sorted(set(tactics[tid]))
    return techniques, tactics


def load_rules(path: Path, min_confidence: float = 0.0) -> List[Dict[str, object]]:
    rules: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pattern = (row.get("pattern") or "").strip()
            technique_id = (row.get("technique_id") or "").strip()
            if not pattern or not technique_id or pattern.startswith("#"):
                continue

            # Parse confidence_score (new column, default 0.5)
            try:
                confidence_score = float(row.get("confidence_score") or "0.5")
            except (ValueError, TypeError):
                confidence_score = 0.5

            if confidence_score < min_confidence:
                continue

            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error:
                continue

            rules.append(
                {
                    "regex": regex,
                    "technique_id": technique_id,
                    "confidence": (row.get("confidence") or "").strip(),
                    "confidence_score": confidence_score,
                    "category": (row.get("category") or "").strip(),
                    "rationale": (row.get("rationale") or "").strip(),
                }
            )
    return rules


def load_templates(path: Path) -> Dict[str, Dict[str, str]]:
    """Load explanation templates if available."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_tokens(feature: str) -> List[str]:
    if feature.startswith("seq:"):
        return feature[4:].split()
    if feature.startswith("skip:"):
        text = feature[5:]
        match = re.match(r"(.+?)__SKIP\d+__(.+)", text)
        if match:
            return [match.group(1), match.group(2)]
        return [text]
    if feature.startswith("desc:"):
        return [feature[5:]]
    return [feature]


def map_feature(
    feature: str,
    rules: Sequence[Dict[str, object]],
) -> List[Tuple[str, float, str]]:
    """Map a feature to ATT&CK techniques.

    Returns list of (technique_id, confidence_score, category) tuples.
    """
    tokens = extract_tokens(feature)
    matched: List[Tuple[str, float, str]] = []
    for rule in rules:
        regex = rule["regex"]
        for token in tokens + [feature]:
            if regex.search(token):
                matched.append((
                    str(rule["technique_id"]),
                    float(rule["confidence_score"]),
                    str(rule.get("category", "")),
                ))
                break
    return matched


def build_explanation(
    top_ids: List[str],
    top_scores: List[float],
    techniques: Dict[str, str],
    tactics: Dict[str, List[str]],
    templates: Dict[str, Dict[str, str]],
    matched_apis: Dict[str, List[str]],
) -> str:
    """Build a natural language explanation from matched techniques."""
    if not top_ids:
        return "ATT&CKマッピングに一致する特徴量はありませんでした。"

    parts: List[str] = []
    for tid, score in zip(top_ids, top_scores):
        tech_name = techniques.get(tid, tid)
        tactic_list = tactics.get(tid, [])
        tactic_str = ", ".join(tactic_list) if tactic_list else "不明"

        # Use template if available
        template = templates.get(tid, {})
        if template and "summary" in template:
            summary = template["summary"]
            # Interpolate matched APIs if available
            apis = matched_apis.get(tid, [])
            api_str = ", ".join(apis[:5]) if apis else "N/A"
            summary = summary.replace("{matched_apis}", api_str)
            summary = summary.replace("{tactic}", tactic_str)
            parts.append(f"{tid} {tech_name} (スコア:{score:.3f}): {summary}")
        else:
            parts.append(
                f"{tid} {tech_name} (スコア:{score:.3f}, 戦術:{tactic_str})"
            )

    return "検出された技術: " + "; ".join(parts)


def ensure_output_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sample_summary (
            sample TEXT,
            label TEXT,
            top_features TEXT,
            technique_ids TEXT,
            technique_names TEXT,
            technique_scores TEXT,
            tactics TEXT,
            explanation TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sample_techniques (
            sample TEXT,
            label TEXT,
            technique_id TEXT,
            technique_name TEXT,
            score REAL,
            confidence_score REAL,
            category TEXT,
            tactics TEXT,
            PRIMARY KEY (sample, label, technique_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sample_features (
            sample TEXT,
            label TEXT,
            rank INTEGER,
            feature TEXT,
            abs_shap REAL,
            technique_ids TEXT,
            PRIMARY KEY (sample, label, rank)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_st_sample ON sample_techniques(sample)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_st_technique ON sample_techniques(technique_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sf_sample ON sample_features(sample)")
    conn.commit()
    return conn


def write_sample_row(
    writer: csv.writer,
    conn: sqlite3.Connection,
    sample: str,
    label: str,
    top_features: List[str],
    feature_shap_values: Dict[str, float],
    technique_scores: Dict[str, float],
    technique_confidences: Dict[str, float],
    technique_categories: Dict[str, str],
    technique_matched_apis: Dict[str, List[str]],
    techniques: Dict[str, str],
    tactics: Dict[str, List[str]],
    templates: Dict[str, Dict[str, str]],
    top_n: int,
) -> None:
    if technique_scores:
        sorted_tech = sorted(
            technique_scores.items(), key=lambda kv: kv[1], reverse=True
        )
    else:
        sorted_tech = []

    top_ids = [tid for tid, _ in sorted_tech[:top_n]]
    top_score_values = [score for _, score in sorted_tech[:top_n]]
    top_names = [techniques.get(tid, "") for tid in top_ids]
    top_scores_str = [f"{tid}:{score:.4f}" for tid, score in sorted_tech[:top_n]]
    top_tactics = [";".join(tactics.get(tid, [])) for tid in top_ids]

    explanation = build_explanation(
        top_ids, top_score_values, techniques, tactics, templates, technique_matched_apis
    )

    row = [
        sample,
        label,
        "|".join(top_features),
        "|".join(top_ids),
        "|".join(top_names),
        "|".join(top_scores_str),
        "|".join(top_tactics),
        explanation,
    ]
    writer.writerow(row)

    conn.execute(
        "INSERT OR REPLACE INTO sample_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        row,
    )

    # Normalized technique rows
    for tid, score in sorted_tech:
        conn.execute(
            "INSERT OR REPLACE INTO sample_techniques VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sample,
                label,
                tid,
                techniques.get(tid, ""),
                score,
                technique_confidences.get(tid, 0.5),
                technique_categories.get(tid, ""),
                ";".join(tactics.get(tid, [])),
            ),
        )

    # Feature rows
    for rank, feature in enumerate(top_features, start=1):
        feat_techniques = []
        tokens = extract_tokens(feature)
        # Simple re-match for feature-level technique tracking
        conn.execute(
            "INSERT OR REPLACE INTO sample_features VALUES (?, ?, ?, ?, ?, ?)",
            (
                sample,
                label,
                rank,
                feature,
                feature_shap_values.get(feature, 0.0),
                "",  # technique_ids populated at query time via JOINs
            ),
        )


def main() -> None:
    args = parse_args()

    techniques, tactics = load_attack_db(args.attack_db)
    rules = load_rules(args.rules, min_confidence=args.min_confidence)
    templates = load_templates(args.templates)

    print(f"[INFO] Loaded {len(rules)} rules (min_confidence={args.min_confidence})")
    print(f"[INFO] Loaded {len(templates)} explanation templates")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_db = ensure_output_db(args.output_db)

    label_summary: Dict[str, Dict[str, Dict[str, object]]] = {}

    for csv_path in sorted(args.shap_per_sample_dir.glob("shap_per_sample_topk_*.csv")):
        label = csv_path.stem.replace("shap_per_sample_topk_", "")
        out_path = output_dir / f"cti_per_sample_summary_{label}.csv"

        with csv_path.open("r", encoding="utf-8") as f_in, out_path.open(
            "w", encoding="utf-8", newline=""
        ) as f_out:
            reader = csv.DictReader(f_in)
            writer = csv.writer(f_out)
            writer.writerow(
                [
                    "sample",
                    "label",
                    "top_features",
                    "technique_ids",
                    "technique_names",
                    "technique_scores",
                    "tactics",
                    "explanation",
                ]
            )

            current_sample: Optional[str] = None
            current_label: Optional[str] = None
            current_features: List[Tuple[int, str]] = []
            current_feature_shap: Dict[str, float] = {}
            current_techniques: Dict[str, float] = {}
            current_confidences: Dict[str, float] = {}
            current_categories: Dict[str, str] = {}
            current_matched_apis: Dict[str, List[str]] = {}

            def flush() -> None:
                if current_sample is None or current_label is None:
                    return
                top_features = [feat for _, feat in sorted(current_features)]
                write_sample_row(
                    writer,
                    output_db,
                    current_sample,
                    current_label,
                    top_features,
                    current_feature_shap,
                    current_techniques,
                    current_confidences,
                    current_categories,
                    current_matched_apis,
                    techniques,
                    tactics,
                    templates,
                    args.top_techniques,
                )

                for tid, score in current_techniques.items():
                    label_summary.setdefault(current_label, {}).setdefault(
                        tid, {"score": 0.0, "samples": set()}
                    )
                    label_summary[current_label][tid]["score"] = (
                        label_summary[current_label][tid]["score"] + score
                    )
                    label_summary[current_label][tid]["samples"].add(
                        current_sample)

            for row in reader:
                sample = row.get("sample") or ""
                row_label = row.get("label") or label
                feature = row.get("feature") or ""
                rank = int(row.get("rank") or 0)
                abs_shap = float(row.get("abs_shap") or 0.0)

                if current_sample is None:
                    current_sample = sample
                    current_label = row_label

                if sample != current_sample:
                    flush()
                    current_sample = sample
                    current_label = row_label
                    current_features = []
                    current_feature_shap = {}
                    current_techniques = {}
                    current_confidences = {}
                    current_categories = {}
                    current_matched_apis = {}

                current_features.append((rank, feature))
                current_feature_shap[feature] = abs_shap

                matches = map_feature(feature, rules)
                for tid, conf_score, category in matches:
                    # Confidence-weighted scoring
                    if args.use_confidence_weight:
                        weighted = abs_shap * conf_score
                    else:
                        weighted = abs_shap

                    current_techniques[tid] = current_techniques.get(tid, 0.0) + weighted
                    current_confidences[tid] = max(
                        current_confidences.get(tid, 0.0), conf_score
                    )
                    if category:
                        current_categories[tid] = category

                    # Track matched API tokens for explanations
                    tokens = extract_tokens(feature)
                    current_matched_apis.setdefault(tid, []).extend(tokens)

            flush()

    output_db.commit()
    output_db.close()

    summary_path = output_dir / "cti_label_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "technique_id",
                "technique_name",
                "tactics",
                "sample_count",
                "total_score",
            ]
        )
        for label, tech_map in sorted(label_summary.items()):
            for tid, stats in sorted(
                tech_map.items(), key=lambda kv: kv[1]["score"], reverse=True
            ):
                writer.writerow(
                    [
                        label,
                        tid,
                        techniques.get(tid, ""),
                        ";".join(tactics.get(tid, [])),
                        len(stats["samples"]),
                        f"{stats['score']:.4f}",
                    ]
                )

    meta_path = output_dir / "cti_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rules": str(args.rules),
                "attack_db": str(args.attack_db),
                "templates": str(args.templates),
                "top_techniques": args.top_techniques,
                "min_confidence": args.min_confidence,
                "use_confidence_weight": args.use_confidence_weight,
                "num_rules_loaded": len(rules),
                "num_templates_loaded": len(templates),
            },
            f,
            indent=2,
        )

    print(f"[INFO] output dir: {output_dir}")


if __name__ == "__main__":
    main()
