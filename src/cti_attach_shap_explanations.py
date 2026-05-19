#!/usr/bin/env python3
"""Attach ATT&CK technique hints to per-sample SHAP results."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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


def load_rules(path: Path) -> List[Dict[str, object]]:
    rules: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pattern = (row.get("pattern") or "").strip()
            technique_id = (row.get("technique_id") or "").strip()
            if not pattern or not technique_id:
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
                    "rationale": (row.get("rationale") or "").strip(),
                }
            )
    return rules


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


def map_feature(feature: str, rules: Sequence[Dict[str, object]]) -> List[str]:
    tokens = extract_tokens(feature)
    matched: List[str] = []
    for rule in rules:
        regex = rule["regex"]
        for token in tokens + [feature]:
            if regex.search(token):
                matched.append(str(rule["technique_id"]))
                break
    return matched


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
    conn.commit()
    return conn


def write_sample_row(
    writer: csv.writer,
    conn: sqlite3.Connection,
    sample: str,
    label: str,
    top_features: List[str],
    technique_scores: Dict[str, float],
    techniques: Dict[str, str],
    tactics: Dict[str, List[str]],
    top_n: int,
) -> None:
    if technique_scores:
        sorted_tech = sorted(
            technique_scores.items(), key=lambda kv: kv[1], reverse=True
        )
    else:
        sorted_tech = []

    top_ids = [tid for tid, _ in sorted_tech[:top_n]]
    top_names = [techniques.get(tid, "") for tid in top_ids]
    top_scores = [f"{tid}:{score:.4f}" for tid, score in sorted_tech[:top_n]]
    top_tactics = [";".join(tactics.get(tid, [])) for tid in top_ids]

    if top_ids:
        pairs = [f"{tid} {techniques.get(tid, '')}" for tid in top_ids]
        explanation = "Mapped techniques: " + "; ".join(pairs)
    else:
        explanation = "No ATT&CK mapping matched."

    row = [
        sample,
        label,
        "|".join(top_features),
        "|".join(top_ids),
        "|".join(top_names),
        "|".join(top_scores),
        "|".join(top_tactics),
        explanation,
    ]
    writer.writerow(row)

    conn.execute(
        "INSERT INTO sample_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        row,
    )


def main() -> None:
    args = parse_args()

    techniques, tactics = load_attack_db(args.attack_db)
    rules = load_rules(args.rules)

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
            current_scores: Dict[str, float] = {}
            current_techniques: Dict[str, float] = {}
            current_seen_tech: set[str] = set()

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
                    current_techniques,
                    techniques,
                    tactics,
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
                    current_scores = {}
                    current_techniques = {}
                    current_seen_tech = set()

                current_features.append((rank, feature))
                matches = map_feature(feature, rules)
                for tid in matches:
                    current_techniques[tid] = current_techniques.get(
                        tid, 0.0) + abs_shap
                    current_scores[tid] = current_scores.get(
                        tid, 0.0) + abs_shap
                    current_seen_tech.add(tid)

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
                "top_techniques": args.top_techniques,
                "note": "Heuristic mapping from API patterns to ATT&CK techniques.",
            },
            f,
            indent=2,
        )

    print(f"[INFO] output dir: {output_dir}")


if __name__ == "__main__":
    main()
