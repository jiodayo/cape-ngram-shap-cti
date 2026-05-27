#!/usr/bin/env python3
"""Advanced CTI analysis: co-occurrence, attack chains, ATT&CK Navigator export.

Features:
  1. Technique co-occurrence matrix — which techniques appear together
  2. Per-sample attack chain reconstruction — kill chain narrative
  3. ATT&CK Navigator layer export — for MITRE ATT&CK Navigator webapp
  4. Label threat profiles — quantified comparison of attack patterns
  5. Campaign-level clustering — group samples by technique similarity

Usage:
  python3 src/cti_advanced_analysis.py \
      --cti-db shap_analysis/.../cti/cti_results.sqlite \
      --attack-db reference/mitre/attack.sqlite \
      --templates reference/mitre/explanation_templates.json \
      --output-dir reports/cti_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Kill chain order
# ---------------------------------------------------------------------------

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

TACTIC_DISPLAY_JA = {
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
    "command-and-control": "C2通信",
    "exfiltration": "持ち出し",
    "impact": "影響",
}

TACTIC_DISPLAY_EN = {
    "reconnaissance": "Reconnaissance",
    "resource-development": "Resource Development",
    "initial-access": "Initial Access",
    "execution": "Execution",
    "persistence": "Persistence",
    "privilege-escalation": "Privilege Escalation",
    "defense-evasion": "Defense Evasion",
    "credential-access": "Credential Access",
    "discovery": "Discovery",
    "lateral-movement": "Lateral Movement",
    "collection": "Collection",
    "command-and-control": "Command and Control",
    "exfiltration": "Exfiltration",
    "impact": "Impact",
}

SEVERITY_WEIGHTS = {"high": 3, "medium": 2, "low": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advanced CTI analysis from SHAP-mapped ATT&CK results."
    )
    parser.add_argument(
        "--cti-db", type=Path,
        default=Path("shap_analysis/ngram_desc_lgbm_group_nometa_106/cti/cti_results.sqlite"),
    )
    parser.add_argument(
        "--attack-db", type=Path,
        default=Path("reference/mitre/attack.sqlite"),
    )
    parser.add_argument(
        "--templates", type=Path,
        default=Path("reference/mitre/explanation_templates.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("reports/cti_analysis"),
    )
    parser.add_argument("--top-samples", type=int, default=50,
                        help="Max samples for detailed attack chain output.")
    parser.add_argument("--labels", type=str, default="",
                        help="Comma-separated labels (empty = all).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sample_techniques(db_path: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Load {label: {sample: [{technique_id, score, confidence_score, category, tactics}]}}."""
    conn = sqlite3.connect(str(db_path))
    data: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    try:
        rows = conn.execute(
            "SELECT sample, label, technique_id, technique_name, score, "
            "confidence_score, category, tactics FROM sample_techniques"
        ).fetchall()
        for sample, label, tid, name, score, conf, cat, tactics in rows:
            data[label][sample].append({
                "technique_id": tid,
                "technique_name": name or tid,
                "score": float(score or 0),
                "confidence_score": float(conf or 0.5),
                "category": cat or "",
                "tactics": [t.strip() for t in (tactics or "").split(";") if t.strip()],
            })
    except sqlite3.OperationalError:
        pass
    conn.close()
    return dict(data)


def load_attack_details(db_path: Path) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, str]]:
    """Load technique names, tactics, and descriptions from ATT&CK DB."""
    if not db_path.exists():
        return {}, {}, {}
    conn = sqlite3.connect(str(db_path))
    names: Dict[str, str] = {}
    tactics: Dict[str, List[str]] = defaultdict(list)
    descriptions: Dict[str, str] = {}

    try:
        for tid, name, desc in conn.execute("SELECT technique_id, name, description FROM techniques"):
            names[tid] = name
            descriptions[tid] = desc or ""
    except sqlite3.OperationalError:
        pass

    try:
        for tid, tactic in conn.execute("SELECT technique_id, tactic FROM technique_tactics"):
            tactics[tid].append(tactic)
    except sqlite3.OperationalError:
        pass

    conn.close()
    return names, dict(tactics), descriptions


def load_templates(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Technique Co-occurrence
# ---------------------------------------------------------------------------

def compute_cooccurrence(
    data: Dict[str, Dict[str, List[Dict]]],
    filter_labels: Optional[List[str]] = None,
) -> Tuple[Dict[Tuple[str, str], int], Dict[str, int]]:
    """Compute how often technique pairs appear together in the same sample."""
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    tech_counts: Dict[str, int] = defaultdict(int)

    for label, samples in data.items():
        if filter_labels and label not in filter_labels:
            continue
        for sample, techniques in samples.items():
            tids = sorted(set(t["technique_id"] for t in techniques))
            for tid in tids:
                tech_counts[tid] += 1
            for t1, t2 in combinations(tids, 2):
                pair_counts[(t1, t2)] += 1

    return dict(pair_counts), dict(tech_counts)


def export_cooccurrence(
    pair_counts: Dict[Tuple[str, str], int],
    tech_counts: Dict[str, int],
    tech_names: Dict[str, str],
    output_dir: Path,
) -> None:
    """Export co-occurrence matrix as CSV and JSON."""

    # CSV: pair list sorted by frequency
    csv_path = output_dir / "technique_cooccurrence.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["technique_1", "name_1", "technique_2", "name_2",
                         "co_count", "jaccard", "pmi"])
        for (t1, t2), count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
            c1 = tech_counts.get(t1, 1)
            c2 = tech_counts.get(t2, 1)
            total = sum(tech_counts.values()) or 1
            jaccard = count / (c1 + c2 - count) if (c1 + c2 - count) > 0 else 0
            # PMI (pointwise mutual information)
            p_joint = count / total
            p1 = c1 / total
            p2 = c2 / total
            pmi = np.log2(p_joint / (p1 * p2)) if p1 > 0 and p2 > 0 and p_joint > 0 else 0

            writer.writerow([
                t1, tech_names.get(t1, ""), t2, tech_names.get(t2, ""),
                count, f"{jaccard:.4f}", f"{pmi:.4f}",
            ])

    # JSON: adjacency list for visualization
    adj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (t1, t2), count in pair_counts.items():
        c1 = tech_counts.get(t1, 1)
        c2 = tech_counts.get(t2, 1)
        jaccard = count / (c1 + c2 - count) if (c1 + c2 - count) > 0 else 0
        adj[t1].append({"target": t2, "count": count, "jaccard": round(jaccard, 4)})
        adj[t2].append({"target": t1, "count": count, "jaccard": round(jaccard, 4)})

    json_path = output_dir / "technique_cooccurrence.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"adjacency": dict(adj), "tech_counts": tech_counts}, f, indent=2, ensure_ascii=False)

    print(f"  [1] Co-occurrence: {csv_path} ({len(pair_counts)} pairs)")


# ---------------------------------------------------------------------------
# 2. Per-Sample Attack Chain
# ---------------------------------------------------------------------------

def build_attack_chain(
    techniques: List[Dict[str, Any]],
    tactic_mapping: Dict[str, List[str]],
    tech_names: Dict[str, str],
    templates: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    """Build a kill-chain ordered attack narrative for one sample."""

    # Group techniques by tactic phase
    tactic_techniques: Dict[str, List[Dict]] = defaultdict(list)
    for tech in techniques:
        tid = tech["technique_id"]
        tactics = tactic_mapping.get(tid, tech.get("tactics", []))
        if not tactics:
            tactics = ["unknown"]
        for tactic in tactics:
            tactic_techniques[tactic].append({
                "technique_id": tid,
                "technique_name": tech_names.get(tid, tech.get("technique_name", tid)),
                "score": tech["score"],
                "confidence": tech.get("confidence_score", 0.5),
                "category": tech.get("category", ""),
            })

    # Build ordered chain
    chain: List[Dict[str, Any]] = []
    for tactic in TACTIC_ORDER:
        if tactic not in tactic_techniques:
            continue
        techs = sorted(tactic_techniques[tactic], key=lambda x: x["score"], reverse=True)
        chain.append({
            "tactic": tactic,
            "tactic_ja": TACTIC_DISPLAY_JA.get(tactic, tactic),
            "tactic_en": TACTIC_DISPLAY_EN.get(tactic, tactic),
            "techniques": techs,
        })

    # Unknown tactics
    if "unknown" in tactic_techniques:
        chain.append({
            "tactic": "unknown",
            "tactic_ja": "未分類",
            "tactic_en": "Unknown",
            "techniques": tactic_techniques["unknown"],
        })

    # Generate narrative
    narrative_parts: List[str] = []
    for phase in chain:
        tactic_ja = phase["tactic_ja"]
        tech_strs = []
        for t in phase["techniques"][:3]:  # Top 3 per tactic
            tid = t["technique_id"]
            name = t["technique_name"]
            template = templates.get(tid, {})
            if template and "summary" in template:
                tech_strs.append(f"{tid} {name}: {template['summary']}")
            else:
                tech_strs.append(f"{tid} {name} (score:{t['score']:.3f})")
        narrative_parts.append(f"【{tactic_ja}】{'; '.join(tech_strs)}")

    narrative = " → ".join(narrative_parts)

    # Threat level
    total_score = sum(t["score"] for t in techniques)
    high_conf_count = sum(1 for t in techniques if t.get("confidence_score", 0) >= 0.7)
    n_tactics = len(chain)

    if total_score > 5 and n_tactics >= 4:
        threat_level = "critical"
    elif total_score > 2 and n_tactics >= 3:
        threat_level = "high"
    elif total_score > 1 and n_tactics >= 2:
        threat_level = "medium"
    else:
        threat_level = "low"

    return {
        "chain": chain,
        "narrative": narrative,
        "threat_level": threat_level,
        "total_score": total_score,
        "n_tactics": n_tactics,
        "n_techniques": len(techniques),
        "high_confidence_techniques": high_conf_count,
    }


def export_attack_chains(
    data: Dict[str, Dict[str, List[Dict]]],
    tactic_mapping: Dict[str, List[str]],
    tech_names: Dict[str, str],
    templates: Dict[str, Dict[str, str]],
    output_dir: Path,
    top_samples: int,
    filter_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Export per-sample attack chains as JSON and summary CSV."""

    all_chains: Dict[str, Dict[str, Any]] = {}
    threat_summary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for label, samples in sorted(data.items()):
        if filter_labels and label not in filter_labels:
            continue

        # Sort samples by total technique score
        sorted_samples = sorted(
            samples.items(),
            key=lambda x: sum(t["score"] for t in x[1]),
            reverse=True,
        )

        for sample, techniques in sorted_samples[:top_samples]:
            chain_result = build_attack_chain(techniques, tactic_mapping, tech_names, templates)
            key = f"{label}/{sample}"
            all_chains[key] = {
                "label": label,
                "sample": sample,
                **chain_result,
            }
            threat_summary[label][chain_result["threat_level"]] += 1

    # Export detailed chains
    json_path = output_dir / "attack_chains.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_chains, f, indent=2, ensure_ascii=False)

    # Export summary CSV
    csv_path = output_dir / "attack_chain_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "sample", "threat_level", "n_tactics",
                         "n_techniques", "total_score", "narrative"])
        for key, chain in sorted(all_chains.items(), key=lambda x: x[1]["total_score"], reverse=True):
            writer.writerow([
                chain["label"], chain["sample"], chain["threat_level"],
                chain["n_tactics"], chain["n_techniques"],
                f"{chain['total_score']:.3f}", chain["narrative"],
            ])

    # Export threat level summary
    threat_path = output_dir / "threat_level_summary.csv"
    with threat_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "critical", "high", "medium", "low", "total"])
        for label in sorted(threat_summary.keys()):
            levels = threat_summary[label]
            total = sum(levels.values())
            writer.writerow([
                label, levels.get("critical", 0), levels.get("high", 0),
                levels.get("medium", 0), levels.get("low", 0), total,
            ])

    print(f"  [2] Attack chains: {json_path} ({len(all_chains)} samples)")
    return threat_summary


# ---------------------------------------------------------------------------
# 3. ATT&CK Navigator Layer Export
# ---------------------------------------------------------------------------

def export_navigator_layers(
    data: Dict[str, Dict[str, List[Dict]]],
    tech_names: Dict[str, str],
    output_dir: Path,
    filter_labels: Optional[List[str]] = None,
) -> None:
    """Export ATT&CK Navigator v4 layer JSON files."""

    nav_dir = output_dir / "navigator_layers"
    nav_dir.mkdir(parents=True, exist_ok=True)

    # Color gradient for scores
    def score_to_color(score: float, max_score: float) -> str:
        if max_score <= 0:
            return "#ffffff"
        ratio = min(score / max_score, 1.0)
        # White → Yellow → Orange → Red
        if ratio < 0.33:
            r, g, b = 255, 255, int(255 * (1 - ratio * 3))
        elif ratio < 0.66:
            r = 255
            g = int(255 * (1 - (ratio - 0.33) * 3))
            b = 0
        else:
            r = 255
            g = int(80 * (1 - (ratio - 0.66) * 3))
            b = 0
        return f"#{r:02x}{g:02x}{b:02x}"

    for label, samples in sorted(data.items()):
        if filter_labels and label not in filter_labels:
            continue

        # Aggregate technique scores
        tech_scores: Dict[str, float] = defaultdict(float)
        tech_sample_counts: Dict[str, int] = defaultdict(int)

        for sample, techniques in samples.items():
            seen: Set[str] = set()
            for tech in techniques:
                tid = tech["technique_id"]
                tech_scores[tid] += tech["score"]
                if tid not in seen:
                    tech_sample_counts[tid] += 1
                    seen.add(tid)

        if not tech_scores:
            continue

        max_score = max(tech_scores.values())

        # Build Navigator layer
        techniques_layer = []
        for tid, score in sorted(tech_scores.items(), key=lambda x: x[1], reverse=True):
            techniques_layer.append({
                "techniqueID": tid,
                "tactic": "",  # Navigator auto-resolves
                "color": score_to_color(score, max_score),
                "comment": (
                    f"Score: {score:.3f}\n"
                    f"Samples: {tech_sample_counts[tid]}\n"
                    f"Name: {tech_names.get(tid, '')}"
                ),
                "enabled": True,
                "metadata": [],
                "links": [],
                "showSubtechniques": True,
                "score": round(score, 4),
            })

        layer = {
            "name": f"SHAP-CTI: {label}",
            "versions": {
                "attack": "14",
                "navigator": "4.9.1",
                "layer": "4.5",
            },
            "domain": "enterprise-attack",
            "description": (
                f"ATT&CK technique heatmap for malware label '{label}' "
                f"generated from SHAP feature analysis.\n"
                f"Total samples: {len(samples)}\n"
                f"Techniques detected: {len(tech_scores)}\n"
                f"Generated: {datetime.now().isoformat()}"
            ),
            "filters": {
                "platforms": ["Windows"],
            },
            "sorting": 3,  # Sort by score descending
            "layout": {
                "layout": "side",
                "aggregateFunction": "average",
                "showID": True,
                "showName": True,
                "showAggregateScores": True,
                "countUnscored": False,
            },
            "hideDisabled": False,
            "techniques": techniques_layer,
            "gradient": {
                "colors": ["#ffffff", "#ffe066", "#ff6b6b"],
                "minValue": 0,
                "maxValue": round(max_score, 2),
            },
            "legendItems": [],
            "metadata": [
                {"name": "source", "value": "CAPE N-gram SHAP CTI Pipeline"},
                {"name": "label", "value": label},
                {"name": "samples", "value": str(len(samples))},
            ],
            "links": [],
            "showTacticRowBackground": True,
            "tacticRowBackground": "#dddddd",
            "selectTechniquesAcrossTactics": True,
            "selectSubtechniquesWithParent": False,
            "selectVisibleTechniques": False,
        }

        layer_path = nav_dir / f"layer_{label}.json"
        with layer_path.open("w", encoding="utf-8") as f:
            json.dump(layer, f, indent=2, ensure_ascii=False)

    # Combined layer (all labels)
    all_tech_scores: Dict[str, float] = defaultdict(float)
    all_sample_counts: Dict[str, int] = defaultdict(int)
    total_samples = 0

    for label, samples in data.items():
        if filter_labels and label not in filter_labels:
            continue
        total_samples += len(samples)
        for sample, techniques in samples.items():
            seen: Set[str] = set()
            for tech in techniques:
                tid = tech["technique_id"]
                all_tech_scores[tid] += tech["score"]
                if tid not in seen:
                    all_sample_counts[tid] += 1
                    seen.add(tid)

    if all_tech_scores:
        max_score = max(all_tech_scores.values())
        combined_techniques = []
        for tid, score in sorted(all_tech_scores.items(), key=lambda x: x[1], reverse=True):
            combined_techniques.append({
                "techniqueID": tid,
                "tactic": "",
                "color": score_to_color(score, max_score),
                "comment": f"Score: {score:.3f}\nSamples: {all_sample_counts[tid]}\nName: {tech_names.get(tid, '')}",
                "enabled": True,
                "metadata": [],
                "links": [],
                "showSubtechniques": True,
                "score": round(score, 4),
            })

        combined_layer = {
            "name": "SHAP-CTI: All Labels",
            "versions": {"attack": "14", "navigator": "4.9.1", "layer": "4.5"},
            "domain": "enterprise-attack",
            "description": f"Combined ATT&CK technique heatmap across all labels.\nTotal samples: {total_samples}\nGenerated: {datetime.now().isoformat()}",
            "filters": {"platforms": ["Windows"]},
            "sorting": 3,
            "layout": {"layout": "side", "aggregateFunction": "average", "showID": True, "showName": True, "showAggregateScores": True, "countUnscored": False},
            "hideDisabled": False,
            "techniques": combined_techniques,
            "gradient": {"colors": ["#ffffff", "#ffe066", "#ff6b6b"], "minValue": 0, "maxValue": round(max_score, 2)},
            "legendItems": [],
            "metadata": [{"name": "source", "value": "CAPE N-gram SHAP CTI Pipeline"}],
            "links": [],
            "showTacticRowBackground": True,
            "tacticRowBackground": "#dddddd",
            "selectTechniquesAcrossTactics": True,
            "selectSubtechniquesWithParent": False,
            "selectVisibleTechniques": False,
        }

        combined_path = nav_dir / "layer_ALL.json"
        with combined_path.open("w", encoding="utf-8") as f:
            json.dump(combined_layer, f, indent=2, ensure_ascii=False)

    n_layers = len(list(nav_dir.glob("layer_*.json")))
    print(f"  [3] Navigator layers: {nav_dir} ({n_layers} layers)")
    print(f"      → https://mitre-attack.github.io/attack-navigator/ にアップロードして使用")


# ---------------------------------------------------------------------------
# 4. Label Threat Profiles
# ---------------------------------------------------------------------------

def export_threat_profiles(
    data: Dict[str, Dict[str, List[Dict]]],
    tactic_mapping: Dict[str, List[str]],
    tech_names: Dict[str, str],
    templates: Dict[str, Dict[str, str]],
    output_dir: Path,
    filter_labels: Optional[List[str]] = None,
) -> None:
    """Generate comparative threat profiles per label."""

    profiles: Dict[str, Dict[str, Any]] = {}

    for label, samples in sorted(data.items()):
        if filter_labels and label not in filter_labels:
            continue

        # Aggregate techniques
        tech_scores: Dict[str, float] = defaultdict(float)
        tech_counts: Dict[str, int] = defaultdict(int)
        tactic_scores: Dict[str, float] = defaultdict(float)
        categories: Dict[str, float] = defaultdict(float)

        for sample, techniques in samples.items():
            seen: Set[str] = set()
            for tech in techniques:
                tid = tech["technique_id"]
                tech_scores[tid] += tech["score"]
                if tid not in seen:
                    tech_counts[tid] += 1
                    seen.add(tid)
                for tactic in tactic_mapping.get(tid, tech.get("tactics", [])):
                    tactic_scores[tactic] += tech["score"]
                if tech.get("category"):
                    categories[tech["category"]] += tech["score"]

        # Top techniques
        top_techs = sorted(tech_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        # Dominant tactic
        dominant_tactic = max(tactic_scores.items(), key=lambda x: x[1])[0] if tactic_scores else "unknown"

        # Kill chain coverage
        covered_tactics = set()
        for tid in tech_scores:
            for tactic in tactic_mapping.get(tid, []):
                covered_tactics.add(tactic)
        kill_chain_coverage = len(covered_tactics) / len(TACTIC_ORDER)

        # Severity assessment
        high_severity = sum(1 for tid in tech_scores if templates.get(tid, {}).get("severity") == "high")
        medium_severity = sum(1 for tid in tech_scores if templates.get(tid, {}).get("severity") == "medium")

        # Normalize tactic shares
        total_tactic = sum(tactic_scores.values()) or 1
        tactic_shares = {t: v / total_tactic for t, v in tactic_scores.items()}

        profiles[label] = {
            "n_samples": len(samples),
            "n_unique_techniques": len(tech_scores),
            "kill_chain_coverage": round(kill_chain_coverage, 3),
            "dominant_tactic": dominant_tactic,
            "dominant_tactic_ja": TACTIC_DISPLAY_JA.get(dominant_tactic, dominant_tactic),
            "high_severity_techniques": high_severity,
            "medium_severity_techniques": medium_severity,
            "tactic_shares": {k: round(v, 4) for k, v in sorted(tactic_shares.items(), key=lambda x: x[1], reverse=True)},
            "category_shares": {k: round(v / (sum(categories.values()) or 1), 4) for k, v in sorted(categories.items(), key=lambda x: x[1], reverse=True)},
            "top_techniques": [
                {
                    "technique_id": tid,
                    "technique_name": tech_names.get(tid, ""),
                    "total_score": round(score, 4),
                    "sample_count": tech_counts.get(tid, 0),
                    "severity": templates.get(tid, {}).get("severity", "unknown"),
                }
                for tid, score in top_techs
            ],
        }

    # Export JSON
    json_path = output_dir / "threat_profiles.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    # Export comparison CSV
    csv_path = output_dir / "threat_profile_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "n_samples", "n_techniques", "kill_chain_coverage",
            "dominant_tactic", "dominant_tactic_ja",
            "high_severity", "medium_severity",
            "top_technique_1", "top_technique_2", "top_technique_3",
        ])
        for label, profile in sorted(profiles.items()):
            top3 = profile["top_techniques"][:3]
            top3_strs = [f"{t['technique_id']} {t['technique_name']}" for t in top3]
            while len(top3_strs) < 3:
                top3_strs.append("")
            writer.writerow([
                label, profile["n_samples"], profile["n_unique_techniques"],
                profile["kill_chain_coverage"],
                profile["dominant_tactic"], profile["dominant_tactic_ja"],
                profile["high_severity_techniques"], profile["medium_severity_techniques"],
                *top3_strs,
            ])

    # Print summary
    print(f"  [4] Threat profiles: {json_path} ({len(profiles)} labels)")
    print()
    print("  ┌──────────────────┬────────┬──────┬────────┬─────────────────┐")
    print("  │ Label            │Samples │Techs │Coverage│Dominant Tactic  │")
    print("  ├──────────────────┼────────┼──────┼────────┼─────────────────┤")
    for label, p in sorted(profiles.items(), key=lambda x: x[1]["kill_chain_coverage"], reverse=True):
        print(f"  │ {label:<16s} │{p['n_samples']:>6d}  │{p['n_unique_techniques']:>4d}  │{p['kill_chain_coverage']:>6.1%}  │{p['dominant_tactic_ja']:<16s} │")
    print("  └──────────────────┴────────┴──────┴────────┴─────────────────┘")


# ---------------------------------------------------------------------------
# 5. Technique Similarity Between Labels
# ---------------------------------------------------------------------------

def export_label_similarity(
    data: Dict[str, Dict[str, List[Dict]]],
    output_dir: Path,
    filter_labels: Optional[List[str]] = None,
) -> None:
    """Compute Jaccard similarity of technique sets between labels."""

    label_techs: Dict[str, Set[str]] = {}
    for label, samples in data.items():
        if filter_labels and label not in filter_labels:
            continue
        techs = set()
        for sample, techniques in samples.items():
            for tech in techniques:
                techs.add(tech["technique_id"])
        label_techs[label] = techs

    labels = sorted(label_techs.keys())
    if len(labels) < 2:
        print("  [5] Skipped similarity (< 2 labels)")
        return

    # Compute matrix
    n = len(labels)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                intersection = len(label_techs[labels[i]] & label_techs[labels[j]])
                union = len(label_techs[labels[i]] | label_techs[labels[j]])
                matrix[i, j] = intersection / union if union > 0 else 0

    # Export CSV
    csv_path = output_dir / "label_technique_similarity.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for i, label in enumerate(labels):
            writer.writerow([label] + [f"{matrix[i, j]:.3f}" for j in range(n)])

    print(f"  [5] Similarity: {csv_path} ({n}x{n} matrix)")


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
        return

    data = load_sample_techniques(args.cti_db)
    tech_names, tactic_mapping, tech_descs = load_attack_details(args.attack_db)
    templates = load_templates(args.templates)

    total_samples = sum(len(s) for s in data.values())
    print(f"[INFO] Labels: {len(data)}, Samples: {total_samples}")
    print(f"[INFO] ATT&CK techniques: {len(tech_names)}, Tactic mappings: {len(tactic_mapping)}")
    print(f"[INFO] Templates: {len(templates)}")
    print()

    # 1. Co-occurrence
    print("=== Technique Co-occurrence ===")
    pair_counts, tech_counts = compute_cooccurrence(data, filter_labels)
    export_cooccurrence(pair_counts, tech_counts, tech_names, args.output_dir)

    # 2. Attack chains
    print("\n=== Attack Chain Reconstruction ===")
    threat_summary = export_attack_chains(
        data, tactic_mapping, tech_names, templates,
        args.output_dir, args.top_samples, filter_labels,
    )

    # 3. Navigator layers
    print("\n=== ATT&CK Navigator Export ===")
    export_navigator_layers(data, tech_names, args.output_dir, filter_labels)

    # 4. Threat profiles
    print("\n=== Threat Profiles ===")
    export_threat_profiles(data, tactic_mapping, tech_names, templates, args.output_dir, filter_labels)

    # 5. Label similarity
    print("\n=== Label Similarity ===")
    export_label_similarity(data, args.output_dir, filter_labels)

    print(f"\n[INFO] All outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
