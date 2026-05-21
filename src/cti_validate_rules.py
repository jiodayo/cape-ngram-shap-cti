#!/usr/bin/env python3
"""Validate api_to_attack_rules.csv against an API list.

Reports:
- Which rules match which APIs (and how many)
- APIs with no rule coverage
- Potential false-positive patterns (rules matching too many APIs)
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

from common import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ATT&CK mapping rules against API list."
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path("reference/mitre/api_to_attack_rules.csv"),
    )
    parser.add_argument(
        "--api-list",
        type=Path,
        default=Path("data/api.json"),
        help="JSON file containing the list of known API names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV with per-rule match details.",
    )
    parser.add_argument(
        "--max-match-warn",
        type=int,
        default=20,
        help="Warn if a single rule matches more than this many APIs.",
    )
    return parser.parse_args()


def load_rules(path: Path) -> List[Dict[str, str]]:
    rules: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pattern = (row.get("pattern") or "").strip()
            if not pattern or pattern.startswith("#"):
                continue
            technique_id = (row.get("technique_id") or "").strip()
            if not technique_id:
                continue
            rules.append(row)
    return rules


def main() -> None:
    args = parse_args()

    rules = load_rules(args.rules)
    print(f"[INFO] Loaded {len(rules)} rules from {args.rules}")

    # Load API list
    if args.api_list.exists():
        api_list: List[str] = load_json(args.api_list)
        print(f"[INFO] Loaded {len(api_list)} APIs from {args.api_list}")
    else:
        print(f"[WARN] API list not found: {args.api_list}")
        print("       Running in rules-only mode (no coverage analysis).")
        api_list = []

    # Validate each rule
    results: List[Dict[str, object]] = []
    all_matched_apis: set = set()
    invalid_patterns: List[str] = []

    for rule in rules:
        pattern = rule["pattern"]
        technique_id = rule.get("technique_id", "")
        confidence = rule.get("confidence", "")
        confidence_score = rule.get("confidence_score", "")
        category = rule.get("category", "")

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            invalid_patterns.append(f"{pattern}: {e}")
            results.append({
                "pattern": pattern,
                "technique_id": technique_id,
                "valid": False,
                "error": str(e),
                "matched_apis": [],
                "match_count": 0,
            })
            continue

        matched = [api for api in api_list if regex.search(api)]
        all_matched_apis.update(matched)

        results.append({
            "pattern": pattern,
            "technique_id": technique_id,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "category": category,
            "valid": True,
            "matched_apis": matched,
            "match_count": len(matched),
        })

    # Summary
    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal rules:       {len(rules)}")
    print(f"Valid patterns:    {sum(1 for r in results if r['valid'])}")
    print(f"Invalid patterns:  {len(invalid_patterns)}")

    if invalid_patterns:
        print("\n[ERROR] Invalid regex patterns:")
        for msg in invalid_patterns:
            print(f"  - {msg}")

    if api_list:
        covered = len(all_matched_apis)
        total = len(api_list)
        uncovered = total - covered
        print(f"\nAPI coverage:      {covered}/{total} ({100*covered/total:.1f}%)")
        print(f"Uncovered APIs:    {uncovered}")

        # Unique techniques
        techniques = set()
        for r in results:
            if r["valid"]:
                techniques.add(r["technique_id"])
        print(f"Unique techniques: {len(techniques)}")

        # Unique categories
        categories = set()
        for r in results:
            if r["valid"] and r.get("category"):
                categories.add(r["category"])
        print(f"Categories:        {', '.join(sorted(categories))}")

    # Warnings
    print()
    warnings = 0
    for r in results:
        if r["valid"] and r["match_count"] > args.max_match_warn:
            print(f"[WARN] Rule '{r['pattern']}' matches {r['match_count']} APIs (threshold: {args.max_match_warn})")
            warnings += 1

    zero_match = [r for r in results if r["valid"] and r["match_count"] == 0]
    if zero_match:
        print(f"\n[WARN] {len(zero_match)} rules match zero APIs in the list:")
        for r in zero_match:
            print(f"  - {r['technique_id']}: {r['pattern']}")
        warnings += len(zero_match)

    if warnings == 0:
        print("[OK] No warnings.")

    # Top matched APIs
    if api_list:
        uncovered_apis = sorted(set(api_list) - all_matched_apis)
        if uncovered_apis:
            print(f"\n--- Uncovered APIs (first 30) ---")
            for api in uncovered_apis[:30]:
                print(f"  {api}")
            if len(uncovered_apis) > 30:
                print(f"  ... and {len(uncovered_apis) - 30} more")

    # Write detailed output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "pattern", "technique_id", "confidence", "confidence_score",
                "category", "valid", "match_count", "matched_apis",
            ])
            for r in results:
                writer.writerow([
                    r["pattern"],
                    r["technique_id"],
                    r.get("confidence", ""),
                    r.get("confidence_score", ""),
                    r.get("category", ""),
                    r["valid"],
                    r["match_count"],
                    "|".join(r["matched_apis"]),
                ])
        print(f"\n[INFO] Detailed output: {args.output}")

    print()


if __name__ == "__main__":
    main()
