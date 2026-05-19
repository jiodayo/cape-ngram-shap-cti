#!/usr/bin/env python3
"""Build a local SQLite DB from MITRE ATT&CK Enterprise STIX data."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ATT&CK SQLite DB from STIX JSON.")
    parser.add_argument(
        "--attack-json",
        type=Path,
        default=Path("reference/mitre/enterprise-attack.json"),
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=Path("reference/mitre/attack.sqlite"),
    )
    return parser.parse_args()


def extract_technique_id(obj: Dict[str, object]) -> Optional[str]:
    refs = obj.get("external_references", [])
    if not isinstance(refs, list):
        return None
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
            return str(ref["external_id"])
    return None


def extract_tactics(obj: Dict[str, object]) -> List[str]:
    phases = obj.get("kill_chain_phases", [])
    tactics: List[str] = []
    if not isinstance(phases, list):
        return tactics
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        if phase.get("kill_chain_name") != "mitre-attack":
            continue
        name = phase.get("phase_name")
        if isinstance(name, str) and name:
            tactics.append(name)
    return sorted(set(tactics))


def load_stix(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    objects = data.get("objects", []) if isinstance(data, dict) else []
    return [obj for obj in objects if isinstance(obj, dict)]


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS techniques (
            technique_id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS technique_tactics (
            technique_id TEXT,
            tactic TEXT,
            PRIMARY KEY (technique_id, tactic)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )


def main() -> None:
    args = parse_args()

    objects = load_stix(args.attack_json)
    conn = sqlite3.connect(str(args.output_db))
    init_db(conn)

    techniques: List[Tuple[str, str, str]] = []
    technique_tactics: List[Tuple[str, str]] = []

    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        technique_id = extract_technique_id(obj)
        if not technique_id:
            continue
        name = obj.get("name") or ""
        desc = obj.get("description") or ""
        techniques.append((technique_id, str(name), str(desc)))
        for tactic in extract_tactics(obj):
            technique_tactics.append((technique_id, tactic))

    conn.executemany(
        "INSERT OR REPLACE INTO techniques (technique_id, name, description) VALUES (?, ?, ?)",
        techniques,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO technique_tactics (technique_id, tactic) VALUES (?, ?)",
        technique_tactics,
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("source", str(args.attack_json)),
    )
    conn.commit()
    conn.close()

    print(f"[INFO] techniques: {len(techniques)}")
    print(f"[INFO] output db: {args.output_db}")


if __name__ == "__main__":
    main()
