#!/usr/bin/env python3
"""Query per-sample SHAP and CTI results from SQLite databases.

Examples:
  # Search by sample name
  python3 src/query_shap_db.py --sample "abc123.json"

  # Search by label
  python3 src/query_shap_db.py --label "Trojan" --limit 10

  # Search by technique
  python3 src/query_shap_db.py --technique "T1055" --limit 20

  # Export results as JSON
  python3 src/query_shap_db.py --sample "abc123.json" --format json
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query SHAP and CTI results from SQLite databases."
    )
    parser.add_argument(
        "--cti-db",
        type=Path,
        default=Path("shap_analysis/ngram_desc_lgbm_group_nometa_106/cti/cti_results.sqlite"),
    )
    parser.add_argument(
        "--shap-db",
        type=Path,
        default=None,
        help="Optional: separate SHAP per-sample SQLite DB.",
    )

    # Query filters
    parser.add_argument("--sample", type=str, default=None, help="Sample name (partial match)")
    parser.add_argument("--label", type=str, default=None, help="Label name")
    parser.add_argument("--technique", type=str, default=None, help="ATT&CK technique ID")
    parser.add_argument("--category", type=str, default=None, help="Attack category")
    parser.add_argument("--min-score", type=float, default=None, help="Minimum technique score")

    # Output
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table")
    parser.add_argument("--output", type=Path, default=None, help="Output file (default: stdout)")

    # Query type
    parser.add_argument(
        "--query",
        choices=["summary", "techniques", "features", "stats"],
        default="summary",
        help="What to query: summary (per-sample overview), techniques (technique details), "
             "features (SHAP features), stats (aggregate statistics).",
    )

    return parser.parse_args()


def query_summary(
    conn: sqlite3.Connection,
    sample: Optional[str],
    label: Optional[str],
    limit: int,
) -> List[Dict[str, str]]:
    """Query sample_summary table."""
    conditions = []
    params = []

    if sample:
        conditions.append("sample LIKE ?")
        params.append(f"%{sample}%")
    if label:
        conditions.append("label = ?")
        params.append(label)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"SELECT * FROM sample_summary {where} LIMIT ?"
    params.append(limit)

    cursor = conn.execute(sql, params)
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def query_techniques(
    conn: sqlite3.Connection,
    sample: Optional[str],
    label: Optional[str],
    technique: Optional[str],
    category: Optional[str],
    min_score: Optional[float],
    limit: int,
) -> List[Dict[str, object]]:
    """Query sample_techniques table."""
    conditions = []
    params = []

    if sample:
        conditions.append("sample LIKE ?")
        params.append(f"%{sample}%")
    if label:
        conditions.append("label = ?")
        params.append(label)
    if technique:
        conditions.append("technique_id LIKE ?")
        params.append(f"%{technique}%")
    if category:
        conditions.append("category = ?")
        params.append(category)
    if min_score is not None:
        conditions.append("score >= ?")
        params.append(min_score)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"SELECT * FROM sample_techniques {where} ORDER BY score DESC LIMIT ?"
    params.append(limit)

    try:
        cursor = conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        print("[WARN] sample_techniques table not found. Use updated cti_attach_shap_explanations.py.")
        return []


def query_features(
    conn: sqlite3.Connection,
    sample: Optional[str],
    label: Optional[str],
    limit: int,
) -> List[Dict[str, object]]:
    """Query sample_features table."""
    conditions = []
    params = []

    if sample:
        conditions.append("sample LIKE ?")
        params.append(f"%{sample}%")
    if label:
        conditions.append("label = ?")
        params.append(label)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"SELECT * FROM sample_features {where} ORDER BY sample, rank LIMIT ?"
    params.append(limit)

    try:
        cursor = conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        print("[WARN] sample_features table not found.")
        return []


def query_stats(conn: sqlite3.Connection) -> List[Dict[str, object]]:
    """Compute aggregate statistics from the database."""
    stats = []

    # Table counts
    for table in ["sample_summary", "sample_techniques", "sample_features"]:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats.append({"metric": f"rows_in_{table}", "value": count})
        except sqlite3.OperationalError:
            stats.append({"metric": f"rows_in_{table}", "value": "table_missing"})

    # Unique samples
    try:
        count = conn.execute("SELECT COUNT(DISTINCT sample) FROM sample_summary").fetchone()[0]
        stats.append({"metric": "unique_samples", "value": count})
    except sqlite3.OperationalError:
        pass

    # Unique labels
    try:
        count = conn.execute("SELECT COUNT(DISTINCT label) FROM sample_summary").fetchone()[0]
        stats.append({"metric": "unique_labels", "value": count})
    except sqlite3.OperationalError:
        pass

    # Unique techniques
    try:
        count = conn.execute("SELECT COUNT(DISTINCT technique_id) FROM sample_techniques").fetchone()[0]
        stats.append({"metric": "unique_techniques", "value": count})
    except sqlite3.OperationalError:
        pass

    # Top techniques by total score
    try:
        rows = conn.execute(
            "SELECT technique_id, technique_name, SUM(score) as total, COUNT(DISTINCT sample) as samples "
            "FROM sample_techniques GROUP BY technique_id ORDER BY total DESC LIMIT 10"
        ).fetchall()
        for tid, name, total, samples in rows:
            stats.append({
                "metric": f"top_technique",
                "value": f"{tid} {name} (score={total:.2f}, samples={samples})",
            })
    except sqlite3.OperationalError:
        pass

    return stats


def format_table(rows: List[Dict], max_col_width: int = 60) -> str:
    """Format rows as an ASCII table."""
    if not rows:
        return "(no results)"

    columns = list(rows[0].keys())
    col_widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            val = str(row.get(col, ""))
            if len(val) > max_col_width:
                val = val[:max_col_width - 3] + "..."
            col_widths[col] = max(col_widths[col], len(val))

    header = " | ".join(col.ljust(col_widths[col]) for col in columns)
    separator = "-+-".join("-" * col_widths[col] for col in columns)

    lines = [header, separator]
    for row in rows:
        cells = []
        for col in columns:
            val = str(row.get(col, ""))
            if len(val) > max_col_width:
                val = val[:max_col_width - 3] + "..."
            cells.append(val.ljust(col_widths[col]))
        lines.append(" | ".join(cells))

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    if not args.cti_db.exists():
        print(f"[ERROR] Database not found: {args.cti_db}")
        sys.exit(1)

    conn = sqlite3.connect(str(args.cti_db))

    # Run query
    if args.query == "summary":
        results = query_summary(conn, args.sample, args.label, args.limit)
    elif args.query == "techniques":
        results = query_techniques(
            conn, args.sample, args.label, args.technique,
            args.category, args.min_score, args.limit,
        )
    elif args.query == "features":
        results = query_features(conn, args.sample, args.label, args.limit)
    elif args.query == "stats":
        results = query_stats(conn)
    else:
        results = []

    conn.close()

    # Format output
    if args.format == "json":
        output = json.dumps(results, indent=2, ensure_ascii=False)
    elif args.format == "csv":
        if results:
            import io
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
            output = buf.getvalue()
        else:
            output = ""
    else:
        output = format_table(results)

    # Write output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"[INFO] Output written to {args.output}")
    else:
        print(output)
        print(f"\n({len(results)} rows)")


if __name__ == "__main__":
    main()
