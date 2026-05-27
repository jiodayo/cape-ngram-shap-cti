#!/usr/bin/env python3
"""Experiment leaderboard: register, compare, and rank experiment runs.

Tracks model performance + explanation quality across experiments in a
central SQLite database. Generates HTML leaderboard with sortable tables,
sparkline comparisons, and per-label drill-down.

Usage:
  # Register a new experiment
  python3 src/leaderboard.py register \
      --name "lgbm_seed42_skip_conf03" \
      --metrics-json logs/.../lgbm/metrics_overall.json \
      --per-label-csv logs/.../lgbm/per_label_metrics.csv \
      --eval-json evaluation/explanation_quality.json \
      --config '{"seed":42, "model":"lgbm", "use_skipgram":true, "min_confidence":0.3}'

  # Show leaderboard (terminal)
  python3 src/leaderboard.py show

  # Generate HTML leaderboard
  python3 src/leaderboard.py html --output reports/leaderboard.html

  # Compare two experiments
  python3 src/leaderboard.py compare --names "exp_A" "exp_B"

  # Delete an experiment
  python3 src/leaderboard.py delete --name "old_experiment"
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_DB = Path("leaderboard.sqlite")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            name TEXT PRIMARY KEY,
            timestamp TEXT,
            config TEXT,
            micro_f1 REAL,
            macro_f1 REAL,
            samples_f1 REAL,
            num_labels INTEGER,
            num_test_samples INTEGER,
            avg_rule_match_rate REAL,
            avg_technique_diversity REAL,
            avg_technique_concentration REAL,
            avg_group_entropy REAL,
            avg_explanation_coverage REAL,
            notes TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS per_label_metrics (
            experiment TEXT,
            label TEXT,
            precision REAL,
            recall REAL,
            f1 REAL,
            accuracy REAL,
            train_positive INTEGER,
            test_positive INTEGER,
            rule_match_rate REAL,
            technique_diversity REAL,
            explanation_coverage REAL,
            PRIMARY KEY (experiment, label),
            FOREIGN KEY (experiment) REFERENCES experiments(name)
        )
    """)
    conn.commit()


def get_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

def cmd_register(args: argparse.Namespace) -> None:
    conn = get_db(args.db)

    # Load model metrics
    metrics = {}
    if args.metrics_json and Path(args.metrics_json).exists():
        with open(args.metrics_json, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    # Load explanation quality
    eval_data: Dict[str, Dict[str, float]] = {}
    if args.eval_json and Path(args.eval_json).exists():
        with open(args.eval_json, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

    # Compute averages for explanation metrics
    def avg_metric(key: str) -> Optional[float]:
        values = [v[key] for v in eval_data.values() if key in v and isinstance(v[key], (int, float))]
        return sum(values) / len(values) if values else None

    # Load config
    config = {}
    if args.config:
        try:
            config = json.loads(args.config)
        except json.JSONDecodeError:
            config = {"raw": args.config}

    # Feature summary
    if args.feature_summary and Path(args.feature_summary).exists():
        with open(args.feature_summary, "r", encoding="utf-8") as f:
            feat_summary = json.load(f)
        config["feature_summary"] = feat_summary

    # CTI metadata
    if args.cti_metadata and Path(args.cti_metadata).exists():
        with open(args.cti_metadata, "r", encoding="utf-8") as f:
            cti_meta = json.load(f)
        config["cti"] = cti_meta

    # Insert experiment
    conn.execute("""
        INSERT OR REPLACE INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        args.name,
        datetime.now().isoformat(),
        json.dumps(config, ensure_ascii=False),
        metrics.get("micro_f1"),
        metrics.get("macro_f1"),
        metrics.get("samples_f1"),
        metrics.get("num_labels"),
        metrics.get("num_test_samples"),
        avg_metric("rule_match_rate"),
        avg_metric("technique_diversity"),
        avg_metric("technique_concentration"),
        avg_metric("group_entropy"),
        avg_metric("explanation_coverage"),
        args.notes or "",
    ))

    # Load per-label metrics
    if args.per_label_csv and Path(args.per_label_csv).exists():
        with open(args.per_label_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("label", "")
                if not label:
                    continue

                # Get eval data for this label
                label_eval = eval_data.get(label, {})

                conn.execute("""
                    INSERT OR REPLACE INTO per_label_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    args.name,
                    label,
                    float(row.get("precision", 0)),
                    float(row.get("recall", 0)),
                    float(row.get("f1", 0)),
                    float(row.get("accuracy", 0)),
                    int(row.get("train_positive", 0)),
                    int(row.get("test_positive", 0)),
                    label_eval.get("rule_match_rate"),
                    label_eval.get("technique_diversity"),
                    label_eval.get("explanation_coverage"),
                ))

    conn.commit()
    conn.close()
    print(f"[OK] Registered: {args.name}")


# ---------------------------------------------------------------------------
# Show (Terminal)
# ---------------------------------------------------------------------------

def cmd_show(args: argparse.Namespace) -> None:
    conn = get_db(args.db)

    sort_col = args.sort or "macro_f1"
    order = "DESC" if not args.asc else "ASC"

    rows = conn.execute(f"""
        SELECT name, timestamp, micro_f1, macro_f1, samples_f1,
               num_labels, num_test_samples,
               avg_rule_match_rate, avg_technique_diversity,
               avg_explanation_coverage, notes
        FROM experiments
        ORDER BY {sort_col} {order}
    """).fetchall()

    if not rows:
        print("(no experiments registered)")
        return

    # Print table
    header = f"{'#':>3} {'Name':<30} {'Macro F1':>9} {'Micro F1':>9} {'Samples F1':>11} {'RuleMatch':>10} {'TechDiv':>8} {'ExplCov':>8} {'Labels':>7}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(rows, 1):
        def fmt(v, d=4):
            return f"{v:.{d}f}" if v is not None else "—"

        print(
            f"{i:>3} {r['name']:<30} "
            f"{fmt(r['macro_f1']):>9} {fmt(r['micro_f1']):>9} {fmt(r['samples_f1']):>11} "
            f"{fmt(r['avg_rule_match_rate']):>10} {fmt(r['avg_technique_diversity'], 1):>8} "
            f"{fmt(r['avg_explanation_coverage']):>8} {r['num_labels'] or '—':>7}"
        )

    print(f"\n({len(rows)} experiments)")
    conn.close()


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> None:
    conn = get_db(args.db)

    names = args.names
    experiments = []
    for name in names:
        row = conn.execute("SELECT * FROM experiments WHERE name = ?", (name,)).fetchone()
        if not row:
            print(f"[WARN] Experiment not found: {name}")
            continue
        experiments.append(dict(row))

    if len(experiments) < 2:
        print("[ERROR] Need at least 2 experiments to compare")
        return

    # Overall comparison
    metrics = ["micro_f1", "macro_f1", "samples_f1",
               "avg_rule_match_rate", "avg_technique_diversity",
               "avg_technique_concentration", "avg_group_entropy",
               "avg_explanation_coverage"]

    metric_display = {
        "micro_f1": "Micro F1",
        "macro_f1": "Macro F1",
        "samples_f1": "Samples F1",
        "avg_rule_match_rate": "Rule Match Rate",
        "avg_technique_diversity": "Tech Diversity",
        "avg_technique_concentration": "Tech Concentration",
        "avg_group_entropy": "Group Entropy",
        "avg_explanation_coverage": "Explanation Coverage",
    }

    # Header
    header = f"{'Metric':<25}"
    for exp in experiments:
        header += f" {exp['name']:<20}"
    header += "  Δ"
    print(header)
    print("-" * len(header))

    for m in metrics:
        line = f"{metric_display.get(m, m):<25}"
        values = []
        for exp in experiments:
            v = exp.get(m)
            if v is not None:
                line += f" {v:<20.4f}"
                values.append(v)
            else:
                line += f" {'—':<20}"

        if len(values) == 2:
            delta = values[1] - values[0]
            sign = "+" if delta > 0 else ""
            color = "\033[32m" if delta > 0 else "\033[31m" if delta < 0 else ""
            reset = "\033[0m"
            line += f"  {color}{sign}{delta:.4f}{reset}"

        print(line)

    # Per-label comparison
    print(f"\n--- Per-Label F1 Comparison ---")
    for name in names:
        label_rows = conn.execute(
            "SELECT label, f1, accuracy FROM per_label_metrics WHERE experiment = ? ORDER BY label",
            (name,),
        ).fetchall()
        if label_rows:
            print(f"\n  [{name}]")
            for r in label_rows:
                print(f"    {r['label']:<20} F1={r['f1']:.4f}  Acc={r['accuracy']:.4f}")

    conn.close()


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def cmd_delete(args: argparse.Namespace) -> None:
    conn = get_db(args.db)
    conn.execute("DELETE FROM per_label_metrics WHERE experiment = ?", (args.name,))
    conn.execute("DELETE FROM experiments WHERE name = ?", (args.name,))
    conn.commit()
    conn.close()
    print(f"[OK] Deleted: {args.name}")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def cmd_html(args: argparse.Namespace) -> None:
    conn = get_db(args.db)

    experiments = conn.execute("""
        SELECT * FROM experiments ORDER BY macro_f1 DESC
    """).fetchall()

    if not experiments:
        print("[WARN] No experiments to display")
        return

    # Per-label data
    per_label: Dict[str, List[Dict]] = {}
    for exp in experiments:
        rows = conn.execute(
            "SELECT * FROM per_label_metrics WHERE experiment = ? ORDER BY label",
            (exp["name"],),
        ).fetchall()
        per_label[exp["name"]] = [dict(r) for r in rows]

    conn.close()

    # Build HTML
    html = _build_leaderboard_html(
        [dict(e) for e in experiments],
        per_label,
        args.title,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    print(f"[OK] Leaderboard: {output} ({output.stat().st_size / 1024:.1f} KB)")


def _build_leaderboard_html(
    experiments: List[Dict],
    per_label: Dict[str, List[Dict]],
    title: str,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Summary cards
    best_macro = max((e for e in experiments if e.get("macro_f1")), key=lambda e: e["macro_f1"], default=None)
    best_rule = max((e for e in experiments if e.get("avg_rule_match_rate")), key=lambda e: e["avg_rule_match_rate"], default=None)

    cards = ""
    cards += f'<div class="card"><div class="card-label">Experiments</div><div class="card-value">{len(experiments)}</div></div>'
    if best_macro:
        cards += f'<div class="card"><div class="card-label">Best Macro F1</div><div class="card-value best">{best_macro["macro_f1"]:.4f}</div><div class="card-sub">{escape(best_macro["name"])}</div></div>'
    if best_rule:
        cards += f'<div class="card"><div class="card-label">Best Rule Match</div><div class="card-value">{best_rule["avg_rule_match_rate"]:.4f}</div><div class="card-sub">{escape(best_rule["name"])}</div></div>'

    # Main table
    def cell(v, fmt=".4f", best_val=None):
        if v is None:
            return '<td class="na">—</td>'
        s = f"{v:{fmt}}"
        cls = ' class="best"' if best_val is not None and abs(v - best_val) < 1e-8 else ""
        return f"<td{cls}>{s}</td>"

    # Find best values for highlighting
    def best(key):
        vals = [e[key] for e in experiments if e.get(key) is not None]
        return max(vals) if vals else None

    b_micro = best("micro_f1")
    b_macro = best("macro_f1")
    b_samples = best("samples_f1")
    b_rule = best("avg_rule_match_rate")
    b_div = best("avg_technique_diversity")
    b_cov = best("avg_explanation_coverage")

    table_rows = ""
    for i, e in enumerate(experiments, 1):
        # Config tooltip
        config = json.loads(e.get("config") or "{}")
        config_str = escape(json.dumps(config, indent=2, ensure_ascii=False))

        medal = ""
        if i == 1:
            medal = "🥇"
        elif i == 2:
            medal = "🥈"
        elif i == 3:
            medal = "🥉"

        table_rows += f"""<tr>
            <td class="rank">{medal} {i}</td>
            <td class="name" title="{config_str}">{escape(e['name'])}</td>
            {cell(e.get('macro_f1'), best_val=b_macro)}
            {cell(e.get('micro_f1'), best_val=b_micro)}
            {cell(e.get('samples_f1'), best_val=b_samples)}
            {cell(e.get('avg_rule_match_rate'), best_val=b_rule)}
            {cell(e.get('avg_technique_diversity'), '.1f', best_val=b_div)}
            {cell(e.get('avg_explanation_coverage'), best_val=b_cov)}
            <td>{e.get('num_labels') or '—'}</td>
            <td>{e.get('num_test_samples') or '—'}</td>
            <td class="ts">{(e.get('timestamp') or '')[:16]}</td>
        </tr>"""

    # Per-label detail sections
    detail_sections = ""
    for e in experiments:
        name = e["name"]
        labels = per_label.get(name, [])
        if not labels:
            continue

        label_rows = ""
        for l in labels:
            label_rows += f"""<tr>
                <td>{escape(l.get('label', ''))}</td>
                <td>{l.get('f1', 0):.4f}</td>
                <td>{l.get('precision', 0):.4f}</td>
                <td>{l.get('recall', 0):.4f}</td>
                <td>{l.get('accuracy', 0):.4f}</td>
                <td>{l.get('rule_match_rate', 0):.4f if l.get('rule_match_rate') is not None else '—'}</td>
                <td>{l.get('train_positive', 0)}</td>
                <td>{l.get('test_positive', 0)}</td>
            </tr>"""

        detail_sections += f"""
        <details class="exp-detail">
            <summary>{escape(name)} — Per-Label Metrics</summary>
            <div class="detail-body">
                <table class="detail-table">
                    <thead><tr>
                        <th>Label</th><th>F1</th><th>Precision</th><th>Recall</th>
                        <th>Accuracy</th><th>Rule Match</th><th>Train +</th><th>Test +</th>
                    </tr></thead>
                    <tbody>{label_rows}</tbody>
                </table>
            </div>
        </details>"""

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escape(title)}</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --surface2: #1c2333;
    --border: #30363d; --text: #e6edf3; --text2: #8b949e;
    --gold: #ffd700; --accent: #58a6ff; --green: #3fb950; --red: #f85149;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter','Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--text); }}
  .container {{ max-width:1500px; margin:0 auto; padding:24px; }}

  header {{ text-align:center; padding:40px 0 30px; }}
  header h1 {{ font-size:2rem; background:linear-gradient(135deg,var(--gold),var(--accent)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  header .sub {{ color:var(--text2); margin-top:8px; }}

  .cards {{ display:flex; gap:16px; justify-content:center; margin:24px 0; flex-wrap:wrap; }}
  .card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:20px 28px; min-width:200px; text-align:center; }}
  .card-label {{ color:var(--text2); font-size:.8rem; text-transform:uppercase; letter-spacing:.5px; }}
  .card-value {{ font-size:2rem; font-weight:700; margin-top:4px; color:var(--accent); }}
  .card-value.best {{ color:var(--gold); }}
  .card-sub {{ color:var(--text2); font-size:.75rem; margin-top:2px; }}

  .board {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow-x:auto; margin:24px 0; }}
  table {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
  th {{ background:var(--surface2); color:var(--text2); font-weight:600; font-size:.75rem; letter-spacing:.5px;
       padding:12px 14px; text-align:left; cursor:pointer; user-select:none; white-space:nowrap;
       position:sticky; top:0; }}
  th:hover {{ color:var(--accent); }}
  td {{ padding:10px 14px; border-top:1px solid var(--border); }}
  tr:hover td {{ background:rgba(88,166,255,0.06); }}
  .rank {{ font-weight:700; text-align:center; width:60px; }}
  .name {{ font-weight:600; color:var(--accent); max-width:250px; overflow:hidden; text-overflow:ellipsis; cursor:help; }}
  .best {{ color:var(--gold) !important; font-weight:700; }}
  .na {{ color:var(--text2); }}
  .ts {{ color:var(--text2); font-size:.75rem; }}

  details.exp-detail {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; margin:8px 0; }}
  details.exp-detail summary {{ padding:12px 16px; cursor:pointer; font-weight:500; }}
  details.exp-detail[open] summary {{ border-bottom:1px solid var(--border); }}
  .detail-body {{ padding:16px; overflow-x:auto; }}
  .detail-table {{ font-size:.82rem; }}
  .detail-table th {{ background:var(--bg); }}

  h2 {{ font-size:1.2rem; margin:32px 0 12px; color:var(--text2); }}

  footer {{ text-align:center; padding:24px; color:var(--text2); font-size:.8rem; border-top:1px solid var(--border); margin-top:40px; }}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>🏆 {escape(title)}</h1>
  <div class="sub">Generated: {now}</div>
</header>

<div class="cards">{cards}</div>

<div class="board">
  <table id="lb-table">
    <thead><tr>
      <th>#</th>
      <th>Experiment</th>
      <th onclick="sortTable(2)">Macro F1 ⇅</th>
      <th onclick="sortTable(3)">Micro F1 ⇅</th>
      <th onclick="sortTable(4)">Samples F1 ⇅</th>
      <th onclick="sortTable(5)">Rule Match ⇅</th>
      <th onclick="sortTable(6)">Tech Diversity ⇅</th>
      <th onclick="sortTable(7)">Expl Coverage ⇅</th>
      <th>Labels</th>
      <th>Test Samples</th>
      <th>Timestamp</th>
    </tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>

<h2>📋 Per-Label Details</h2>
{detail_sections if detail_sections else '<p style="color:var(--text2)">Per-label data not registered.</p>'}

</div>

<footer>Experiment Leaderboard — CAPE N-gram SHAP CTI Pipeline</footer>

<script>
function sortTable(col) {{
  const table = document.getElementById('lb-table');
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.rows);
  const asc = table.dataset.sortCol == col ? !JSON.parse(table.dataset.sortAsc || 'false') : false;
  table.dataset.sortCol = col;
  table.dataset.sortAsc = asc;
  rows.sort((a, b) => {{
    let va = a.cells[col]?.textContent.trim() || '';
    let vb = b.cells[col]?.textContent.trim() || '';
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}
</script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# List labels
# ---------------------------------------------------------------------------

def cmd_labels(args: argparse.Namespace) -> None:
    conn = get_db(args.db)
    name = args.name

    rows = conn.execute(
        "SELECT * FROM per_label_metrics WHERE experiment = ? ORDER BY f1 DESC",
        (name,),
    ).fetchall()

    if not rows:
        print(f"No per-label data for: {name}")
        return

    print(f"{'Label':<24} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Acc':>8} {'RuleMatch':>10}")
    print("-" * 76)
    for r in rows:
        rm = f"{r['rule_match_rate']:.4f}" if r['rule_match_rate'] is not None else "—"
        print(f"{r['label']:<24} {r['f1']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['accuracy']:>8.4f} {rm:>10}")

    conn.close()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def cmd_export(args: argparse.Namespace) -> None:
    conn = get_db(args.db)

    experiments = conn.execute("SELECT * FROM experiments ORDER BY macro_f1 DESC").fetchall()
    per_label: Dict[str, List[Dict]] = {}
    for exp in experiments:
        rows = conn.execute(
            "SELECT * FROM per_label_metrics WHERE experiment = ?",
            (exp["name"],),
        ).fetchall()
        per_label[exp["name"]] = [dict(r) for r in rows]

    output = {
        "experiments": [dict(e) for e in experiments],
        "per_label": per_label,
        "exported_at": datetime.now().isoformat(),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[OK] Exported: {out_path}")
    conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment leaderboard for SHAP/CTI analysis."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Leaderboard SQLite DB path.")

    sub = parser.add_subparsers(dest="command")

    # register
    p_reg = sub.add_parser("register", help="Register a new experiment.")
    p_reg.add_argument("--name", required=True, help="Unique experiment name.")
    p_reg.add_argument("--metrics-json", type=str, default=None)
    p_reg.add_argument("--per-label-csv", type=str, default=None)
    p_reg.add_argument("--eval-json", type=str, default=None)
    p_reg.add_argument("--feature-summary", type=str, default=None, help="Path to feature_summary.json")
    p_reg.add_argument("--cti-metadata", type=str, default=None, help="Path to cti_metadata.json")
    p_reg.add_argument("--config", type=str, default=None, help="JSON string or key=value config")
    p_reg.add_argument("--notes", type=str, default=None)

    # show
    p_show = sub.add_parser("show", help="Show leaderboard in terminal.")
    p_show.add_argument("--sort", type=str, default="macro_f1")
    p_show.add_argument("--asc", action="store_true")

    # html
    p_html = sub.add_parser("html", help="Generate HTML leaderboard.")
    p_html.add_argument("--output", type=str, default="reports/leaderboard.html")
    p_html.add_argument("--title", type=str, default="Experiment Leaderboard")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare experiments side-by-side.")
    p_cmp.add_argument("--names", nargs="+", required=True)

    # labels
    p_lbl = sub.add_parser("labels", help="Show per-label metrics for an experiment.")
    p_lbl.add_argument("--name", required=True)

    # delete
    p_del = sub.add_parser("delete", help="Delete an experiment.")
    p_del.add_argument("--name", required=True)

    # export
    p_exp = sub.add_parser("export", help="Export all data as JSON.")
    p_exp.add_argument("--output", type=str, default="reports/leaderboard_export.json")

    args = parser.parse_args()

    if args.command == "register":
        cmd_register(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "html":
        cmd_html(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "labels":
        cmd_labels(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
