#!/usr/bin/env python3
"""Generate interactive HTML reports from SHAP analysis and CTI results.

Produces a self-contained HTML file with:
  - SHAP top features table (sortable/searchable)
  - Feature group contribution charts (inline SVG)
  - CTI technique mapping summary
  - Per-sample SHAP detail (expandable)
  - Explanation quality metrics

Usage:
  python3 src/generate_html_report.py \
      --group-csv shap_analysis/.../shap_group_contributions.csv \
      --top-csv shap_analysis/.../shap_top_features.csv \
      --per-sample-dir shap_analysis/.../per_sample \
      --cti-dir shap_analysis/.../cti \
      --eval-json evaluation/explanation_quality.json \
      --output reports/shap_report.html
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML report from SHAP/CTI results."
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
        default=Path("shap_analysis/ngram_desc_lgbm_group_nometa_106/per_sample"),
    )
    parser.add_argument(
        "--cti-dir",
        type=Path,
        default=Path("shap_analysis/ngram_desc_lgbm_group_nometa_106/cti"),
    )
    parser.add_argument(
        "--cti-db",
        type=Path,
        default=None,
        help="CTI SQLite DB (auto-detected from cti-dir if not set).",
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=Path("evaluation/explanation_quality.json"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures/slide_set"),
        help="Directory with slide-set PNGs to embed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/shap_cti_report.html"),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="SHAP / CTI Analysis Report",
    )
    parser.add_argument(
        "--max-per-sample-rows",
        type=int,
        default=200,
        help="Max per-sample rows to include in report.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="Comma-separated labels (empty = all).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json_file(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_cti_summary_from_db(db_path: Path, limit: int = 500) -> List[Dict[str, str]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    rows = []
    try:
        cursor = conn.execute(
            "SELECT sample, label, technique_ids, technique_names, "
            "technique_scores, tactics, explanation "
            "FROM sample_summary LIMIT ?",
            (limit,),
        )
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            rows.append(dict(zip(columns, row)))
    except sqlite3.OperationalError:
        pass
    conn.close()
    return rows


def load_technique_stats_from_db(db_path: Path) -> List[Dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    rows = []
    try:
        cursor = conn.execute(
            "SELECT technique_id, technique_name, "
            "SUM(score) as total_score, COUNT(DISTINCT sample) as sample_count, "
            "AVG(confidence_score) as avg_confidence, "
            "GROUP_CONCAT(DISTINCT category) as categories, "
            "GROUP_CONCAT(DISTINCT tactics) as tactics "
            "FROM sample_techniques "
            "GROUP BY technique_id "
            "ORDER BY total_score DESC"
        )
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            rows.append(dict(zip(columns, row)))
    except sqlite3.OperationalError:
        pass
    conn.close()
    return rows


def encode_image_base64(path: Path) -> Optional[str]:
    """Read image file and encode as base64 for inline embedding."""
    import base64
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{data}"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242836;
    --border: #2e3345;
    --text: #e4e6f0;
    --text2: #9a9eb8;
    --accent: #6c7bf7;
    --accent2: #4ecdc4;
    --red: #ff6b6b;
    --orange: #ffa94d;
    --green: #51cf66;
    --blue: #339af0;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px 30px; }}

  /* Header */
  header {{
    background: linear-gradient(135deg, #1a1d27 0%, #242836 100%);
    border-bottom: 1px solid var(--border);
    padding: 30px 0;
    margin-bottom: 30px;
  }}
  header h1 {{
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  header .meta {{ color: var(--text2); font-size: 0.85rem; margin-top: 6px; }}

  /* Navigation */
  nav {{
    position: sticky; top: 0; z-index: 100;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 10px 30px;
    display: flex; gap: 4px; flex-wrap: wrap;
  }}
  nav a {{
    color: var(--text2);
    text-decoration: none;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 0.85rem;
    transition: all 0.2s;
  }}
  nav a:hover, nav a.active {{
    background: var(--accent);
    color: white;
  }}

  /* Sections */
  section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
  }}
  section h2 {{
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 10px;
  }}
  section h2 .icon {{ font-size: 1.4rem; }}
  section h3 {{
    font-size: 1.05rem;
    color: var(--accent);
    margin: 18px 0 10px;
  }}

  /* Cards */
  .card-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    margin-bottom: 16px;
  }}
  .card {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px;
  }}
  .card .label {{ color: var(--text2); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; }}
  .card .value {{ font-size: 1.8rem; font-weight: 700; margin-top: 4px; }}
  .card .value.green {{ color: var(--green); }}
  .card .value.blue {{ color: var(--blue); }}
  .card .value.orange {{ color: var(--orange); }}
  .card .value.accent {{ color: var(--accent); }}

  /* Tables */
  .table-wrap {{
    overflow-x: auto;
    border-radius: 8px;
    border: 1px solid var(--border);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }}
  th {{
    background: var(--surface2);
    color: var(--text2);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
    padding: 10px 14px;
    text-align: left;
    cursor: pointer;
    user-select: none;
    position: sticky; top: 0;
    white-space: nowrap;
  }}
  th:hover {{ color: var(--accent); }}
  th .sort-arrow {{ margin-left: 4px; opacity: 0.4; }}
  td {{
    padding: 8px 14px;
    border-top: 1px solid var(--border);
    vertical-align: top;
  }}
  tr:hover td {{ background: rgba(108, 123, 247, 0.05); }}

  /* Feature name styling */
  .feat-seq {{ color: var(--blue); }}
  .feat-skip {{ color: var(--orange); }}
  .feat-desc {{ color: var(--green); }}

  /* SHAP bar */
  .shap-bar-wrap {{ display: flex; align-items: center; gap: 8px; }}
  .shap-bar {{
    height: 14px;
    border-radius: 3px;
    min-width: 2px;
    transition: width 0.3s;
  }}
  .shap-bar.pos {{ background: linear-gradient(90deg, var(--red), #ff8787); }}
  .shap-bar.neg {{ background: linear-gradient(90deg, #74c0fc, var(--blue)); }}

  /* Technique badges */
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 1px 2px;
  }}
  .badge-high {{ background: rgba(255,107,107,0.2); color: var(--red); border: 1px solid rgba(255,107,107,0.3); }}
  .badge-medium {{ background: rgba(255,169,77,0.2); color: var(--orange); border: 1px solid rgba(255,169,77,0.3); }}
  .badge-low {{ background: rgba(81,207,102,0.2); color: var(--green); border: 1px solid rgba(81,207,102,0.3); }}
  .badge-cat {{
    background: rgba(108,123,247,0.15); color: var(--accent); border: 1px solid rgba(108,123,247,0.25);
  }}

  /* Search */
  .search-box {{
    margin-bottom: 12px;
    display: flex; gap: 10px; align-items: center;
  }}
  .search-box input {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 14px;
    color: var(--text);
    font-size: 0.85rem;
    width: 300px;
    outline: none;
    transition: border-color 0.2s;
  }}
  .search-box input:focus {{ border-color: var(--accent); }}
  .search-box .count {{ color: var(--text2); font-size: 0.8rem; }}

  /* Tab system */
  .tabs {{ display: flex; gap: 2px; margin-bottom: 16px; flex-wrap: wrap; }}
  .tab-btn {{
    padding: 8px 16px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px 8px 0 0;
    color: var(--text2);
    cursor: pointer;
    font-size: 0.85rem;
    transition: all 0.2s;
  }}
  .tab-btn:hover {{ color: var(--text); }}
  .tab-btn.active {{
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  /* Explanation text */
  .explanation {{
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    line-height: 1.7;
    margin: 8px 0;
  }}

  /* Collapsible */
  details {{
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 8px;
  }}
  details summary {{
    padding: 10px 14px;
    cursor: pointer;
    font-weight: 500;
    background: var(--surface2);
    border-radius: 8px;
  }}
  details[open] summary {{ border-radius: 8px 8px 0 0; border-bottom: 1px solid var(--border); }}
  details .detail-body {{ padding: 14px; }}

  /* Images */
  .fig-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 16px;
  }}
  .fig-card {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }}
  .fig-card img {{ width: 100%; display: block; }}
  .fig-card .caption {{
    padding: 10px 14px;
    font-size: 0.8rem;
    color: var(--text2);
    text-align: center;
  }}

  /* Donut inline SVG */
  .donut-row {{ display: flex; gap: 24px; flex-wrap: wrap; align-items: center; }}
  .donut-legend {{ font-size: 0.85rem; }}
  .donut-legend li {{ list-style: none; margin: 4px 0; display: flex; align-items: center; gap: 6px; }}
  .donut-legend .swatch {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}

  /* Footer */
  footer {{
    text-align: center;
    padding: 24px;
    color: var(--text2);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
    margin-top: 40px;
  }}

  @media (max-width: 768px) {{
    .container {{ padding: 12px; }}
    .card-grid {{ grid-template-columns: 1fr; }}
    .fig-grid {{ grid-template-columns: 1fr; }}
    .search-box input {{ width: 100%; }}
  }}
</style>
</head>
<body>

<header>
  <div class="container">
    <h1>{title}</h1>
    <div class="meta">Generated: {timestamp} ｜ Pipeline: CAPE + N-gram + SHAP + ATT&CK CTI</div>
  </div>
</header>

<nav id="main-nav">
  <a href="#overview">📊 Overview</a>
  <a href="#top-features">🔬 Top Features</a>
  <a href="#groups">📦 Group Contributions</a>
  <a href="#cti">🛡️ ATT&CK Mapping</a>
  <a href="#per-sample">🔍 Per-Sample</a>
  <a href="#evaluation">📈 Evaluation</a>
  <a href="#figures">🖼️ Figures</a>
</nav>

<div class="container">
{content}
</div>

<footer>
  CAPE N-gram SHAP CTI Report — Auto-generated by generate_html_report.py
</footer>

<script>
// Table sorting
document.querySelectorAll('th[data-sort]').forEach(th => {{
  th.addEventListener('click', () => {{
    const table = th.closest('table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const col = parseInt(th.dataset.sort);
    const type = th.dataset.type || 'string';
    const asc = th.classList.toggle('asc');

    rows.sort((a, b) => {{
      let va = a.cells[col]?.textContent.trim() || '';
      let vb = b.cells[col]?.textContent.trim() || '';
      if (type === 'number') {{
        va = parseFloat(va) || 0;
        vb = parseFloat(vb) || 0;
        return asc ? va - vb : vb - va;
      }}
      return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    }});
    rows.forEach(r => tbody.appendChild(r));
  }});
}});

// Table search
document.querySelectorAll('.search-input').forEach(input => {{
  input.addEventListener('input', () => {{
    const q = input.value.toLowerCase();
    const tableId = input.dataset.table;
    const table = document.getElementById(tableId);
    if (!table) return;
    const rows = table.querySelectorAll('tbody tr');
    let visible = 0;
    rows.forEach(r => {{
      const match = r.textContent.toLowerCase().includes(q);
      r.style.display = match ? '' : 'none';
      if (match) visible++;
    }});
    const counter = input.parentElement.querySelector('.count');
    if (counter) counter.textContent = visible + ' / ' + rows.length;
  }});
}});

// Tabs
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const group = btn.dataset.group;
    const target = btn.dataset.tab;
    document.querySelectorAll(`.tab-btn[data-group="${{group}}"]`).forEach(b => b.classList.remove('active'));
    document.querySelectorAll(`.tab-content[data-group="${{group}}"]`).forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(target)?.classList.add('active');
  }});
}});

// Smooth scroll nav
document.querySelectorAll('#main-nav a').forEach(a => {{
  a.addEventListener('click', (e) => {{
    e.preventDefault();
    const id = a.getAttribute('href').slice(1);
    document.getElementById(id)?.scrollIntoView({{ behavior: 'smooth' }});
    document.querySelectorAll('#main-nav a').forEach(n => n.classList.remove('active'));
    a.classList.add('active');
  }});
}});
</script>
</body>
</html>"""


def fmt(val: Any, decimals: int = 4) -> str:
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return escape(str(val)) if val is not None else ""


def shap_bar_html(value: float, max_abs: float) -> str:
    pct = min(abs(value) / max_abs * 100, 100) if max_abs > 0 else 0
    cls = "pos" if value >= 0 else "neg"
    return (
        f'<div class="shap-bar-wrap">'
        f'<div class="shap-bar {cls}" style="width:{pct:.1f}%"></div>'
        f'<span>{value:.4f}</span></div>'
    )


def feature_class(name: str) -> str:
    if name.startswith("seq:"):
        return "feat-seq"
    if name.startswith("skip:"):
        return "feat-skip"
    if name.startswith("desc:"):
        return "feat-desc"
    return ""


def confidence_badge(score: float) -> str:
    if score >= 0.7:
        return f'<span class="badge badge-high">{score:.2f}</span>'
    elif score >= 0.4:
        return f'<span class="badge badge-medium">{score:.2f}</span>'
    return f'<span class="badge badge-low">{score:.2f}</span>'


def make_donut_svg(shares: List[Tuple[str, float, str]], size: int = 160) -> str:
    """Generate inline SVG donut chart."""
    cx, cy, r = size // 2, size // 2, size // 2 - 10
    inner_r = r * 0.6
    total = sum(s for _, s, _ in shares) or 1
    svg = f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'

    import math
    angle = -90
    for name, share, color in shares:
        pct = share / total
        sweep = pct * 360
        large = 1 if sweep > 180 else 0

        start_rad = math.radians(angle)
        end_rad = math.radians(angle + sweep)

        x1o = cx + r * math.cos(start_rad)
        y1o = cy + r * math.sin(start_rad)
        x2o = cx + r * math.cos(end_rad)
        y2o = cy + r * math.sin(end_rad)

        x1i = cx + inner_r * math.cos(end_rad)
        y1i = cy + inner_r * math.sin(end_rad)
        x2i = cx + inner_r * math.cos(start_rad)
        y2i = cy + inner_r * math.sin(start_rad)

        path = (
            f"M {x1o:.1f} {y1o:.1f} "
            f"A {r} {r} 0 {large} 1 {x2o:.1f} {y2o:.1f} "
            f"L {x1i:.1f} {y1i:.1f} "
            f"A {inner_r} {inner_r} 0 {large} 0 {x2i:.1f} {y2i:.1f} Z"
        )
        svg += f'<path d="{path}" fill="{color}" opacity="0.85"/>'
        angle += sweep

    svg += "</svg>"
    return svg


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def build_overview(
    top_data: List[Dict],
    group_data: List[Dict],
    cti_stats: List[Dict],
    eval_data: Optional[Dict],
) -> str:
    labels = set()
    for row in top_data:
        labels.add(row.get("label", ""))
    for row in group_data:
        labels.add(row.get("label", ""))

    n_labels = len(labels)
    n_features = len(top_data)
    n_techniques = len(cti_stats)

    cards = f"""
    <section id="overview">
      <h2><span class="icon">📊</span> Overview</h2>
      <div class="card-grid">
        <div class="card">
          <div class="label">Labels Analyzed</div>
          <div class="value accent">{n_labels}</div>
        </div>
        <div class="card">
          <div class="label">Top Features Tracked</div>
          <div class="value blue">{n_features}</div>
        </div>
        <div class="card">
          <div class="label">ATT&CK Techniques Detected</div>
          <div class="value orange">{n_techniques}</div>
        </div>"""

    if eval_data:
        # Avg rule match rate
        rates = [v.get("rule_match_rate", 0) for v in eval_data.values() if "rule_match_rate" in v]
        avg_rate = sum(rates) / len(rates) if rates else 0
        cards += f"""
        <div class="card">
          <div class="label">Avg Rule Match Rate</div>
          <div class="value green">{avg_rate:.1%}</div>
        </div>"""

    cards += """
      </div>
    </section>"""
    return cards


def build_top_features_section(top_data: List[Dict], labels: List[str]) -> str:
    if not top_data:
        return '<section id="top-features"><h2><span class="icon">🔬</span> Top Features</h2><p>データなし</p></section>'

    # Group by label
    by_label: Dict[str, List[Dict]] = {}
    for row in top_data:
        by_label.setdefault(row.get("label", ""), []).append(row)

    if labels:
        by_label = {k: v for k, v in by_label.items() if k in labels}

    # Build tabs
    tab_btns = ""
    tab_contents = ""
    for i, (label, rows) in enumerate(sorted(by_label.items())):
        active = " active" if i == 0 else ""
        tab_id = f"feat-{label}"
        tab_btns += f'<button class="tab-btn{active}" data-group="feat-tabs" data-tab="{tab_id}">{escape(label)}</button>'

        max_shap = max((float(r.get("mean_abs_shap", 0)) for r in rows), default=1)
        table_rows = ""
        for r in sorted(rows, key=lambda x: float(x.get("mean_abs_shap", 0)), reverse=True):
            feat = r.get("feature", "")
            cls = feature_class(feat)
            val = float(r.get("mean_abs_shap", 0))
            table_rows += f"""<tr>
              <td>{r.get('rank', '')}</td>
              <td class="{cls}">{escape(feat)}</td>
              <td>{shap_bar_html(val, max_shap)}</td>
            </tr>"""

        tab_contents += f"""
        <div id="{tab_id}" class="tab-content{active}" data-group="feat-tabs">
          <div class="table-wrap">
            <table id="feat-table-{label}">
              <thead><tr>
                <th data-sort="0" data-type="number">Rank <span class="sort-arrow">⇅</span></th>
                <th data-sort="1">Feature <span class="sort-arrow">⇅</span></th>
                <th data-sort="2" data-type="number">Mean |SHAP| <span class="sort-arrow">⇅</span></th>
              </tr></thead>
              <tbody>{table_rows}</tbody>
            </table>
          </div>
        </div>"""

    return f"""
    <section id="top-features">
      <h2><span class="icon">🔬</span> Top Features (SHAP)</h2>
      <div class="search-box">
        <input type="text" class="search-input" placeholder="🔍 特徴量を検索..." data-table="feat-table-{sorted(by_label.keys())[0] if by_label else ''}">
        <span class="count"></span>
      </div>
      <div class="tabs">{tab_btns}</div>
      {tab_contents}
    </section>"""


def build_group_section(group_data: List[Dict]) -> str:
    if not group_data:
        return '<section id="groups"><h2><span class="icon">📦</span> Feature Group Contributions</h2><p>データなし</p></section>'

    by_label: Dict[str, List[Dict]] = {}
    for row in group_data:
        by_label.setdefault(row.get("label", ""), []).append(row)

    group_colors = {"seq_ngram": "#339af0", "skipgram": "#ffa94d", "desc_tfidf": "#51cf66"}
    group_names = {"seq_ngram": "Sequence N-gram", "skipgram": "Skip-gram", "desc_tfidf": "Description TF-IDF"}

    content = ""
    for label, rows in sorted(by_label.items()):
        shares = []
        for r in rows:
            gname = r.get("group", "")
            share = float(r.get("share", 0))
            color = group_colors.get(gname, "#999")
            shares.append((gname, share, color))

        svg = make_donut_svg(shares)
        legend = "<ul class='donut-legend'>"
        for gname, share, color in shares:
            display_name = group_names.get(gname, gname)
            legend += f'<li><span class="swatch" style="background:{color}"></span> {display_name}: {share:.1%}</li>'
        legend += "</ul>"

        n_samples = rows[0].get("num_samples", "?") if rows else "?"
        content += f"""
        <details>
          <summary>{escape(label)} (n={n_samples})</summary>
          <div class="detail-body">
            <div class="donut-row">{svg}{legend}</div>
          </div>
        </details>"""

    return f"""
    <section id="groups">
      <h2><span class="icon">📦</span> Feature Group Contributions</h2>
      {content}
    </section>"""


def build_cti_section(cti_stats: List[Dict], cti_summary: List[Dict]) -> str:
    if not cti_stats and not cti_summary:
        return '<section id="cti"><h2><span class="icon">🛡️</span> ATT&CK Technique Mapping</h2><p>データなし</p></section>'

    # Technique overview table
    tech_rows = ""
    for r in cti_stats[:50]:
        tid = r.get("technique_id", "")
        name = r.get("technique_name", "")
        score = float(r.get("total_score", 0))
        samples = r.get("sample_count", 0)
        conf = float(r.get("avg_confidence", 0))
        cats = r.get("categories", "")
        tactics = r.get("tactics", "")

        cat_badges = "".join(f'<span class="badge badge-cat">{escape(c.strip())}</span>' for c in (cats or "").split(",") if c.strip())

        tech_rows += f"""<tr>
          <td><strong>{escape(tid)}</strong></td>
          <td>{escape(name)}</td>
          <td>{score:.3f}</td>
          <td>{samples}</td>
          <td>{confidence_badge(conf)}</td>
          <td>{cat_badges}</td>
          <td style="font-size:0.75rem">{escape(tactics or '')}</td>
        </tr>"""

    # Per-sample CTI (first N)
    sample_rows = ""
    for r in cti_summary[:100]:
        explanation = r.get("explanation", "")
        tech_ids = r.get("technique_ids", "")
        sample_rows += f"""
        <details>
          <summary>{escape(r.get('sample', ''))} — {escape(r.get('label', ''))}</summary>
          <div class="detail-body">
            <div><strong>Techniques:</strong> {escape(tech_ids)}</div>
            <div class="explanation">{escape(explanation)}</div>
          </div>
        </details>"""

    return f"""
    <section id="cti">
      <h2><span class="icon">🛡️</span> ATT&CK Technique Mapping</h2>
      <h3>Technique Overview</h3>
      <div class="search-box">
        <input type="text" class="search-input" placeholder="🔍 Technique ID/名前で検索..." data-table="cti-table">
        <span class="count"></span>
      </div>
      <div class="table-wrap">
        <table id="cti-table">
          <thead><tr>
            <th data-sort="0">ID <span class="sort-arrow">⇅</span></th>
            <th data-sort="1">Name <span class="sort-arrow">⇅</span></th>
            <th data-sort="2" data-type="number">Score <span class="sort-arrow">⇅</span></th>
            <th data-sort="3" data-type="number">Samples <span class="sort-arrow">⇅</span></th>
            <th data-sort="4" data-type="number">Confidence <span class="sort-arrow">⇅</span></th>
            <th>Category</th>
            <th>Tactics</th>
          </tr></thead>
          <tbody>{tech_rows}</tbody>
        </table>
      </div>

      <h3>Per-Sample Explanations</h3>
      {sample_rows if sample_rows else '<p style="color:var(--text2)">CTI結果なし</p>'}
    </section>"""


def build_per_sample_section(per_sample_dir: Path, labels: List[str], max_rows: int) -> str:
    if not per_sample_dir.exists():
        return '<section id="per-sample"><h2><span class="icon">🔍</span> Per-Sample SHAP</h2><p>データなし</p></section>'

    content = ""
    for csv_path in sorted(per_sample_dir.glob("shap_per_sample_topk_*.csv")):
        label = csv_path.stem.replace("shap_per_sample_topk_", "")
        if labels and label not in labels:
            continue

        rows = load_csv(csv_path)[:max_rows]
        if not rows:
            continue

        # Group by sample
        by_sample: Dict[str, List[Dict]] = {}
        for r in rows:
            by_sample.setdefault(r.get("sample", ""), []).append(r)

        sample_details = ""
        for sample, feats in list(by_sample.items())[:30]:
            max_abs = max((float(f.get("abs_shap", 0)) for f in feats), default=1)
            feat_rows = ""
            for f in feats:
                feat = f.get("feature", "")
                cls = feature_class(feat)
                shap_val = float(f.get("shap_value", 0))
                feat_rows += f"""<tr>
                  <td>{f.get('rank','')}</td>
                  <td class="{cls}">{escape(feat)}</td>
                  <td>{shap_bar_html(shap_val, max_abs)}</td>
                  <td>{fmt(float(f.get('feature_value', 0)), 2)}</td>
                </tr>"""

            sample_details += f"""
            <details>
              <summary>{escape(sample)}</summary>
              <div class="detail-body">
                <table><thead><tr><th>Rank</th><th>Feature</th><th>SHAP Value</th><th>Feature Value</th></tr></thead>
                <tbody>{feat_rows}</tbody></table>
              </div>
            </details>"""

        content += f"""
        <h3>{escape(label)} ({len(by_sample)} samples)</h3>
        {sample_details}"""

    return f"""
    <section id="per-sample">
      <h2><span class="icon">🔍</span> Per-Sample SHAP Details</h2>
      {content if content else '<p style="color:var(--text2)">Per-sample データなし</p>'}
    </section>"""


def build_eval_section(eval_data: Optional[Dict]) -> str:
    if not eval_data:
        return '<section id="evaluation"><h2><span class="icon">📈</span> Explanation Quality</h2><p>データなし</p></section>'

    # Build table
    all_metrics = sorted(set(k for v in eval_data.values() for k in v.keys()))
    metric_names = {
        "rule_match_rate": "ルール一致率",
        "technique_diversity": "技術多様性",
        "technique_concentration": "技術集中度",
        "group_entropy": "グループエントロピー",
        "explanation_coverage": "説明カバレッジ",
    }

    header = "<th data-sort='0'>Label <span class='sort-arrow'>⇅</span></th>"
    for i, m in enumerate(all_metrics, 1):
        display = metric_names.get(m, m)
        header += f"<th data-sort='{i}' data-type='number'>{escape(display)} <span class='sort-arrow'>⇅</span></th>"

    rows = ""
    for label in sorted(eval_data.keys()):
        vals = eval_data[label]
        cells = f"<td><strong>{escape(label)}</strong></td>"
        for m in all_metrics:
            v = vals.get(m)
            if isinstance(v, float):
                cells += f"<td>{v:.4f}</td>"
            elif v is not None:
                cells += f"<td>{v}</td>"
            else:
                cells += "<td>—</td>"
        rows += f"<tr>{cells}</tr>"

    # Averages
    avg_cells = "<td><strong>平均</strong></td>"
    for m in all_metrics:
        values = [v[m] for v in eval_data.values() if m in v and isinstance(v[m], (int, float))]
        if values:
            avg = sum(values) / len(values)
            avg_cells += f"<td><strong>{avg:.4f}</strong></td>"
        else:
            avg_cells += "<td>—</td>"

    return f"""
    <section id="evaluation">
      <h2><span class="icon">📈</span> Explanation Quality Metrics</h2>
      <div class="table-wrap">
        <table id="eval-table">
          <thead><tr>{header}</tr></thead>
          <tbody>{rows}<tr style="border-top:2px solid var(--accent)">{avg_cells}</tr></tbody>
        </table>
      </div>
    </section>"""


def build_figures_section(figures_dir: Path, labels: List[str]) -> str:
    if not figures_dir.exists():
        return '<section id="figures"><h2><span class="icon">🖼️</span> Figures</h2><p>図なし（figures_dirが見つかりません）</p></section>'

    content = ""
    for label_dir in sorted(figures_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if labels and label not in labels:
            continue

        figs = ""
        for img_name, caption in [
            ("summary_bar.png", "Top Features (Summary Bar)"),
            ("group_donut.png", "Feature Group Contribution"),
            ("representative_waterfall.png", "Representative Sample (Waterfall)"),
        ]:
            img_path = label_dir / img_name
            b64 = encode_image_base64(img_path)
            if b64:
                figs += f'<div class="fig-card"><img src="{b64}" alt="{caption}"><div class="caption">{caption}</div></div>'

        if figs:
            content += f"""
            <h3>{escape(label)}</h3>
            <div class="fig-grid">{figs}</div>"""

    return f"""
    <section id="figures">
      <h2><span class="icon">🖼️</span> Visualization</h2>
      {content if content else '<p style="color:var(--text2)">生成済み図なし（先に visualize_shap_slide_set.py を実行してください）</p>'}
    </section>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    labels = [l.strip() for l in args.labels.split(",") if l.strip()] if args.labels else []

    # Load data
    top_data = load_csv(args.top_csv)
    group_data = load_csv(args.group_csv)

    cti_db_path = args.cti_db or (args.cti_dir / "cti_results.sqlite")
    cti_stats = load_technique_stats_from_db(cti_db_path)
    cti_summary = load_cti_summary_from_db(cti_db_path, limit=args.max_per_sample_rows)

    eval_data = load_json_file(args.eval_json)

    print(f"[INFO] Top features: {len(top_data)} rows")
    print(f"[INFO] Group data: {len(group_data)} rows")
    print(f"[INFO] CTI technique stats: {len(cti_stats)} techniques")
    print(f"[INFO] CTI per-sample: {len(cti_summary)} samples")
    print(f"[INFO] Eval data: {'loaded' if eval_data else 'not found'}")

    # Build sections
    content = ""
    content += build_overview(top_data, group_data, cti_stats, eval_data)
    content += build_top_features_section(top_data, labels)
    content += build_group_section(group_data)
    content += build_cti_section(cti_stats, cti_summary)
    content += build_per_sample_section(args.per_sample_dir, labels, args.max_per_sample_rows)
    content += build_eval_section(eval_data)
    content += build_figures_section(args.figures_dir, labels)

    # Render
    html = HTML_TEMPLATE.format(
        title=escape(args.title),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        content=content,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")

    size_kb = args.output.stat().st_size / 1024
    print(f"\n[INFO] Report generated: {args.output} ({size_kb:.1f} KB)")
    print(f"[INFO] Open in browser: file://{args.output.resolve()}")


if __name__ == "__main__":
    main()
