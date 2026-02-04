"""
Padel schedule stats dashboard generator (offline).

Drop this file into a folder containing the exported JSON files created by the
Streamlit app upload, then run:

  python padel_stats_dashboard.py

It will create:
  - padel_stats_dashboard.html

Expected input files (in the same folder):
  - Schedule_Statistics_*_pairings.json

Those JSON files must include:
  - date: e.g. "Monday 13 October 2025"  (or session_date: "2025-10-13")
  - schedule_number: integer
  - pairings_lines: list of strings like:
      "Aideen ♀ [0]: Anna ♀ [2,1], Anny ♀ [1,2]"

Heuristic for "actual schedule used":
  - players usually play Tue/Thu mornings, but schedules are generated in the lead-up
  - treat the session cutoff as 14:00 (2pm) local Spain time on Tue/Thu
  - assign each file to the *next* Tue/Thu session whose cutoff is >= generated time
  - then pick the latest file per session date

We prefer JSON field "generated_at" (UTC ISO). If missing, we fall back to file modified time.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None


# --------------------------- user-editable settings -------------------------- #

# Sessions happen Tue/Thu morning, but schedules may be generated days before.
# We assign each generated file to the *next* Tue/Thu session where the timestamp
# is <= 14:00 local (Spain) time, then choose the latest file within that session window.
SPAIN_TZ = "Europe/Madrid"
CUTOFF_HOUR_LOCAL = 14  # 2pm

# Output filename
OUTPUT_HTML = "padel_stats_dashboard.html"


# ------------------------------ parsing helpers ----------------------------- #

_LINE_RE = re.compile(r"^(?P<p>.+?)(?:\s[♀♂])?\s\[(?P<rest>\d+)\]:\s(?P<rhs>.*)$")
_PAIR_RE = re.compile(r"(?P<q>.+?)(?:\s[♀♂])?\s\[(?P<with>\d+),(?P<against>\d+)\]$")


def _parse_time_hhmm(s: str) -> time:
    hh, mm = s.strip().split(":")
    return time(hour=int(hh), minute=int(mm))


def _parse_session_date(obj: Dict[str, Any], filename: str) -> Optional[date]:
    # Preferred: ISO
    if isinstance(obj.get("session_date"), str):
        try:
            return datetime.strptime(obj["session_date"], "%Y-%m-%d").date()
        except Exception:
            pass
    # Next: "Monday 13 October 2025"
    if isinstance(obj.get("date"), str):
        try:
            return datetime.strptime(obj["date"], "%A %d %B %Y").date()
        except Exception:
            pass
    # Fallback: attempt to parse from filename tokens (best-effort)
    # Example: Schedule_Statistics_1_Monday_13_October_2025_pairings.json
    parts = filename.replace(".json", "").split("_")
    # Find 4-token pattern: Weekday, DD, Month, YYYY
    for i in range(len(parts) - 3):
        w, dd, mon, yyyy = parts[i : i + 4]
        if dd.isdigit() and yyyy.isdigit():
            cand = f"{w} {dd} {mon} {yyyy}"
            try:
                return datetime.strptime(cand, "%A %d %B %Y").date()
            except Exception:
                continue
    return None


def _parse_generated_at(obj: Dict[str, Any], file_path: Path) -> datetime:
    if isinstance(obj.get("generated_at"), str):
        try:
            return datetime.fromisoformat(obj["generated_at"].replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pass
    # Fallback: file mtime (local)
    return datetime.fromtimestamp(file_path.stat().st_mtime)

def _to_spain_local(dt_utc_naive: datetime) -> datetime:
    if ZoneInfo is None:
        return dt_utc_naive
    return dt_utc_naive.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(SPAIN_TZ)).replace(tzinfo=None)


def assign_session_date_from_generated_at(dt_utc_naive: datetime) -> date:
    """Assign generation time to the most likely session date.

    Rule: pick the next Tue/Thu whose local cutoff (14:00) is >= dt_local.
    """
    dt_local = _to_spain_local(dt_utc_naive)
    cutoff_t = time(hour=CUTOFF_HOUR_LOCAL, minute=0)
    for delta in range(0, 8):
        d = dt_local.date() + timedelta(days=delta)
        if d.weekday() in (1, 3):  # Tue=1, Thu=3
            if dt_local <= datetime.combine(d, cutoff_t):
                return d
    return dt_local.date()


def _strip_gender_symbol(name: str) -> str:
    return name.replace(" ♀", "").replace(" ♂", "").strip()


@dataclass(frozen=True)
class SessionFile:
    path: Path
    session_date: date
    schedule_number: int
    generated_at: datetime
    pairings_lines: List[str]


def parse_pairings_lines(lines: Iterable[str]) -> Tuple[Dict[str, int], Dict[Tuple[str, str], Tuple[int, int]]]:
    """Return (rest_by_player, pair_counts) for one schedule.

    pair_counts key is (a,b) sorted; value is (teammates, opponents).
    """
    rests: Dict[str, int] = {}
    pairs: Dict[Tuple[str, str], Tuple[int, int]] = {}

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue

        m = _LINE_RE.match(raw)
        if not m:
            continue

        p = _strip_gender_symbol(m.group("p"))
        rest = int(m.group("rest"))
        rhs = m.group("rhs").strip()
        rests[p] = rest

        if not rhs:
            continue

        entries = [e.strip() for e in rhs.split(",") if e.strip()]
        for entry in entries:
            m2 = _PAIR_RE.match(entry)
            if not m2:
                continue

            q = _strip_gender_symbol(m2.group("q"))
            w = int(m2.group("with"))
            a = int(m2.group("against"))

            if p == q:
                continue

            a_name, b_name = sorted([p, q])
            # Only count once using lexical direction, to avoid double counting
            if p != a_name:
                continue

            tw, ta = pairs.get((a_name, b_name), (0, 0))
            pairs[(a_name, b_name)] = (tw + w, ta + a)

    return rests, pairs


def choose_actual_files(files: List[SessionFile]) -> List[SessionFile]:
    """Pick one 'most likely used' file per session_date."""
    by_date: Dict[date, List[SessionFile]] = {}
    for f in files:
        by_date.setdefault(f.session_date, []).append(f)

    chosen: List[SessionFile] = []
    for d, fs in sorted(by_date.items(), key=lambda x: x[0]):
        cutoff_dt_local = datetime.combine(d, time(hour=CUTOFF_HOUR_LOCAL, minute=0))
        before = [f for f in fs if _to_spain_local(f.generated_at) <= cutoff_dt_local]
        chosen.append(max(before or fs, key=lambda f: f.generated_at))
    return chosen


def aggregate(files: List[SessionFile]) -> Dict[str, Any]:
    """Aggregate rest + pairings across the provided session files."""
    rest_total: Dict[str, int] = {}
    pair_total: Dict[Tuple[str, str], Tuple[int, int]] = {}

    for sf in files:
        rests, pairs = parse_pairings_lines(sf.pairings_lines)
        for p, r in rests.items():
            rest_total[p] = rest_total.get(p, 0) + r
        for (a, b), (tw, ta) in pairs.items():
            cw, ca = pair_total.get((a, b), (0, 0))
            pair_total[(a, b)] = (cw + tw, ca + ta)

    # Build tables
    players = sorted(rest_total.keys())
    player_rows = [{"player": p, "rests": rest_total.get(p, 0)} for p in players]

    pair_rows = []
    for (a, b), (tw, ta) in pair_total.items():
        pair_rows.append(
            {
                "a": a,
                "b": b,
                "teammates": tw,
                "opponents": ta,
                "total": tw + ta,
            }
        )
    pair_rows.sort(key=lambda r: (r["total"], r["teammates"], r["opponents"]), reverse=True)

    return {
        "players": player_rows,
        "pairs": pair_rows,
    }


def aggregate_by_year(files: List[SessionFile]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    by_year: Dict[int, List[SessionFile]] = {}
    for f in files:
        by_year.setdefault(f.session_date.year, []).append(f)
    out["All time"] = aggregate(files)
    for y, fs in sorted(by_year.items()):
        out[str(y)] = aggregate(fs)
    return out


# --------------------------------- HTML ----------------------------------- #

def _html_page(data_by_year: Dict[str, Dict[str, Any]], chosen_sessions: List[SessionFile]) -> str:
    sessions = [
        {
            "date": sf.session_date.isoformat(),
            "date_label": sf.session_date.strftime("%A %d %B %Y"),
            "schedule_number": sf.schedule_number,
            "generated_at": sf.generated_at.isoformat(timespec="seconds"),
            "filename": sf.path.name,
        }
        for sf in sorted(chosen_sessions, key=lambda s: s.session_date)
    ]

    # Use Plotly via CDN (no pip install required)
    payload = {
        "sessions": sessions,
        "byYear": data_by_year,
    }

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Padel Schedule Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      margin: 0; padding: 18px; background: #f6f7fb; color: #111;
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; }}
    .card {{
      background: #fff; border: 1px solid #e6e8ef; border-radius: 12px;
      padding: 14px 16px; margin: 12px 0;
      box-shadow: 0 2px 10px rgba(10, 20, 40, 0.04);
    }}
    h1 {{ margin: 6px 0 4px; font-size: 22px; }}
    .sub {{ color:#556; margin: 0 0 10px; font-size: 13px; }}
    .row {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
    @media (min-width: 900px) {{ .row.two {{ grid-template-columns: 1fr 1fr; }} }}
    select {{ padding: 6px 8px; border-radius: 8px; border: 1px solid #ccd; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 8px; border-bottom: 1px solid #eef; text-align: left; font-size: 13px; }}
    th {{ background: #f3f5fb; position: sticky; top: 0; z-index: 1; }}
    .small {{ font-size: 12px; color:#667; }}
    .pill {{ display:inline-block; padding:2px 8px; border-radius: 999px; background:#eef; color:#334; font-size:12px; }}
    .grid3 {{ display:grid; grid-template-columns: 1fr; gap: 12px; }}
    @media (min-width: 900px) {{ .grid3 {{ grid-template-columns: 1fr 1fr 1fr; }} }}
    .scroll {{ max-height: 420px; overflow: auto; border: 1px solid #eef; border-radius: 10px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Padel Schedule Dashboard</h1>
      <p class="sub">Aggregated from exported schedule pairing JSON files. Year selector updates plots/tables.</p>
      <label class="small">View: </label>
      <select id="yearSelect"></select>
      <span class="pill" id="sessionCount"></span>
    </div>

    <div class="row two">
      <div class="card">
        <h3 style="margin:0 0 8px;">Sessions over time (chosen “used” schedules)</h3>
        <div id="sessionsPlot" style="height:320px;"></div>
        <p class="small">Heuristic: for each session date, pick the latest schedule generated before the session start time (Tue/Thu).</p>
      </div>
      <div class="card">
        <h3 style="margin:0 0 8px;">Top pairs (teammates + opponents)</h3>
        <div class="scroll">
          <table id="pairsTable">
            <thead><tr><th>Pair</th><th>Teammates</th><th>Opponents</th><th>Total</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="row two">
      <div class="card">
        <h3 style="margin:0 0 8px;">Rests by player</h3>
        <div id="restsPlot" style="height:360px;"></div>
      </div>
      <div class="card">
        <h3 style="margin:0 0 8px;">Chosen sessions list</h3>
        <div class="scroll">
          <table id="sessionsTable">
            <thead><tr><th>Date</th><th>Schedule #</th><th>Generated at</th><th>File</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <script>
    const DATA = {json.dumps(payload, ensure_ascii=False)};

    function byDateCount(sessions) {{
      const m = new Map();
      sessions.forEach(s => {{
        m.set(s.date, (m.get(s.date) || 0) + 1);
      }});
      const dates = Array.from(m.keys()).sort();
      return {{
        x: dates,
        y: dates.map(d => m.get(d))
      }};
    }}

    function setSessionTable(sessions) {{
      const tbody = document.querySelector('#sessionsTable tbody');
      tbody.innerHTML = '';
      sessions.forEach(s => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${{s.date_label}}</td><td style="text-align:center;">${{s.schedule_number}}</td><td>${{s.generated_at}}</td><td>${{s.filename}}</td>`;
        tbody.appendChild(tr);
      }});
    }}

    function setPairsTable(rows) {{
      const tbody = document.querySelector('#pairsTable tbody');
      tbody.innerHTML = '';
      rows.slice(0, 200).forEach(r => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${{r.a}} &amp; ${{r.b}}</td><td style="text-align:center;">${{r.teammates}}</td><td style="text-align:center;">${{r.opponents}}</td><td style="text-align:center;">${{r.total}}</td>`;
        tbody.appendChild(tr);
      }});
    }}

    function renderRests(rows) {{
      // top 30
      const top = rows.slice().sort((a,b) => b.rests - a.rests).slice(0, 30);
      const x = top.map(r => r.rests).reverse();
      const y = top.map(r => r.player).reverse();
      Plotly.newPlot('restsPlot', [{{
        type: 'bar',
        orientation: 'h',
        x, y,
        marker: {{color: '#5b7cfa'}}
      }}], {{
        margin: {{l: 120, r: 20, t: 10, b: 40}},
        xaxis: {{title: 'Total rests'}},
        yaxis: {{automargin: true}},
      }}, {{displayModeBar: false}});
    }}

    function renderSessionsPlot(sessions) {{
      const bc = byDateCount(sessions);
      Plotly.newPlot('sessionsPlot', [{{
        type: 'scatter',
        mode: 'lines+markers',
        x: bc.x,
        y: bc.y,
        line: {{color: '#222'}},
        marker: {{size: 6, color: '#222'}}
      }}], {{
        margin: {{l: 40, r: 10, t: 10, b: 40}},
        yaxis: {{title: 'Chosen schedules'}},
        xaxis: {{
          type: 'date',
          rangeslider: {{visible: true}},
          rangeselector: {{
            buttons: [
              {{count: 1, label: '1m', step: 'month', stepmode: 'backward'}},
              {{count: 6, label: '6m', step: 'month', stepmode: 'backward'}},
              {{count: 1, label: '1y', step: 'year', stepmode: 'backward'}},
              {{step: 'all', label: 'All'}}
            ]
          }}
        }}
      }}, {{displayModeBar: false}});
    }}

    function setYearOptions() {{
      const sel = document.getElementById('yearSelect');
      sel.innerHTML = '';
      const years = Object.keys(DATA.byYear);
      years.forEach(y => {{
        const opt = document.createElement('option');
        opt.value = y;
        opt.textContent = y;
        sel.appendChild(opt);
      }});
      sel.value = 'All time';
      return sel;
    }}

    function refreshYear(y) {{
      const agg = DATA.byYear[y];
      document.getElementById('sessionCount').textContent = `Pairs: ${{agg.pairs.length}} | Players: ${{agg.players.length}}`;
      setPairsTable(agg.pairs);
      renderRests(agg.players);
      // sessions plot remains "chosen schedules" all-time (still useful)
      renderSessionsPlot(DATA.sessions);
    }}

    const yearSel = setYearOptions();
    setSessionTable(DATA.sessions);
    refreshYear(yearSel.value);
    yearSel.addEventListener('change', (e) => refreshYear(e.target.value));
  </script>
</body>
</html>
"""


def main() -> int:
    here = Path(__file__).resolve().parent
    json_files = sorted(here.glob("*_pairings.json"))
    if not json_files:
        print("No *_pairings.json files found in:", here)
        return 2

    parsed: List[SessionFile] = []
    for p in json_files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        schedule_number = int(obj.get("schedule_number", 0) or 0)
        gen = _parse_generated_at(obj, p)
        if "generated_at" in obj:
            sd = assign_session_date_from_generated_at(gen)
        else:
            # fallback: try session date from the JSON/filename
            sd = _parse_session_date(obj, p.name) or assign_session_date_from_generated_at(gen)
        if not sd:
            continue
        lines = obj.get("pairings_lines") or []
        if not isinstance(lines, list):
            continue

        parsed.append(
            SessionFile(
                path=p,
                session_date=sd,
                schedule_number=schedule_number,
                generated_at=gen,
                pairings_lines=[str(x) for x in lines],
            )
        )

    if not parsed:
        print("No valid session JSON files parsed.")
        return 2

    chosen = choose_actual_files(parsed)
    data_by_year = aggregate_by_year(chosen)

    out_html = _html_page(data_by_year, chosen)
    out_path = here / OUTPUT_HTML
    out_path.write_text(out_html, encoding="utf-8")

    print("Wrote:", out_path)
    print(f"Sessions (chosen): {len(chosen)} | Source files parsed: {len(parsed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

