#!/usr/bin/env python3
"""
Padel schedule stats dashboard builder (no server required).

Usage (run inside the folder that contains the schedule JSON files):
  python build_dashboard.py --tz Europe/Madrid

This scans the current directory for JSON files containing:
  { "pairings_lines": [ "Player ♀ [R]: Other ♀ [with,against], ...", ... ] }

It then selects the "Used schedule" for each Tue/Thu session window:
  - assign each file to the NEXT Tue/Thu session cutoff (14:00 local time)
  - pick the latest such file as the Used schedule for that session

Finally, it writes a self-contained dashboard.html in the same folder.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import platform
import re
import sys
import bisect
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


SESSION_WEEKDAYS = {1, 3}  # Tue=1, Thu=3 (datetime.weekday: Mon=0)
SESSION_CUTOFF_HOUR = 14
SESSION_CUTOFF_MINUTE = 0


_GENDER_SYMBOLS = {"♀", "♂"}


def _strip_gender_symbols(s: str) -> str:
    s = s.replace("♀", "").replace("♂", "")
    return " ".join(s.split()).strip()


_LEFT_RE = re.compile(r"^(?P<name>.+?)\s*\[(?P<rest>\d+)\]\s*$")
_RIGHT_RE = re.compile(r"^(?P<name>.+?)\s*\[(?P<with>\d+)\s*,\s*(?P<against>\d+)\]\s*$")

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html_tags(s: str) -> str:
    """Remove HTML tags like <span ...> to support exported pairings JSON."""
    return _HTML_TAG_RE.sub("", s)


@dataclasses.dataclass(frozen=True)
class FileMeta:
    filename: str
    path: str
    ts_source: str  # created|modified
    ts_iso: str
    ts_epoch: float


@dataclasses.dataclass(frozen=True)
class ParsedSchedule:
    meta: FileMeta
    # edges for unordered pairs: (a,b,with,against)
    edges: list[tuple[str, str, int, int]]
    players: list[str]
    rests: dict[str, int]


def _local_tzinfo() -> dt.tzinfo:
    return dt.datetime.now().astimezone().tzinfo or dt.timezone.utc


def resolve_tz(tz_name: str | None) -> tuple[dt.tzinfo, str]:
    if not tz_name or tz_name.lower() in {"local", "system"}:
        tz = _local_tzinfo()
        return tz, getattr(tz, "key", str(tz))
    if ZoneInfo is None:
        raise RuntimeError("zoneinfo is unavailable; use --tz local")
    tz = ZoneInfo(tz_name)
    return tz, tz_name


def file_generation_time(path: Path) -> tuple[float, str]:
    """Return (epoch_seconds, source_label).

    Preference order:
      1) Filename timestamp prefix (preferred)
      2) Filesystem created time where reliably available
      3) Filesystem modified time
    """
    # Preferred: filename timestamp prefix like:
    #   YYYY-MM-DD_HHMMZ_...
    #   YYYY-MM-DD_HHMM_...
    # (seconds optional, and optional trailing 'Z' to indicate UTC)
    m = re.match(r"^(?P<ymd>\d{4}-\d{2}-\d{2})_(?P<hm>\d{4})(?P<sec>\d{2})?(?P<z>Z)?_", path.name)
    if m:
        y, mo, d = (int(x) for x in m.group("ymd").split("-"))
        hh = int(m.group("hm")[:2])
        mm = int(m.group("hm")[2:])
        ss = int(m.group("sec")) if m.group("sec") else 0
        # Interpret Z as UTC; otherwise treat as local machine time (naive -> local)
        if m.group("z"):
            ts = dt.datetime(y, mo, d, hh, mm, ss, tzinfo=dt.timezone.utc)
            return ts.timestamp(), "filename_utc"
        # local
        local_tz = _local_tzinfo()
        ts = dt.datetime(y, mo, d, hh, mm, ss, tzinfo=local_tz)
        return ts.timestamp(), "filename_local"

    st = path.stat()
    # Windows: st_ctime is creation time
    if platform.system().lower().startswith("win"):
        return float(st.st_ctime), "created"
    # macOS: birthtime may exist
    birth = getattr(st, "st_birthtime", None)
    if birth is not None:
        return float(birth), "created"
    # POSIX: no true creation time -> use modified
    return float(st.st_mtime), "modified"


def parse_pairings_lines(lines: list[str]) -> ParsedSchedule | None:
    # This function is used after JSON is read; meta is added later.
    raise AssertionError("use parse_schedule_json()")


def parse_schedule_json(path: Path, tz: dt.tzinfo) -> ParsedSchedule | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    pairings_lines = data.get("pairings_lines")
    if not isinstance(pairings_lines, list) or not all(isinstance(x, str) for x in pairings_lines):
        return None

    ts_epoch, ts_source = file_generation_time(path)
    ts_dt = dt.datetime.fromtimestamp(ts_epoch, tz=tz)
    meta = FileMeta(
        filename=path.name,
        path=str(path),
        ts_source=ts_source,
        ts_iso=ts_dt.isoformat(timespec="seconds"),
        ts_epoch=ts_epoch,
    )

    # Build directed map: p -> q -> (with, against)
    directed: dict[str, dict[str, tuple[int, int]]] = {}
    rests: dict[str, int] = {}
    players_set: set[str] = set()

    for raw in pairings_lines:
        raw = raw.strip()
        if not raw:
            continue
        # The lines may contain HTML (e.g. style='color:cyan'), so splitting on ':'
        # is unsafe. The reliable separator is the rest bracket close followed by ':'
        # e.g. "... [0]: Other [1,2], ..."
        sep = raw.find("]:")
        if sep == -1:
            continue
        left = raw[: sep + 1].strip()
        right = raw[sep + 2 :].strip()

        left_clean = _strip_gender_symbols(_strip_html_tags(left))
        lm = _LEFT_RE.match(left_clean)
        if not lm:
            continue
        p = lm.group("name").strip()
        rests[p] = int(lm.group("rest"))
        players_set.add(p)
        directed.setdefault(p, {})

        if not right:
            continue
        parts = [x.strip() for x in right.split(",") if x.strip()]
        # parts are like: "Anna ♀ [2,1]" but commas also separate counts list items.
        # We need to re-split carefully: tokens end with ']'.
        tokens: list[str] = []
        buf: list[str] = []
        for part in [x.strip() for x in right.split(",")]:
            if not part:
                continue
            buf.append(part)
            if part.endswith("]"):
                tokens.append(", ".join(buf).strip())
                buf = []
        if buf:
            # trailing partial, ignore
            pass

        for tok in tokens:
            tok_clean = _strip_gender_symbols(_strip_html_tags(tok))
            rm = _RIGHT_RE.match(tok_clean)
            if not rm:
                continue
            q = rm.group("name").strip()
            w = int(rm.group("with"))
            a = int(rm.group("against"))
            players_set.add(q)
            directed.setdefault(p, {})[q] = (w, a)

    players = sorted(players_set, key=lambda x: x.lower())

    # Convert to undirected edges (a<b)
    edges: list[tuple[str, str, int, int]] = []
    for i, a in enumerate(players):
        for b in players[i + 1 :]:
            w_a = directed.get(a, {}).get(b)
            w_b = directed.get(b, {}).get(a)
            if w_a is None and w_b is None:
                continue
            w, ag = (w_a or w_b or (0, 0))
            edges.append((a, b, int(w), int(ag)))

    return ParsedSchedule(meta=meta, edges=edges, players=players, rests=rests)


def cutoff_dt_for_day(d: dt.date, tz: dt.tzinfo) -> dt.datetime:
    return dt.datetime(d.year, d.month, d.day, SESSION_CUTOFF_HOUR, SESSION_CUTOFF_MINUTE, tzinfo=tz)


def _iter_session_days(start: dt.date, end: dt.date):
    d = start
    one = dt.timedelta(days=1)
    while d <= end:
        if d.weekday() in SESSION_WEEKDAYS:
            yield d
        d += one


def pick_used_per_session(schedules: list[ParsedSchedule], tz: dt.tzinfo):
    """Assign each file to the *next* Tue/Thu session cutoff (14:00) after its timestamp.

    This matches the real workflow: files are generated/downloaded before the session,
    sometimes days earlier. Example: Wednesday file (no later files) becomes the
    candidate for Thursday's session.
    """
    if not schedules:
        return [], []

    times = [dt.datetime.fromtimestamp(s.meta.ts_epoch, tz=tz) for s in schedules]
    min_d = min(times).date() - dt.timedelta(days=7)
    max_d = max(times).date() + dt.timedelta(days=14)

    cutoffs = [cutoff_dt_for_day(d, tz) for d in _iter_session_days(min_d, max_d)]
    cutoff_epochs = [c.timestamp() for c in cutoffs]

    # session_key = cutoff datetime (unique)
    by_cutoff: dict[dt.datetime, list[ParsedSchedule]] = {c: [] for c in cutoffs}
    unassigned: list[ParsedSchedule] = []

    for s, t in zip(schedules, times):
        idx = bisect.bisect_right(cutoff_epochs, t.timestamp())
        if idx >= len(cutoffs):
            unassigned.append(s)
            continue
        by_cutoff[cutoffs[idx]].append(s)

    sessions = []
    for c in cutoffs:
        cands = by_cutoff.get(c) or []
        if not cands:
            continue
        cands = sorted(cands, key=lambda s: s.meta.ts_epoch)
        used = cands[-1]
        sessions.append(
            {
                "session_date": c.date().isoformat(),
                "cutoff_local": c.isoformat(timespec="minutes"),
                "used": used.meta.filename,
                "candidates": [x.meta.filename for x in cands],
            }
        )

    return sessions, unassigned


def load_regular_players_from_app_py(app_py_path: Path) -> list[str]:
    """Extract REGULAR_PLAYERS names from app.py.

    Keeps only the 'name' fields and returns sorted unique names.
    """
    try:
        txt = app_py_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    # Rough but effective for this codebase: pull {"name": "X", ...} occurrences.
    names = re.findall(r'{"name"\s*:\s*"([^"]+)"\s*,\s*"gender"', txt)
    out = sorted(set(names), key=lambda x: x.lower())
    return out


def load_regular_players(cwd: Path) -> list[str]:
    """Load fixed regular-player list.

    Priority:
      1) ./app.py
      2) ../app.py
      3) ./regular_players.txt (one name per line)
    """
    for p in [cwd / "app.py", cwd.parent / "app.py"]:
        if p.exists():
            names = load_regular_players_from_app_py(p)
            if names:
                return names

    rp = cwd / "regular_players.txt"
    if rp.exists():
        names = []
        for ln in rp.read_text(encoding="utf-8", errors="ignore").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                names.append(ln)
        return sorted(set(names), key=lambda x: x.lower())
    return []


def build_dashboard_html(data: dict) -> str:
    data_json = json.dumps(data, ensure_ascii=False)
    # New UI: heatmaps + rested bars (regular players only), with live range filtering.
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mijas Padellers Dashboard</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; color:#111; }
    h1,h2,h3 { margin: 0.4rem 0; }
    .muted { color:#555; }
    .row { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-end; }
    .card { border:1px solid #ddd; border-radius:10px; padding:12px; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.05); }
    .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; border:1px solid #ccc; background:#f5f5f5; }
    .badge.used { background:#e8f5e9; border-color:#a5d6a7; }
    .badge.warn { background:#fff3e0; border-color:#ffcc80; }
    .controls label { font-size: 12px; display:block; margin-bottom:4px; }
    .controls select, .controls input { padding:6px 8px; }
    .inline { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
    .gridbound { width: fit-content; max-width: 100%; }
    .mini { font-size:12px; color:#333; display:flex; gap:6px; align-items:center; white-space:nowrap; }
    .mini input[type="checkbox"] { transform: translateY(1px); }
    .range2 {
      --trackH: 6px;
      --thumb: 20px; /* ~20% larger */
      position: relative;
      height: calc(var(--thumb) + 10px); /* room for hover scaling */
      width: min(560px, 100%);
      flex:1;
      min-width: 280px;
      overflow: visible;
    }
    .range2 .track { position:absolute; left:0; right:0; top:50%; height:var(--trackH); transform: translateY(-50%); background:#e0e0e0; border-radius:999px; pointer-events:none; }
    .range2 .fill { position:absolute; top:50%; height:var(--trackH); transform: translateY(-50%); background:#1976d2; border-radius:999px; left:0; right:0; pointer-events:none; }
    /* Extend the input so thumbs can reach the visible track ends */
    .range2 input[type="range"] { position:absolute; top:50%; transform: translateY(-50%); left: calc(var(--thumb) / -2); width: calc(100% + var(--thumb)); height: var(--thumb); margin:0; padding:0; background:transparent; pointer-events:all; -webkit-appearance:none; appearance:none; }
    .range2 input[type="range"]::-webkit-slider-thumb { -webkit-appearance:none; width:var(--thumb); height:var(--thumb); border-radius:50%; background:#1976d2; border:2px solid #145ea8; box-shadow:0 0 0 1px rgba(0,0,0,.05); pointer-events:all; margin-top: 0; transition: transform .12s ease, background .12s ease, border-color .12s ease; }
    .range2 input[type="range"]::-webkit-slider-runnable-track { height:var(--trackH); background:transparent; }
    .range2 input[type="range"]::-moz-range-thumb { width:var(--thumb); height:var(--thumb); border-radius:50%; background:#1976d2; border:2px solid #145ea8; pointer-events:all; transition: transform .12s ease, background .12s ease, border-color .12s ease; }
    .range2 input[type="range"]::-moz-range-track { height:var(--trackH); background:transparent; }
    .range2 input[type="range"]:hover::-webkit-slider-thumb { transform: scale(1.10); background:#2f86dd; border-color:#0f4f8e; }
    .range2 input[type="range"]:hover::-moz-range-thumb { transform: scale(1.10); background:#2f86dd; border-color:#0f4f8e; }
    .range2 input[type="range"]:focus-visible::-webkit-slider-thumb { box-shadow: 0 0 0 3px rgba(25,118,210,.25), 0 0 0 1px rgba(0,0,0,.05); }
    .range2 input[type="range"]:focus-visible::-moz-range-thumb { box-shadow: 0 0 0 3px rgba(25,118,210,.25); }
    /* Stack heatmaps vertically (Played Against below Played With) */
    .two-col { display:grid; grid-template-columns: 1fr; gap:12px; }
    canvas { width: 100%; height: 140px; }
    details summary { cursor: pointer; }
    code { background:#f6f6f6; padding:2px 4px; border-radius:6px; }

    .seg { display:inline-flex; border:1px solid #ddd; border-radius:10px; overflow:hidden; }
    .seg button { border:0; background:#fff; padding:6px 10px; cursor:pointer; font-size:12px; }
    .seg button + button { border-left:1px solid #ddd; }
    .seg button.active { background:#1976d2; color:#fff; }

    /* No internal grid scrolling: page scrolls instead */
    .heatwrap { overflow: visible; max-height: none; padding-right: 0; }
    .heatframe { display:inline-block; border:1px solid #eee; border-radius:10px; overflow:hidden; }
    :root { --cell: 14px; --xlab: 84px; }
    table.heatmap { border-collapse: collapse; width: max-content; min-width: 0; display:inline-block; }
    .heatmap { font-size: 11px; }
    .heatmap th, .heatmap td { border:1px solid #ddd; padding:0; box-sizing:border-box; font-weight: 400; vertical-align: middle; }
    /* Y-axis labels */
    .heatmap th.rowhdr { position: sticky; left: 0; z-index: 3; text-align:left; background:#f3f3f3; }
    .heatmap td.rowhdr {
      position: sticky;
      left: 0;
      z-index: 1;
      background:#fff;
      font-weight: 400;
      white-space: nowrap;
      padding: 0 3px;
      height: var(--cell);
      line-height: var(--cell);
      font-size: 12px;
    }
    .heatmap td { text-align: center; font-variant-numeric: tabular-nums; }
    /* Bottom X-axis labels + Rested row */
    .heatmap tfoot th { background:#f3f3f3; text-align:center; }
    .heatmap tfoot td { background:#fff; text-align:center; }
    .heatmap th.colhdr { width: var(--cell); height: var(--xlab); padding:0; position: relative; overflow: hidden; }
    .heatmap th.colhdr .rot {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: flex-start;       /* top of the header cell */
      justify-content: center;       /* centered in the column */
      padding-top: 4px;
      writing-mode: vertical-rl;     /* stable vertical text layout */
      transform: rotate(180deg);     /* make it read bottom->top visually */
      white-space: nowrap;
      font-size: 10px;
      font-weight: 400;
      pointer-events: none;
    }
    /* Square data cells */
    .heatmap td.cell, .heatmap td.diag {
      width: var(--cell);
      height: var(--cell);
      min-width: var(--cell);
      line-height: var(--cell);
      font-size: 10px;
    }
    .heatmap td.diag { background:#b7d5ff; color: transparent; }
    .legend { display:flex; align-items:center; gap:10px; margin-top:8px; flex-wrap:wrap; }
    .grad { height:10px; width:180px; border-radius:999px; background: linear-gradient(90deg, #e8f5ff, #ffd9c8, #d7301f); border:1px solid #ddd; }
    .bars { overflow:auto; }
    .bars svg { width: max(900px, 100%); height: 260px; }
  </style>
</head>
<body>
  <h1>Mijas Padellers Dashboard</h1>
  <div class="muted">
    Built at: <code id="builtAt"></code>
  </div>

  <div class="row" style="margin-top:12px;">
    <div class="card controls" style="flex:1; min-width:320px;">
      <div class="inline gridbound" id="controlsBound">
            <div class="range2">
              <div class="track"></div>
              <div class="fill" id="rangeFill"></div>
              <input type="range" id="rangeMin" min="0" max="0" value="0" />
              <input type="range" id="rangeMax" min="0" max="0" value="0" />
            </div>
            <span class="muted" id="rangeLabel"></span>
            <label class="mini" style="margin-left:auto;">
              <input type="checkbox" id="includeCandidates" />
              Include all candidates
            </label>
      </div>
    </div>

  </div>

  <div class="card" style="margin-top:12px;">
    <div class="row gridbound" id="modesBound" style="align-items:center; justify-content:space-between;">
      <h3 id="gridTitle" style="margin:0;">Played With (regulars)</h3>
      <div class="inline" style="gap:10px; justify-content:flex-end;">
        <div class="seg" role="group" aria-label="Heatmap mode">
          <button id="modeWithBtn" type="button" class="active">With</button>
          <button id="modeAgainstBtn" type="button">Against</button>
        </div>
        <div class="seg" role="group" aria-label="Metric mode">
          <button id="metricRawBtn" type="button" class="active">Raw</button>
          <button id="metricNormBtn" type="button">Norm</button>
        </div>
      </div>
    </div>
    <div class="muted" style="margin-top:6px;">Shortcuts: <code>T</code> toggle with/against · <code>N</code> normalize · <code>C</code> include candidates</div>
    <div class="heatwrap" id="heatGrid"></div>
    <div class="legend"><span class="muted">cool</span><div class="grad"></div><span class="muted">hot</span> <span class="muted" id="heatMinMax"></span></div>
  </div>

  <details class="card" style="margin-top:12px;">
    <summary><b>Sessions & file audit</b> <span class="muted">(collapsed)</span></summary>
    <div style="margin-top:10px;">
      <h3 style="margin:0;">Selection logic</h3>
      <div class="muted" style="margin-top:6px;">
        Tue/Thu sessions. Each file is assigned to the next session cutoff at <code>14:00</code> local.
        Used schedule is the latest file assigned to that session.
      </div>
      <div class="muted" id="regularNote" style="margin-top:6px;"></div>
    </div>
    <div id="audit" style="margin-top:10px;"></div>
  </details>

  <script>
  const DATA = __EMBEDDED_DATA__;

  function byId(x) { return document.getElementById(x); }
  let gridMode = 'with'; // 'with' | 'against'
  let metricMode = 'raw'; // 'raw' | 'norm'
  let gridData = null;   // { usedPlayers, withM, againstM, withN, againstN, maxRawBoth, maxNormBoth, rests, restsNorm }

  function fmtDateUK(isoYmd) {
    // "YYYY-MM-DD" -> "DD-MM-YYYY"
    if (!isoYmd || isoYmd.length < 10) return isoYmd || '';
    const y = isoYmd.slice(0, 4), mo = isoYmd.slice(5, 7), d = isoYmd.slice(8, 10);
    if (isoYmd[4] !== '-' || isoYmd[7] !== '-') return isoYmd;
    return `${d}-${mo}-${y}`;
  }

  function fmtTsUK(isoTs) {
    // "YYYY-MM-DDTHH:MM...(+HH:MM|Z)" -> "DD-MM-YYYY HH:MM +HH:MM"
    if (!isoTs || isoTs.length < 16) return isoTs || '';
    const date = isoTs.slice(0, 10);
    const time = isoTs.slice(11, 16);
    if (isoTs[10] !== 'T') return isoTs;
    let off = '';
    if (isoTs.endsWith('Z')) off = '+00:00';
    else {
      const pPlus = isoTs.lastIndexOf('+');
      const pMinus = isoTs.lastIndexOf('-');
      const p = Math.max(pPlus, pMinus);
      if (p > 15 && isoTs.length - p >= 6) off = isoTs.slice(p, p + 6);
    }
    return `${fmtDateUK(date)} ${time}${off ? ' ' + off : ''}`;
  }

  function setGridMode(mode) {
    gridMode = mode === 'against' ? 'against' : 'with';
    const withBtn = byId('modeWithBtn');
    const againstBtn = byId('modeAgainstBtn');
    if (withBtn && againstBtn) {
      withBtn.classList.toggle('active', gridMode === 'with');
      againstBtn.classList.toggle('active', gridMode === 'against');
    }
    const title = byId('gridTitle');
    if (title) title.textContent = gridMode === 'with' ? 'Played With (regulars)' : 'Played Against (regulars)';
    renderCurrentGrid();
  }

  function setMetricMode(mode) {
    metricMode = mode === 'norm' ? 'norm' : 'raw';
    const rawBtn = byId('metricRawBtn');
    const normBtn = byId('metricNormBtn');
    if (rawBtn && normBtn) {
      rawBtn.classList.toggle('active', metricMode === 'raw');
      normBtn.classList.toggle('active', metricMode === 'norm');
    }
    renderCurrentGrid();
  }

  function computeIncludedFiles(includeCandidates, startIdx, endIdx) {
    const sessions = DATA.sessions;
    const used = [];
    const cands = [];
    sessions.forEach((s, idx) => {
      const d = new Date(s.session_date + 'T00:00:00');
      used.push({idx, date: d, file: s.used});
      s.candidates.forEach(fn => cands.push({idx, date: d, file: fn}));
    });

    let rows = includeCandidates ? cands : used;
    const a = Math.min(startIdx, endIdx);
    const b = Math.max(startIdx, endIdx);
    rows = rows.filter(r => r.idx >= a && r.idx <= b);
    return rows.map(r => r.file);
  }

  function colorFor(value, max) {
    if (max <= 0) return '#fff';
    const t = Math.min(1, Math.max(0, value / max));
    const stops = [
      {t:0.0, c:[232,245,255]},  // #e8f5ff
      {t:0.6, c:[255,217,200]},  // #ffd9c8
      {t:1.0, c:[215,48,31]}     // #d7301f
    ];
    let a = stops[0], b = stops[stops.length-1];
    for (let i=0;i<stops.length-1;i++){
      if (t >= stops[i].t && t <= stops[i+1].t){ a=stops[i]; b=stops[i+1]; break; }
    }
    const u = (t - a.t) / (b.t - a.t || 1);
    const r = Math.round(a.c[0] + (b.c[0]-a.c[0])*u);
    const g = Math.round(a.c[1] + (b.c[1]-a.c[1])*u);
    const bl = Math.round(a.c[2] + (b.c[2]-a.c[2])*u);
    return `rgb(${r},${g},${bl})`;
  }

  function renderHeatmap(containerId, playersOrdered, matrix, maxVal, modeLabel, restsMap) {
    const el = byId(containerId);
    if (!playersOrdered.length) {
      el.innerHTML = '<div class="muted">No regular players loaded.</div>';
      return;
    }
    let html = '<div class="heatframe"><table class="heatmap"><tbody>';
    for (let i=0;i<playersOrdered.length;i++){
      const rp = playersOrdered[i];
      html += `<tr><td class="rowhdr">${rp}</td>`;
      for (let j=0;j<playersOrdered.length;j++){
        if (i===j){
          html += `<td class="diag"></td>`;
          continue;
        }
        const v = matrix[i][j] || 0;
        const bg = colorFor(v, maxVal);
        const lbl = modeLabel || 'vs';
        const show = (metricMode === 'raw') ? (v ? String(v) : '') : '';
        const tipVal = (metricMode === 'raw') ? v : (Math.round(v * 1000) / 1000);
        html += `<td class="cell" title="${rp} ${lbl} ${playersOrdered[j]}: ${tipVal}" style="background:${bg};">${show}</td>`;
      }
      html += '</tr>';
    }
    html += '</tbody><tfoot><tr><th class="rowhdr">Player</th>';
    playersOrdered.forEach(p => { html += `<th class="colhdr"><span class="rot">${p}</span></th>`; });
    html += '</tr>';
    html += '<tr><td class="rowhdr">Rested</td>';
    playersOrdered.forEach(p => {
      const v = restsMap ? (restsMap.get(p) || 0) : 0;
      const show = (metricMode === 'raw') ? String(v) : String(Math.round(v * 100) / 100);
      html += `<td class="cell restcell" title="${p} rested: ${show}">${show}</td>`;
    });
    html += '</tr></tfoot></table></div>';
    el.innerHTML = html;
  }

  function renderCurrentGrid() {
    if (!gridData) return;
    const m =
      (metricMode === 'norm')
        ? (gridMode === 'against' ? gridData.againstN : gridData.withN)
        : (gridMode === 'against' ? gridData.againstM : gridData.withM);
    const lbl = gridMode === 'against' ? 'against' : 'with';
    const maxVal = (metricMode === 'norm') ? gridData.maxNormBoth : gridData.maxRawBoth;
    const restsMap = (metricMode === 'norm') ? gridData.restsNorm : gridData.rests;
    renderHeatmap('heatGrid', gridData.usedPlayers, m, maxVal, lbl, restsMap);
    const mm = byId('heatMinMax');
    if (mm) mm.textContent = `scale min 0 → max ${metricMode === 'norm' ? (Math.round(maxVal * 1000) / 1000) : maxVal}`;
  }

  function renderRests(containerId, playersOrdered, restsMap) {
    const el = byId(containerId);
    const counts = playersOrdered.map(p => restsMap.get(p) || 0);
    const maxVal = Math.max(0, ...counts);
    const w = Math.max(900, playersOrdered.length * 22 + 180);
    const h = 260;
    const padL = 60, padR = 20, padT = 20, padB = 80;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    const bw = plotW / Math.max(1, playersOrdered.length);
    let svg = `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMinYMin meet">`;
    svg += `<line x1="${padL}" y1="${padT+plotH}" x2="${padL+plotW}" y2="${padT+plotH}" stroke="#999" />`;
    playersOrdered.forEach((p, i) => {
      const v = counts[i];
      const bh = maxVal ? (v / maxVal) * plotH : 0;
      const x = padL + i * bw + 2;
      const y = padT + plotH - bh;
      svg += `<rect x="${x}" y="${y}" width="${Math.max(1, bw-4)}" height="${bh}" fill="#1976d2" opacity="0.85"><title>${p}: rested ${v}</title></rect>`;
      const tx = padL + i * bw + bw/2;
      svg += `<text x="${tx}" y="${padT+plotH+14}" font-size="10" text-anchor="end" transform="rotate(-60 ${tx} ${padT+plotH+14})">${p}</text>`;
    });
    svg += `<text x="${padL-10}" y="${padT+12}" font-size="11" text-anchor="end">rest</text>`;
    svg += `</svg>`;
    el.innerHTML = `<div class="muted" style="margin-bottom:6px;">Max rested in range: ${maxVal}</div>` + svg;
  }

  function renderAudit() {
    const el = byId('audit');
    if (DATA.sessions.length === 0) {
      el.innerHTML = '<div class="muted">No sessions found in this folder.</div>';
      return;
    }
    let html = '';
    DATA.sessions.forEach((s) => {
      html += `<details style="margin-bottom:8px;"><summary><span class="badge used">Used</span> <b>${fmtDateUK(s.session_date)}</b> → <code>${s.used}</code> <span class="muted">(cutoff ${fmtTsUK(s.cutoff_local)})</span></summary>`;
      html += '<div style="padding:8px 0 0 0;">';
      html += '<div class="muted">Candidates considered for this session window:</div>';
      html += '<ul style="margin-top:6px;">';
      s.candidates.forEach(fn => {
        const meta = DATA.files[fn]?.meta;
        const ts = meta ? fmtTsUK(meta.ts_iso) : 'unknown';
        const src = meta ? meta.ts_source : 'unknown';
        const badge = fn === s.used ? '<span class="badge used">Used schedule picked for session</span>' : '<span class="badge">Candidate</span>';
        html += `<li>${badge} <code>${fn}</code> <span class="muted">(${ts}, ${src})</span></li>`;
      });
      html += '</ul>';
      html += '</div></details>';
    });
    el.innerHTML = html;
  }

  function refresh() {
    byId('builtAt').textContent = DATA.built_at_display || DATA.built_at;

    const includeCandidates = byId('includeCandidates').checked;
    const startIdxRaw = parseInt(byId('rangeMin').value, 10);
    const endIdxRaw = parseInt(byId('rangeMax').value, 10);
    const startIdx = startIdxRaw;
    const endIdx = endIdxRaw;

    // Keep thumbs from crossing
    if (startIdxRaw > endIdxRaw) {
      byId('rangeMax').value = String(startIdxRaw);
    }
    const included = computeIncludedFiles(includeCandidates, startIdx, endIdx);

    const regularPlayers = DATA.regular_players || [];
    const regularSet = new Set(regularPlayers);
    if (regularPlayers.length) {
      byId('regularNote').textContent = `Regulars loaded: ${regularPlayers.length} (guests ignored).`;
    } else {
      byId('regularNote').textContent = "No regular list found (put app.py or regular_players.txt next to build_dashboard.py). Falling back to all names in data.";
    }

    // Aggregate directly from included files
    const playersOrdered = regularPlayers.length ? regularPlayers : [];
    const idxMap = new Map();
    const usedPlayers = playersOrdered.length ? playersOrdered : [];
    // If no regular list, gather from included files
    if (!playersOrdered.length) {
      const set = new Set();
      included.forEach(fn => {
        const f = DATA.files[fn];
        if (!f) return;
        (f.players || []).forEach(p => set.add(p));
      });
      usedPlayers.push(...Array.from(set).sort((a,b)=>a.localeCompare(b)));
    }
    usedPlayers.forEach((p,i)=>idxMap.set(p,i));

    const n = usedPlayers.length;
    const withM = Array.from({length:n}, ()=>Array.from({length:n}, ()=>0));
    const againstM = Array.from({length:n}, ()=>Array.from({length:n}, ()=>0));
    const rests = new Map();
    const attendance = new Map();
    usedPlayers.forEach(p => attendance.set(p, 0));
    let maxWith = 0, maxAgainst = 0;

    included.forEach(fn => {
      const f = DATA.files[fn];
      if (!f) return;
      (f.edges || []).forEach(e => {
        const a = e[0], b = e[1], w = e[2], ag = e[3];
        if (regularPlayers.length && (!regularSet.has(a) || !regularSet.has(b))) return;
        if (!idxMap.has(a) || !idxMap.has(b)) return;
        const i = idxMap.get(a), j = idxMap.get(b);
        withM[i][j] += w; withM[j][i] += w;
        againstM[i][j] += ag; againstM[j][i] += ag;
        if (withM[i][j] > maxWith) maxWith = withM[i][j];
        if (againstM[i][j] > maxAgainst) maxAgainst = againstM[i][j];
      });
      const r = f.rests || {};
      Object.keys(r).forEach(p => {
        if (regularPlayers.length && !regularSet.has(p)) return;
        if (!idxMap.has(p)) return;
        rests.set(p, (rests.get(p) || 0) + (r[p] || 0));
      });
      // Attendance: count sessions a player appears in (using rests keys is robust)
      Object.keys(r).forEach(p => {
        if (regularPlayers.length && !regularSet.has(p)) return;
        if (!attendance.has(p)) return;
        attendance.set(p, (attendance.get(p) || 0) + 1);
      });
    });

    const maxRawBoth = Math.max(maxWith, maxAgainst);
    // Normalized by row player's attendance (may be asymmetric)
    const withN = Array.from({length:n}, ()=>Array.from({length:n}, ()=>0));
    const againstN = Array.from({length:n}, ()=>Array.from({length:n}, ()=>0));
    const restsNorm = new Map();
    let maxNormBoth = 0;
    for (let i=0;i<n;i++){
      const p = usedPlayers[i];
      const denom = attendance.get(p) || 0;
      if (denom <= 0) { restsNorm.set(p, 0); continue; }
      restsNorm.set(p, (rests.get(p) || 0) / denom);
      for (let j=0;j<n;j++){
        if (i===j) continue;
        withN[i][j] = withM[i][j] / denom;
        againstN[i][j] = againstM[i][j] / denom;
        if (withN[i][j] > maxNormBoth) maxNormBoth = withN[i][j];
        if (againstN[i][j] > maxNormBoth) maxNormBoth = againstN[i][j];
      }
    }
    gridData = { usedPlayers, withM, againstM, withN, againstN, maxRawBoth, maxNormBoth, rests, restsNorm };
    renderCurrentGrid();
    syncToGridWidth();
    if (DATA.sessions.length) {
      const a = Math.min(startIdx, endIdx);
      const b = Math.max(startIdx, endIdx);
      byId('rangeLabel').textContent = `#${a+1}–#${b+1} (${fmtDateUK(DATA.sessions[a].session_date)}→${fmtDateUK(DATA.sessions[b].session_date)})`;
    } else {
      byId('rangeLabel').textContent = '(no sessions)';
    }

    // Update slider fill
    updateRangeFill();
  }

  function updateRangeFill() {
    const fill = byId('rangeFill');
    const n = DATA.sessions.length;
    if (!fill || n <= 1) return;
    const min = parseInt(byId('rangeMin').value, 10);
    const max = parseInt(byId('rangeMax').value, 10);
    const lo = Math.min(min, max) / (n - 1);
    const hi = Math.max(min, max) / (n - 1);
    fill.style.left = (lo * 100) + '%';
    fill.style.right = (100 - hi * 100) + '%';
  }

  function syncToGridWidth() {
    const frame = document.querySelector('#heatGrid .heatframe');
    const w = frame ? frame.getBoundingClientRect().width : 0;
    if (!w) return;
    const px = Math.ceil(w) + 'px';
    const controls = byId('controlsBound');
    const modes = byId('modesBound');
    if (controls) controls.style.width = px;
    if (modes) modes.style.width = px;
  }

  function init() {
    const n = DATA.sessions.length;
    byId('rangeMin').max = Math.max(0, n-1);
    byId('rangeMax').max = Math.max(0, n-1);
    byId('rangeMin').value = 0;
    byId('rangeMax').value = Math.max(0, n-1);
    // Ensure handles are always draggable and ordered
    byId('rangeMin').addEventListener('input', () => {
      const a = parseInt(byId('rangeMin').value, 10);
      const b = parseInt(byId('rangeMax').value, 10);
      if (a > b) byId('rangeMax').value = String(a);
      updateRangeFill();
    });
    byId('rangeMax').addEventListener('input', () => {
      const a = parseInt(byId('rangeMin').value, 10);
      const b = parseInt(byId('rangeMax').value, 10);
      if (b < a) byId('rangeMin').value = String(b);
      updateRangeFill();
    });

    ['includeCandidates','rangeMin','rangeMax'].forEach(id => {
      byId(id).addEventListener('input', refresh);
      byId(id).addEventListener('change', refresh);
    });
    byId('modeWithBtn').addEventListener('click', () => setGridMode('with'));
    byId('modeAgainstBtn').addEventListener('click', () => setGridMode('against'));
    byId('metricRawBtn').addEventListener('click', () => setMetricMode('raw'));
    byId('metricNormBtn').addEventListener('click', () => setMetricMode('norm'));
    document.addEventListener('keydown', (e) => {
      // Ignore typing in inputs/selects
      const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
      if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
      const k = (e.key || '').toLowerCase();
      if (k === 't') setGridMode(gridMode === 'with' ? 'against' : 'with');
      else if (k === 'n') setMetricMode(metricMode === 'raw' ? 'norm' : 'raw');
      else if (k === 'c') {
        const cb = byId('includeCandidates');
        cb.checked = !cb.checked;
        refresh();
      }
    });

    renderAudit();
    setGridMode('with');
    setMetricMode('raw');
    updateRangeFill();
    refresh();
    // After first render, align controls to grid
    setTimeout(syncToGridWidth, 0);
    window.addEventListener('resize', () => { syncToGridWidth(); });
  }

  init();
  </script>
</body>
</html>
"""
    return html.replace("__EMBEDDED_DATA__", data_json)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tz", default="local", help="Timezone (IANA like Europe/Madrid) or 'local' (default).")
    ap.add_argument("--copy-renamed", action="store_true",
                    help="Copy JSON files to a timestamped name YYYY-MM-DD_HHMM_<original>.json (optional).")
    args = ap.parse_args()

    tz, tz_label = resolve_tz(args.tz)

    cwd = Path(".").resolve()
    json_files = sorted([p for p in cwd.iterdir() if p.is_file() and p.suffix.lower() == ".json"],
                        key=lambda p: p.name.lower())

    schedules: list[ParsedSchedule] = []
    for p in json_files:
        ps = parse_schedule_json(p, tz)
        if ps:
            schedules.append(ps)

    # Optional copy/rename for readability
    if args.copy_renamed:
        for s in schedules:
            t = dt.datetime.fromtimestamp(s.meta.ts_epoch, tz=tz)
            stamp = t.strftime("%Y-%m-%d_%H%M")
            new_name = f"{stamp}_{s.meta.filename}"
            src = cwd / s.meta.filename
            dst = cwd / new_name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())

    sessions, unassigned = pick_used_per_session(schedules, tz)

    files_dict: dict[str, dict] = {}
    for s in schedules:
        files_dict[s.meta.filename] = {
            "meta": dataclasses.asdict(s.meta),
            "players": s.players,
            "edges": s.edges,
            "rests": s.rests,
        }

    tz_now_iso = dt.datetime.now(tz=tz).isoformat(timespec="seconds")
    built_at_dt = dt.datetime.now(tz=tz)
    built_at_iso = built_at_dt.isoformat(timespec="seconds")
    built_at_display = built_at_dt.strftime("%d-%m-%Y %H:%M %z")
    if len(built_at_display) >= 5:
        built_at_display = built_at_display[:-2] + ":" + built_at_display[-2:]
    regular_players = load_regular_players(cwd)
    if not regular_players:
        print(
            "ERROR: Could not load regular players list. Put app.py next to this script, "
            "or create regular_players.txt (one name per line).",
            file=sys.stderr,
        )
        return 2
    payload = {
        "built_at": built_at_iso,
        "built_at_display": built_at_display,
        "tz_used": tz_label,
        "tz_now_iso": tz_now_iso,
        "sessions": sessions,
        "unassigned": [s.meta.filename for s in unassigned],
        "files": files_dict,
        "regular_players": regular_players,
    }

    out_html = build_dashboard_html(payload)
    (cwd / "dashboard.html").write_text(out_html, encoding="utf-8")
    print(f"Wrote dashboard.html in {cwd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

