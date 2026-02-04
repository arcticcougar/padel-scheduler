#!/usr/bin/env python3
"""
Padel schedule stats dashboard builder (no server required).

Usage (run inside the folder that contains the schedule JSON files):
  python build_dashboard.py

This scans the current directory for JSON files containing:
  { "pairings_lines": [ "Player ♀ [R]: Other ♀ [with,against], ...", ... ] }

It then selects the "Used schedule" for each Tue/Thu session window:
  - consider only files timestamped (from filename) BEFORE 14:00 local time on that day
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

    return ParsedSchedule(meta=meta, edges=edges, players=players)


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


def build_dashboard_html(data: dict) -> str:
    # Self-contained HTML with embedded data + vanilla JS (no f-string to avoid JS braces escaping)
    data_json = json.dumps(data, ensure_ascii=False)
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Padel Schedule Dashboard</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; color:#111; }
    h1,h2,h3 { margin: 0.4rem 0; }
    .muted { color:#555; }
    .row { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-end; }
    .card { border:1px solid #ddd; border-radius:10px; padding:12px; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.05); }
    .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; border:1px solid #ccc; background:#f5f5f5; }
    .badge.used { background:#e8f5e9; border-color:#a5d6a7; }
    .badge.warn { background:#fff3e0; border-color:#ffcc80; }
    table { border-collapse: collapse; width:100%; }
    th, td { border:1px solid #ddd; padding:6px 8px; }
    th { background:#f3f3f3; }
    .controls label { font-size: 12px; display:block; margin-bottom:4px; }
    .controls select, .controls input { padding:6px 8px; }
    .two-col { display:grid; grid-template-columns: 1fr 1fr; gap:12px; }
    @media (max-width: 900px) { .two-col { grid-template-columns:1fr; } }
    canvas { width: 100%; height: 160px; }
    details summary { cursor: pointer; }
    code { background:#f6f6f6; padding:2px 4px; border-radius:6px; }
  </style>
</head>
<body>
  <h1>Padel Schedule Dashboard</h1>
  <div class="muted">
    TZ used: <code id="tzUsed"></code> · Built at: <code id="builtAt"></code>
  </div>

  <div class="row" style="margin-top:12px;">
    <div class="card controls" style="flex:1; min-width:280px;">
      <div class="row">
        <div style="min-width:240px;">
          <label>Player</label>
          <select id="playerSelect" style="min-width:240px;"></select>
        </div>
        <div style="min-width:200px;">
          <label>Scope</label>
          <select id="scopeSelect">
            <option value="all">All-time (Used schedules)</option>
            <option value="ytd">Year-to-date (Used schedules)</option>
            <option value="custom">Custom session range (Used schedules)</option>
          </select>
        </div>
        <div>
          <label>&nbsp;</label>
          <label style="display:flex;gap:8px;align-items:center;">
            <input type="checkbox" id="includeCandidates" />
            Include all candidates (auditing)
          </label>
        </div>
      </div>
      <div id="customRange" style="margin-top:10px; display:none;">
        <div class="row">
          <div style="flex:1; min-width:220px;">
            <label>Start session</label>
            <input type="range" id="rangeStart" min="0" max="0" value="0" />
          </div>
          <div style="flex:1; min-width:220px;">
            <label>End session</label>
            <input type="range" id="rangeEnd" min="0" max="0" value="0" />
          </div>
          <div class="muted" id="rangeLabel" style="min-width:260px;"></div>
        </div>
        <canvas id="timeline"></canvas>
      </div>
    </div>

    <div class="card" style="min-width:280px;">
      <h3>Selection logic</h3>
      <div class="muted">
        Tue/Thu sessions. Used schedule is the latest file timestamp before <code>14:00</code> local on that day.
        Timestamps come from filesystem (created preferred; else modified).
      </div>
    </div>
  </div>

  <div class="two-col" style="margin-top:12px;">
    <div class="card">
      <h3>Played with (selected player)</h3>
      <div id="withTable"></div>
    </div>
    <div class="card">
      <h3>Played against (selected player)</h3>
      <div id="againstTable"></div>
    </div>
  </div>

  <div class="card" style="margin-top:12px;">
    <h3>Sessions & file audit</h3>
    <div id="audit"></div>
  </div>

  <script>
  const DATA = __EMBEDDED_DATA__;

  function byId(x) { return document.getElementById(x); }
  function uniq(arr) { return Array.from(new Set(arr)); }

  function ytdStartDate(tzNowISO) {
    const now = new Date(tzNowISO);
    return new Date(now.getFullYear(), 0, 1);
  }

  function computeIncludedFiles(scope, includeCandidates, customStartIdx, customEndIdx) {
    const sessions = DATA.sessions;
    const used = [];
    const cands = [];
    sessions.forEach((s, idx) => {
      const d = new Date(s.session_date + 'T00:00:00');
      used.push({idx, date: d, file: s.used});
      s.candidates.forEach(fn => cands.push({idx, date: d, file: fn}));
    });

    let rows = includeCandidates ? cands : used;

    if (scope === 'ytd') {
      const start = ytdStartDate(DATA.tz_now_iso);
      rows = rows.filter(r => r.date >= start);
    } else if (scope === 'custom') {
      const a = Math.min(customStartIdx, customEndIdx);
      const b = Math.max(customStartIdx, customEndIdx);
      rows = rows.filter(r => r.idx >= a && r.idx <= b);
    }

    return rows.map(r => r.file);
  }

  function aggregateEdges(fileNames) {
    const edgesByFile = DATA.files;
    const acc = new Map();
    const players = new Set();
    fileNames.forEach(fn => {
      const f = edgesByFile[fn];
      if (!f) return;
      f.players.forEach(p => players.add(p));
      f.edges.forEach(e => {
        const a = e[0], b = e[1], w = e[2], ag = e[3];
        const aa = a < b ? a : b;
        const bb = a < b ? b : a;
        const key = aa + '||' + bb;
        const prev = acc.get(key) || [aa, bb, 0, 0];
        prev[2] += w;
        prev[3] += ag;
        acc.set(key, prev);
      });
    });
    return { players: Array.from(players).sort((x,y)=>x.localeCompare(y)), edges: Array.from(acc.values()) };
  }

  function buildPlayerMaps(agg) {
    const withMap = new Map();
    const againstMap = new Map();
    agg.players.forEach(p => { withMap.set(p, new Map()); againstMap.set(p, new Map()); });
    agg.edges.forEach(([a,b,w,ag]) => {
      withMap.get(a).set(b, (withMap.get(a).get(b)||0) + w);
      withMap.get(b).set(a, (withMap.get(b).get(a)||0) + w);
      againstMap.get(a).set(b, (againstMap.get(a).get(b)||0) + ag);
      againstMap.get(b).set(a, (againstMap.get(b).get(a)||0) + ag);
    });
    return {withMap, againstMap};
  }

  function renderTable(m, player) {
    const mp = m.get(player);
    if (!mp) return '<div class="muted">No data in this range.</div>';
    const rows = Array.from(mp.entries())
      .filter(([_,v]) => v > 0)
      .sort((a,b)=>b[1]-a[1]);
    if (rows.length === 0) return '<div class="muted">No data in this range.</div>';
    let html = '<table><thead><tr><th>Other</th><th style="text-align:right;">Count</th></tr></thead><tbody>';
    rows.forEach(([name, count]) => {
      html += `<tr><td>${name}</td><td style="text-align:right;">${count}</td></tr>`;
    });
    html += '</tbody></table>';
    return html;
  }

  function renderAudit() {
    const el = byId('audit');
    if (DATA.sessions.length === 0) {
      el.innerHTML = '<div class="muted">No Tue/Thu session candidates found (before 14:00) in this folder.</div>';
      return;
    }
    let html = '';
    DATA.sessions.forEach((s) => {
      html += `<details style="margin-bottom:8px;"><summary><span class="badge used">Used</span> <b>${s.session_date}</b> → <code>${s.used}</code> <span class="muted">(cutoff ${s.cutoff_local})</span></summary>`;
      html += '<div style="padding:8px 0 0 0;">';
      html += '<div class="muted">Candidates considered for this session window:</div>';
      html += '<ul style="margin-top:6px;">';
      s.candidates.forEach(fn => {
        const meta = DATA.files[fn]?.meta;
        const ts = meta ? meta.ts_iso : 'unknown';
        const src = meta ? meta.ts_source : 'unknown';
        const badge = fn === s.used ? '<span class="badge used">Used schedule picked for session</span>' : '<span class="badge">Candidate</span>';
        html += `<li>${badge} <code>${fn}</code> <span class="muted">(${ts}, ${src})</span></li>`;
      });
      html += '</ul>';
      html += '</div></details>';
    });

    if (DATA.unassigned.length) {
      html += `<details><summary><span class="badge warn">Unassigned</span> Files not considered (non Tue/Thu or after cutoff)</summary>`;
      html += '<ul style="margin-top:6px;">';
      DATA.unassigned.forEach(fn => {
        const meta = DATA.files[fn]?.meta;
        const ts = meta ? meta.ts_iso : 'unknown';
        const src = meta ? meta.ts_source : 'unknown';
        html += `<li><code>${fn}</code> <span class="muted">(${ts}, ${src})</span></li>`;
      });
      html += '</ul></details>';
    }
    el.innerHTML = html;
  }

  function drawTimeline() {
    const canvas = byId('timeline');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,w,h);

    const sessions = DATA.sessions;
    if (!sessions.length) return;
    const start = parseInt(byId('rangeStart').value, 10);
    const end = parseInt(byId('rangeEnd').value, 10);
    const a = Math.min(start, end);
    const b = Math.max(start, end);

    const pad = 18;
    const barW = Math.max(2, (w - pad*2) / sessions.length);
    for (let i=0;i<sessions.length;i++) {
      const x = pad + i * barW;
      const inRange = i>=a && i<=b;
      ctx.fillStyle = inRange ? '#1976d2' : '#cfd8dc';
      ctx.fillRect(x, h-40, Math.max(1, barW-1), 22);
    }
    ctx.fillStyle = '#555';
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText(sessions[a].session_date, pad, 18);
    ctx.fillText(sessions[b].session_date, pad, 34);
  }

  function refresh() {
    byId('tzUsed').textContent = DATA.tz_used;
    byId('builtAt').textContent = DATA.built_at;

    const scope = byId('scopeSelect').value;
    byId('customRange').style.display = scope === 'custom' ? '' : 'none';

    const includeCandidates = byId('includeCandidates').checked;
    const startIdx = parseInt(byId('rangeStart').value, 10);
    const endIdx = parseInt(byId('rangeEnd').value, 10);

    const included = computeIncludedFiles(scope, includeCandidates, startIdx, endIdx);
    const agg = aggregateEdges(included);
    const maps = buildPlayerMaps(agg);

    const player = byId('playerSelect').value;
    byId('withTable').innerHTML = renderTable(maps.withMap, player);
    byId('againstTable').innerHTML = renderTable(maps.againstMap, player);

    if (scope === 'custom' && DATA.sessions.length) {
      const a = Math.min(startIdx, endIdx);
      const b = Math.max(startIdx, endIdx);
      byId('rangeLabel').textContent = `Sessions ${a+1}..${b+1} (${DATA.sessions[a].session_date} → ${DATA.sessions[b].session_date})`;
      drawTimeline();
    }
  }

  function init() {
    const playerNames = uniq(Object.values(DATA.files).flatMap(f => f.players));
    const sel = byId('playerSelect');
    playerNames.sort((a,b)=>a.localeCompare(b));
    playerNames.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p;
      opt.textContent = p;
      sel.appendChild(opt);
    });
    if (playerNames.length) sel.value = playerNames[0];

    const n = DATA.sessions.length;
    byId('rangeStart').max = Math.max(0, n-1);
    byId('rangeEnd').max = Math.max(0, n-1);
    byId('rangeStart').value = 0;
    byId('rangeEnd').value = Math.max(0, n-1);

    ['playerSelect','scopeSelect','includeCandidates','rangeStart','rangeEnd'].forEach(id => {
      byId(id).addEventListener('input', refresh);
      byId(id).addEventListener('change', refresh);
    });
    window.addEventListener('resize', () => { if (byId('scopeSelect').value === 'custom') drawTimeline(); });

    renderAudit();
    refresh();
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
        }

    tz_now_iso = dt.datetime.now(tz=tz).isoformat(timespec="seconds")
    payload = {
        "built_at": dt.datetime.now(tz=tz).isoformat(timespec="seconds"),
        "tz_used": tz_label,
        "tz_now_iso": tz_now_iso,
        "sessions": sessions,
        "unassigned": [s.meta.filename for s in unassigned],
        "files": files_dict,
    }

    out_html = build_dashboard_html(payload)
    (cwd / "dashboard.html").write_text(out_html, encoding="utf-8")
    print(f"Wrote dashboard.html in {cwd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

