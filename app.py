###############################################################################
#  Mijas Padellers Match Scheduler – Streamlit application                    #
#  Last updated: 23 Oct 2026 – Combined player skills and court preferences    #
###############################################################################
import streamlit as st
import itertools
import numpy as np
import random
from datetime import date, time, timedelta, datetime, timezone
import json
import urllib.request
import urllib.error
import html as _html
import math
from math import comb, factorial

# --------------------------------------------------------------------------- #
#  Page config & compact CSS                                                  #
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Mijas Padellers Match Scheduler",
                   layout="centered",
                   initial_sidebar_state="auto")

COMPACT_CSS = """
<style>
div[data-testid="stProgressBar"] > div[role="progressbar"] {
    background-color: #ff7f50;
}
div[data-testid="stProgressBar"] { margin: 2px 0; padding: 2px 0; }
.css-1n76uvr, .stAlert          { padding: .3rem .5rem!important; margin: .2rem 0!important; }
[data-testid="stMainBlockContainer"] {
    padding-top: 2.5rem; padding-bottom: 2.5rem;
}
[data-testid="stVerticalBlock"] { gap: .5rem; }
[data-testid="stCheckbox"] { margin-top: 0 !important; margin-bottom: 0 !important; }
[data-testid="stMarkdownContainer"] li {
    line-height: 1.35; margin-top: .075rem; margin-bottom: .075rem;
}
[data-testid="stMarkdownContainer"] hr { margin: .6rem 0; }
[data-testid="stExpander"] summary {
    padding: .5rem .75rem; min-height: 2.5rem;
}
[data-testid="stExpanderDetails"] { padding: .6rem; }
.st-key-session_details h1 { font-size: clamp(1.75rem, 4.5vw, 2.25rem); }
[class*="st-key-guest_controls_"] {
    padding-bottom: .75rem; border-bottom: 1px solid rgba(128, 128, 128, .2);
}
.st-key-guest_controls_7 { padding-bottom: 0; border-bottom: 0; }
/* Keep the existing dense grid, with a clear selectable row and keyboard focus. */
:is(.st-key-regular_player_grid, .st-key-regular_court_grid) [data-testid="stCheckbox"] label {
    box-sizing: border-box; width: 100%; min-height: 1.5rem;
    padding: 0 .375rem; border: 0; border-radius: .375rem;
    align-items: center; background: rgba(128, 128, 128, .035);
}
:is(.st-key-regular_player_grid, .st-key-regular_court_grid) [data-testid="stCheckbox"] label:hover {
    background: rgba(128, 128, 128, .1);
}
:is(.st-key-regular_player_grid, .st-key-regular_court_grid) [data-testid="stCheckbox"] label:has(input:checked) {
    background: rgba(59, 130, 246, .12);
    box-shadow: inset 0 0 0 1px rgba(59, 130, 246, .6);
}
:is(.st-key-regular_player_grid, .st-key-regular_court_grid) [data-testid="stCheckbox"] label:has(input:checked) p {
    font-weight: 600;
}
:is(.st-key-regular_player_grid, .st-key-regular_court_grid) [data-testid="stCheckbox"] label:has(input:focus-visible) {
    outline: 2px solid #60a5fa; outline-offset: 2px;
}
@media (max-width: 640px) {
    /* Give mobile options a forgiving tap area, including guest and court radios. */
    [data-testid="stCheckbox"] label[data-baseweb="checkbox"] {
        min-height: 2.75rem; width: 100%; align-items: center;
    }
    [data-testid="stRadio"] label[data-baseweb="radio"] {
        min-height: 2.75rem; min-width: 2.75rem; align-items: center;
    }
    [class*="st-key-guest_controls_"] > [data-testid="stHorizontalBlock"] {
        display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .5rem .75rem;
    }
    [class*="st-key-guest_controls_"] [data-testid="stColumn"] {
        min-width: 0 !important; width: 100% !important;
    }
    [class*="st-key-guest_controls_"] [data-testid="stColumn"]:first-child,
    [class*="st-key-guest_controls_"] [data-testid="stColumn"]:last-child { grid-column: 1 / -1; }
    .st-key-session_settings [data-testid="stHorizontalBlock"] {
        display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .5rem .75rem;
    }
    .st-key-session_settings [data-testid="stColumn"] {
        min-width: 0 !important; width: 100% !important;
    }
    .st-key-session_settings [data-testid="stColumn"]:first-child { grid-column: 1 / -1; }
    .st-key-regular_player_grid > [data-testid="stHorizontalBlock"],
    .st-key-regular_court_grid > [data-testid="stHorizontalBlock"] {
        display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .75rem;
    }
    .st-key-regular_player_grid > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    .st-key-regular_court_grid > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important; min-width: 0 !important;
    }
    .st-key-regular_player_grid [data-testid="stVerticalBlock"],
    .st-key-regular_court_grid [data-testid="stVerticalBlock"] { gap: .125rem; }
    .st-key-regular_player_grid [data-testid="stCheckbox"] label,
    .st-key-regular_court_grid [data-testid="stCheckbox"] label {
        min-height: 2.75rem; width: 100%; align-items: center;
    }
    [data-testid="stButton"] button,
    [data-testid="stDownloadButton"] button { width: 100%; min-height: 2.75rem; }
}
@media (max-width: 400px) {
    .st-key-session_settings [data-testid="stHorizontalBlock"] { grid-template-columns: minmax(0, 1fr); }
}
</style>
"""
st.markdown(COMPACT_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
#  Helper functions                                                           #
# --------------------------------------------------------------------------- #
def initialize_matrices(n):
    return (np.zeros((n, n), int), np.zeros((n, n), int))

def initialize_rest_tracker(n):
    return np.zeros(n, int)

def compute_max_unique_matchups(N, K, S):
    M = K * S
    return 0 if N < M else comb(N, M) * factorial(M) // ((factorial(S) ** K) * factorial(K))

def canonical_config(group):
    """Return an order-invariant tuple identifying a 4-player matchup."""
    t1 = tuple(sorted(group[:2])); t2 = tuple(sorted(group[2:4]))
    return tuple(sorted([t1, t2]))

# --------------------------------------------------------------------------- #
#  Match-quality scoring & search                                             #
# --------------------------------------------------------------------------- #
def evaluate_match(groups, teammate_mtx, opponent_mtx, reject_mixed, genders,
                   available, court_size, enable_skill, skills, skill_wt,
                   history=None):
    score = 0
    for g in groups:
        t1, t2 = g[:2], g[2:]
        team_score = sum(teammate_mtx[i][j] for i, j in itertools.combinations(t1, 2)) + \
                     sum(teammate_mtx[i][j] for i, j in itertools.combinations(t2, 2))
        opp_score  = sum(opponent_mtx[i][j] for i, j in itertools.product(t1, t2))
        score += team_score + opp_score + (team_score + opp_score)  # diversity penalty

    if enable_skill:
        for g in groups:
            a1 = (skills[g[0]] + skills[g[1]]) / 2
            a2 = (skills[g[2]] + skills[g[3]]) / 2
            score += skill_wt * abs(a1 - a2)

    if reject_mixed and len(available) >= court_size:
        for g in groups:
            gs = [genders[i] for i in g]
            if len(set(gs)) > 1:
                pool = [genders[i] for i in available]
                if pool.count('F') >= court_size or pool.count('M') >= court_size:
                    score += 1000

    if history:
        for g in groups:
            if canonical_config(g) in history:
                score += 10000

    return score


def find_best_match(players, n_courts, court_sz, teammate_mtx, opponent_mtx,
                    match_no, samples=100000, reject_mixed=False,
                    genders=None, enable_skill=False, skills=None, skill_wt=20,
                    history=None, match_offset: int = 0):
    if len(players) != n_courts * court_sz or len(set(players)) != len(players):
        raise ValueError("Every active court must have four different players.")
    if n_courts == 0:
        return []
    best, best_score = None, float('inf')
    bar = st.progress(0); status = st.empty()
    for i in range(samples):
        shuffled = players[:]; random.shuffle(shuffled)
        groups = [tuple(shuffled[j*court_sz:(j+1)*court_sz]) for j in range(n_courts)]
        if len(set(itertools.chain(*groups))) != len(players):
            continue
        s = evaluate_match(groups, teammate_mtx, opponent_mtx, reject_mixed, genders,
                           players, court_sz, enable_skill, skills, skill_wt,
                           history)
        if s < best_score:
            best, best_score = groups, s
        if (i+1) % 1000 == 0:
            bar.progress(int((i+1)/samples*100))
            status.write(f"Match {match_no+1+match_offset}: sample {(i+1):,}/{samples:,}")
    bar.progress(100); status.write(f"Match {match_no+1+match_offset} best match found!")
    return best


def update_matrices_for_match(groups, teammate_mtx, opponent_mtx):
    for g in groups:
        t1, t2 = g[:2], g[2:]
        for i, j in itertools.combinations(t1, 2):
            teammate_mtx[i][j] += 1; teammate_mtx[j][i] += 1
        for i, j in itertools.combinations(t2, 2):
            teammate_mtx[i][j] += 1; teammate_mtx[j][i] += 1
        for i in t1:
            for j in t2:
                opponent_mtx[i][j] += 1; opponent_mtx[j][i] += 1


def select_bench_players(players, rest_track, n_bench):
    return sorted(players, key=lambda x: rest_track[x])[:n_bench]


def determine_courts_to_use(n_players, available_courts, court_sz=4):
    return min(available_courts, n_players // court_sz)


def deduplicate_names(names):
    seen, out = {}, []
    for n in names:
        if n not in seen:
            seen[n] = 1; out.append(n)
        else:
            seen[n] += 1
            new = f"{n} ({seen[n]})"
            while new in seen:
                seen[new] = 1; new = f"{n} ({seen[new]})"
            seen[new] = 1; out.append(new)
    return out

# --------------------------------------------------------------------------- #
#  Optional upload: send Statistics HTML to Google Drive via webhook           #
# --------------------------------------------------------------------------- #
def upload_stats_html_to_drive(stats_html: str, filename: str, meta: dict | None = None) -> None:
    """Best-effort upload of stats HTML to a configured webhook.

    Requires Streamlit secrets:
      - STATS_WEBHOOK_URL
      - STATS_WEBHOOK_TOKEN
    """
    try:
        url = st.secrets.get("STATS_WEBHOOK_URL")
        token = st.secrets.get("STATS_WEBHOOK_TOKEN")
    except Exception:
        return

    if not url or not token:
        return

    payload = {"token": token, "filename": filename, "html": stats_html}
    if meta:
        payload["meta"] = meta

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except Exception:
        # Silent failure to preserve existing UX.
        return


def extract_pairings_lines_from_stats_html(stats_html: str) -> list[str]:
    """Extract just the per-player pairing lines from the stats <pre> block."""
    try:
        pre_i = stats_html.find("<pre")
        if pre_i == -1:
            return []
        gt_i = stats_html.find(">", pre_i)
        if gt_i == -1:
            return []
        end_i = stats_html.find("</pre>", gt_i)
        if end_i == -1:
            return []
        raw = stats_html[gt_i + 1:end_i]
        text = _html.unescape(raw)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        out = []
        for ln in lines:
            if ln.startswith("Teammate Total:") or ln.startswith("Opponent Total:") or \
               ln.startswith("Teammate Count Frequencies:") or ln.startswith("Opponent Count Frequencies:") or \
               ln.startswith("All-Male Games:") or ln.startswith("All-Female Games:") or ln.startswith("Mixed Games:"):
                break
            out.append(ln)
        return out
    except Exception:
        return []

# -------------------- downstairs rotation helpers --------------------------- #
def identify_downstairs_courts(court_names):
    """Return (downstairs_idx, upstairs_idx) based on courts 12/13 aliases."""
    downstairs_aliases = {"Court 12", "12", "Court 13", "13"}
    downstairs_idx = [i for i, c in enumerate(court_names)
                      if c in downstairs_aliases or c.endswith("12") or c.endswith("13")]
    upstairs_idx = [i for i in range(len(court_names)) if i not in downstairs_idx]
    return downstairs_idx, upstairs_idx

def validate_downstairs_limits(players, total_matches, downstairs_slots, limits=None):
    """Return stay limits or explain why the selected courts cannot be filled."""
    limits = [2] * len(players) if limits is None else list(limits)
    if len(limits) != len(players) or any(n not in (0, 2, 4) for n in limits):
        raise ValueError("Each player must have a downstairs limit of 0, 2 or 4 matches.")
    blocks = math.ceil(total_matches / 2)
    stays = {p: min(limit // 2, blocks) for p, limit in zip(players, limits)}
    eligible = sum(n > 0 for n in stays.values())
    if eligible < downstairs_slots:
        raise ValueError(
            f"The downstairs courts need {downstairs_slots} available players at a time, "
            f"but only {eligible} are available. Adjust Keep upstairs, add players, "
            "or select fewer downstairs courts."
        )
    missing = downstairs_slots * blocks - sum(stays.values())
    if missing > 0:
        can_extend = sum(limit == 2 for limit in limits) if blocks > 1 else 0
        if missing <= can_extend:
            raise ValueError(
                f"To fill the downstairs courts for all {total_matches} matches, "
                f"mark at least {missing} more {'player' if missing == 1 else 'players'} as Keep downstairs "
                "in the player adjustments above, "
                "add available players, or select fewer downstairs courts."
            )
        raise ValueError(
            "These players cannot cover all the downstairs matches within their limits. "
            "Add available players, select fewer downstairs courts, or schedule fewer matches. "
            "Nobody will be assigned more than four downstairs matches."
        )
    return stays


def build_downstairs_rotation(players, total_matches, downstairs_slots, upstairs_slots,
                              limits=None, genders=None, skills=None, enable_skill=False,
                              skill_wt=20, same_sex_indices=()):
    """Plan full two-match stays, respecting each player's firm downstairs limit.

    Compare complete session plans for rest fairness and match quality. Preview
    pairings use the same social/skill/same-sex score as the main match search.
    """
    if not players or total_matches <= 0 or downstairs_slots <= 0:
        return []
    stay_limits = validate_downstairs_limits(players, total_matches, downstairs_slots, limits)
    blocks = math.ceil(total_matches / 2)
    bench_count = max(0, len(players) - downstairs_slots - upstairs_slots)
    reserve_count = min(2 * bench_count, len(players) - downstairs_slots)
    reserved = set(players[:reserve_count])
    base_order = players[downstairs_slots:] + players[:downstairs_slots]
    rng = random.Random(0)
    orders = [base_order]
    for _ in range(127):
        order = players[:]
        rng.shuffle(order)
        orders.append(order)

    best_rotation, best_score = None, None
    for order in orders:
        remaining = stay_limits.copy()
        visits = {p: 0 for p in players}
        rotation = []
        for block in range(blocks):
            eligible = [p for p in order if remaining[p] > 0]
            priority = {p: i for i, p in enumerate(order)}
            eligible.sort(key=lambda p: (
                visits[p], p in reserved if block == 0 else False, priority[p]
            ))
            # With two blocks left, enough players with two remaining stays must
            # play now to leave a full group available for the final block.
            required_twice = max(0, 2 * downstairs_slots - len(eligible)) \
                             if blocks - block == 2 else 0
            required = [p for p in eligible if remaining[p] == 2][:required_twice]
            group = required + [p for p in eligible if p not in required][:downstairs_slots - len(required)]
            downstairs = set(group)
            rotation.append(downstairs)
            for p in downstairs:
                remaining[p] -= 1
                visits[p] += 1

        rests = {p: 0 for p in players}
        downstairs_matches = {p: 0 for p in players}
        tm, om = initialize_matrices(len(players))
        preview_history = set()
        preview_rng = random.Random(1)
        match_quality = 0
        for match_no in range(total_matches):
            downstairs = rotation[match_no // 2]
            upstairs = [p for p in players if p not in downstairs]
            bench = select_bench_players(upstairs, rests, bench_count)
            for p in bench:
                rests[p] += 1
            for p in downstairs:
                downstairs_matches[p] += 1
            pools = [[p for p in players if p in downstairs],
                     [p for p in upstairs if p not in bench]]
            for available in pools:
                if not available:
                    continue
                best_groups, quality = None, float("inf")
                for _ in range(16):
                    shuffled = available[:]
                    preview_rng.shuffle(shuffled)
                    groups = [tuple(shuffled[i:i+4]) for i in range(0, len(shuffled), 4)]
                    score = evaluate_match(
                        groups, tm, om, match_no in same_sex_indices, genders,
                        available, 4, enable_skill, skills, skill_wt, preview_history,
                    )
                    if score < quality:
                        best_groups, quality = groups, score
                match_quality += quality
                update_matrices_for_match(best_groups, tm, om)
                preview_history.update(canonical_config(g) for g in best_groups)
        rest_counts = list(rests.values())
        down_counts = [downstairs_matches[p] for p in players if stay_limits[p] > 0]
        score = (
            max(rest_counts) - min(rest_counts),
            sum(n * n for n in rest_counts),
            sum(n * n for n in down_counts),
            match_quality,
            -sum(len(a & b) for a, b in zip(rotation, rotation[1:])),
        )
        if best_score is None or score < best_score:
            best_rotation, best_score = rotation, score
    return best_rotation

# Determine which match indices (0-based) should prefer same-sex play.
# Strategy: evenly space N indices across 0..total_matches-1, always including the last match.
def compute_same_sex_match_indices(total_matches: int, count: int):
    if count <= 0 or total_matches <= 0:
        return []
    last_idx = total_matches - 1
    # Evenly spaced using rounding; i runs 1..count, guaranteeing last_idx is included for i=count
    raw = [round(i * last_idx / count) for i in range(1, count + 1)]
    # Deduplicate while preserving order
    seen = set(); indices = []
    for idx in raw:
        if 0 <= idx <= last_idx and idx not in seen:
            indices.append(idx); seen.add(idx)
    # Ensure we have exactly 'count' indices by adding earlier slots if duplicates collapsed
    probe = last_idx - 1
    while len(indices) < count and probe >= 0:
        if probe not in seen:
            indices.insert(0, probe); seen.add(probe)
        probe -= 1
    return sorted(indices)

# --------------------------------------------------------------------------- #
#  HTML builders                                                              #
# --------------------------------------------------------------------------- #
def format_court_preference_icon(limit):
    arrow, colour, description = {
        0: ("↑", "#facc15", "Keep upstairs: never assigned a downstairs court."),
        2: ("↕", "#ff9f43", "Standard rotation: up to 2 downstairs matches."),
        4: ("↓", "#ff4b4b", "Keep downstairs: allows up to 4 downstairs matches in two-match stays."),
    }[limit]
    description = _html.escape(description, quote=True)
    return (
        f"<span class='court-preference-icon' style='color:{colour};font-weight:600;"
        "margin-left:.3em;' "
        f"title='{description}' role='img' aria-label='{description}'>{arrow}&#xfe0e;</span>"
    )


def format_downstairs_court_icon():
    return (
        "<span class='downstairs-court-icon' style='color:#ff4b4b;font-weight:600;"
        "margin-left:.3em;' title='Downstairs court' role='img' "
        "aria-label='Downstairs court'>↓&#xfe0e;</span>"
    )


def schedule_background_html():
    """Original, closely packed padel doodles in portable, light-grey vector art."""
    return """
    <svg class="sched-doodles" xmlns="http://www.w3.org/2000/svg"
         viewBox="0 0 884 1223" preserveAspectRatio="none" aria-hidden="true" focusable="false">
      <defs>
        <g id="doodle-racket" fill="none" stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 4C8 3 1 14 3 28c1 10 9 15 13 21v18c0 4 11 4 11 0V49c4-6 12-11 13-21C42 14 36 4 22 4Z"/>
          <path d="M16 49h11M16 54h11M17 61l9 4"/>
          <circle cx="14" cy="18" r="2"/><circle cx="24" cy="15" r="2"/>
          <circle cx="32" cy="21" r="2"/><circle cx="12" cy="28" r="2"/>
          <circle cx="23" cy="25" r="2"/><circle cx="30" cy="32" r="2"/>
          <circle cx="20" cy="36" r="2"/>
          <path d="M8 11c-3 4-4 8-3 12"/>
        </g>
        <pattern id="padel-doodle-tile" width="240" height="240" patternUnits="userSpaceOnUse">
          <g fill="none" stroke="#cdcdcd" stroke-width="1.35" stroke-linecap="round" stroke-linejoin="round">
            <use href="#doodle-racket" transform="translate(12 9) rotate(-18 22 35)"/>
            <use href="#doodle-racket" transform="translate(17 143) rotate(25 22 35) scale(.85)"/>
            <!-- Padel balls with their curved seams. -->
            <g transform="translate(105 20) rotate(25)">
              <circle r="12"/><path d="M-7-10c8 5 8 15 0 20M7-10c-8 5-8 15 0 20"/>
            </g>
            <g transform="translate(222 179) rotate(-20)">
              <circle r="9"/><path d="M-5-7c5 4 5 10 0 14M5-7c-5 4-5 10 0 14"/>
            </g>
            <!-- A glass-sided court and centre net. -->
            <g transform="translate(158 10) rotate(10 26 40)">
              <rect x="0" y="0" width="52" height="80" rx="3"/>
              <path d="M0 40h52M3 36v8m46-8v8M0 21h52M0 59h52M26 21v38M-5 4v21m62 30v21"/>
            </g>
            <!-- Water bottle, with a loose hand-drawn label. -->
            <g transform="translate(86 76) rotate(12 12 27)">
              <path d="M7 4h10v8l5 7v34c0 5-22 5-22 0V19l7-7ZM6 0h12v5H6ZM0 29c8-3 14 3 22 0M0 43c8-3 14 3 22 0"/>
            </g>
            <!-- Court net. -->
            <g transform="translate(134 147) rotate(-8 36 18)">
              <path d="M0 0v43m72-43v43M0 8c24 5 48 5 72 0M0 32h72M12 11v21m12-19v19m12-18v18m12-19v19m12-21v21M0 20h72"/>
            </g>
            <!-- Trainers and towel. -->
            <g transform="translate(85 196) rotate(-12)">
              <path d="M0 18 7 0l16 8 8 10 22 5c8 2 8 13 0 13H5c-7 0-9-11-5-18ZM0 28h58M13 9l12 6m-15 0 13 6m4-4-5 7"/>
            </g>
            <g transform="translate(179 201) rotate(10)">
              <path d="M0 0h44v31H0ZM8 0v31m28-31v31M0 26h44M3 31v4m6-4v4m6-4v4m6-4v4m6-4v4m6-4v4m6-4v4"/>
            </g>
            <!-- Sunny Mijas, plus small movement marks and sparkles. -->
            <g transform="translate(211 116)">
              <circle r="8"/><path d="M0-13v-5m0 31v5m13-18h5m-31 0h-5m27-9 4-4m-22 22-4 4m22-4 4 4m-22-22-4-4"/>
            </g>
            <!-- Smaller sports doodles and bubbles fill the gaps between motifs. -->
            <g transform="translate(14 100) rotate(-12 18 12)">
              <path d="M0 15C3 1 20-4 28 13l10 6c-11 6-25 7-38-4ZM2 14c9 3 17 2 26-1M13 2l-2 11"/>
            </g>
            <g transform="translate(122 45) rotate(-9 13 10)">
              <rect width="26" height="21" rx="3"/><path d="M13 0v21M0 10h26M7 5v1m12-1v1M7 15v1m12-1v1"/>
            </g>
            <g transform="translate(143 116) rotate(55)">
              <circle r="10"/><path d="M-6-8c6 4 6 12 0 16M6-8c-6 4-6 12 0 16"/>
            </g>
            <path d="M61 136c-6-9 5-14 9-7 5-7 14-1 8 6l-8 9ZM117 163c5-7 10-5 9 0-2 6-9 5-7 11m8-41 2 5 5 2-5 2-2 5-2-5-5-2 5-2ZM61 219l7 5-7 5-4-5ZM73 50l2 6 6 2-6 2-2 6-2-6-6-2 6-2ZM163 214l2 4m-9 1 2 4M68 88l5-2m-3 8 5-2M8 232c7-7 15-7 21-2"/>
            <circle cx="74" cy="13" r="3"/><circle cx="134" cy="8" r="4"/>
            <circle cx="234" cy="69" r="3"/><circle cx="130" cy="85" r="3"/>
            <circle cx="74" cy="172" r="4"/><circle cx="158" cy="231" r="3"/>
            <circle cx="7" cy="93" r="2"/><circle cx="235" cy="142" r="3"/>
            <circle cx="234" cy="215" r="2"/><circle cx="232" cy="19" r="3"/>
          </g>
        </pattern>
      </defs>
      <rect width="884" height="1223" fill="url(#padel-doodle-tile)"/>
    </svg>
    """


def format_match_table_html(match_no, groups, court_names, player_names, bench,
                           block_tag=None, downstairs_idx=None, time_label=None,
                           match_offset: int = 0):
    blk = f" – Block {block_tag}" if block_tag else ""
    time_str = f" [{time_label}]" if time_label else ""
    html = f"""
    <div class="match-block" style="margin-bottom:10px;box-shadow:0 2px 4px rgba(0,0,0,.1);
                border-radius:8px;overflow:hidden;background:#fff;">
      <div class="match-content" style="padding:10px;">
        <h2 class="match-title" style="text-align:center;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;
                   margin:0 0 10px 0;">Match {match_no+1+match_offset}{time_str}{blk}</h2>
        <table style="width:100%;border-collapse:collapse;table-layout:fixed;
                      font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;">
          <thead><tr style="background:#f0f0f0;">
            <th style="width:20%;padding:8px;text-align:left;border-bottom:2px solid #ddd;">Court</th>
            <th style="width:30%;padding:8px;text-align:left;border-bottom:2px solid #ddd;">Team&nbsp;1</th>
            <th style="width:10%;padding:8px;text-align:center;border-bottom:2px solid #ddd;">vs</th>
            <th style="width:30%;padding:8px;text-align:left;border-bottom:2px solid #ddd;">Team&nbsp;2</th>
          </tr></thead><tbody>
    """
    for cid, g in enumerate(groups):
        t1 = " & ".join(player_names[p] for p in g[:2])
        t2 = " & ".join(player_names[p] for p in g[2:])
        row_bg = "#ffffff" if cid % 2 == 0 else "#f9f9f9"
        is_down = (downstairs_idx is not None and cid in downstairs_idx)
        badge = " <span class='downstairs-badge' style='background:#222;color:#fff;border-radius:6px;padding:2px 6px;font-size:.8em;-webkit-print-color-adjust:exact;print-color-adjust:exact;'>Downstairs</span>" if is_down else ""
        html += f"""
            <tr style="background:{row_bg};">
              <td style="width:20%;padding:6px;border-bottom:1px solid #ddd;">{court_names[cid]}{badge}</td>
              <td style="width:30%;padding:6px;border-bottom:1px solid #ddd;">{t1}</td>
              <td style="width:10%;padding:6px;border-bottom:1px solid #ddd;text-align:center;">vs</td>
              <td style="width:30%;padding:6px;border-bottom:1px solid #ddd;">{t2}</td>
            </tr>"""
    resting = ", ".join(player_names[p] for p in bench)
    html += f"""
          </tbody><tfoot><tr>
            <td style="padding:6px;background:#f0f0f0;" colspan="4"><strong>Resting:</strong> {resting}</td>
          </tr></tfoot></table></div></div>"""
    return html

# -------------------- statistics / debug helpers --------------------------- #
def format_debug_schedule_html(all_matches, genders, skills, names, courts, match_offset: int = 0):
    dbg = "<h2>Debug Schedule with Skill Info</h2>\n"
    for idx, (groups, bench) in enumerate(all_matches):
        dbg += f"""
        <div style='margin-bottom:10px;border:1px solid #ccc;border-radius:8px;background:#fff;padding:10px;'>
          <h3 style='text-align:center;margin-top:0;'>Match {idx+1+match_offset}</h3>
          <table style='width:100%;border-collapse:collapse;table-layout:fixed;'>
            <thead><tr style='background:#f0f0f0;'>
              <th style='width:15%;padding:6px;border-bottom:2px solid #ddd;'>Court</th>
              <th style='width:25%;padding:6px;border-bottom:2px solid #ddd;'>Team&nbsp;1</th>
              <th style='width:25%;padding:6px;border-bottom:2px solid #ddd;'>Team&nbsp;2</th>
              <th style='width:10%;padding:6px;border-bottom:2px solid #ddd;text-align:center;'>Same Sex</th>
              <th style='width:10%;padding:6px;border-bottom:2px solid #ddd;text-align:center;'>T1 Skill</th>
              <th style='width:10%;padding:6px;border-bottom:2px solid #ddd;text-align:center;'>T2 Skill</th>
              <th style='width:15%;padding:6px;border-bottom:2px solid #ddd;text-align:center;'>Diff</th>
            </tr></thead><tbody>"""
        for cid, g in enumerate(groups):
            t1, t2 = g[:2], g[2:]
            t1_s = sum(skills[i] for i in t1); t2_s = sum(skills[i] for i in t2)
            t1_str = " & ".join(f"{names[i]} ({skills[i]})" for i in t1)
            t2_str = " & ".join(f"{names[i]} ({skills[i]})" for i in t2)
            # Determine if the game is same-sex (all players M or all F)
            gs = [genders[i] for i in g]
            same_sex_cell = (
                "<span style='color:cyan;'>♂</span>" if all(x == "M" for x in gs)
                else ("<span style='color:magenta;'>♀</span>" if all(x == "F" for x in gs) else "No")
            )
            row_bg = "#ffffff" if cid % 2 == 0 else "#f9f9f9"
            dbg += f"""
              <tr style='background:{row_bg};'><td style='padding:6px;border-bottom:1px solid #ddd;'>{courts[cid]}</td>
              <td style='padding:6px;border-bottom:1px solid #ddd;'>{t1_str}</td>
              <td style='padding:6px;border-bottom:1px solid #ddd;'>{t2_str}</td>
              <td style='padding:6px;border-bottom:1px solid #ddd;text-align:center;'>{same_sex_cell}</td>
              <td style='padding:6px;border-bottom:1px solid #ddd;text-align:center;'>{t1_s}</td>
              <td style='padding:6px;border-bottom:1px solid #ddd;text-align:center;'>{t2_s}</td>
              <td style='padding:6px;border-bottom:1px solid #ddd;text-align:center;'>{abs(t1_s-t2_s)}</td></tr>"""
        rest_names = ", ".join(f"{names[p]} ({skills[p]})" for p in bench) if bench else "None"
        dbg += f"""
            </tbody><tfoot><tr><td style='padding:6px;background:#f0f0f0;font-weight:bold;'>Resting</td>
            <td style='padding:6px;background:#f0f0f0;' colspan='6'>{rest_names}</td></tr></tfoot>
          </table></div>"""
    return dbg


def build_player_schedule_table(all_matches, names, courts, match_offset: int = 0):
    n_matches = len(all_matches)
    sorted_idx = sorted(range(len(names)), key=lambda i: names[i].lower().strip())
    col_w = 80 / n_matches if n_matches else 80
    table = ["<h2>Player Schedule Summary</h2>",
             "<table style='width:100%;border-collapse:collapse;table-layout:fixed;'><thead><tr style='background:#f0f0f0;text-align:center;'>",
             "<th style='width:20%;padding:6px;border-bottom:2px solid #ddd;'>Player</th>"]
    table += [f"<th style='width:{col_w}%;padding:6px;border-bottom:2px solid #ddd;text-align:center;'>Match {i+1+match_offset}</th>"
              for i in range(n_matches)]
    table += ["</tr></thead><tbody>"]
    for idx in sorted_idx:
        table.append(f"<tr><td style='padding:6px;border-bottom:1px solid #ddd;'>{names[idx]}</td>")
        for groups, bench in all_matches:
            cell = "<b>Rest</b>" if idx in bench else ""
            if not cell:
                for c_i, g in enumerate(groups):
                    if idx in g:
                        cell = courts[c_i] if c_i < len(courts) else ""
                        break
            table.append(f"<td style='padding:6px;border-bottom:1px solid #ddd;text-align:center;'>{cell}</td>")
        table.append("</tr>")
    table.append("</tbody></table>")
    return "".join(table)


def format_statistics_html(stats, date_str_uk, all_matches, genders, match_offset: int = 0):
    explanation = (
        "<h3>Session Overview</h3>"
        "<p>This page shows how many times each player rested, and how often they teamed up or faced each other.</p>"
        "<p>Reading the lines:<br><strong>P1 [R]</strong>: P2 [X,Y], ...</p>"
        "<p>- [R] = times rested.&nbsp; [X,Y] = with&nbsp;X / against&nbsp;Y.</p>"
        "<p>Totals and gender distribution follow the list.</p>"
    )
    # configuration summary (injected later by caller)
    return f"""
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
    <title>Mijas Padellers Match Statistics - {date_str_uk}</title>
    <style>
        body{{font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;background:#fff;color:#000;margin:0;padding:20px;}}
        table{{width:100%;border-collapse:collapse;margin-bottom:10px;table-layout:fixed;}}
        th,td{{padding:6px;border:1px solid #ddd;text-align:left;}}
        th{{background:#f2f2f2;font-size:1.1em;}}
        tr:nth-child(even){{background:#f9f9f9;}}
        @media print{{body{{margin:0;padding:0;}}}}
    </style></head><body><div class='container'>
    <h1 style='text-align:center;'>Mijas Padellers Match Statistics</h1>
    <h2 style='text-align:center;'>{date_str_uk}</h2>
    <div style='background:#f9f9f9;padding:10px;border:1px solid #ddd;border-radius:8px;margin-bottom:20px;'>
      {explanation}<pre style='white-space:pre-wrap;line-height:1.5em;'>{stats}</pre>
    </div></div></body></html>"""


# --------------------------------------------------------------------------- #
#  Statistics generator (bug-fixed)                                           #
# --------------------------------------------------------------------------- #
def generate_Schedule_Statistics(team_mtx, opp_mtx, rest_track, names,
                                 all_matches, genders, skills, courts,
                                 match_offset: int = 0):
    lines = []; team_total = opp_total = 0
    tfreq, ofreq = {}, {}
    n = len(names)

    for p in range(n):
        rest = rest_track[p]; pair_list = []
        for q in range(n):
            if p == q: continue
            t, o = team_mtx[p][q], opp_mtx[p][q]
            tfreq[t] = tfreq.get(t, 0) + 1
            ofreq[o] = ofreq.get(o, 0) + 1
            pair_list.append(f"{names[q]} [{t},{o}]")
            team_total += t; opp_total += o
        lines.append(f"{names[p]} [{rest}]: " + ", ".join(pair_list))

    lines += [f"\nTeammate Total: {team_total}",
              f"Opponent Total: {opp_total}",
              "\nTeammate Count Frequencies:"]
    lines += [f"  Count {k}: {v}" for k, v in sorted(tfreq.items())]
    lines.append("\nOpponent Count Frequencies:")
    lines += [f"  Count {k}: {v}" for k, v in sorted(ofreq.items())]
    stats_text = "\n".join(lines)

    all_male = all_fem = mixed = 0
    for groups, _ in all_matches:
        for g in groups:
            gs = [genders[i] for i in g]
            if len(set(gs)) == 1:
                if gs[0] == "M":
                    all_male += 1
                else:
                    all_fem += 1
            else:
                mixed += 1
    stats_text += (f"\nAll-Male Games: {all_male}\n"
                   f"All-Female Games: {all_fem}\n"
                   f"Mixed Games: {mixed}\n")

    base = format_statistics_html(stats_text, st.session_state["date_str_uk"],
                                  all_matches, genders, match_offset=match_offset)
    debug = format_debug_schedule_html(all_matches, genders, skills, names, courts, match_offset=match_offset)
    p_table = build_player_schedule_table(all_matches, names, courts, match_offset=match_offset)
    return base.replace("</body>",
                        f"{debug}<div style='page-break-before: always;'>{p_table}</div></body>")


# --------------------------------------------------------------------------- #
#  Core scheduler with reliable page-break insertion                          #
# --------------------------------------------------------------------------- #
def assign_matches(names, genders, skills, courts, court_sz=4, total_matches=6,
                   samples=100000, reject_mixed=False, enable_skill=False,
                   skill_wt=20, same_sex_matches_count=0,
                   downstairs_limits=None,
                   start_dt: datetime | None = None,
                   match_offset: int = 0,
                   header_html: str = "",
                   repeat_header_html: str = "",
                   first_page_footer_html: str = ""):
    n_players = len(names)
    courts_used = determine_courts_to_use(n_players, len(courts), court_sz)
    active_courts = courts[:courts_used]
    down_idx, up_idx = identify_downstairs_courts(active_courts)
    teammate_mtx, opponent_mtx = initialize_matrices(n_players)
    rest_track = initialize_rest_tracker(n_players)
    players = list(range(n_players))
    history = set()
    all_matches = []

    eff_same_sex_count = same_sex_matches_count if reject_mixed else 0
    spaced_same_sex = compute_same_sex_match_indices(total_matches, eff_same_sex_count)
    preferred_indices = spaced_same_sex if same_sex_matches_count > 0 else (
        range(total_matches) if reject_mixed else []
    )
    # Every active downstairs court follows the same eligibility/stay limits,
    # including sessions with only downstairs courts.
    rotation = build_downstairs_rotation(
        players, total_matches, len(down_idx) * court_sz, len(up_idx) * court_sz,
        limits=downstairs_limits, genders=genders, skills=skills,
        enable_skill=enable_skill, skill_wt=skill_wt, same_sex_indices=preferred_indices,
    ) if down_idx else None

    for match_no in range(total_matches):
        st.info(f"--- Assigning players for Match {match_no+1+match_offset} ---")
        if rotation:
            downstairs = rotation[match_no // 2]
            upstairs = [p for p in players if p not in downstairs]
            benched = select_bench_players(
                upstairs, rest_track, n_players - courts_used * court_sz
            )
            court_pools = [
                (down_idx, [p for p in players if p in downstairs]),
                (up_idx, [p for p in upstairs if p not in benched]),
            ]
        else:
            benched = select_bench_players(
                players, rest_track, n_players - courts_used * court_sz
            )
            court_pools = [
                (list(range(courts_used)), [p for p in players if p not in benched])
            ]
        effective_reject_mixed = (
            match_no in spaced_same_sex
            if same_sex_matches_count > 0 else bool(reject_mixed)
        )
        groups = [None] * courts_used
        for court_indices, available in court_pools:
            best = find_best_match(
                available, len(court_indices), court_sz,
                teammate_mtx, opponent_mtx, match_no, samples,
                effective_reject_mixed, genders, enable_skill, skills,
                skill_wt, history, match_offset=match_offset,
            )
            for cid, group in zip(court_indices, best):
                groups[cid] = group
        for group in groups:
            history.add(canonical_config(group))
        update_matrices_for_match(groups, teammate_mtx, opponent_mtx)
        for p in benched:
            rest_track[p] += 1
        all_matches.append((groups, benched))

    # Preserve the established print layout, times, numbering and repeated focus.
    matches_per_page = 3 if courts_used <= 4 else 2
    html_blocks = [header_html] if header_html else []
    base_dt = start_dt if start_dt else datetime.combine(date.today(), time(hour=10, minute=30))
    for idx, (groups, bench) in enumerate(all_matches):
        if idx and idx % matches_per_page == 0:
            html_blocks.append("<div style='page-break-after: always;'></div>")
            if repeat_header_html:
                html_blocks.append(repeat_header_html)
        slot_start = base_dt + timedelta(minutes=15 * idx)
        time_label = slot_start.strftime('%H:%M')
        html_blocks.append(format_match_table_html(
            idx, groups, active_courts, names, bench,
            downstairs_idx=down_idx, time_label=time_label,
            match_offset=match_offset,
        ))
    return "".join(html_blocks), teammate_mtx, opponent_mtx, rest_track, all_matches, courts_used


# --------------------------------------------------------------------------- #
#  Streamlit UI                                                               #
# --------------------------------------------------------------------------- #
def main():
    # ----------------- session defaults -----------------
    defaults = {
        "all_schedules": [], "all_stats": [], "date_str_uk": "",
        "player_selection_order": {}, "player_counter": 1
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ----------------- page header ----------------------
    with st.container(key="session_details"):
        st.title("Mijas Padellers Match Scheduler")
    with st.container(key="session_settings", border=True):
        date_col, time_col, number_col = st.columns([1.3, 1, 1])
        with date_col:
            sess_date = st.date_input("📅 Session Date", value=date.today())
        with time_col:
            sess_time = st.time_input("⏰ Start Time", value=time(hour=10, minute=30))
        with number_col:
            start_match_number = st.number_input("Start match number", min_value=1, value=1)
        date_str = sess_date.strftime("%A %d %B %Y")
        st.caption(f"Selected date: {date_str} at {sess_time.strftime('%H:%M')}")

    # ----------------- player master list ---------------
    # Permanent court defaults: 2 = Standard rotation, 0 = Keep upstairs,
    # 4 = Keep downstairs (up to four matches, in two-match stays).
    REGULAR_PLAYERS = [
        {"name": "Agelina", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "Aideen", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "Anna", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "Anne", "gender": "F", "skill": 4, "downstairs_limit": 0},
        {"name": "Asmo", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "Bevan", "gender": "M", "skill": 5, "downstairs_limit": 2},
        {"name": "Chris", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "Christine", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "Collin", "gender": "M", "skill": 6, "downstairs_limit": 2},
        {"name": "Cris B.", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "Daryoush", "gender": "M", "skill": 6, "downstairs_limit": 0},
        {"name": "Dave", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "Declan", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "Dee", "gender": "F", "skill": 4, "downstairs_limit": 2},
        {"name": "Eugene", "gender": "M", "skill": 6, "downstairs_limit": 2},
        {"name": "Geordie", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "Gregor", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "Glynis", "gender": "F", "skill": 4, "downstairs_limit": 0},
        {"name": "Heather", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "Janet", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "John F.", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "John S.", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "Joyce", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "Judy", "gender": "F", "skill": 6, "downstairs_limit": 2},
        {"name": "Julia", "gender": "F", "skill": 4, "downstairs_limit": 2},
        {"name": "Julie", "gender": "F", "skill": 4, "downstairs_limit": 2},
        {"name": "Katty", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "Kevan", "gender": "M", "skill": 5, "downstairs_limit": 2},
        {"name": "Kim", "gender": "F", "skill": 6, "downstairs_limit": 2},
        {"name": "Linda", "gender": "F", "skill": 6, "downstairs_limit": 2},
        {"name": "Lindsey", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "Louise", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "Lynn", "gender": "F", "skill": 4, "downstairs_limit": 2},
        {"name": "Lynsey", "gender": "F", "skill": 4, "downstairs_limit": 2},
        {"name": "Maria", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "Melvyn", "gender": "M", "skill": 6, "downstairs_limit": 2},
        {"name": "Mieke", "gender": "F", "skill": 6, "downstairs_limit": 2},
        {"name": "Mike", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "Norman", "gender": "M", "skill": 6, "downstairs_limit": 2},
        {"name": "Outi", "gender": "F", "skill": 6, "downstairs_limit": 0},
        {"name": "Paul", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "Pieter", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "Ray", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "Roger", "gender": "M", "skill": 6, "downstairs_limit": 2},
        {"name": "Ronan", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "Ruth", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "Sandy", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "Sarah M", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "Sarah R", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "Scott", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "Sharon", "gender": "F", "skill": 4, "downstairs_limit": 2},
        {"name": "Soraya", "gender": "F", "skill": 6, "downstairs_limit": 2},
        {"name": "Sue", "gender": "F", "skill": 5, "downstairs_limit": 2},
        {"name": "Tania", "gender": "F", "skill": 6, "downstairs_limit": 2},
        {"name": "Tony", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "Travis", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "Vic", "gender": "M", "skill": 6, "downstairs_limit": 2},
        {"name": "Walker", "gender": "M", "skill": 7, "downstairs_limit": 2},
        {"name": "Wendy", "gender": "F", "skill": 4, "downstairs_limit": 0},
    ]
    REGULAR_PLAYERS = sorted(REGULAR_PLAYERS, key=lambda x: x["name"])

    # ------------ today's player adjustments (optional) ------------
    court_preferences = {"Standard rotation": 2, "Keep upstairs": 0, "Keep downstairs": 4}
    court_labels = list(court_preferences)
    court_help = (
        "Standard rotation: up to 2 downstairs matches. "
        "Keep upstairs: never assigned a downstairs court. "
        "Keep downstairs: allows up to 4 downstairs matches in two-match stays. "
        "The scheduler chooses how many are needed; nobody plays 6 downstairs matches."
    )
    # Scope the joined controls to regular-player rows. Keep the inner columns
    # together on small screens while the outer player grid can still stack.
    st.markdown("""
    <style>
    [class*="st-key-player_controls_"] {
        gap: .35rem;
    }
    .player-control-label {
        font-size: .875rem; line-height: 1.4;
    }
    [class*="st-key-player_controls_"] > [data-testid="stElementContainer"] > [data-testid="stMarkdown"] > [data-testid="stMarkdownContainer"] {
        margin-bottom: 0;
    }
    [class*="st-key-player_controls_"] [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important; gap: 0 !important;
    }
    [class*="st-key-player_controls_"] [data-testid="stColumn"]:first-child {
        flex: 1 1 0 !important; min-width: 0 !important; width: auto !important;
    }
    [class*="st-key-player_controls_"] [data-testid="stColumn"]:last-child {
        flex: 0 0 2.5rem !important; min-width: 2.5rem !important; width: 2.5rem !important;
    }
    [class*="st-key-player_controls_"] [data-testid="stNumberInputContainer"],
    [class*="st-key-player_controls_"] [data-testid="stNumberInputContainer"] > div {
        border-top-right-radius: 0; border-bottom-right-radius: 0;
    }
    [class*="st-key-player_controls_"] [data-testid="stPopover"] > div[aria-haspopup] [data-testid="stTooltipHoverTarget"] {
        width: 100%;
    }
    [class*="st-key-player_controls_"] [data-testid="stPopover"] button {
        width: 100%; height: 2.5rem; min-height: 2.5rem; padding: 0;
        border-radius: 0 .5rem .5rem 0; border: 1px solid #2563eb;
        background: #2563eb; color: #fff; justify-content: center; gap: 0;
    }
    [class*="st-key-player_controls_"] [data-testid="stPopover"] button:hover {
        background: #1d4ed8; border-color: #1d4ed8; color: #fff;
    }
    [class*="st-key-player_controls_"] [data-testid="stPopover"] button:focus-visible {
        outline: 2px solid #93c5fd; outline-offset: 2px;
    }
    [class*="st-key-player_controls_"] [data-testid="stPopover"] button [data-testid="stMarkdownContainer"] {
        position: absolute; width: 1px; height: 1px; overflow: hidden;
        clip-path: inset(50%); white-space: nowrap;
    }
    </style>
    """, unsafe_allow_html=True)

    def current_court_preference(name, key, default_limit=2):
        # Carry forward choices from the previous version in an existing session.
        if key not in st.session_state:
            previous = st.session_state.get("downstairs_player_preferences", {})
            limit = previous.get(name, default_limit)
            st.session_state[key] = next(
                label for label, value in court_preferences.items() if value == limit
            )
        return st.session_state[key]

    def player_court_preference(name, key, default_limit=2, compact=False):
        current = current_court_preference(name, key, default_limit)
        if compact:
            with st.popover(
                f"Court preference for {name}: {current}",
                help=f"{name}: {current}. {court_help}",
                use_container_width=True,
            ):
                choice = st.radio(f"Court preference for {name}", court_labels, key=key,
                                  help=court_help)
        else:
            choice = st.selectbox("Court preference", court_labels, key=key, help=court_help)
        return court_preferences[choice]

    st.header("👥 Select Regular Players")
    with st.expander("⚙️ Adjust Regular Player Skills & Court Preferences (today)",
                     expanded=False):
        st.caption("Change a player's skill or court preference for today's schedule. "
                   "New sessions start from the saved defaults; speak to Scott for permanent changes.")
        cols_s = st.columns(3)
        per_col = math.ceil(len(REGULAR_PLAYERS) / 3)
        for c in range(3):
            with cols_s[c]:
                for r in range(per_col):
                    idx = c * per_col + r
                    if idx < len(REGULAR_PLAYERS):
                        p = REGULAR_PLAYERS[idx]
                        with st.container(key=f"player_controls_{idx}"):
                            current = current_court_preference(
                                p["name"], f"court_pref_{p['name']}", p["downstairs_limit"]
                            )
                            st.markdown(
                                "<div class='player-control-label'>"
                                f"{_html.escape(p['name'])}{format_court_preference_icon(court_preferences[current])}"
                                "</div>", unsafe_allow_html=True,
                            )
                            skill_col, pref_col = st.columns([1, .2], vertical_alignment="bottom")
                            with skill_col:
                                p["skill"] = st.number_input(p["name"], 1, 10, p["skill"],
                                                             key=f"skill_{p['name']}",
                                                             label_visibility="collapsed")
                            with pref_col:
                                p["downstairs_limit"] = player_court_preference(
                                    p["name"], f"court_pref_{p['name']}", p["downstairs_limit"],
                                    compact=True,
                                )

        # ------- live grouped-by-skill summary (inside expander) -------
        st.markdown("**📊 Current Regulars Grouped by Skill (live):**", unsafe_allow_html=True)
        skills_present = sorted({int(p["skill"]) for p in REGULAR_PLAYERS}, reverse=True)
        for s in skills_present:
            names_for_skill = []
            for p in REGULAR_PLAYERS:
                if int(p["skill"]) == s:
                    sym = "<span style='color:magenta;'>♀</span>" if p["gender"] == "F" \
                          else "<span style='color:cyan;'>♂</span>"
                    names_for_skill.append(f"{p['name']} {sym}")
            if names_for_skill:
                names_for_skill = sorted(names_for_skill, key=lambda x: x.lower())
                st.markdown(f"<b><u>Skill {s}:</u></b>", unsafe_allow_html=True)
                st.markdown(", ".join(names_for_skill), unsafe_allow_html=True)
                st.markdown("&nbsp;", unsafe_allow_html=True)

        st.markdown("**Current Regulars by Court Preference:**")
        for limit, label in ((0, "Keep upstairs"), (4, "Keep downstairs (up to 4 matches)")):
            names_for_preference = []
            for p in REGULAR_PLAYERS:
                if p["downstairs_limit"] == limit:
                    sym = "<span style='color:magenta;'>♀</span>" if p["gender"] == "F" \
                          else "<span style='color:cyan;'>♂</span>"
                    names_for_preference.append(
                        f"{_html.escape(p['name'])} {sym}{format_court_preference_icon(limit)}"
                    )
            st.markdown(f"<b><u>{label}:</u></b>", unsafe_allow_html=True)
            st.markdown(
                ", ".join(names_for_preference) or "None",
                unsafe_allow_html=True,
            )


    # ---- apply queued Scott testing preset (before widgets) ----
    # We MUST apply this before creating the player/court widgets, otherwise Streamlit
    # will throw when we try to programmatically change widget-backed session_state.
    if st.session_state.get("scott_schedule_testing_request"):
        req = st.session_state.get("scott_schedule_testing_request") or {}
        preset = req.get("preset")
        seed = int(req.get("seed") or 0)
        rng = random.Random(seed)

        # Ensure maps exist
        if "player_selection_order" not in st.session_state:
            st.session_state["player_selection_order"] = {}

        # Clear all regular-player checkboxes + orders
        for idx, p in enumerate(REGULAR_PLAYERS):
            st.session_state[f"p_sel_{p['name']}{idx}"] = False
            st.session_state["player_selection_order"][p["name"]] = None

        # Clear guest inputs (so tests are consistent)
        for gi in range(8):
            st.session_state[f"g_nm_{gi}"] = ""
            st.session_state[f"g_gen_{gi}"] = "F"
            st.session_state[f"g_sk_{gi}"] = 5
            st.session_state[f"g_court_pref_{gi}"] = "Standard rotation"

        # Clear custom courts
        for i in range(4):
            st.session_state[f"cust_{i}"] = ""

        # Clear court checkboxes
        for i in range(16):
            st.session_state[f"court_{i}"] = False

        # Apply preset selection (or just reset)
        if preset and preset != "Reset selections":
            n_players = 18
            court_nums = [7, 8, 9, 10]
            if str(preset).startswith("16 "):
                n_players = 16
            if str(preset).endswith("7–11"):
                court_nums = [7, 8, 9, 10, 11]

            n_players = max(0, min(int(n_players), len(REGULAR_PLAYERS)))
            picked_idx = rng.sample(range(len(REGULAR_PLAYERS)), n_players)

            st.session_state["player_counter"] = 1
            for idx in picked_idx:
                nm = REGULAR_PLAYERS[idx]["name"]
                st.session_state[f"p_sel_{nm}{idx}"] = True
                st.session_state["player_selection_order"][nm] = st.session_state["player_counter"]
                st.session_state["player_counter"] += 1

            for cn in court_nums:
                ci = int(cn) - 1
                if 0 <= ci < 16:
                    st.session_state[f"court_{ci}"] = True
        else:
            st.session_state["player_counter"] = 1

        # Clear request so it only applies once
        st.session_state["scott_schedule_testing_request"] = None

    # ------------ player selection ------------------------
    per_col = math.ceil(len(REGULAR_PLAYERS) / 4)
    with st.container(key="regular_player_grid"):
        cols_p = st.columns(4)
        for c in range(4):
            with cols_p[c]:
                for r in range(per_col):
                    idx = c * per_col + r
                    if idx < len(REGULAR_PLAYERS):
                        p = REGULAR_PLAYERS[idx]
                        sel = st.checkbox(p["name"], key=f"p_sel_{p['name']}{idx}")
                        if sel:
                            if st.session_state["player_selection_order"].get(p["name"]) is None:
                                st.session_state["player_selection_order"][p["name"]] = st.session_state["player_counter"]
                                st.session_state["player_counter"] += 1
                        else:
                            st.session_state["player_selection_order"][p["name"]] = None

    # ------------ guest players ---------------------------
    with st.expander("➕ Add Up to 8 Guest Players (optional)", expanded=False):
        # A visible radio label changes Streamlit's widget identity. Carry forward
        # existing guests' genders once, before registering the labelled widgets.
        if not st.session_state.get("guest_gender_label_migrated", False):
            for gi in range(8):
                gender_key = f"g_gen_{gi}"
                if gender_key in st.session_state:
                    st.session_state[gender_key] = st.session_state[gender_key]
            st.session_state["guest_gender_label_migrated"] = True
        guest_players = []
        for gi in range(8):
            with st.container(key=f"guest_controls_{gi}"):
                cn, cg, cs, cp = st.columns([1.5, 1.2, 1.5, 1.8])
                gname  = cn.text_input(f"Guest Player {gi+1}", key=f"g_nm_{gi}")
                ggen   = cg.radio("Gender", ["F", "M"], horizontal=True, key=f"g_gen_{gi}")
                gskill = cs.number_input("Skill", 1, 10, 5, key=f"g_sk_{gi}")
                with cp:
                    glimit = player_court_preference(gname.strip(), f"g_court_pref_{gi}")

            if gname.strip():
                name = gname.strip()
                # give every new guest an order slot
                if st.session_state["player_selection_order"].get(name) is None:
                    st.session_state["player_selection_order"][name] = st.session_state["player_counter"]
                    st.session_state["player_counter"] += 1
                guest_players.append((name, ggen, gskill, glimit))
            else:
                # if the box was cleared, drop it from the order map
                prior = st.session_state["player_selection_order"].get(gname.strip())
                if prior is not None:
                    st.session_state["player_selection_order"][gname.strip()] = None

    # ----------- build unified ordered list ---------------
    # ticked regulars
    chosen_regulars = [p["name"]
                       for idx, p in enumerate(REGULAR_PLAYERS)
                       if st.session_state.get(f"p_sel_{p['name']}{idx}", False)]
    # guests with non-blank names
    guest_names = [g[0] for g in guest_players]

    all_current = set(chosen_regulars + guest_names)
    ordered = sorted(
        [(n, o) for n, o in st.session_state["player_selection_order"].items()
         if o is not None and n in all_current],
        key=lambda x: x[1]
    )
    sel_names = [n for n, _ in ordered]   # ✅ regulars + guests in true entry order

    # assemble final player arrays
    raw_wg, raw_ng, genders, skills, downstairs_limits = [], [], [], [], []
    for n in sel_names:
        # is it a regular?
        reg = next((x for x in REGULAR_PLAYERS if x["name"] == n), None)
        if reg:
            p = reg
            sym = "<span style='color:magenta;font-weight:bold;'>♀</span>" if p["gender"] == "F" \
                  else "<span style='color:cyan;font-weight:bold;'>♂</span>"
            raw_wg.append(f"{p['name']} {sym}")
            raw_ng.append(p["name"])
            genders.append(p["gender"]); skills.append(p["skill"])
            downstairs_limits.append(p["downstairs_limit"])
        else:
            g = next(x for x in guest_players if x[0] == n)
            sym = "<span style='color:magenta;font-weight:bold;'>♀</span>" if g[1] == "F" \
                  else "<span style='color:cyan;font-weight:bold;'>♂</span>"
            raw_wg.append(f"{g[0]} {sym}"); raw_ng.append(g[0])
            genders.append(g[1]); skills.append(g[2])
            downstairs_limits.append(g[3])

    if len({x.lower() for x in raw_ng}) < len(raw_ng):
        st.error("Duplicate player names detected."); st.stop()

    players_wg = deduplicate_names(raw_wg)
    players_ng = deduplicate_names(raw_ng)

    # Totals by gender for selected players
    male_count   = sum(1 for g in genders if g == "M")
    female_count = sum(1 for g in genders if g == "F")
    male_sym     = "<span style='color:cyan;'>♂</span>"
    female_sym   = "<span style='color:magenta;'>♀</span>"
    male_word    = "<span style='color:cyan;'>Men</span>"
    female_word  = "<span style='color:magenta;'>Women</span>"
    male_num     = f"<span style='color:cyan;'>{male_count}</span>"
    female_num   = f"<span style='color:magenta;'>{female_count}</span>"
    st.markdown(
        f"**Totals:** {len(players_ng)} Players [ {male_num} {male_word} {male_sym}, "
        f"{female_num} {female_word} {female_sym} ]",
        unsafe_allow_html=True
    )

    st.markdown("---\n**📋 Final Player List (rest order):**", unsafe_allow_html=True)
    st.markdown(
        "\n".join(f"{i+1}. {p}{format_court_preference_icon(limit) if limit != 2 else ''}"
                  for i, (p, limit) in enumerate(zip(players_wg, downstairs_limits))),
        unsafe_allow_html=True,
    )

    # ------------ court selection ------------------------
    st.header("🎾 Select Regular Courts")
    REG_COURTS  = [f"Court {i}" for i in range(1, 17)]
    regular_downstairs, _ = identify_downstairs_courts(REG_COURTS)
    per_col_c   = math.ceil(len(REG_COURTS) / 4)
    sel_courts  = []
    with st.container(key="regular_court_grid"):
        cols_c = st.columns(4)
        for c in range(4):
            with cols_c[c]:
                for r in range(per_col_c):
                    idx = c * per_col_c + r
                    if idx < len(REG_COURTS):
                        court_label = REG_COURTS[idx]
                        if idx in regular_downstairs:
                            court_label += " :red[↓\ufe0e]"
                        if st.checkbox(court_label, key=f"court_{idx}"):
                            sel_courts.append(REG_COURTS[idx])

    with st.expander("➕ Add Up to 4 Custom Courts (optional)", expanded=False):
        custom_courts = []
        for i in range(4):
            cc = st.text_input(f"Custom Court {i+1}", key=f"cust_{i}")
            if cc.strip():
                custom_courts.append(cc.strip())

    courts = sel_courts + custom_courts
    selected_downstairs, _ = identify_downstairs_courts(courts)
    st.markdown("---\n**🏟️ Final Court List:**")
    st.markdown(
        "\n".join(f"{i+1}. {_html.escape(c)}"
                  f"{format_downstairs_court_icon() if i in selected_downstairs else ''}"
                  for i, c in enumerate(courts)) or "No courts selected.",
        unsafe_allow_html=True,
    )

    # ------------- configuration inputs ------------------
    n_sel       = len(players_wg)
    courts_used = determine_courts_to_use(n_sel, len(courts), 4)
    max_unique  = compute_max_unique_matchups(n_sel, courts_used, 4) if n_sel >= courts_used*4 else 0
    active_down, active_up = identify_downstairs_courts(courts[:courts_used])
    if active_down:
        st.caption("Downstairs courts are included automatically in the same schedule, "
                   "in two-match stays. Match times remain 15 minutes apart.")
    if courts_used < len(courts):
        st.caption(f"With {n_sel} players, the first {courts_used} selected courts can be filled.")

    st.header("⚙️ Configuration")
    total_matches = st.number_input("🔢 Matches per schedule", 1, value=6)
    num_sched     = st.number_input("🗂 Number of schedules", 1, value=1)
    default_samples = int(min(max_unique, 200_000)) if max_unique else 200_000

    downstairs_error = None
    if active_down:
        st.caption("To change who goes downstairs, open Adjust Regular Player Skills & Court "
                   "Preferences above. Guest preferences are beside each guest's details.")
        try:
            validate_downstairs_limits(
                list(range(n_sel)), total_matches, len(active_down) * 4, downstairs_limits,
            )
        except ValueError as exc:
            downstairs_error = str(exc)
            st.warning(downstairs_error)

    reject_mixed = st.checkbox("🏳️‍🌈 Prefer Same-Sex Play", value=False)
    # Only show the count input when same-sex mode is enabled.
    # Default should remain 3 when/if enabled.
    if reject_mixed:
        same_sex_matches_count = st.number_input(
            "Number of same-sex matches",
            min_value=0,
            value=3,
            key="same_sex_matches_count",
        )
    else:
        same_sex_matches_count = 3
    enable_skill = st.checkbox("🎯 Skill-based Matches", value=True)

    # John Virgo "Trick Shot Challenge" (feature name stays, content is now a daily focus)
    show_john_virgo_trickshot = st.checkbox(
        "John Virgo Trick Shot Challenge",
        value=False,
        help="When enabled, pick a single focus for today; it will appear under the date in the downloaded schedule.",
    )

    def _slug_key(s: str) -> str:
        out = []
        for ch in s.lower():
            if ch.isalnum():
                out.append(ch)
            else:
                out.append("_")
        return "".join(out).strip("_")

    def render_single_choice_checkbox_grid(
        *,
        title: str,
        options: list[str],
        selected_key: str,
        key_prefix: str,
        cols: int = 4,
    ) -> str | None:
        """Render options as a 4-across checkbox grid, but enforce single choice.

        Returns the selected option (or None).
        """
        if selected_key not in st.session_state:
            st.session_state[selected_key] = None

        selected: str | None = st.session_state.get(selected_key)
        st.markdown(f"**{title}**")
        col_objs = st.columns(cols)
        all_keys = [f"{key_prefix}_{_slug_key(opt)}" for opt in options]

        def _on_change(opt: str, ck: str):
            if st.session_state.get(ck, False):
                st.session_state[selected_key] = opt
                # Uncheck all other boxes
                for other in all_keys:
                    if other != ck:
                        st.session_state[other] = False
            else:
                # If the selected option was unchecked, clear selection
                if st.session_state.get(selected_key) == opt:
                    st.session_state[selected_key] = None

        for i, opt in enumerate(options):
            ck = f"{key_prefix}_{_slug_key(opt)}"
            # Keep UI synced to selection each run (after callbacks)
            if st.session_state.get(selected_key) == opt:
                st.session_state[ck] = True
            elif ck not in st.session_state:
                st.session_state[ck] = False
            with col_objs[i % cols]:
                st.checkbox(
                    opt,
                    key=ck,
                    on_change=_on_change,
                    kwargs={"opt": opt, "ck": ck},
                )
        return st.session_state.get(selected_key)

    trickshot_focus_html = ""
    if show_john_virgo_trickshot:
        # Beginner-friendly + fun shots (+ Custom) (16 total to fit 4x4 grid)
        focus_options = [
            "Custom",
            "Racket up",
            "Split-step",
            "Cross-court",
            "Safe middle",
            "Mine / yours",
            "Middle balls",
            "Net together",
            "Mix it up",
            "Drop shot",
            "Lob",
            "Wall-shot",
            "Smash",
            "Slice",
            "Topspin",
            "Side-spin",
        ]
        focus_choice = render_single_choice_checkbox_grid(
            title="Trick-shot focus (pick one)",
            options=focus_options,
            selected_key="trickshot_focus_choice",
            key_prefix="trickshot_focus",
            cols=4,
        )

        # Editable schedule header text with a hard character limit (tweet-style).
        # Keep this reasonably short so printing stays stable.
        FOCUS_MAX_CHARS = 280
        SCHEDULE_FOCUS_PREFIX = "Today's skill focus is:"

        default_focus_text = {
            "Racket up": (
                f"{SCHEDULE_FOCUS_PREFIX} Racket up. "
                "Set early and stay ready between shots. Keep the racket in front of you."
            ),
            "Split-step": (
                f"{SCHEDULE_FOCUS_PREFIX} Split-step. "
                "Do a tiny hop as your opponent hits the ball. Land balanced, then move."
            ),
            "Cross-court": (
                f"{SCHEDULE_FOCUS_PREFIX} Cross-court. "
                "Aim cross-court most of the time—bigger target and higher margin."
            ),
            "Safe middle": (
                f"{SCHEDULE_FOCUS_PREFIX} Safe middle. "
                "When under pressure, play safely through the middle to reset the point."
            ),
            "Mine / yours": (
                f"{SCHEDULE_FOCUS_PREFIX} Mine / yours. "
                "Call early and commit—loud and simple: mine or yours."
            ),
            "Middle balls": (
                f"{SCHEDULE_FOCUS_PREFIX} Middle balls. "
                "Agree now who takes balls down the middle (e.g. forehand takes it), then trust it."
            ),
            "Net together": (
                f"{SCHEDULE_FOCUS_PREFIX} Net together. "
                "Move forward as a pair—don’t leave one player back alone."
            ),
            "Mix it up": (
                f"{SCHEDULE_FOCUS_PREFIX} Mix it up. "
                "Change pace and direction on purpose—one rally ball, then a softer one."
            ),
            "Drop shot": (
                f"{SCHEDULE_FOCUS_PREFIX} Drop shot. "
                "Soften the hands and keep it low. Try it only when you’re balanced."
            ),
            "Lob": (
                f"{SCHEDULE_FOCUS_PREFIX} Lob. "
                "When under pressure, go high and deep to buy time and take the net away."
            ),
            "Wall-shot": (
                f"{SCHEDULE_FOCUS_PREFIX} Wall-shot. "
                "If it’s going past you, let it bounce. Wait, then play it after the wall."
            ),
            "Smash": (
                f"{SCHEDULE_FOCUS_PREFIX} Smash. "
                "Choose the right one—placement over power. If it’s not on, don’t force it."
            ),
            "Slice": (
                f"{SCHEDULE_FOCUS_PREFIX} Slice. "
                "A gentle cut under the ball for control—short swing, firm wrist, keep it low."
            ),
            "Topspin": (
                f"{SCHEDULE_FOCUS_PREFIX} Topspin. "
                "Brush up the back of the ball to help it dip in and land deeper."
            ),
            "Side-spin": (
                f"{SCHEDULE_FOCUS_PREFIX} Side-spin. "
                "Add a little curve around the ball—keep it controlled and use it occasionally."
            ),
        }

        if focus_choice:
            if focus_choice == "Custom":
                # Custom focus: same single editable field as the other options.
                txt_key = "trickshot_custom_text_custom"
                if txt_key not in st.session_state:
                    st.session_state[txt_key] = (
                        f"{SCHEDULE_FOCUS_PREFIX} XXXXX. "
                        "Write your custom instructions here."
                    )

                edited = st.text_area(
                    "Schedule header text (editable)",
                    key=txt_key,
                    max_chars=FOCUS_MAX_CHARS,
                    height=110,
                    help=f"Max {FOCUS_MAX_CHARS} characters to keep printing stable.",
                )
                remaining = max(0, FOCUS_MAX_CHARS - len(edited or ""))
                st.caption(f"{remaining} characters remaining.")

                raw = (edited or "").strip()
                # Normalize older wording to avoid "trick-shot" on the schedule
                raw = raw.replace("Today's trick-shot focus is:", SCHEDULE_FOCUS_PREFIX)
                raw = raw.replace("Today's trickshot focus is:", SCHEDULE_FOCUS_PREFIX)
                raw = raw.replace("\r\n", "\n").replace("\r", "\n")
                head = raw.strip()
                body = ""
                if "." in head:
                    h, b = head.split(".", 1)
                    head = h.strip() + "."
                    body = b.strip()

                head_html = _html.escape(head)
                body_html = _html.escape(body).replace("\n", "<br>") if body else ""
                if head_html:
                    body_div = f"<div class='sched-focus-body'>{body_html}</div>" if body_html else ""
                    trickshot_focus_html = (
                        "<div class='sched-focus'><div class='sched-focus-inner'>"
                        f"<div class='sched-focus-head'>{head_html}</div>"
                        f"{body_div}"
                        "</div></div>"
                    )
            else:
                txt_key = f"trickshot_custom_text_{_slug_key(focus_choice)}"
                if txt_key not in st.session_state:
                    st.session_state[txt_key] = default_focus_text.get(focus_choice, "")

                edited = st.text_area(
                    "Schedule header text (editable)",
                    key=txt_key,
                    max_chars=FOCUS_MAX_CHARS,
                    height=110,
                    help=f"Max {FOCUS_MAX_CHARS} characters to keep printing stable.",
                )
                remaining = max(0, FOCUS_MAX_CHARS - len(edited or ""))
                st.caption(f"{remaining} characters remaining.")

                raw = (edited or "").strip()
                raw = raw.replace("Today's trick-shot focus is:", SCHEDULE_FOCUS_PREFIX)
                raw = raw.replace("Today's trickshot focus is:", SCHEDULE_FOCUS_PREFIX)
                raw = raw.replace("\r\n", "\n").replace("\r", "\n")
                head = raw.strip()
                body = ""
                if "." in head:
                    h, b = head.split(".", 1)
                    head = h.strip() + "."
                    body = b.strip()

                head_html = _html.escape(head)
                body_html = _html.escape(body).replace("\n", "<br>") if body else ""
                if head_html:
                    body_div = f"<div class='sched-focus-body'>{body_html}</div>" if body_html else ""
                    trickshot_focus_html = (
                        "<div class='sched-focus'><div class='sched-focus-inner'>"
                        f"<div class='sched-focus-head'>{head_html}</div>"
                        f"{body_div}"
                        "</div></div>"
                    )
    with st.expander("🧰 Miscellaneous", expanded=False):
        skill_wt = st.number_input("⚖️ Skill penalty weight", 1, value=20)
        samples = st.number_input("🔍 Random combinations to try (count)", 1, value=default_samples, step=50_000)
        st.caption(f"Approx. unique combinations: {int(max_unique):,}")

        st.markdown("---")
        scott_testing = st.checkbox(
            "Scott's schedule testing",
            value=False,
            help="Quickly preselect a typical test setup (players + courts) to save time while testing.",
            key="scott_schedule_testing_enabled",
        )
        if scott_testing:
            preset_options = [
                "18 random regulars + Courts 7–10",
                "16 random regulars + Courts 7–10",
                "18 random regulars + Courts 7–11",
                "Reset selections",
            ]
            # Default preset (only used until you pick one)
            if "scott_schedule_testing_preset" not in st.session_state:
                st.session_state["scott_schedule_testing_preset"] = preset_options[0]
            preset_choice = render_single_choice_checkbox_grid(
                title="Testing preset (pick one)",
                options=preset_options,
                selected_key="scott_schedule_testing_preset",
                key_prefix="scott_test",
                cols=4,
            )

            def _queue_scott_testing_request(*, reroll: bool = False) -> None:
                # Queue a request and let Streamlit rerun. The actual widget state
                # updates will be applied BEFORE widgets are created on the next run.
                preset = st.session_state.get("scott_schedule_testing_preset") or preset_options[0]
                seed = random.randint(1, 1_000_000_000) if reroll else random.randint(1, 1_000_000_000)
                st.session_state["scott_schedule_testing_request"] = {"preset": preset, "seed": seed}

            c_apply, c_reroll = st.columns([1, 1])
            with c_apply:
                st.button(
                    "Apply preset",
                    use_container_width=True,
                    on_click=_queue_scott_testing_request,
                    kwargs={"reroll": False},
                    key="scott_apply_preset_btn",
                )
            with c_reroll:
                st.button(
                    "Re-roll players",
                    use_container_width=True,
                    on_click=_queue_scott_testing_request,
                    kwargs={"reroll": True},
                    key="scott_reroll_players_btn",
                )

    # ------------- generate button -----------------------
    if st.button("📅 Generate Schedule(s)", disabled=downstairs_error is not None):
        if n_sel < 4:
            st.error("Need at least 4 players."); st.stop()
        if not courts:
            st.error("Please select at least one court."); st.stop()

        st.session_state["all_schedules"].clear()
        st.session_state["all_stats"].clear()
        st.session_state["date_str_uk"] = date_str

        with st.spinner("🔄 Generating schedule..."):
            for s_no in range(num_sched):
                st.info(f"Generating Schedule {s_no+1}/{num_sched}...")
                sched_html, tm, om, rest, all_matches, used = assign_matches(
                    players_ng, genders, skills, courts,
                    total_matches=total_matches, samples=samples,
                    reject_mixed=reject_mixed, enable_skill=enable_skill,
                    skill_wt=skill_wt,
                    same_sex_matches_count=same_sex_matches_count,
                    downstairs_limits=downstairs_limits,
                    start_dt=datetime.combine(sess_date, sess_time),
                    match_offset=max(0, int(start_match_number) - 1),
                    header_html=(
                        "<header class='sched-heading'>"
                        "<h1 class='sched-title'>Mijas Padellers Match Schedule</h1>"
                        f"<h2 class='sched-date'>{date_str}</h2>"
                        + trickshot_focus_html
                        + "</header>"
                    ),
                    repeat_header_html=(trickshot_focus_html or ""),
                    first_page_footer_html="",
                )

                full_html = f"""
                <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
                <title>Mijas Padellers Match Schedule - {date_str} (v{s_no+1})</title>
                <style>
                    body{{font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;background:#fff;color:#000;margin:0;padding:20px;}}
                    .sched-doodles{{position:fixed;inset:0;width:100%;height:100%;z-index:0;pointer-events:none;}}
                    .container{{position:relative;z-index:1;}}
                    .sched-heading{{background:transparent;}}
                    table{{width:100%;border-collapse:collapse;margin-bottom:10px;table-layout:fixed;}}
                    th,td{{padding:6px;border:1px solid #ddd;text-align:left;}}
                    th{{background:#f2f2f2;font-size:1.1em;}}
                    tr:nth-child(even){{background:#f9f9f9;}}
                    .sched-title{{display:table;max-width:100%;text-align:center;margin:0 auto 4px;padding:4px 10px;font-size:1.7em;background:#fff;}}
                    .sched-date{{display:table;max-width:100%;text-align:center;margin:0 auto 10px;padding:3px 10px;font-size:1.15em;font-weight:600;background:#fff;}}
                    .sched-focus{{margin:0 auto 12px auto;max-width:980px;padding:0 28px;box-sizing:border-box;}}
                    .sched-focus-inner{{background:#eef6ff;border:1px solid #cce0ff;border-radius:10px;padding:10px 12px;text-align:center;line-height:1.25;color:#111;}}
                    .sched-focus-inner{{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
                    .sched-focus-head{{font-weight:800;font-size:1.05em;margin:0 0 4px 0;}}
                    .sched-focus-body{{margin:0;color:#222;}}
                    .downstairs-badge{{background:#222;color:#fff;border-radius:6px;padding:2px 6px;font-size:.8em;-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
                    @page{{size:A4;margin:10mm;}}
                    @media print{{
                        body{{margin:0;padding:0;font-size:12pt;line-height:1.2;}}
                        .sched-doodles{{inset:auto;top:-10mm;left:-10mm;width:210mm;height:297mm;}}
                        .match-block{{break-inside:avoid;page-break-inside:avoid;margin:0 3mm 12px!important;border:.75pt solid #888;box-shadow:none!important;border-radius:5px!important;overflow:visible!important;background:#fff!important;}}
                        .match-content{{padding:6px 8px!important;}}
                        .match-title{{font-size:15pt;margin:0 0 5px!important;padding:0;}}
                        .match-block table{{margin-bottom:0;}}
                        .match-block th,.match-block td{{border:0!important;border-bottom:.5pt solid #aaa!important;padding:4px 5px!important;background:#fff!important;color:#000!important;vertical-align:middle;}}
                        .match-block thead th{{font-size:11pt;border-bottom:1pt solid #555!important;}}
                        .match-block thead th:nth-child(2),.match-block tbody td:nth-child(2){{text-align:right!important;}}
                        .match-block tr{{break-inside:avoid;background:#fff!important;}}
                        .match-block tfoot td{{font-size:11pt;border-bottom:0!important;}}
                        .sched-title{{font-size:18pt;}}
                        .sched-date{{font-size:12pt;}}
                        .sched-focus{{padding:0;margin-bottom:8px;}}
                        .sched-focus-inner{{background:#fff!important;border:.6pt solid #777;border-radius:0;padding:6px 8px;color:#000;}}
                        .sched-focus-head,.sched-focus-body{{font-size:11pt;color:#000;}}
                        .downstairs-badge{{background:#fff!important;color:#000!important;border:.6pt solid #555;border-radius:2px;padding:1px 4px!important;font-size:8.5pt!important;-webkit-print-color-adjust:economy;print-color-adjust:economy;}}
                    }}
                </style></head><body>{schedule_background_html()}<div class='container'>
                {sched_html}
                </div></body></html>"""

                st.session_state["all_schedules"].append(full_html)

                stats_html = generate_Schedule_Statistics(tm, om, rest,
                                                          players_wg,
                                                          all_matches,
                                                          genders, skills,
                                                          courts,
                                                          match_offset=max(0, int(start_match_number) - 1))
                # Inject configuration summary at top of stats page
                # Use effective count (0 when checkbox off) for summary as well
                eff_same_sex_count = same_sex_matches_count if reject_mixed else 0
                spaced_same_sex = compute_same_sex_match_indices(total_matches, eff_same_sex_count)
                cfg_same_sex = "Yes" if reject_mixed else "No"
                # Apply offset to same-sex match indices for display
                cfg_same_sex_matches = ", ".join(str(i+1 + max(0, int(start_match_number) - 1)) for i in spaced_same_sex) if spaced_same_sex else ("0")
                cfg_skill = "Yes" if enable_skill else "No"
                active_down, active_up = identify_downstairs_courts(courts[:used])
                cfg_down = "Automatic (two-match stays, maximum 4 per player)" if active_down else "No"
                cfg_keep = ", ".join(_html.escape(n) for n, limit in zip(players_ng, downstairs_limits) if limit == 0) or "None"
                cfg_four = ", ".join(_html.escape(n) for n, limit in zip(players_ng, downstairs_limits) if limit == 4) or "None"
                cfg_preferences = (
                    f"<li>Keep upstairs: {cfg_keep}</li>"
                    f"<li>Keep downstairs (up to 4 matches): {cfg_four}</li>"
                ) if active_down else ""
                cfg_html = (
                    f"<div style='background:#eef6ff;border:1px solid #cce0ff;padding:10px;border-radius:8px;margin-bottom:10px;'>"
                    f"<h3 style='margin:0 0 8px 0;'>Configuration</h3>"
                    f"<ul style='margin:0 0 0 18px;padding:0;'>"
                    f"<li>Skill-based: {cfg_skill}</li>"
                    f"<li>Same-sex preferred: {cfg_same_sex} (matches: {cfg_same_sex_matches})</li>"
                    f"<li>Downstairs rotation (12/13): {cfg_down}</li>"
                    f"{cfg_preferences}"
                    f"</ul></div>"
                )
                stats_html = stats_html.replace(
                    "<div class='container'>",
                    "<div class='container'>" + cfg_html,
                    1
                )
                st.session_state["all_stats"].append(stats_html)
        st.success("Schedule(s) generated successfully!")

    # ---------------- downloads --------------------------
    if st.session_state["all_schedules"]:
        st.subheader("📥 Download")
        for i, sched in enumerate(st.session_state["all_schedules"]):
            fn = f"Schedule_{i+1}_{date_str.replace(' ','_')}.html"
            clicked_schedule = st.download_button(f"📄 Schedule {i+1}", sched, fn, "text/html")
            fn_stat = f"Schedule_Statistics_{i+1}_{date_str.replace(' ','_')}.html"
            st.download_button(f"📄 Statistics {i+1}",
                               st.session_state["all_stats"][i],
                               fn_stat, "text/html")
            # When the schedule is downloaded, also (best-effort) upload the statistics
            # HTML to the developer's Google Drive via configured webhook.
            if clicked_schedule:
                downloaded_at_utc = datetime.now(timezone.utc).replace(second=0, microsecond=0)
                # Prefix uploads with download timestamp so downstream tooling can
                # select the "used" schedule based on filenames (not filesystem times).
                stamp = downloaded_at_utc.strftime("%Y-%m-%d_%H%MZ")
                fn_stat_upload = f"{stamp}_{fn_stat}"
                upload_stats_html_to_drive(
                    st.session_state["all_stats"][i],
                    fn_stat_upload,
                    meta={"date": date_str, "schedule_number": i + 1, "downloaded_at_utc": downloaded_at_utc.isoformat()},
                )
                # Also upload a small JSON containing just the pairing lines section
                pairings_lines = extract_pairings_lines_from_stats_html(st.session_state["all_stats"][i])
                json_payload = json.dumps(
                    {
                        "date": date_str,
                        "schedule_number": i + 1,
                        "downloaded_at_utc": downloaded_at_utc.isoformat(),
                        "pairings_lines": pairings_lines,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                fn_json = f"Schedule_Statistics_{i+1}_{date_str.replace(' ','_')}_pairings.json"
                upload_stats_html_to_drive(
                    json_payload,
                    f"{stamp}_{fn_json}",
                    meta={"date": date_str, "schedule_number": i + 1, "type": "pairings_json"},
                )
        st.divider()

    # ------------- footer note ----------------------------
    st.caption("In loving memory of Frankie. The bestest boy 🐶")


# --------------------------------------------------------------------------- #
#  program entry                                                              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
