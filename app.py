###############################################################################
#  Mijas Padellers Match Scheduler – Streamlit application                    #
#  Last updated: 29 Apr 2025 – Guest-player order now preserved               #
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
                   history=None, forced_pair=None):
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

    if forced_pair:
        a_idx, o_idx = forced_pair
        found_same_side = False
        for g in groups:
            if a_idx in g and o_idx in g:
                idx_a = g.index(a_idx); idx_o = g.index(o_idx)
                if (idx_a < 2) == (idx_o < 2):   # same side
                    found_same_side = True
                else:
                    score += 10000
                break
        in_groups = {p for grp in groups for p in grp}
        if a_idx in in_groups and o_idx in in_groups and not found_same_side:
            score += 10000
    return score


def find_best_match(players, n_courts, court_sz, teammate_mtx, opponent_mtx,
                    match_no, samples=100000, reject_mixed=False,
                    genders=None, enable_skill=False, skills=None, skill_wt=20,
                    history=None, forced_pair=None, match_offset: int = 0):
    best, best_score = None, float('inf')
    bar = st.progress(0); status = st.empty()
    for i in range(samples):
        shuffled = players[:]; random.shuffle(shuffled)
        groups = [tuple(shuffled[j*court_sz:(j+1)*court_sz]) for j in range(n_courts)]
        if len(set(itertools.chain(*groups))) != len(players):
            continue
        s = evaluate_match(groups, teammate_mtx, opponent_mtx, reject_mixed, genders,
                           players, court_sz, enable_skill, skills, skill_wt,
                           history, forced_pair)
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

def build_downstairs_rotation(players, blocks, d_slots, offset_first_block=True):
    """Map each 2-match block to a set of player indices assigned downstairs.
    Tries to give everyone exactly one block when possible; uses entry order fairness.
    Allows ±1 size variance when needed.
    """
    n = len(players)
    rotation = [set() for _ in range(blocks)]
    if blocks <= 0 or d_slots <= 0:
        return rotation
    max_unique = blocks * d_slots
    eligible = players[:]  # entry/rest-order
    # Everyone gets one block if capacity permits, else as many as possible from the front
    pick = eligible if n <= max_unique else eligible[:max_unique]
    # Rest-aware tweak: avoid placing the very first rest-priority players
    # downstairs in the first block by rotating the pick list by d_slots.
    if offset_first_block and d_slots > 0 and len(pick) > 1:
        rot = min(d_slots, len(pick))
        pick = pick[rot:] + pick[:rot]
    idx = 0
    for p in pick:
        rotation[idx % blocks].add(p)
        idx += 1
    return rotation

_BLOCK_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def block_label(i):
    return _BLOCK_LABELS[i] if 0 <= i < len(_BLOCK_LABELS) else f"Block {i+1}"
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
def format_match_table_html(match_no, groups, court_names, player_names, bench,
                           block_tag=None, downstairs_idx=None, time_label=None,
                           match_offset: int = 0):
    blk = f" – Block {block_tag}" if block_tag else ""
    time_str = f" [{time_label}]" if time_label else ""
    html = f"""
    <div style="margin-bottom:10px;box-shadow:0 2px 4px rgba(0,0,0,.1);
                border-radius:8px;overflow:hidden;background:#fff;">
      <div style="padding:10px;">
        <h2 style="text-align:center;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;
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
            <td style="width:20%;padding:6px;background:#f0f0f0;font-weight:bold;">Resting</td>
            <td style="width:80%;padding:6px;background:#f0f0f0;" colspan="3">{resting}</td>
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
                   skill_wt=20, force_outi_asmo=False,
                   same_sex_matches_count=0,
                   use_downstairs_rotation=False,
                   start_dt: datetime | None = None,
                   match_offset: int = 0,
                   header_html: str = "",
                   repeat_header_html: str = "",
                   first_page_footer_html: str = ""):
    n_players   = len(names)
    courts_used = determine_courts_to_use(n_players, len(courts), court_sz)

    teammate_mtx, opponent_mtx = initialize_matrices(n_players)
    rest_track  = initialize_rest_tracker(n_players)
    players     = list(range(n_players))
    history     = set()
    all_matches = []

    # detect Outi + Asmo indices
    forced_pair = None
    if force_outi_asmo:
        try:
            forced_pair = (names.index("Asmo"), names.index("Outi"))
        except ValueError:
            pass   # one or both not present

    # If using downstairs rotation and we have downstairs courts, schedule by 2-match blocks
    down_idx, up_idx = identify_downstairs_courts(courts)
    d_courts = len(down_idx)
    u_courts = len(up_idx)
    if use_downstairs_rotation and d_courts > 0:
        blocks = math.ceil(total_matches / 2)
        d_slots = d_courts * court_sz
        rotation = build_downstairs_rotation(players, blocks, d_slots)
        # Force N=0 when checkbox is off
        eff_same_sex_count = same_sex_matches_count if reject_mixed else 0
        spaced_same_sex = compute_same_sex_match_indices(total_matches, eff_same_sex_count)

        all_matches.clear()
        teammate_mtx, opponent_mtx = initialize_matrices(n_players)
        rest_track  = initialize_rest_tracker(n_players)
        history     = set()

        match_counter = 0
        for b in range(blocks):
            matches_in_block = 2 if (match_counter + 2) <= total_matches else 1
            down_group = sorted([p for p in rotation[b] if p in players])

            for _ in range(matches_in_block):
                st.info(f"--- Assigning players for Match {match_counter+1+match_offset} (Block {block_label(b)}) ---")

                # Upstairs capacity and candidates (downstairs group never benches within block)
                capacity_up = u_courts * court_sz
                candidates_up = [p for p in players if p not in down_group]
                to_bench_up = select_bench_players(candidates_up, rest_track,
                                                   max(0, len(candidates_up) - capacity_up))
                available_up = [p for p in candidates_up if p not in to_bench_up]

                # Same-sex preference per match index
                effective_reject_mixed = (match_counter in spaced_same_sex) if same_sex_matches_count > 0 else bool(reject_mixed)

                # Build downstairs groups
                groups_down = []
                if d_courts > 0:
                    groups_down = find_best_match(
                        down_group, d_courts, court_sz,
                        teammate_mtx, opponent_mtx,
                        match_counter, samples,
                        effective_reject_mixed, genders, enable_skill, skills, skill_wt, history, forced_pair,
                        match_offset=match_offset
                    ) or []

                # Build upstairs groups
                groups_up = []
                if u_courts > 0:
                    groups_up = find_best_match(
                        available_up, u_courts, court_sz,
                        teammate_mtx, opponent_mtx,
                        match_counter, samples,
                        effective_reject_mixed, genders, enable_skill, skills, skill_wt, history, forced_pair,
                        match_offset=match_offset
                    ) or []

                # Merge aligned to original court order
                merged = [None] * (d_courts + u_courts)
                for i, g in zip(down_idx, groups_down):
                    merged[i] = g
                for i, g in zip(up_idx, groups_up):
                    merged[i] = g
                groups_full = merged

                # Update matrices, history, rest (only upstairs benched)
                for grp in groups_full:
                    if grp:
                        history.add(canonical_config(grp))
                update_matrices_for_match([g for g in groups_full if g], teammate_mtx, opponent_mtx)
                for p in to_bench_up:
                    rest_track[p] += 1

                all_matches.append((groups_full, to_bench_up))
                match_counter += 1

        # Build schedule HTML with reliable page breaks (print-friendly).
        # Keep the HTML structure simple (matches + explicit page-break divs)
        # to avoid browser print quirks with flex/min-height wrappers.
        matches_per_page = 3 if len(courts) <= 4 else 2
        html_blocks: list[str] = []
        if header_html:
            html_blocks.append(header_html)
        for idx, (groups, bench) in enumerate(all_matches):
            if idx and idx % matches_per_page == 0:
                html_blocks.append("<div style='page-break-after: always;'></div>")
                if repeat_header_html:
                    html_blocks.append(repeat_header_html)
            # Compute block label from match index
            bidx = idx // 2
            # When downstairs rotation is enabled, don't show block tag in downloadable HTML
            block_for_title = None if use_downstairs_rotation else block_label(bidx)
            base_dt = start_dt if start_dt else datetime.combine(date.today(), time(hour=10, minute=30))
            slot_start = base_dt + timedelta(minutes=15 * idx)
            slot_end = slot_start + timedelta(minutes=15)
            time_label = f"{slot_start.strftime('%H:%M')}-{slot_end.strftime('%H:%M')}"
            html_blocks.append(
                format_match_table_html(
                    idx, groups, courts, names, bench,
                    block_tag=block_for_title,
                    downstairs_idx=down_idx,
                    time_label=time_label,
                    match_offset=match_offset
                )
            )

        schedule_html = "".join(html_blocks)
        return schedule_html, teammate_mtx, opponent_mtx, rest_track, all_matches, courts_used

    # -------- default flow (no downstairs rotation) --------
    # Force N=0 when checkbox is off
    eff_same_sex_count = same_sex_matches_count if reject_mixed else 0
    spaced_same_sex = compute_same_sex_match_indices(total_matches, eff_same_sex_count)

    for m_no in range(total_matches):
        st.info(f"--- Assigning players for Match {m_no+1+match_offset} ---")

        capacity = courts_used * court_sz
        benched  = select_bench_players(players, rest_track,
                                        max(0, n_players - capacity))
        available = [p for p in players if p not in benched]

        # ensure forced pair both play or both rest
        if forced_pair:
            a_idx, o_idx = forced_pair
            if a_idx in available and o_idx not in available:
                available.remove(a_idx); benched.append(a_idx)
                if o_idx in benched:
                    benched.remove(o_idx)
                available.append(o_idx)

        # Apply same-sex preference for specifically chosen matches when count>0,
        # otherwise fall back to the checkbox behavior for all matches.
        effective_reject_mixed = (
            (m_no in spaced_same_sex)
            if same_sex_matches_count > 0
            else bool(reject_mixed)
        )

        best = find_best_match(available, courts_used, court_sz,
                               teammate_mtx, opponent_mtx, m_no, samples,
                               effective_reject_mixed, genders, enable_skill, skills,
                               skill_wt, history, forced_pair)

        if best:
            for grp in best:
                history.add(canonical_config(grp))
            all_matches.append((best, benched))
            update_matrices_for_match(best, teammate_mtx, opponent_mtx)
            for p in benched:
                rest_track[p] += 1

    # ------------ build schedule HTML with reliable page breaks -------------
    matches_per_page = 3 if courts_used <= 4 else 2
    html_blocks: list[str] = []
    if header_html:
        html_blocks.append(header_html)
    base_dt = start_dt if start_dt else datetime.combine(date.today(), time(hour=10, minute=30))
    for idx, (groups, bench) in enumerate(all_matches):
        if idx and idx % matches_per_page == 0:
            html_blocks.append("<div style='page-break-after: always;'></div>")
            if repeat_header_html:
                html_blocks.append(repeat_header_html)
        slot_start = base_dt + timedelta(minutes=15 * idx)
        slot_end = slot_start + timedelta(minutes=15)
        time_label = f"{slot_start.strftime('%H:%M')}-{slot_end.strftime('%H:%M')}"
        html_blocks.append(
            format_match_table_html(
                idx, groups, courts, names, bench,
                time_label=time_label,
                match_offset=match_offset
            )
        )
    schedule_html = "".join(html_blocks)
    # ----------------------------------------------------------------------

    return schedule_html, teammate_mtx, opponent_mtx, rest_track, all_matches, courts_used


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
    st.title("Mijas Padellers Match Scheduler")
    sess_date = st.date_input("📅 Session Date", value=date.today())
    sess_time = st.time_input("⏰ Start Time", value=time(hour=10, minute=30))
    # Start match number placed directly below time input for clarity
    start_match_number = st.number_input("Start match number", min_value=1, value=1)
    date_str  = sess_date.strftime("%A %d %B %Y")
    st.write("Selected date:", date_str, "at", sess_time.strftime("%H:%M"))

    # ----------------- player master list ---------------
    REGULAR_PLAYERS = [
        {"name": "Aideen", "gender": "F", "skill": 5},
        {"name": "Anna", "gender": "F", "skill": 5},
        {"name": "Anne", "gender": "F", "skill": 4},
        {"name": "Asmo", "gender": "M", "skill": 7},
        {"name": "Bevan", "gender": "M", "skill": 5},
        {"name": "Chris", "gender": "M", "skill": 7},
        {"name": "Christine", "gender": "F", "skill": 5},
        {"name": "Collin T.", "gender": "M", "skill": 6},
        {"name": "Colin B.", "gender": "M", "skill": 6},
        {"name": "Cris B.", "gender": "F", "skill": 5},
        {"name": "Daryoush", "gender": "M", "skill": 6},
        {"name": "Dave", "gender": "M", "skill": 7},
        {"name": "Declan", "gender": "M", "skill": 7},
        {"name": "Dee", "gender": "F", "skill": 4},
        {"name": "Eugene", "gender": "M", "skill": 6},
        {"name": "Geordie", "gender": "M", "skill": 7},
        {"name": "Gregor", "gender": "M", "skill": 7},
        {"name": "Glynis", "gender": "F", "skill": 4},
        {"name": "Heather", "gender": "F", "skill": 5},
        {"name": "Janet", "gender": "F", "skill": 5},
        {"name": "John F.", "gender": "M", "skill": 7},
        {"name": "John S.", "gender": "M", "skill": 7},
        {"name": "Joyce", "gender": "F", "skill": 5},
        {"name": "Judy", "gender": "F", "skill": 6},
        {"name": "Julia", "gender": "F", "skill": 4},
        {"name": "Julie", "gender": "F", "skill": 4},
        {"name": "Katty", "gender": "F", "skill": 5},
        {"name": "Kevan", "gender": "M", "skill": 5},
        {"name": "Kim", "gender": "F", "skill": 6},
        {"name": "Linda", "gender": "F", "skill": 6},
        {"name": "Lindsey", "gender": "F", "skill": 5},
        {"name": "Louise", "gender": "F", "skill": 5},
        {"name": "Lynn", "gender": "F", "skill": 4},
        {"name": "Lynsey", "gender": "F", "skill": 4},
        {"name": "Maria", "gender": "F", "skill": 5},
        {"name": "Melvyn", "gender": "M", "skill": 6},
        {"name": "Mieke", "gender": "F", "skill": 6},
        {"name": "Mike", "gender": "M", "skill": 7},
        {"name": "Norman", "gender": "M", "skill": 6},
        {"name": "Outi", "gender": "F", "skill": 6},
        {"name": "Paul", "gender": "M", "skill": 7},
        {"name": "Pieter", "gender": "M", "skill": 7},
        {"name": "Ray", "gender": "M", "skill": 7},
        {"name": "Roger", "gender": "M", "skill": 6},
        {"name": "Ronan", "gender": "M", "skill": 7},
        {"name": "Ruth", "gender": "F", "skill": 5},
        {"name": "Sandy", "gender": "F", "skill": 5},
        {"name": "Sarah M", "gender": "F", "skill": 5},
        {"name": "Sarah R", "gender": "F", "skill": 5},
        {"name": "Scott", "gender": "M", "skill": 7},
        {"name": "Sharon", "gender": "F", "skill": 4},
        {"name": "Soraya", "gender": "F", "skill": 6},
        {"name": "Sue", "gender": "F", "skill": 5},
        {"name": "Tania", "gender": "F", "skill": 6},
        {"name": "Tony", "gender": "M", "skill": 7},
        {"name": "Travis", "gender": "M", "skill": 7},
        {"name": "Vic", "gender": "M", "skill": 6},
        {"name": "Walker", "gender": "M", "skill": 7},
        {"name": "Wendy", "gender": "F", "skill": 4},
    ]
    REGULAR_PLAYERS = sorted(REGULAR_PLAYERS, key=lambda x: x["name"])

    # ------------ skill adjustments (optional) ------------
    st.header("👥 Select Regular Players")
    with st.expander("⚙️ Adjust Regular Player Skill Levels "
                     "(temporary). Speak to Scott for permanent changes.",
                     expanded=False):
        cols_s  = st.columns(4)
        per_col = math.ceil(len(REGULAR_PLAYERS) / 4)
        for c in range(4):
            with cols_s[c]:
                for r in range(per_col):
                    idx = c * per_col + r
                    if idx < len(REGULAR_PLAYERS):
                        p = REGULAR_PLAYERS[idx]
                        p["skill"] = st.number_input(p["name"], 1, 10, p["skill"],
                                                     key=f"skill_{p['name']}")

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
    cols_p  = st.columns(4)
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
        guest_players = []
        for gi in range(8):
            cn, cg, cs = st.columns([2, 1, 1])
            gname  = cn.text_input(f"Guest Player {gi+1}", key=f"g_nm_{gi}")
            ggen   = cg.radio("", ["F", "M"], horizontal=True, key=f"g_gen_{gi}")
            gskill = cs.number_input("Skill", 1, 10, 5, key=f"g_sk_{gi}")

            if gname.strip():
                name = gname.strip()
                # give every new guest an order slot
                if st.session_state["player_selection_order"].get(name) is None:
                    st.session_state["player_selection_order"][name] = st.session_state["player_counter"]
                    st.session_state["player_counter"] += 1
                guest_players.append((name, ggen, gskill))
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
    raw_wg, raw_ng, genders, skills = [], [], [], []
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
        else:
            g = next(x for x in guest_players if x[0] == n)
            sym = "<span style='color:magenta;font-weight:bold;'>♀</span>" if g[1] == "F" \
                  else "<span style='color:cyan;font-weight:bold;'>♂</span>"
            raw_wg.append(f"{g[0]} {sym}"); raw_ng.append(g[0])
            genders.append(g[1]); skills.append(g[2])

    if len({x.lower() for x in raw_ng}) < len(raw_ng):
        st.error("Duplicate player names detected."); st.stop()

    players_wg = deduplicate_names(raw_wg)
    players_ng = deduplicate_names(raw_ng)

    st.markdown("---\n**📋 Final Player List (rest order):**", unsafe_allow_html=True)
    st.markdown("\n".join(f"{i+1}. {p}" for i, p in enumerate(players_wg)),
                unsafe_allow_html=True)

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
        f"**Totals:** {male_num} {male_word} {male_sym}, {female_num} {female_word} {female_sym}",
        unsafe_allow_html=True
    )

    # ------------ court selection ------------------------
    st.header("🎾 Select Regular Courts")
    REG_COURTS  = [f"Court {i}" for i in range(1, 17)]
    per_col_c   = math.ceil(len(REG_COURTS) / 4)
    cols_c      = st.columns(4)
    sel_courts  = []
    for c in range(4):
        with cols_c[c]:
            for r in range(per_col_c):
                idx = c * per_col_c + r
                if idx < len(REG_COURTS):
                    if st.checkbox(REG_COURTS[idx], key=f"court_{idx}"):
                        sel_courts.append(REG_COURTS[idx])

    with st.expander("➕ Add Up to 4 Custom Courts (optional)", expanded=False):
        custom_courts = []
        for i in range(4):
            cc = st.text_input(f"Custom Court {i+1}", key=f"cust_{i}")
            if cc.strip():
                custom_courts.append(cc.strip())

    courts = sel_courts + custom_courts
    st.markdown("---\n**🏟️ Final Court List:**")
    st.markdown("\n".join(f"{i+1}. {c}" for i, c in enumerate(courts)) or "No courts selected.")

    # ------------- configuration inputs ------------------
    n_sel       = len(players_wg)
    courts_used = determine_courts_to_use(n_sel, len(courts), 4)
    max_unique  = compute_max_unique_matchups(n_sel, courts_used, 4) if n_sel >= courts_used*4 else 0

    st.header("⚙️ Configuration")
    total_matches = st.number_input("🔢 Matches per schedule", 1, value=6)
    num_sched     = st.number_input("🗂 Number of schedules", 1, value=1)
    default_samples = int(min(max_unique, 200_000)) if max_unique else 200_000
    

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
    # Show downstairs rotation only if courts 12/13 are selected
    has_downstairs = any(c.endswith("12") or c == "Court 12" or c == "12" for c in courts) or \
                     any(c.endswith("13") or c == "Court 13" or c == "13" for c in courts)
    use_downstairs_rotation = False
    if has_downstairs:
        use_downstairs_rotation = st.checkbox(
            "Use downstairs rotation for courts 12/13", value=True,
            help=("Splits the session into 2-match blocks and rotates a fixed subgroup "
                  "downstairs per block; teams reshuffle between the two matches.")
        )
    with st.expander("🧰 Miscellaneous", expanded=False):
        force_pairing = st.checkbox("Always pair Outi & Asmo", value=False)
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
    if st.button("📅 Generate Schedule(s)"):
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
                    skill_wt=skill_wt, force_outi_asmo=force_pairing,
                    same_sex_matches_count=same_sex_matches_count,
                    use_downstairs_rotation=use_downstairs_rotation,
                    start_dt=datetime.combine(sess_date, sess_time),
                    match_offset=max(0, int(start_match_number) - 1),
                    header_html=(
                        "<h1 class='sched-title'>Mijas Padellers Match Schedule</h1>"
                        f"<h2 class='sched-date'>{date_str}</h2>"
                        + trickshot_focus_html
                    ),
                    repeat_header_html=(trickshot_focus_html or ""),
                    first_page_footer_html="",
                )

                full_html = f"""
                <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
                <title>Mijas Padellers Match Schedule - {date_str} (v{s_no+1})</title>
                <style>
                    body{{font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;background:#fff;color:#000;margin:0;padding:20px;}}
                    table{{width:100%;border-collapse:collapse;margin-bottom:10px;table-layout:fixed;}}
                    th,td{{padding:6px;border:1px solid #ddd;text-align:left;}}
                    th{{background:#f2f2f2;font-size:1.1em;}}
                    tr:nth-child(even){{background:#f9f9f9;}}
                    .sched-title{{text-align:center;margin:0 0 4px 0;font-size:1.7em;}}
                    .sched-date{{text-align:center;margin:0 0 6px 0;font-size:1.15em;font-weight:600;}}
                    .sched-focus{{margin:0 auto 12px auto;max-width:980px;padding:0 28px;box-sizing:border-box;}}
                    .sched-focus-inner{{background:#eef6ff;border:1px solid #cce0ff;border-radius:10px;padding:10px 12px;text-align:center;line-height:1.25;color:#111;}}
                    .sched-focus-inner{{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
                    .sched-focus-head{{font-weight:800;font-size:1.05em;margin:0 0 4px 0;}}
                    .sched-focus-body{{margin:0;color:#222;}}
                    .downstairs-badge{{background:#222;color:#fff;border-radius:6px;padding:2px 6px;font-size:.8em;-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
                    @media print{{body{{margin:0;padding:0;}} .downstairs-badge{{-webkit-print-color-adjust:exact;print-color-adjust:exact;background:#000;color:#fff;}}}}
                </style></head><body><div class='container'>
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
                cfg_pair = "Yes" if force_pairing else "No"
                cfg_down = "Yes" if (use_downstairs_rotation and any(x in courts for x in ("Court 12","12","Court 13","13"))) else "No"
                cfg_html = (
                    f"<div style='background:#eef6ff;border:1px solid #cce0ff;padding:10px;border-radius:8px;margin-bottom:10px;'>"
                    f"<h3 style='margin:0 0 8px 0;'>Configuration</h3>"
                    f"<ul style='margin:0 0 0 18px;padding:0;'>"
                    f"<li>Skill-based: {cfg_skill}</li>"
                    f"<li>Same-sex preferred: {cfg_same_sex} (matches: {cfg_same_sex_matches})</li>"
                    f"<li>Downstairs rotation (12/13): {cfg_down}</li>"
                    f"<li>Always pair Outi & Asmo: {cfg_pair}</li>"
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
