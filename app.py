###############################################################################
#  Mijas Padellers Match Scheduler – Streamlit application                    #
#  Last updated: 29 Apr 2025 – Guest-player order now preserved               #
###############################################################################
import streamlit as st
import itertools
import numpy as np
import random
from datetime import date
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
                    history=None, forced_pair=None):
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
            status.write(f"Match {match_no+1}: sample {(i+1):,}/{samples:,}")
    bar.progress(100); status.write(f"Match {match_no+1} best match found!")
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
def format_match_table_html(match_no, groups, court_names, player_names, bench):
    html = f"""
    <div style="margin-bottom:10px;box-shadow:0 2px 4px rgba(0,0,0,.1);
                border-radius:8px;overflow:hidden;background:#fff;">
      <div style="padding:10px;">
        <h2 style="text-align:center;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;
                   margin:0 0 10px 0;">Match {match_no+1}</h2>
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
        html += f"""
            <tr style="background:{row_bg};">
              <td style="width:20%;padding:6px;border-bottom:1px solid #ddd;">{court_names[cid]}</td>
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
def format_debug_schedule_html(all_matches, genders, skills, names, courts):
    dbg = "<h2>Debug Schedule with Skill Info</h2>\n"
    for idx, (groups, bench) in enumerate(all_matches):
        dbg += f"""
        <div style='margin-bottom:10px;border:1px solid #ccc;border-radius:8px;background:#fff;padding:10px;'>
          <h3 style='text-align:center;margin-top:0;'>Match {idx+1}</h3>
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


def build_player_schedule_table(all_matches, names, courts):
    n_matches = len(all_matches)
    sorted_idx = sorted(range(len(names)), key=lambda i: names[i].lower().strip())
    col_w = 80 / n_matches if n_matches else 80
    table = ["<h2>Player Schedule Summary</h2>",
             "<table style='width:100%;border-collapse:collapse;table-layout:fixed;'><thead><tr style='background:#f0f0f0;text-align:center;'>",
             "<th style='width:20%;padding:6px;border-bottom:2px solid #ddd;'>Player</th>"]
    table += [f"<th style='width:{col_w}%;padding:6px;border-bottom:2px solid #ddd;'>Match {i+1}</th>"
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


def format_statistics_html(stats, date_str_uk, all_matches, genders):
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
                                 all_matches, genders, skills, courts):
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
                                  all_matches, genders)
    debug = format_debug_schedule_html(all_matches, genders, skills, names, courts)
    p_table = build_player_schedule_table(all_matches, names, courts)
    return base.replace("</body>",
                        f"{debug}<div style='page-break-before: always;'>{p_table}</div></body>")


# --------------------------------------------------------------------------- #
#  Core scheduler with reliable page-break insertion                          #
# --------------------------------------------------------------------------- #
def assign_matches(names, genders, skills, courts, court_sz=4, total_matches=6,
                   samples=100000, reject_mixed=False, enable_skill=False,
                   skill_wt=20, force_outi_asmo=False,
                   same_sex_matches_count=0):
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

    # Pre-compute which matches will prefer same-sex (N spaced matches ending with the last)
    spaced_same_sex = compute_same_sex_match_indices(total_matches, same_sex_matches_count)

    for m_no in range(total_matches):
        st.info(f"--- Assigning players for Match {m_no+1} ---")

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

    # ------------ build schedule HTML with page breaks ---------------------
    matches_per_page = 3 if courts_used <= 4 else 2
    html_blocks = []
    for idx, (groups, bench) in enumerate(all_matches):
        if idx and idx % matches_per_page == 0:
            html_blocks.append("<div style='page-break-after: always;'></div>")
        html_blocks.append(format_match_table_html(idx, groups, courts,
                                                   names, bench))
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
    date_str  = sess_date.strftime("%A %d %B %Y")
    st.write("Selected date:", date_str)

    # ----------------- player master list ---------------
    REGULAR_PLAYERS = [
        {"name": "Agneta", "gender": "F", "skill": 5},
        {"name": "Aideen", "gender": "F", "skill": 5},
        {"name": "Anna", "gender": "F", "skill": 5},
        {"name": "Anny", "gender": "F", "skill": 6},
        {"name": "Asmo", "gender": "M", "skill": 6},
        {"name": "Bevan", "gender": "M", "skill": 5},
        {"name": "Bill", "gender": "M", "skill": 7},
        {"name": "Chris", "gender": "M", "skill": 6},
        {"name": "Christine", "gender": "F", "skill": 5},
        {"name": "Colin", "gender": "M", "skill": 6},
        {"name": "Collin", "gender": "M", "skill": 6},
        {"name": "Cris Burgos", "gender": "F", "skill": 5},
        {"name": "Daryoush", "gender": "M", "skill": 6},
        {"name": "Dave", "gender": "M", "skill": 7},
        {"name": "Declan", "gender": "M", "skill": 7},
        {"name": "Dee", "gender": "F", "skill": 4},
        {"name": "Eugene", "gender": "M", "skill": 6},
        {"name": "Fabian", "gender": "M", "skill": 5},
        {"name": "Geordie", "gender": "M", "skill": 7},
        {"name": "Gregor", "gender": "M", "skill": 7},
        {"name": "Glynis", "gender": "F", "skill": 4},
        {"name": "Heather", "gender": "F", "skill": 5},
        {"name": "Janet", "gender": "F", "skill": 5},
        {"name": "John F.", "gender": "M", "skill": 6},
        {"name": "John S.", "gender": "M", "skill": 7},
        {"name": "Joyce", "gender": "F", "skill": 5},
        {"name": "Judy", "gender": "F", "skill": 6},
        {"name": "Julia", "gender": "F", "skill": 4},
        {"name": "Julie", "gender": "F", "skill": 4},
        {"name": "Katty", "gender": "F", "skill": 5},
        {"name": "Kevan", "gender": "M", "skill": 5},
        {"name": "Kim", "gender": "F", "skill": 4},
        {"name": "Leah", "gender": "F", "skill": 4},
        {"name": "Leigh", "gender": "F", "skill": 4},
        {"name": "Lilli", "gender": "F", "skill": 5},   
        {"name": "Linda", "gender": "F", "skill": 6},
        {"name": "Lindsey", "gender": "F", "skill": 5},
        {"name": "Lynn", "gender": "F", "skill": 4},
        {"name": "Lynsey", "gender": "F", "skill": 4},
        {"name": "Maria", "gender": "F", "skill": 4},
        {"name": "Maxine", "gender": "F", "skill": 5},
        {"name": "Mike", "gender": "M", "skill": 7},
        {"name": "Norman", "gender": "M", "skill": 6},
        {"name": "Outi", "gender": "F", "skill": 5},
        {"name": "Paola", "gender": "F", "skill": 4},
        {"name": "Paul", "gender": "M", "skill": 7},
        {"name": "Ray", "gender": "M", "skill": 7},
        {"name": "Roger", "gender": "M", "skill": 6},
        {"name": "Ronan", "gender": "M", "skill": 7},
        {"name": "Ruth", "gender": "F", "skill": 5},
        {"name": "Sandy", "gender": "F", "skill": 5},
        {"name": "Scott", "gender": "M", "skill": 7},
        {"name": "Sharon", "gender": "F", "skill": 4},
        {"name": "Soraya", "gender": "F", "skill": 6},
        {"name": "Tania", "gender": "F", "skill": 5},
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
    samples       = st.number_input("🔍 Random combinations to try", 1,
                                    value=default_samples, step=50_000)
    st.info(f"Approx. unique combinations: {int(max_unique):,}")

    reject_mixed   = st.checkbox("🏳️‍🌈 Prefer Same-Sex Play", value=True)
    same_sex_matches_count = st.number_input(
        "Number of initial matches to prefer same-sex play", 0, value=3
    )
    enable_skill   = st.checkbox("🎯 Skill-based Matches", value=True)
    force_pairing  = st.checkbox("Always pair Outi & Asmo", value=False)
    with st.expander("⚖️ Skill Penalty Weight", expanded=False):
        skill_wt = st.number_input("Penalty weight", 1, value=20)

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
                    same_sex_matches_count=same_sex_matches_count)

                p_table = build_player_schedule_table(all_matches, players_ng, courts)
                full_html = f"""
                <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
                <title>Mijas Padellers Match Schedule - {date_str} (v{s_no+1})</title>
                <style>
                    body{{font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;background:#fff;color:#000;margin:0;padding:20px;}}
                    table{{width:100%;border-collapse:collapse;margin-bottom:10px;table-layout:fixed;}}
                    th,td{{padding:6px;border:1px solid #ddd;text-align:left;}}
                    th{{background:#f2f2f2;font-size:1.1em;}}
                    tr:nth-child(even){{background:#f9f9f9;}}
                    @media print{{body{{margin:0;padding:0;}}}}
                </style></head><body><div class='container'>
                <h1 style='text-align:center;'>Mijas Padellers Match Schedule</h1>
                <h2 style='text-align:center;'>{date_str}</h2>
                {sched_html}<hr><div style='page-break-before:always;'>{p_table}</div>
                </div></body></html>"""

                st.session_state["all_schedules"].append(full_html)

                stats_html = generate_Schedule_Statistics(tm, om, rest,
                                                          players_wg,
                                                          all_matches,
                                                          genders, skills,
                                                          courts)
                # Inject configuration summary at top of stats page
                spaced_same_sex = compute_same_sex_match_indices(total_matches, same_sex_matches_count)
                cfg_same_sex = "Yes" if (reject_mixed or same_sex_matches_count > 0) else "No"
                cfg_same_sex_matches = ", ".join(str(i+1) for i in spaced_same_sex) if spaced_same_sex else ("All" if reject_mixed else "0")
                cfg_skill = "Yes" if enable_skill else "No"
                cfg_pair = "Yes" if force_pairing else "No"
                cfg_html = (
                    f"<div style='background:#eef6ff;border:1px solid #cce0ff;padding:10px;border-radius:8px;margin-bottom:10px;'>"
                    f"<h3 style='margin:0 0 8px 0;'>Configuration</h3>"
                    f"<ul style='margin:0 0 0 18px;padding:0;'>"
                    f"<li>Skill-based: {cfg_skill}</li>"
                    f"<li>Same-sex preferred: {cfg_same_sex} (matches: {cfg_same_sex_matches})</li>"
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
            st.download_button(f"📄 Schedule {i+1}", sched, fn, "text/html")
            fn_stat = f"Schedule_Statistics_{i+1}_{date_str.replace(' ','_')}.html"
            st.download_button(f"📄 Statistics {i+1}",
                               st.session_state["all_stats"][i],
                               fn_stat, "text/html")
        st.divider()

    # ------------- footer note ----------------------------
    st.caption("In loving memory of Frankie. The bestest boy 🐶")


# --------------------------------------------------------------------------- #
#  program entry                                                              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
