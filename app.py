import streamlit as st
import itertools
import numpy as np
import random
from datetime import date
import math
from math import comb, factorial

# ---------------------- 1) SET PAGE CONFIG FIRST ----------------------
st.set_page_config(page_title="Mijas Padellers Match Scheduler", layout="centered", initial_sidebar_state="auto")

# ---------------------- Custom CSS for Compact Progress/Info ----------------------
COMPACT_CSS = """
<style>
div[data-testid="stProgressBar"] > div[role="progressbar"] {
    background-color: #ff7f50;
}
div[data-testid="stProgressBar"] {
    margin: 2px 0;
    padding: 2px 0;
}
.css-1n76uvr, .stAlert {
    padding: 0.3rem 0.5rem !important;
    margin: 0.2rem 0 !important;
}
</style>
"""
st.markdown(COMPACT_CSS, unsafe_allow_html=True)

# ---------------------- Helper Functions ----------------------
def initialize_matrices(num_players):
    return (np.zeros((num_players, num_players), dtype=int),
            np.zeros((num_players, num_players), dtype=int))

def initialize_rest_tracker(num_players):
    return np.zeros(num_players, dtype=int)

def compute_max_unique_matchups(N, K, S):
    M = K * S
    if N < M:
        return 0
    return comb(N, M) * factorial(M) / ((factorial(S) ** K) * factorial(K))

# NEW: Function to get a canonical representation of a 4-player match configuration.
def canonical_config(group):
    team1 = tuple(sorted(group[:2]))
    team2 = tuple(sorted(group[2:4]))
    return tuple(sorted([team1, team2]))

def evaluate_match(groups, teammate_matrix, opponent_matrix, reject_mixed, player_genders,
                   available_players, court_size, enable_skill, player_skills, skill_weight, match_history=None):
    total_score = 0
    for group in groups:
        team1, team2 = group[:2], group[2:]
        teammate_score = sum(teammate_matrix[i][j] for i, j in itertools.combinations(team1, 2)) + \
                         sum(teammate_matrix[i][j] for i, j in itertools.combinations(team2, 2))
        opponent_score = sum(opponent_matrix[i][j] for i, j in itertools.product(team1, team2))
        diversity_penalty = sum(teammate_matrix[i][j] for i, j in itertools.combinations(team1, 2)) + \
                            sum(teammate_matrix[i][j] for i, j in itertools.combinations(team2, 2)) + \
                            sum(opponent_matrix[i][j] for i, j in itertools.product(team1, team2))
        total_score += (teammate_score + opponent_score + diversity_penalty)

    if enable_skill:
        skill_penalty = 0
        for group in groups:
            team1, team2 = group[:2], group[2:]
            avg1 = sum(player_skills[i] for i in team1) / len(team1)
            avg2 = sum(player_skills[i] for i in team2) / len(team2)
            skill_penalty += skill_weight * abs(avg1 - avg2)
        total_score += skill_penalty

    if reject_mixed and len(available_players) >= court_size:
        for group in groups:
            group_genders = [player_genders[i] for i in group]
            if len(set(group_genders)) > 1:
                available_pool = [player_genders[i] for i in available_players]
                if available_pool.count('F') >= court_size or available_pool.count('M') >= court_size:
                    total_score += 1000

    # NEW: If match_history is provided, penalize any group that has been used before.
    if match_history is not None:
        for group in groups:
            config = canonical_config(group)
            if config in match_history:
                total_score += 10000

    return total_score

def find_best_match(players, num_courts, court_size, teammate_matrix, opponent_matrix,
                    match_number, samples=100000, reject_mixed=False,
                    player_genders=None, enable_skill=False, player_skills=None, skill_weight=20, match_history=None):
    best_match, best_score = None, float('inf')
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(samples):
        shuffled = players[:]
        random.shuffle(shuffled)
        groups = [tuple(shuffled[i*court_size:(i+1)*court_size]) for i in range(num_courts)]
        if len(set(itertools.chain(*groups))) != len(players):
            continue

        score = evaluate_match(groups, teammate_matrix, opponent_matrix,
                               reject_mixed, player_genders, players,
                               court_size, enable_skill, player_skills, skill_weight, match_history=match_history)

        if score < best_score:
            best_score, best_match = score, groups

        if (i+1) % 1000 == 0:
            pct = int((i+1)/samples * 100)
            progress_bar.progress(pct)
            status_text.write(f"Match {match_number+1}: sample {i+1:,} of {samples:,}")

    progress_bar.progress(100)
    status_text.write(f"Match {match_number+1} best match found!")
    return best_match

def update_matrices_for_match(groups, teammate_matrix, opponent_matrix):
    for group in groups:
        team1, team2 = group[:2], group[2:]
        for i, j in itertools.combinations(team1, 2):
            teammate_matrix[i][j] += 1
            teammate_matrix[j][i] += 1
        for i, j in itertools.combinations(team2, 2):
            teammate_matrix[i][j] += 1
            teammate_matrix[j][i] += 1
        for i in team1:
            for j in team2:
                opponent_matrix[i][j] += 1
                opponent_matrix[j][i] += 1

def select_bench_players(players, rest_tracker, num_benched):
    return sorted(players, key=lambda x: rest_tracker[x])[:num_benched]

def determine_courts_to_use(num_players, available_courts, court_size=4):
    return min(available_courts, num_players // court_size)

def deduplicate_names(name_list):
    seen = {}
    result = []
    for name in name_list:
        if name not in seen:
            seen[name] = 1
            result.append(name)
        else:
            seen[name] += 1
            new_name = f"{name} ({seen[name]})"
            while new_name in seen:
                seen[new_name] = 1
                new_name = f"{name} ({seen[new_name]})"
            seen[new_name] = 1
            result.append(new_name)
    return result

def format_match_table_html(match_number, best_match, court_names, player_names, bench_players):
    html = f"""
    <div style="margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; background-color: #ffffff;">
      <div style="padding: 10px;">
        <h2 style="text-align: center; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; margin: 0 0 10px 0;">Match {match_number+1}</h2>
        <table style="width: 100%; border-collapse: collapse; table-layout: fixed; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;">
          <thead>
            <tr style="background-color: #f0f0f0;">
              <th style="width: 20%; padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Court</th>
              <th style="width: 30%; padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Team 1</th>
              <th style="width: 10%; padding: 8px; text-align: center; border-bottom: 2px solid #ddd;">vs</th>
              <th style="width: 30%; padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Team 2</th>
            </tr>
          </thead>
          <tbody>
    """
    for court_id, group in enumerate(best_match):
        team1 = " & ".join([player_names[p] for p in group[:2]])
        team2 = " & ".join([player_names[p] for p in group[2:]])
        cname = court_names[court_id]
        row_bg = "#ffffff" if court_id % 2 == 0 else "#f9f9f9"
        html += f"""
            <tr style="background-color: {row_bg};">
              <td style="width: 20%; padding: 6px; border-bottom: 1px solid #ddd;">{cname}</td>
              <td style="width: 30%; padding: 6px; border-bottom: 1px solid #ddd;">{team1}</td>
              <td style="width: 10%; padding: 6px; border-bottom: 1px solid #ddd; text-align: center;">vs</td>
              <td style="width: 30%; padding: 6px; border-bottom: 1px solid #ddd;">{team2}</td>
            </tr>
        """
    resting_names = ", ".join([player_names[p] for p in bench_players])
    html += f"""
          </tbody>
          <tfoot>
            <tr>
              <td colspan="4" style="padding: 6px; background-color: #f0f0f0; font-weight: bold;">Resting: {resting_names}</td>
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
    """
    return html

def format_statistics_html(stats, date_str_uk, all_matches_info, final_player_genders):
    explanation_text = (
        "<h3>Session Overview</h3>"
        "<p>This page shows how many times each player rested, and how often they teamed up or faced each other.</p>"
        "<p>"
        "Reading the lines:<br>"
        "<strong>P1 [R]</strong>: P2 [X,Y], P3 [X,Y], ...<br>"
        "- [R] is how many times Player 1 rested.<br>"
        "- [X,Y] after another player's name means Player 1 played <strong>with</strong> that player X times and <strong>against</strong> that player Y times.<br>"
        "</p>"
        "<p>For example, <em>P1 [2]: P2 [1,3]</em> means Player 1 rested twice total, teamed with Player 2 once, and played against Player 2 three times.</p>"
        "<p>"
        "After the player breakdown, you'll see totals for how many times people were teammates and opponents overall,<br>"
        "and how many all-male, all-female, or mixed games occurred."
        "</p>"
    )

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Mijas Padellers Match Statistics - {date_str_uk}</title>
      <style>
        body {{
          font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
          background-color: #ffffff;
          color: #000000;
          margin: 0;
          padding: 20px;
        }}
        .container {{
          width: 100%;
          margin: 0;
        }}
        h1 {{
          text-align: center;
          color: #333333;
          margin-bottom: 10px;
        }}
        h2 {{
          text-align: center;
          color: #666666;
          margin-bottom: 20px;
          font-size: 1.2em;
        }}
        h3 {{
          margin-top: 0;
        }}
        .stats-box {{
          background-color: #f9f9f9;
          padding: 10px;
          border: 1px solid #dddddd;
          border-radius: 8px;
          margin-bottom: 20px;
        }}
        .stats-content {{
          white-space: pre-wrap;
          line-height: 1.5em;
        }}
        table {{
          width: 100%;
          border-collapse: collapse;
          margin-bottom: 10px;
          table-layout: fixed;
        }}
        th, td {{
          padding: 6px;
          text-align: left;
          border: 1px solid #dddddd;
        }}
        th {{
          background-color: #f2f2f2;
          font-size: 1.1em;
        }}
        tr:nth-child(even) {{
          background-color: #f9f9f9;
        }}
        tr:nth-child(odd) {{
          background-color: #ffffff;
        }}
        tfoot td {{
          background-color: #f2f2f2;
          font-weight: bold;
          padding: 6px;
        }}
        @media print {{
          body {{
            margin: 0;
            padding: 0;
          }}
          .container {{
            width: 100%;
            margin: 0;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Mijas Padellers Match Statistics</h1>
        <h2>{date_str_uk}</h2>
        <div class="stats-box">
          {explanation_text}
          <div class="stats-content">{stats}</div>
        </div>
      </div>
    </body>
    </html>
    """
    return html

def format_debug_schedule_html(all_matches_info, final_player_genders, final_player_skills,
                               player_names, court_names):
    debug_html = "<h2>Debug Schedule with Skill Info</h2>\n"
    for match_index, (match_groups, bench_players) in enumerate(all_matches_info):
        debug_html += f"""
        <div style="margin-bottom: 10px; border: 1px solid #ccc; border-radius: 8px; background-color: #ffffff; padding: 10px;">
          <h3 style="text-align: center; margin-top: 0;">Match {match_index+1}</h3>
          <table style="width: 100%; border-collapse: collapse; table-layout: fixed; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;">
            <thead>
              <tr style="background-color: #f0f0f0;">
                <th style="width: 15%; padding: 6px; border-bottom: 2px solid #ddd;">Court</th>
                <th style="width: 25%; padding: 6px; border-bottom: 2px solid #ddd;">Team 1</th>
                <th style="width: 25%; padding: 6px; border-bottom: 2px solid #ddd;">Team 2</th>
                <th style="width: 10%; padding: 6px; border-bottom: 2px solid #ddd; text-align: center;">Team1 Skill</th>
                <th style="width: 10%; padding: 6px; border-bottom: 2px solid #ddd; text-align: center;">Team2 Skill</th>
                <th style="width: 15%; padding: 6px; border-bottom: 2px solid #ddd; text-align: center;">Difference</th>
              </tr>
            </thead>
            <tbody>
        """
        for court_id, group in enumerate(match_groups):
            team1 = group[:2]
            team2 = group[2:]
            def player_str(idx):
                return f"{player_names[idx]} ({final_player_skills[idx]})"
            team1_str = " & ".join([player_str(idx) for idx in team1])
            team2_str = " & ".join([player_str(idx) for idx in team2])
            team1_skill = sum(final_player_skills[idx] for idx in team1)
            team2_skill = sum(final_player_skills[idx] for idx in team2)
            diff = abs(team1_skill - team2_skill)
            cname = court_names[court_id]
            row_bg = "#ffffff" if court_id % 2 == 0 else "#f9f9f9"
            debug_html += f"""
              <tr style="background-color: {row_bg};">
                <td style="width: 15%; padding: 6px; border-bottom: 1px solid #ddd;">{cname}</td>
                <td style="width: 25%; padding: 6px; border-bottom: 1px solid #ddd;">{team1_str}</td>
                <td style="width: 25%; padding: 6px; border-bottom: 1px solid #ddd;">{team2_str}</td>
                <td style="width: 10%; padding: 6px; border-bottom: 1px solid #ddd; text-align: center;">{team1_skill}</td>
                <td style="width: 10%; padding: 6px; border-bottom: 1px solid #ddd; text-align: center;">{team2_skill}</td>
                <td style="width: 15%; padding: 6px; border-bottom: 1px solid #ddd; text-align: center;">{diff}</td>
              </tr>
            """
        if bench_players:
            resting_names = ", ".join([f"{player_names[p]} ({final_player_skills[p]})" for p in bench_players])
        else:
            resting_names = "None"
        debug_html += f"""
            </tbody>
          </table>
          <p style="font-weight: bold;">Resting: {resting_names}</p>
        </div>
        """
    return debug_html

def build_player_schedule_table(all_matches_info, player_names, court_names):
    num_matches = len(all_matches_info)
    sorted_indices = sorted(range(len(player_names)), key=lambda i: player_names[i].strip().lower())
    table_html = "<h2>Player Schedule Summary</h2>\n"
    table_html += """
    <table style="width: 100%; border-collapse: collapse; table-layout: fixed; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;">
      <thead>
        <tr style="background-color: #f0f0f0; text-align: center;">
          <th style="width: 20%; padding: 6px; border-bottom: 2px solid #ddd;">Player</th>
    """
    col_width = 80 / num_matches if num_matches > 0 else 80
    for m in range(num_matches):
        table_html += f'<th style="width: {col_width}%; padding: 6px; border-bottom: 2px solid #ddd; text-align: center;">Match {m+1}</th>'
    table_html += "</tr></thead><tbody>"

    for idx in sorted_indices:
        pname = player_names[idx]
        table_html += "<tr>"
        table_html += f'<td style="padding: 6px; border-bottom: 1px solid #ddd;">{pname}</td>'
        for (best_match, bench_players) in all_matches_info:
            assigned = "Rest"
            if idx not in bench_players:
                for court_index, group in enumerate(best_match):
                    if idx in group:
                        assigned = court_names[court_index] if court_index < len(court_names) else ""
                        break
            table_html += f'<td style="padding: 6px; border-bottom: 1px solid #ddd; text-align: center;">{assigned}</td>'
        table_html += "</tr>"
    table_html += "</tbody></table>"
    return table_html

def assign_matches(player_names, player_genders, player_skills, court_names, court_size=4,
                   total_matches=8, samples=100000, reject_mixed=False,
                   enable_skill=False, skill_weight=20):
    num_players = len(player_names)
    available_courts = len(court_names)
    courts_to_use = determine_courts_to_use(num_players, available_courts, court_size)
    teammate_matrix, opponent_matrix = initialize_matrices(num_players)
    rest_tracker = initialize_rest_tracker(num_players)
    players = list(range(num_players))
    schedule_output = []
    all_matches_info = []
    # NEW: Maintain a set of canonical configurations that have already been used
    match_history = set()
    for match_number in range(total_matches):
        st.info(f"--- Assigning players for Match {match_number+1} ---")
        total_court_capacity = courts_to_use * court_size
        num_benched = max(0, num_players - total_court_capacity)
        bench_players = select_bench_players(players, rest_tracker, num_benched)
        available_players = [p for p in players if p not in bench_players]

        best_match = find_best_match(
            available_players, courts_to_use, court_size,
            teammate_matrix, opponent_matrix,
            match_number, samples=samples,
            reject_mixed=reject_mixed, player_genders=player_genders,
            enable_skill=enable_skill, player_skills=player_skills,
            skill_weight=skill_weight, match_history=match_history
        )
        if best_match:
            # NEW: For each court configuration chosen in this match, add its canonical form to match_history
            for group in best_match:
                config = canonical_config(group)
                match_history.add(config)
            all_matches_info.append((best_match, bench_players))
            update_matrices_for_match(best_match, teammate_matrix, opponent_matrix)
            match_table = format_match_table_html(match_number, best_match, court_names, player_names, bench_players)
            schedule_output.append(match_table)
            st.success(f"Match {match_number+1} assigned successfully.")
            for bp in bench_players:
                rest_tracker[bp] += 1
    return "".join(schedule_output), teammate_matrix, opponent_matrix, rest_tracker, all_matches_info

def generate_Schedule_Statistics(teammate_matrix, opponent_matrix, rest_tracker, player_names,
                                 all_matches_info, final_player_genders, final_player_skills, court_names):
    output = []
    teammate_total, opponent_total = 0, 0
    teammate_frequencies, opponent_frequencies = {}, {}
    num_players = len(player_names)

    for player in range(num_players):
        rest_count = rest_tracker[player]
        teammates_opponents = []
        for other_player in range(num_players):
            if player != other_player:
                teammate_count = teammate_matrix[player][other_player]
                opponent_count = opponent_matrix[player][other_player]
                teammate_frequencies[teammate_count] = teammate_frequencies.get(teammate_count, 0) + 1
                opponent_frequencies[opponent_count] = opponent_frequencies.get(opponent_count, 0) + 1
                teammates_opponents.append(f"{player_names[other_player]} [{teammate_count},{opponent_count}]")
                teammate_total += teammate_count
                opponent_total += opponent_count
        output.append(f"{player_names[player]} [{rest_count}]: " + ", ".join(teammates_opponents))

    output.append(f"\nTeammate Total: {teammate_total}")
    output.append(f"Opponent Total: {opponent_total}")
    output.append("\nTeammate Count Frequencies:")
    for count, freq in sorted(teammate_frequencies.items()):
        output.append(f"  Count {count}: {freq}")
    output.append("\nOpponent Count Frequencies:")
    for count, freq in sorted(opponent_frequencies.items()):
        output.append(f"  Count {count}: {freq}")

    stats_text = "\n".join(output)

    all_male_games = 0
    all_female_games = 0
    mixed_games = 0
    for best_match, bench_players in all_matches_info:
        for group in best_match:
            group_genders = [final_player_genders[idx] for idx in group]
            if len(set(group_genders)) == 1:
                if group_genders[0] == "M":
                    all_male_games += 1
                else:
                    all_female_games += 1
            else:
                mixed_games += 1

    distribution_text = (
        f"\nAll-Male Games: {all_male_games}"
        f"\nAll-Female Games: {all_female_games}"
        f"\nMixed Games: {mixed_games}\n"
    )
    full_stats_text = stats_text + "\n" + distribution_text

    base_html = format_statistics_html(full_stats_text, st.session_state["date_str_uk"], all_matches_info, final_player_genders)
    debug_schedule = format_debug_schedule_html(all_matches_info, final_player_genders, final_player_skills, player_names, court_names)
    player_schedule = build_player_schedule_table(all_matches_info, player_names, court_names)

    final_html = base_html.replace(
        "</body>",
        f"{debug_schedule}<div style='page-break-before: always;'>{player_schedule}</div></body>"
    )
    return final_html

def main():
    if "all_schedules" not in st.session_state:
        st.session_state["all_schedules"] = []
    if "all_stats" not in st.session_state:
        st.session_state["all_stats"] = []
    if "date_str_uk" not in st.session_state:
        st.session_state["date_str_uk"] = ""
    if "player_selection_order" not in st.session_state:
        st.session_state["player_selection_order"] = {}
    if "player_counter" not in st.session_state:
        st.session_state["player_counter"] = 1

    st.title("Mijas Padellers Match Scheduler")

    session_date = st.date_input("📅 Session Date", value=date.today())
    formatted_date = session_date.strftime("%A %d %B %Y")
    st.write("Selected date:", formatted_date)

    REGULAR_PLAYERS = [
        {"name": "Agneta", "gender": "F", "skill": 5},
        {"name": "Anna", "gender": "F", "skill": 4},
        {"name": "Anny", "gender": "F", "skill": 5},
        {"name": "Bevan", "gender": "M", "skill": 5},
        {"name": "Bill", "gender": "M", "skill": 7},
        {"name": "Chris", "gender": "M", "skill": 6},
        {"name": "Christine", "gender": "F", "skill": 4},
        {"name": "Cris Burgos", "gender": "F", "skill": 4},
        {"name": "Daryoush", "gender": "M", "skill": 5},
        {"name": "Declan", "gender": "M", "skill": 7},
        {"name": "Dee", "gender": "F", "skill": 4},
        {"name": "Evgen", "gender": "M", "skill": 6},
        {"name": "Fabian", "gender": "M", "skill": 5},
        {"name": "Geordie", "gender": "M", "skill": 7},
        {"name": "Glynis", "gender": "F", "skill": 4},
        {"name": "Heather", "gender": "F", "skill": 5},
        {"name": "Janet", "gender": "F", "skill": 5},
        {"name": "John", "gender": "M", "skill": 7},
        {"name": "Joyce", "gender": "F", "skill": 5},
        {"name": "Juan", "gender": "M", "skill": 7},
        {"name": "Julia", "gender": "F", "skill": 4},
        {"name": "Julie", "gender": "F", "skill": 4},
        {"name": "Kevan", "gender": "M", "skill": 5},
        {"name": "Leah", "gender": "F", "skill": 4},
        {"name": "Linda", "gender": "F", "skill": 5},
        {"name": "Lindsey", "gender": "F", "skill": 5},
        {"name": "Lynn", "gender": "F", "skill": 4},
        {"name": "Lynsey", "gender": "F", "skill": 4},
        {"name": "Maria", "gender": "F", "skill": 4},
        {"name": "Maxine", "gender": "F", "skill": 5},
        {"name": "Mike", "gender": "M", "skill": 7},
        {"name": "Norman", "gender": "M", "skill": 6},
        {"name": "Paola", "gender": "F", "skill": 4},
        {"name": "Paul", "gender": "M", "skill": 7},
        {"name": "Ruth", "gender": "F", "skill": 4},
        {"name": "Sandy", "gender": "F", "skill": 5},
        {"name": "Scott", "gender": "M", "skill": 7},
        {"name": "Sharon", "gender": "F", "skill": 4},
        {"name": "Soraya", "gender": "F", "skill": 5},
        {"name": "Tania", "gender": "F", "skill": 4},
        {"name": "Tony", "gender": "M", "skill": 7},
        {"name": "Travis", "gender": "M", "skill": 7},
        {"name": "Walker", "gender": "M", "skill": 7},
        {"name": "Wendy", "gender": "F", "skill": 4}
    ]
    REGULAR_PLAYERS = sorted(REGULAR_PLAYERS, key=lambda x: x["name"])

    st.header("👥 Select Regular Players")
    num_cols = 4
    n_players = len(REGULAR_PLAYERS)
    num_per_col = math.ceil(n_players / num_cols)
    cols_players = st.columns(num_cols)
    for col_idx in range(num_cols):
        with cols_players[col_idx]:
            for row_idx in range(num_per_col):
                i = col_idx * num_per_col + row_idx
                if i < n_players:
                    player = REGULAR_PLAYERS[i]
                    label = player["name"]
                    selected = st.checkbox(label, key=f"player_{player['name']}{i}", value=False)
                    if selected:
                        if st.session_state["player_selection_order"].get(player["name"]) is None:
                            st.session_state["player_selection_order"][player["name"]] = st.session_state["player_counter"]
                            st.session_state["player_counter"] += 1
                    else:
                        st.session_state["player_selection_order"][player["name"]] = None

    ordered_selected = []
    for player in REGULAR_PLAYERS:
        order = st.session_state["player_selection_order"].get(player["name"])
        if order is not None:
            ordered_selected.append((player["name"], order))
    ordered_selected.sort(key=lambda x: x[1])
    selected_regular_players = [p[0] for p in ordered_selected]

    with st.expander("➕ Add Up to 8 Guest Players (optional)", expanded=False):
        guest_players = []
        for i in range(8):
            col_name, col_gender, col_skill = st.columns([2, 1, 1])
            guest_name = col_name.text_input(f"Guest Player {i+1}", key=f"guest_player_{i+1}")
            guest_gender = col_gender.radio("", ["F", "M"], key=f"guest_gender_{i+1}", horizontal=True)
            guest_skill = col_skill.number_input("Skill", min_value=1, max_value=10, value=5, key=f"guest_skill_{i+1}")
            if guest_name.strip():
                guest_players.append((guest_name.strip(), guest_gender, guest_skill))

    raw_player_names_with_gender = []
    raw_player_names_no_gender = []
    final_player_genders = []
    final_player_skills = []

    for name in selected_regular_players:
        for p in REGULAR_PLAYERS:
            if p["name"] == name:
                gsymbol = "<span style='color: magenta; font-weight: bold;'>♀</span>" if p["gender"] == "F" else "<span style='color: cyan; font-weight: bold;'>♂</span>"
                raw_player_names_with_gender.append(f"{p['name']} {gsymbol}")
                raw_player_names_no_gender.append(p["name"])
                final_player_genders.append(p["gender"])
                final_player_skills.append(p["skill"])
                break

    for gname, ggender, gskill in guest_players:
        if ggender == "F":
            raw_player_names_with_gender.append(f"{gname} <span style='color: magenta; font-weight: bold;'>♀</span>")
        else:
            raw_player_names_with_gender.append(f"{gname} <span style='color: cyan; font-weight: bold;'>♂</span>")
        raw_player_names_no_gender.append(gname)
        final_player_genders.append(ggender)
        final_player_skills.append(gskill)

    player_names_with_gender = deduplicate_names(raw_player_names_with_gender)
    player_names_no_gender = deduplicate_names(raw_player_names_no_gender)

    st.write("---")
    st.markdown("**📋 Final Player List (players rested in order of selection):**", unsafe_allow_html=True)
    if player_names_with_gender:
        player_markdown = ""
        for idx, player in enumerate(player_names_with_gender, start=1):
            player_markdown += f"{idx}. {player}\n"
        st.markdown(player_markdown, unsafe_allow_html=True)
    else:
        st.write("No players selected.")

    st.header("🎾 Select Regular Courts")
    REGULAR_COURTS = [f"Court {i}" for i in range(1, 17)]
    num_courts = len(REGULAR_COURTS)
    num_per_col_courts = math.ceil(len(REGULAR_COURTS) / num_cols)
    cols_courts = st.columns(num_cols)
    selected_regular_courts = []
    for col_idx in range(num_cols):
        with cols_courts[col_idx]:
            for row_idx in range(num_per_col_courts):
                i = col_idx * num_per_col_courts + row_idx
                if i < len(REGULAR_COURTS):
                    court = REGULAR_COURTS[i]
                    if st.checkbox(court, key=f"court_{court}{i}"):
                        selected_regular_courts.append(court)
    with st.expander("➕ Add Up to 4 Custom Courts (optional)", expanded=False):
        custom_courts = []
        cols_custom = st.columns(4)
        for i in range(4):
            col = cols_custom[i]
            c_court = col.text_input(f"Custom Court {i+1}", key=f"custom_court_{i+1}")
            if c_court.strip():
                custom_courts.append(c_court.strip())
    court_names = selected_regular_courts + custom_courts
    st.write("---")
    st.markdown("**🏟️ Final Court List:**")
    if court_names:
        court_markdown = ""
        for idx, court in enumerate(court_names, start=1):
            court_markdown += f"{idx}. {court}\n"
        st.markdown(court_markdown)
    else:
        st.write("No courts selected.")

    num_players_selected = len(player_names_with_gender)
    courts_used = determine_courts_to_use(num_players_selected, len(court_names), 4)
    max_unique = compute_max_unique_matchups(num_players_selected, courts_used, 4) if num_players_selected >= courts_used * 4 else 0

    st.header("⚙️ Configuration")
    default_samples = int(min(max_unique, 200000)) if max_unique else 200000
    samples = st.number_input(
        "🔍 Number of random combinations to try (larger = better results but takes longer to run)",
        min_value=1,
        value=default_samples,
        step=50000
    )
    st.info(f"Approx. unique combinations: {int(max_unique):,}")

    reject_mixed = st.checkbox("🏳️‍🌈 Same-Sex Court Filling", value=True)
    enable_skill = st.checkbox("🎯 Skill-based Matchups", value=True)
    with st.expander("⚖️ Skill Penalty Weight", expanded=False):
        skill_weight = st.number_input("Enter skill penalty weight", min_value=1, value=20)

    if st.button("📅 Generate Schedule(s)"):
        if num_players_selected < 4:
            st.error("🚫 Need at least 4 players to schedule a match.")
            st.stop()
        if not court_names:
            st.error("🚫 Please select or enter at least one court.")
            st.stop()

        st.session_state["all_schedules"].clear()
        st.session_state["all_stats"].clear()
        st.session_state["date_str_uk"] = formatted_date

        num_schedules = 1
        with st.spinner("🔄 Generating schedule..."):
            for i in range(num_schedules):
                st.info(f"Generating Schedule {i+1} of {num_schedules}...")
                # NEW: Initialize match history to track previously used configurations.
                schedule_output, teammate_matrix, opponent_matrix, rest_tracker, all_matches_info = assign_matches(
                    player_names=player_names_no_gender,
                    player_genders=final_player_genders,
                    player_skills=final_player_skills,
                    court_names=court_names,
                    court_size=4,
                    total_matches=8,
                    samples=samples,
                    reject_mixed=reject_mixed,
                    enable_skill=enable_skill,
                    skill_weight=skill_weight
                )
                player_schedule_table = build_player_schedule_table(all_matches_info, player_names_no_gender, court_names)
                schedule_html = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <title>Mijas Padellers Match Schedule - {formatted_date} (Version {i+1})</title>
                    <style>
                        body {{
                            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                            background-color: #ffffff;
                            color: #000000;
                            margin: 0;
                            padding: 20px;
                        }}
                        .container {{
                            max-width: 1200px;
                            margin: 0 auto;
                        }}
                        h1 {{
                            text-align: center;
                            color: #333333;
                            margin-bottom: 10px;
                        }}
                        h2 {{
                            text-align: center;
                            color: #666666;
                            margin-bottom: 20px;
                            font-size: 1.2em;
                        }}
                        table {{
                            width: 100%;
                            border-collapse: collapse;
                            margin-bottom: 10px;
                            table-layout: fixed;
                        }}
                        th, td {{
                            padding: 6px;
                            text-align: left;
                            border: 1px solid #dddddd;
                        }}
                        th {{
                            background-color: #f2f2f2;
                            font-size: 1.1em;
                        }}
                        tr:nth-child(even) {{
                            background-color: #f9f9f9;
                        }}
                        tr:nth-child(odd) {{
                            background-color: #ffffff;
                        }}
                        tfoot td {{
                            background-color: #f2f2f2;
                            font-weight: bold;
                            padding: 6px;
                        }}
                        @media print {{
                            body {{
                                margin: 0;
                                padding: 0;
                            }}
                            .container {{
                                width: 100%;
                                margin: 0;
                            }}
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Mijas Padellers Match Schedule</h1>
                        <h2>{formatted_date}</h2>
                        {schedule_output}
                        <hr>
                        <div style="page-break-before: always;">{player_schedule_table}</div>
                    </div>
                </body>
                </html>
                """
                st.session_state["all_schedules"].append(schedule_html)

                stats_html = generate_Schedule_Statistics(
                    teammate_matrix,
                    opponent_matrix,
                    rest_tracker,
                    player_names_with_gender,
                    all_matches_info,
                    final_player_genders,
                    final_player_skills,
                    court_names
                )
                st.session_state["all_stats"].append(stats_html)
        st.success("Schedule generated successfully!")

    if st.session_state["all_schedules"]:
        st.subheader("📥 Download Your Schedule")
        date_str_uk = st.session_state["date_str_uk"] or "Schedule"
        for i, schedule_html in enumerate(st.session_state["all_schedules"]):
            schedule_filename = f"Schedule_{i+1}_{date_str_uk.replace(' ', '_')}.html"
            st.download_button(
                label=f"📄 Download Schedule {i+1}",
                data=schedule_html,
                file_name=schedule_filename,
                mime="text/html"
            )
            stats_html = st.session_state["all_stats"][i]
            if stats_html:
                stats_filename = f"Schedule_Statistics_{i+1}_{date_str_uk.replace(' ', '_')}.html"
                st.download_button(
                    label=f"📄 Download Statistics {i+1}",
                    data=stats_html,
                    file_name=stats_filename,
                    mime="text/html"
                )
        st.divider()

if __name__ == "__main__":
    main()
