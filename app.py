import streamlit as st
import itertools
import numpy as np
import random
from datetime import date
import pandas as pd

# ---------------------- Helper Functions ----------------------

def initialize_matrices(num_players):
    """
    Initializes teammate and opponent matrices with zeros.
    """
    return (np.zeros((num_players, num_players), dtype=int),
            np.zeros((num_players, num_players), dtype=int))

def initialize_rest_tracker(num_players):
    """
    Initializes a rest tracker for players.
    """
    return np.zeros(num_players, dtype=int)

def evaluate_match(groups, teammate_matrix, opponent_matrix):
    """
    Evaluates the total score for a set of groups based on teammate and opponent matrices.
    """
    total_score = 0
    for group in groups:
        team1, team2 = group[:2], group[2:]
        teammate_score = sum(teammate_matrix[i][j] for i, j in itertools.combinations(team1, 2)) + \
                         sum(teammate_matrix[i][j] for i, j in itertools.combinations(team2, 2))
        opponent_score = sum(opponent_matrix[i][j] for i in team1 for j in team2)
        diversity_penalty = sum(1 if teammate_matrix[i][j] > 0 else 0 for i, j in itertools.combinations(team1, 2)) + \
                            sum(1 if teammate_matrix[i][j] > 0 else 0 for i, j in itertools.combinations(team2, 2)) + \
                            sum(1 if opponent_matrix[i][j] > 0 else 0 for i in team1 for j in team2)
        total_score += teammate_score + opponent_score + diversity_penalty
    return total_score

def find_best_match(players, num_courts, court_size, teammate_matrix, opponent_matrix,
                   match_number, samples=100_000, show_progress=True):
    """
    Finds the best match arrangement by sampling random player groupings.
    """
    best_match, best_score = None, float('inf')

    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
        st.info(f"Finding best players for Match {match_number + 1} ...")
    else:
        progress_bar = None
        status_text = None

    for i in range(samples):
        shuffled_players = players[:]
        random.shuffle(shuffled_players)
        groups = [
            tuple(shuffled_players[i * court_size : (i + 1) * court_size])
            for i in range(num_courts)
        ]
        if len(set(itertools.chain(*groups))) != len(players):
            continue

        score = evaluate_match(groups, teammate_matrix, opponent_matrix)
        if score < best_score:
            best_score, best_match = score, groups

        if show_progress and (i + 1) % 10_000 == 0:
            pct = int((i + 1) / samples * 100)
            progress_bar.progress(pct)
            status_text.write(f"Sample {i + 1:,} of {samples:,}")

    if show_progress:
        progress_bar.progress(100)
        status_text.write(f"Done searching for best match for Match {match_number + 1}!")
        st.info(f"Best arrangement found for Match {match_number + 1}.")

    return best_match

def update_matrices_for_match(groups, teammate_matrix, opponent_matrix):
    """
    Updates the teammate and opponent matrices based on the groups in a match.
    """
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
    """
    Selects players to bench based on their rest counts.
    """
    return sorted(players, key=lambda x: rest_tracker[x])[:num_benched]

def determine_courts_to_use(num_players, available_courts, court_size=4):
    """
    Determines the number of courts to use based on the number of players and available courts.
    """
    max_full_courts = num_players // court_size
    return min(available_courts, max_full_courts)

def deduplicate_names(name_list):
    """
    Removes duplicates from the name list by appending a count suffix.
    """
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
    """
    Formats the match information into a styled HTML structure with advanced CSS for a modern look.
    Reduces whitespace between matches.
    """
    # Define a container for the match with reduced margin-bottom
    match_html = f"""
    <div style="margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); border-radius: 8px; overflow: hidden; background-color: #ffffff;">
        <div style="padding: 10px;">
            <h2 style="text-align: center; color: #333333; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; margin-bottom: 10px; margin-top: 0;">Match {match_number + 1}</h2>
            <table style="width: 100%; border-collapse: collapse; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;">
                <thead>
                    <tr style="background-color: #f0f0f0;">
                        <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Court</th>
                        <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Team 1</th>
                        <th style="padding: 8px; text-align: center; border-bottom: 2px solid #ddd;">vs</th>
                        <th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">Team 2</th>
                    </tr>
                </thead>
                <tbody>
    """
    # Add each court's match with reduced padding
    for court_id, group in enumerate(best_match):
        team1 = " & ".join([player_names[p] for p in group[:2]])
        team2 = " & ".join([player_names[p] for p in group[2:]])
        cname = court_names[court_id]
        row_bg = "#ffffff" if court_id % 2 == 0 else "#f9f9f9"
        match_html += f"""
                    <tr style="background-color: {row_bg};">
                        <td style="padding: 6px; border-bottom: 1px solid #ddd;">{cname}</td>
                        <td style="padding: 6px; border-bottom: 1px solid #ddd;">{team1}</td>
                        <td style="padding: 6px; border-bottom: 1px solid #ddd; text-align: center;">vs</td>
                        <td style="padding: 6px; border-bottom: 1px solid #ddd;">{team2}</td>
                    </tr>
        """

    # Add the resting players
    resting = ", ".join([player_names[p] for p in bench_players])
    match_html += f"""
                </tbody>
                <tfoot>
                    <tr>
                        <td colspan="4" style="padding: 6px; background-color: #f0f0f0; font-weight: bold; margin-top: 10px;">Resting: {resting}</td>
                    </tr>
                </tfoot>
            </table>
        </div>
    </div>
    """
    return match_html

def format_statistics_html(stats, date_str_uk):
    """
    Formats the statistics information into a styled HTML structure.
    """
    stats_html = f"""
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
                margin-bottom: 40px;
                font-size: 1.2em;
            }}
            pre {{
                background-color: #f9f9f9;
                padding: 10px;
                border: 1px solid #dddddd;
                border-radius: 8px;
                overflow-x: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-size: 1em;
                line-height: 1.5em;
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
            <pre>{stats}</pre>
        </div>
    </body>
    </html>
    """
    return stats_html

def assign_matches(player_names, court_names, court_size=4, total_matches=8,
                   samples=100_000, show_progress=True):
    """
    Assigns players to matches across multiple courts and generates the schedule output.
    """
    num_players = len(player_names)
    available_courts = len(court_names)
    courts_to_use = determine_courts_to_use(num_players, available_courts, court_size)
    teammate_matrix, opponent_matrix = initialize_matrices(num_players)
    rest_tracker = initialize_rest_tracker(num_players)
    players = list(range(num_players))
    schedule_output = []

    for match_number in range(total_matches):
        st.info(f"--- Assigning players for Match {match_number + 1} ---")
        total_court_capacity = courts_to_use * court_size
        num_benched = max(0, num_players - total_court_capacity)
        bench_players = select_bench_players(players, rest_tracker, num_benched)
        available_players = [p for p in players if p not in bench_players]

        best_match = find_best_match(
            available_players,
            courts_to_use,
            court_size,
            teammate_matrix,
            opponent_matrix,
            match_number,
            samples=samples,
            show_progress=show_progress
        )
        if best_match:
            update_matrices_for_match(best_match, teammate_matrix, opponent_matrix)
            match_table = format_match_table_html(
                match_number, best_match, court_names, player_names, bench_players
            )
            schedule_output.append(match_table)
            st.success(f"Match {match_number + 1} assigned successfully.")
            for player in bench_players:
                rest_tracker[player] += 1

    return "".join(schedule_output), teammate_matrix, opponent_matrix, rest_tracker

def generate_Schedule_Statistics(teammate_matrix, opponent_matrix, rest_tracker, player_names):
    """
    Generates statistics based on the matches scheduled.
    """
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
                teammates_opponents.append(
                    f"{player_names[other_player]} [{teammate_count},{opponent_count}]"
                )
                teammate_total += teammate_count
                opponent_total += opponent_count
        output.append(f"{player_names[player]} [{rest_count}]: " + ", ".join(teammates_opponents))

    output.append(f"\nTeammate Total: {teammate_total}")
    output.append(f"Opponent Total: {opponent_total}")
    output.append("\nTeammate Count Frequencies:")
    output.extend([f"  Count {count}: {freq}" for count, freq in sorted(teammate_frequencies.items())])
    output.append("\nOpponent Count Frequencies:")
    output.extend([f"  Count {count}: {freq}" for count, freq in sorted(opponent_frequencies.items())])
    return "\n".join(output)

# ---------------------- Main Function ----------------------

def main():
    # Initialize session state if not present
    if "schedule_html" not in st.session_state:
        st.session_state["schedule_html"] = ""
    if "stats_html" not in st.session_state:
        st.session_state["stats_html"] = ""
    if "show_results" not in st.session_state:
        st.session_state["show_results"] = False
    if "date_str_uk" not in st.session_state:
        st.session_state["date_str_uk"] = ""

    # Page Configuration
    st.set_page_config(page_title="Mijas Padellers Match Scheduler", layout="centered", initial_sidebar_state="auto")
    
    # Hide Streamlit's default menu and footer for a cleaner look
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

    # Title without Emoji
    st.title("Mijas Padellers Match Scheduler")

    # Date Selection
    session_date = st.date_input("📅 Session Date", value=date.today())
    formatted_date = session_date.strftime("%A %d %B %Y")
    st.write("Selected date:", formatted_date)

    # Regular Players List
    REGULAR_PLAYERS = [
        "Agneta", "Anna", "Anny", "Bevan", "Bill", "Chris", "Christine", "Cris Burgos",
        "Daryoush", "Declan", "Dee", "Evgen", "Fabian", "Geordie", "Glynis", "Heather",
        "Janet", "John", "Juan", "Joyce", "Julie", "Julia", "Kevan", "Leah", "Linda",
        "Lindsey", "Lynn", "Lynsey", "Maria", "Maxine", "Mike", "Norman", "Paola",
        "Paul", "Ruth", "Sandy", "Scott", "Sharon", "Soraya", "Tania", "Tony", "Travis",
        "Walker", "Wendy"
    ]
    REGULAR_PLAYERS.sort(key=lambda x: x.lower())

    # Select Regular Players
    st.header("👥 Select Regular Players")
    selected_regular_players = []
    cols_players = st.columns(4)
    for i, player in enumerate(REGULAR_PLAYERS):
        col = cols_players[i % 4]
        if col.checkbox(player, key=f"player_{player}{i}"):
            selected_regular_players.append(player)

    # Add Guest Players
    st.header("➕ Add Up to 8 Guest Players (optional)")
    guest_players = []
    cols_guest = st.columns(4)
    for i in range(8):
        col = cols_guest[i % 4]
        guest_name = col.text_input(f"Guest Player {i+1}", key=f"guest_player_{i+1}")
        if guest_name.strip():
            guest_players.append(guest_name.strip())

    # Combine and Deduplicate Player Names
    raw_player_names = selected_regular_players + guest_players
    player_names = deduplicate_names(raw_player_names)

    # Display Final Player List as Numbered Markdown List Starting at 1
    st.write("---")
    st.markdown("**📋 Final Player List:**")
    if player_names:
        player_markdown = ""
        for idx, player in enumerate(player_names, start=1):
            player_markdown += f"{idx}. {player}\n"
        st.markdown(player_markdown)
    else:
        st.write("No players selected.")

    # Select Regular Courts
    st.header("🎾 Select Regular Courts")
    REGULAR_COURTS = [f"Court {i}" for i in range(1, 17)]
    selected_regular_courts = []
    cols_courts = st.columns(4)
    for i, court in enumerate(REGULAR_COURTS):
        col = cols_courts[i % 4]
        if col.checkbox(court, key=f"court_{court}{i}"):
            selected_regular_courts.append(court)

    # Add Custom Courts
    st.header("➕ Add Up to 4 Custom Courts (optional)")
    custom_courts = []
    cols_custom = st.columns(4)
    for i in range(4):
        col = cols_custom[i]
        c_court = col.text_input(f"Custom Court {i+1}", key=f"custom_court_{i+1}")
        if c_court.strip():
            custom_courts.append(c_court.strip())

    # Combine Final Court List as Numbered Markdown List Starting at 1
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

    # Number of Matches and Samples
    st.header("⚙️ Configuration")
    total_matches = st.number_input(
        "🔢 How many matches do you want to schedule?",
        min_value=1,
        value=8
    )

    samples = st.number_input(
        "🔍 How many random samples to try? (WARNING: large values can be slow)",
        min_value=1000,
        value=100_000,
        step=1000
    )

    # Additional Options
    show_progress = st.checkbox("📈 Show detailed progress updates (every 10,000 samples)", value=True)
    show_stats = st.checkbox("📊 Show Schedule Statistics", value=False)

    # Generate Schedule Button
    if st.button("📅 Generate Schedule"):
        if len(player_names) < 4:
            st.error("🚫 Need at least 4 players to schedule a match.")
            st.stop()
        if not court_names:
            st.error("🚫 Please select or enter at least one court.")
            st.stop()

        with st.spinner("🔄 Generating schedule..."):
            schedule_output, teammate_matrix, opponent_matrix, rest_tracker = assign_matches(
                player_names=player_names,
                court_names=court_names,
                court_size=4,
                total_matches=total_matches,
                samples=samples,
                show_progress=show_progress
            )

        # Format the date as "Monday 13 January 2025"
        date_str_uk = session_date.strftime("%A %d %B %Y")
        st.session_state["date_str_uk"] = date_str_uk

        # Assemble the complete HTML document with reduced whitespace and separate subtitle
        schedule_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Mijas Padellers Match Schedule - {date_str_uk}</title>
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
                <h2>{date_str_uk}</h2>
                {schedule_output}
            </div>
        </body>
        </html>
        """

        st.session_state["schedule_html"] = schedule_html

        if show_stats:
            stats = generate_Schedule_Statistics(teammate_matrix, opponent_matrix, rest_tracker, player_names)
            # Assemble statistics HTML with separate subtitle
            stats_html = format_statistics_html(stats, date_str_uk)
            st.session_state["stats_html"] = stats_html
        else:
            st.session_state["stats_html"] = ""

        st.session_state["show_results"] = True

    # Display Download Buttons
    if st.session_state["show_results"]:
        st.subheader("📥 Download Your Schedule")
        # Retrieve date_str_uk from session_state
        date_str_uk = st.session_state.get("date_str_uk", "Schedule")

        # Download Schedule as HTML
        schedule_filename = f"Schedule_{date_str_uk.replace(' ', '_')}.html"
        st.download_button(
            label="📄 Download Schedule HTML",
            data=st.session_state["schedule_html"],
            file_name=schedule_filename,
            mime="text/html"
        )

        # Display and Download Statistics if Enabled
        if st.session_state["stats_html"]:
            st.subheader("📥 Download Schedule Statistics")
            stats_filename = f"Schedule_Statistics_{date_str_uk.replace(' ', '_')}.html"
            st.download_button(
                label="📄 Download Statistics HTML",
                data=st.session_state["stats_html"],
                file_name=stats_filename,
                mime="text/html"
            )

        # Divider
        st.divider()

# ---------------------- Run the App ----------------------

if __name__ == "__main__":
    main()
