import itertools
import numpy as np
import random

# Initialize matrices for teammate and opponent tracking
def initialize_matrices(num_players):
    return (np.zeros((num_players, num_players), dtype=int),
            np.zeros((num_players, num_players), dtype=int))

# Initialize the rest tracker
def initialize_rest_tracker(num_players):
    return np.zeros(num_players, dtype=int)

# Evaluate the score of a full match based on teammate and opponent counts
def evaluate_match(groups, teammate_matrix, opponent_matrix):
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

# Find the best match assignment using a sampling approach
def find_best_match(players, num_courts, court_size, teammate_matrix, opponent_matrix, match_number, samples=50000):
    best_match, best_score = None, float('inf')

    for i in range(samples):
        shuffled_players = players[:]
        random.shuffle(shuffled_players)
        groups = [tuple(shuffled_players[i * court_size:(i + 1) * court_size]) for i in range(num_courts)]

        if len(set(itertools.chain(*groups))) != len(players):
            continue

        score = evaluate_match(groups, teammate_matrix, opponent_matrix)
        if score < best_score:
            best_score, best_match = score, groups

        if (i + 1) % 10000 == 0:
            print(f"Match {match_number + 1}: {i + 1} samples processed...")

    return best_match

# Update matrices after assigning players to all courts in a match
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

# Select players to rest for the current match
def select_bench_players(players, rest_tracker, num_benched):
    # Sort players by rest_count ascending and select the first num_benched
    return sorted(players, key=lambda x: rest_tracker[x])[:num_benched]

# Determine the number of courts to use based on players and available courts
def determine_courts_to_use(num_players, available_courts, court_size=4):
    max_full_courts = num_players // court_size
    courts_to_use = min(available_courts, max_full_courts)
    return courts_to_use

# Helper function to format match as a table with 'vs' column
def format_match_table(match_number, best_match, court_names, player_names, bench_players):
    column_width = 20
    total_width = column_width * 2 + 29  # Adjust for total row width
    match_header = f"| Match {match_number + 1:<5} | {'Team 1':<{column_width}} |  vs   | {'Team 2':<{column_width}} |"
    separator = "|" + "-" * (total_width - 2) + "|"

    table = []
    table.append(separator)
    table.append(match_header)
    table.append(separator)

    for court_id, group in enumerate(best_match):
        team1 = " & ".join([player_names[p] for p in group[:2]])
        team2 = " & ".join([player_names[p] for p in group[2:]])
        court_name = court_names[court_id]

        row = f"| Court {court_name:<5} | {team1:<{column_width}} |  vs   | {team2:<{column_width}} |"
        table.append(row)

    # Resting players
    resting = ", ".join([player_names[p] for p in bench_players])
    resting_row = f"| Resting     | {resting:<{total_width - 14}}|"
    table.append(separator)
    table.append(resting_row)
    table.append(separator)
    table.append("")  # Add an empty line for separation

    return "\n".join(table)

# Assign matches with teammate and opponent tracking
def assign_matches(player_names, court_names, schedule_date, court_size=4, total_matches=50, samples=50000):
    num_players = len(player_names)
    available_courts = len(court_names)
    courts_to_use = determine_courts_to_use(num_players, available_courts, court_size)

    if courts_to_use < available_courts:
        print(f"Adjusting number of courts from {available_courts} to {courts_to_use} based on the number of players.")
        court_names = court_names[:courts_to_use]

    teammate_matrix, opponent_matrix = initialize_matrices(num_players)
    rest_tracker = initialize_rest_tracker(num_players)
    players = list(range(num_players))
    match_assignments = []
    schedule_output = []

    schedule_output.append(f"Mijas Padellers Playing Schedule - {schedule_date}")
    schedule_output.append("")

    for match_number in range(total_matches):
        total_court_capacity = courts_to_use * court_size
        num_benched = max(0, num_players - total_court_capacity)

        bench_players = select_bench_players(players, rest_tracker, num_benched)
        available_players = [p for p in players if p not in bench_players]

        best_match = find_best_match(available_players, courts_to_use, court_size, \
                                     teammate_matrix, opponent_matrix, match_number, samples=samples)
        if best_match:
            update_matrices_for_match(best_match, teammate_matrix, opponent_matrix)
            match_assignments.append((best_match, bench_players))

            # Use the helper function to format the match table
            match_table = format_match_table(match_number, best_match, court_names, player_names, bench_players)
            schedule_output.append(match_table)

        # Update rest counts
        for player in bench_players:
            rest_tracker[player] += 1

    return match_assignments, teammate_matrix, opponent_matrix, rest_tracker, schedule_output

# Generate human-readable output
def generate_Schedule_Statistics(teammate_matrix, opponent_matrix, rest_tracker, player_names):
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
    output.extend([f"  Count {count}: {freq}" for count, freq in sorted(teammate_frequencies.items())])

    output.append("\nOpponent Count Frequencies:")
    output.extend([f"  Count {count}: {freq}" for count, freq in sorted(opponent_frequencies.items())])

    return "\n".join(output)

# Main execution
if __name__ == "__main__":
    print("Please enter the details for your schedule filename:")

    while True:
        try:
            day_of_week = int(input("1. Enter the day of the week of the session (1 = Monday, 7 = Sunday): ").strip())
            if day_of_week < 1 or day_of_week > 7:
                raise ValueError("Day of the week must be between 1 and 7.")
            break
        except ValueError as e:
            print(e)

    while True:
        try:
            day = input("2. Enter the TWO DIGIT date of the session (e.g., 01 for 1st of the month): ").strip()
            if len(day) != 2 or not day.isdigit() or int(day) < 1 or int(day) > 31:
                raise ValueError("Date must be a two-digit number between 01 and 31.")
            break
        except ValueError as e:
            print(e)

    while True:
        try:
            month = input("3. Enter the TWO DIGIT month of the session (e.g., 02 for February): ").strip()
            if len(month) != 2 or not month.isdigit() or int(month) < 1 or int(month) > 12:
                raise ValueError("Month must be a two-digit number between 01 and 12.")
            break
        except ValueError as e:
            print(e)

    while True:
        try:
            year = input("4. Enter the FOUR DIGIT year of the session (e.g., 2024): ").strip()
            if len(year) != 4 or not year.isdigit() or int(year) < 1900:
                raise ValueError("Year must be a valid four-digit number.")
            break
        except ValueError as e:
            print(e)

    schedule_date = f"{day}_{month}_{year}"
    Padel_Schedule_filename = f"Schedule_{day}_{month}_{year}.txt"

    print("Enter player names (one per line). Press Enter on an empty line to finish:")

    player_names = []
    while True:
        player_name = input().strip()
        if not player_name:
            break
        player_names.append(player_name)

    print(f"Total players entered: {len(player_names)}")

    print("Enter court names (one per line). Press Enter on an empty line to finish:")

    court_names = []
    while True:
        court_name = input().strip()
        if not court_name:
            break
        court_names.append(court_name)

    print(f"Total courts entered: {len(court_names)}")

    court_size = 4
    total_matches = 8  # Adjusted to match your example
    samples = 5000

    match_assignments, teammate_matrix, opponent_matrix, rest_tracker, schedule_output = assign_matches(
        player_names, court_names, schedule_date, court_size, total_matches, samples=samples)

    with open(Padel_Schedule_filename, "w") as f:
        f.write("\n".join(schedule_output))

    Schedule_Statistics = generate_Schedule_Statistics(teammate_matrix, opponent_matrix, rest_tracker, player_names)
    with open("Schedule_Statistics.txt", "w") as f:
        f.write(Schedule_Statistics)

    print(f"Schedule saved as {Padel_Schedule_filename} and human-readable output saved as Schedule_Statistics.txt.")
