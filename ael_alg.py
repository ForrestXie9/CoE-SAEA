import numpy as np

def selection(score_set, total_selection_count, current_time_slot, total_time_slots):
    # Initialize variables
    action_count = 8
    avg_scores = np.zeros(action_count)
    selection_counts = np.zeros(action_count)  # Count of how many times each action has been selected

    # Calculate average scores and selection counts
    for action in range(action_count):
        if action in score_set:
            if score_set[action]:
                avg_scores[action] = np.mean(score_set[action])  # Calculate average score
            selection_counts[action] = len(score_set[action])  # Count selections for the action

    # Calculate UCB values
    ucb_values = np.zeros(action_count)
    for action in range(action_count):
        if selection_counts[action] > 0:
            # UCB formula: avg_score + sqrt(2 * log(total_selections) / selection_count)
            ucb_values[action] = avg_scores[action] + np.sqrt(2 * np.log(total_selection_count) / selection_counts[action])
        else:
            # If an action has not been selected, ensure it is favored
            ucb_values[action] = float('inf')  # So that unselected actions are chosen at least once

    # Select action based on UCB values
    action_index = np.argmax(ucb_values)

    return action_index