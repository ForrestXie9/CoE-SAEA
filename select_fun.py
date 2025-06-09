import numpy as np

def selection(score_set, total_selection_count, current_time_slot, total_time_slots):
    actions = list(score_set.keys())
    num_actions = len(actions)

    # Initialize parameters
    avg_scores = np.zeros(num_actions)
    selection_counts = np.zeros(num_actions)
    min_score_threshold = 0.1  # Minimum baseline for limited data

    # Calculate average scores and selection counts
    for i, action in enumerate(actions):
        scores = score_set[action]
        counts = len(scores)
        selection_counts[i] = counts
        avg_scores[i] = np.mean(scores) if counts > 0 else min_score_threshold

    # Normalize average scores for fair comparison
    max_avg_score = np.max(avg_scores)
    if max_avg_score > 0:
        avg_scores = avg_scores / max_avg_score

    # Exploration-Exploitation trade-off using Epsilon-Greedy strategy
    epsilon = 1.0 - (current_time_slot / total_time_slots)  # Start high, reduce over time
    epsilon = max(epsilon, 0.1)  # Ensure a minimum level of exploration

    # Select action based on epsilon-greedy strategy
    if np.random.rand() < epsilon:
        # Explore: randomly select an action
        action_index = np.random.choice(num_actions)
    else:
        # Exploit: calculate adjusted scores based on selection counts
        adjusted_scores = avg_scores + (1 / (1 + selection_counts))  # Weighting by selection counts
        action_index = np.argmax(adjusted_scores)

    return action_index