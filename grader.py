from reward import compute_reward


def grade(output, expected, constraints=None):
    result = compute_reward(output, expected, constraints=constraints)
    return result["score"]
