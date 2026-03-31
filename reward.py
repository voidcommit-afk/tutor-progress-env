def keyword_score(output, keywords):
    if not keywords:
        return 0
    return sum(1 for k in keywords if k.lower() in output.lower()) / len(keywords)


def compute_reward(output, expected):
    breakdown = {}

    score = 0

    if "concepts" in expected:
        c = keyword_score(output, expected["concepts"])
        breakdown["concepts"] = c
        score += 0.25 * c

    if "weaknesses" in expected:
        w = keyword_score(output, expected["weaknesses"])
        breakdown["weaknesses"] = w
        score += 0.25 * w

    if "issues" in expected:
        i = keyword_score(output, expected["issues"])
        breakdown["issues"] = i
        score += 0.25 * i

    if "plan_features" in expected:
        p = keyword_score(output, expected["plan_features"])
        breakdown["plan"] = p
        score += 0.25 * p

    # structure bonus
    structure = any(x in output.lower() for x in ["summary", "plan", "weakness"])
    breakdown["structure"] = 1 if structure else 0
    if structure:
        score += 0.1

    # verbosity penalty
    if len(output.split()) > 120:
        breakdown["verbosity_penalty"] = -0.1
        score -= 0.1

    final = max(0, min(score, 1.0))

    return {
        "score": final,
        "breakdown": breakdown
    }