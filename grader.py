def keyword_score(output, keywords):
    if not keywords:
        return 0
    return sum(1 for k in keywords if k.lower() in output.lower()) / len(keywords)

def grade(output, expected):
    score = 0

    if "concepts" in expected:
        score += 0.4 * keyword_score(output, expected["concepts"])

    if "summary_points" in expected:
        score += 0.3 * keyword_score(output, expected["summary_points"])

    if "weaknesses" in expected:
        score += 0.4 * keyword_score(output, expected["weaknesses"])

    if "pattern" in expected:
        score += 0.3 * keyword_score(output, expected["pattern"])

    # structure bonus
    if any(x in output.lower() for x in ["summary", "weakness", "issue"]):
        score += 0.2
    
    if "issues" in expected:
        score += 0.4 * keyword_score(output, expected["issues"])

    if "plan_features" in expected:
        score += 0.4 * keyword_score(output, expected["plan_features"])

    return min(score, 1.0)