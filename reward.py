def keyword_score(output, keywords):
    if not keywords:
        return 0.0
    text = output.lower()
    hits = sum(1 for k in keywords if k.lower() in text)
    return hits / len(keywords)


def _constraint_score(output, constraints):
    if not constraints:
        return 1.0
    text = output.lower()
    score_parts = []

    exam_days = constraints.get("exam_in_days")
    if exam_days is not None:
        has_days = str(exam_days) in text or "days" in text
        score_parts.append(1.0 if has_days else 0.0)

    time_per_day = constraints.get("time_per_day")
    if time_per_day:
        hours_number = "".join(ch for ch in str(time_per_day) if ch.isdigit())
        has_hours = "hour" in text or "hours" in text
        has_number = hours_number in text if hours_number else False
        score_parts.append(1.0 if (has_hours or has_number) else 0.0)

    if not score_parts:
        return 1.0
    return sum(score_parts) / len(score_parts)


def compute_reward(output, expected, constraints=None):
    breakdown = {}
    score = 0.0

    coverage_keys = []
    for key in ["concepts", "summary_points", "weaknesses", "pattern", "issues", "plan_features"]:
        if key in expected:
            coverage_keys.append(key)

    if coverage_keys:
        per = 0.6 / len(coverage_keys)
        for key in coverage_keys:
            c = keyword_score(output, expected[key])
            breakdown[key] = c
            score += per * c

    if "must_include" in expected:
        m = keyword_score(output, expected["must_include"])
        breakdown["must_include"] = m
        score += 0.15 * m

    structure = any(x in output.lower() for x in ["summary", "plan", "diagnosis", "issues"])
    breakdown["structure"] = 1.0 if structure else 0.0
    score += 0.1 * breakdown["structure"]

    constraint_score = _constraint_score(output, constraints or {})
    breakdown["constraints"] = constraint_score
    score += 0.15 * constraint_score

    words = len(output.split())
    if words > 180:
        breakdown["verbosity_penalty"] = -0.1
        score -= 0.1
    if words > 250:
        breakdown["verbosity_penalty"] = -0.2
        score -= 0.2
    if words < 15:
        breakdown["brevity_penalty"] = -0.1
        score -= 0.1

    final = max(0.0, min(score, 1.0))
    return {"score": final, "breakdown": breakdown}
