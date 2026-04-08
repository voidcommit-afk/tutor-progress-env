from reward import compute_reward


def test_keyword_stuffing_not_full_score():
    expected = {
        "issues": ["time management", "numericals"],
        "must_include": ["time", "mock"],
        "plan_features": ["timed practice", "prioritization", "mock tests"],
    }
    constraints = {"exam_in_days": 5, "time_per_day": "2 hours"}
    stuffed = "summary diagnosis plan constraints time mock timed practice prioritization mock tests numericals days 2 hours time time time time"
    score = compute_reward(stuffed, expected, constraints=constraints)["score"]
    assert score < 1.0


def test_structured_answer_scores_better_than_unstructured():
    expected = {"concepts": ["fractions", "division"], "summary_points": ["basic math confusion"]}
    constraints = {}
    structured = "Summary: confusion in fractions.\nDiagnosis: division procedure gap.\nPlan: practice worked examples.\nConstraints: none."
    unstructured = "fractions division confusion practice"
    structured_score = compute_reward(structured, expected, constraints=constraints)["score"]
    unstructured_score = compute_reward(unstructured, expected, constraints=constraints)["score"]
    assert structured_score > unstructured_score
