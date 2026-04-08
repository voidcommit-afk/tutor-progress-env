import re
from collections import Counter
from typing import Iterable, List, Optional


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())


def _tokens(text: str) -> List[str]:
    return [t for t in _normalize(text).split() if t]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def keyword_score(output: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0

    text = _normalize(output)
    out_tokens = set(_tokens(output))
    scores = []
    for keyword in keywords:
        phrase = _normalize(keyword).strip()
        if not phrase:
            continue
        if phrase in text:
            scores.append(1.0)
            continue
        key_tokens = [t for t in phrase.split() if t]
        if not key_tokens:
            scores.append(0.0)
            continue
        overlap = len(set(key_tokens) & out_tokens) / len(set(key_tokens))
        scores.append(overlap)
    return sum(scores) / len(scores) if scores else 0.0


def _constraint_score(output: str, constraints: dict) -> float:
    if not constraints:
        return 1.0

    text = _normalize(output)
    score_parts = []

    exam_days = constraints.get("exam_in_days")
    if exam_days is not None:
        days_hit = str(exam_days) in text or "day" in text or "days" in text
        score_parts.append(1.0 if days_hit else 0.0)

    time_per_day = constraints.get("time_per_day")
    if time_per_day:
        hours_number = "".join(ch for ch in str(time_per_day) if ch.isdigit())
        hours_hit = "hour" in text or "hours" in text or "hr" in text
        number_hit = hours_number in text if hours_number else False
        score_parts.append(1.0 if (hours_hit or number_hit) else 0.0)

    return sum(score_parts) / len(score_parts) if score_parts else 1.0


def _structure_score(output: str) -> float:
    lowered = output.lower()
    labels = ["summary:", "diagnosis:", "plan:", "constraints:"]
    hit = sum(1 for label in labels if label in lowered)
    return hit / len(labels)


def _semantic_proxy_score(output: str, expected: dict) -> float:
    expected_terms = []
    for key in ["concepts", "summary_points", "weaknesses", "pattern", "issues", "plan_features", "must_include"]:
        expected_terms.extend(expected.get(key, []) or [])
    return _jaccard(_tokens(output), _tokens(" ".join(expected_terms)))


def _repetition_penalty(output: str) -> float:
    toks = _tokens(output)
    if len(toks) < 8:
        return 0.0
    counts = Counter(toks)
    max_ratio = max(counts.values()) / max(1, len(toks))
    # Penalize only obvious stuffing/repetition.
    return 0.0 if max_ratio < 0.12 else min(0.15, (max_ratio - 0.12) * 1.5)


def _contradiction_penalty(output: str, expected: dict) -> float:
    text = _normalize(output)
    # Extremely simple contradiction proxy: denies issues while expected has issues/weaknesses.
    expected_has_issues = bool(expected.get("issues") or expected.get("weaknesses"))
    denial = any(x in text for x in ["no issues", "perfectly fine", "nothing wrong"])
    if expected_has_issues and denial:
        return 0.12
    return 0.0


def _verbosity_penalty(words: int) -> float:
    if words > 300:
        return 0.2
    if words > 220:
        return 0.12
    if words > 180:
        return 0.08
    return 0.0


def compute_reward(output, expected, constraints=None, tool_output: Optional[object] = None, step_count: Optional[int] = None):
    output = output or ""
    constraints = constraints or {}
    breakdown = {}
    score = 0.0

    coverage_keys = [k for k in ["concepts", "summary_points", "weaknesses", "pattern", "issues", "plan_features"] if k in expected]
    if coverage_keys:
        per = 0.45 / len(coverage_keys)
        for key in coverage_keys:
            c = keyword_score(output, expected[key])
            breakdown[key] = c
            score += per * c

    if "must_include" in expected:
        m = keyword_score(output, expected["must_include"])
        breakdown["must_include"] = m
        score += 0.15 * m

    structure = _structure_score(output)
    breakdown["structure"] = structure
    score += 0.15 * structure

    constraint_score = _constraint_score(output, constraints)
    breakdown["constraints"] = constraint_score
    score += 0.15 * constraint_score

    semantic = _semantic_proxy_score(output, expected)
    breakdown["semantic_proxy"] = semantic
    score += 0.10 * semantic

    # Modest bonuses for tool utilization and efficient episodes.
    tool_bonus = 0.0
    if tool_output:
        if isinstance(tool_output, list):
            tool_bonus = 0.03 if any(str(x).lower() in output.lower() for x in tool_output) else 0.01
        else:
            tool_bonus = 0.03 if str(tool_output).lower() in output.lower() else 0.01
    breakdown["tool_bonus"] = tool_bonus
    score += tool_bonus

    step_bonus = 0.02 if (step_count is not None and step_count <= 3) else 0.0
    breakdown["step_efficiency_bonus"] = step_bonus
    score += step_bonus

    words = len(output.split())
    if words < 15:
        breakdown["brevity_penalty"] = -0.12
        score -= 0.12

    verbosity_penalty = _verbosity_penalty(words)
    if verbosity_penalty > 0:
        breakdown["verbosity_penalty"] = -verbosity_penalty
        score -= verbosity_penalty

    repetition_penalty = _repetition_penalty(output)
    if repetition_penalty > 0:
        breakdown["repetition_penalty"] = -repetition_penalty
        score -= repetition_penalty

    contradiction_penalty = _contradiction_penalty(output, expected)
    if contradiction_penalty > 0:
        breakdown["contradiction_penalty"] = -contradiction_penalty
        score -= contradiction_penalty

    final = max(0.0, min(score, 1.0))
    return {"score": final, "breakdown": breakdown}
