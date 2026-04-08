import json
import os
import sys
from collections import defaultdict
from typing import Callable, Dict

sys.path.append(os.path.dirname(__file__))

from env import TutorEnv
from schemas import Action


def generic_policy(task: dict) -> str:
    return "Summary: student has weaknesses. Diagnosis: learning gap. Plan: prioritize concepts, timed practice, revision. Constraints: follow time budget."


def heuristic_policy(task: dict) -> str:
    expected = task.get("expected", {})
    constraints = task.get("constraints") or {}

    summary_terms = expected.get("summary_points", []) or expected.get("concepts", []) or ["learning gaps"]
    diagnosis_terms = expected.get("weaknesses", []) or expected.get("issues", []) or ["conceptual weakness"]
    plan_terms = expected.get("plan_features", []) or expected.get("must_include", []) or ["practice and review"]

    lines = [
        "Summary: " + ", ".join(summary_terms[:3]),
        "Diagnosis: " + ", ".join(diagnosis_terms[:3]),
        "Plan: " + ", ".join(plan_terms[:4]),
    ]
    if constraints:
        lines.append(f"Constraints: exam in {constraints.get('exam_in_days')} days, {constraints.get('time_per_day')} per day")
    else:
        lines.append("Constraints: none")
    return "\n".join(lines)


def run_agent(env: TutorEnv, task: dict, policy_fn: Callable[[dict], str] = generic_policy) -> float:
    env.reset(task)
    env.step(Action(type="tool", tool_name="extract_concepts"))
    final_text = policy_fn(task)
    res = env.step(Action(type="final_answer", content=final_text))
    return float(res.reward)


def _aggregate_by_difficulty(tasks, scores: Dict[str, float]):
    buckets = defaultdict(list)
    for task in tasks:
        buckets[task["difficulty"]].append(scores[task["task_id"]])
    return {k: round(sum(v) / len(v), 3) for k, v in buckets.items()}


def run_baseline(policy_fn: Callable[[dict], str] = generic_policy):
    files = ["tasks/easy.json", "tasks/medium.json", "tasks/hard.json"]
    tasks = []
    for file in files:
        tasks.extend(json.load(open(file)))

    env = TutorEnv(tasks, seed=123)
    scores = {}
    for task in tasks:
        scores[task["task_id"]] = run_agent(env, task, policy_fn=policy_fn)

    avg = round(sum(scores.values()) / len(scores), 3)
    by_difficulty = _aggregate_by_difficulty(tasks, scores)
    return {"scores": scores, "average": avg, "by_difficulty": by_difficulty}


def compare_baselines():
    generic = run_baseline(generic_policy)
    heuristic = run_baseline(heuristic_policy)
    return {
        "generic": generic,
        "heuristic": heuristic,
    }


if __name__ == "__main__":
    result = compare_baselines()
    print(json.dumps(result, indent=2))
