import json
from collections import defaultdict

from baseline import generic_policy, heuristic_policy
from env import TutorEnv
from schemas import Action


def load_tasks():
    tasks = []
    for file in ["tasks/easy.json", "tasks/medium.json", "tasks/hard.json"]:
        with open(file) as f:
            tasks.extend(json.load(f))
    with open("tasks/splits.json") as f:
        splits = json.load(f)
    return tasks, splits


def evaluate_policy(tasks, task_ids, policy_fn):
    task_map = {t["task_id"]: t for t in tasks}
    selected = [task_map[t] for t in task_ids if t in task_map]
    env = TutorEnv(tasks, seed=42)

    results = {}
    outputs = {}
    by_difficulty = defaultdict(list)

    for task in selected:
        env.reset(task)
        env.step(Action(type="tool", tool_name="extract_concepts"))
        output = policy_fn(task)
        outputs[task["task_id"]] = output
        res = env.step(Action(type="final_answer", content=output))
        score = float(res.reward)
        results[task["task_id"]] = score
        by_difficulty[task["difficulty"]].append(score)

    avg = round(sum(results.values()) / max(1, len(results)), 3)
    diff_avg = {k: round(sum(v) / len(v), 3) for k, v in by_difficulty.items()}
    failure_cases = sorted(results.items(), key=lambda x: x[1])[:3]
    failure_examples = [{"task_id": tid, "score": score, "output": outputs[tid]} for tid, score in failure_cases]

    return {
        "average": avg,
        "by_difficulty": diff_avg,
        "scores": results,
        "failure_examples": failure_examples,
    }


def main():
    tasks, splits = load_tasks()
    report = {}
    for split_name, ids in splits.items():
        report[split_name] = {
            "generic": evaluate_policy(tasks, ids, generic_policy),
            "heuristic": evaluate_policy(tasks, ids, heuristic_policy),
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
