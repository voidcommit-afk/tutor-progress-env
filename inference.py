import json
from env import TutorEnv
from baseline import run_agent


def load_tasks():
    tasks = []
    for file in ["tasks/easy.json", "tasks/medium.json", "tasks/hard.json"]:
        with open(file) as f:
            tasks.extend(json.load(f))
    return tasks


def main():
    tasks = load_tasks()
    env = TutorEnv(tasks)

    results = {}

    for task in tasks:
        score = run_agent(env, task)
        results[task["task_id"]] = score

    # print results (required)
    print("Baseline Results:")
    for k, v in results.items():
        print(f"{k}: {round(v, 3)}")

    # also return avg
    avg = sum(results.values()) / len(results)
    print(f"\nAverage Score: {round(avg, 3)}")


if __name__ == "__main__":
    main()