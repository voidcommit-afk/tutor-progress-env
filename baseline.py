import sys
import os
sys.path.append(os.path.dirname(__file__))

import json
from env import TutorEnv  


from schemas import Action

def run_agent(env, task):
    state = env.reset(task)

    action1 = {
        "type": "tool",
        "tool_name": "extract_concepts"
    }
    env.step(Action(**action1))

    action2 = {
        "type": "final_answer",
        "content": "Summary: student has weaknesses. Plan: prioritize concepts, timed practice, and revision."
    }
    res = env.step(Action(**action2))

    return res.reward


def run_baseline():
    files = ["tasks/easy.json", "tasks/medium.json", "tasks/hard.json"]
    results = {}

    for file in files:
        tasks = json.load(open(file))
        env = TutorEnv(tasks)

        for task in tasks:
            score = run_agent(env, task)
            results[task["task_id"]] = score

    return results


if __name__ == "__main__":
    print(run_baseline())