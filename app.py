import sys
import os
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI
import json

from env import TutorEnv
from schemas import Action
from baseline import run_agent

app = FastAPI()

tasks_easy = json.load(open("tasks/easy.json"))
tasks_medium = json.load(open("tasks/medium.json"))
tasks_hard = json.load(open("tasks/hard.json"))

ALL_TASKS = tasks_easy + tasks_medium + tasks_hard
env = TutorEnv(ALL_TASKS)

@app.get("/")
def root():
    return {
        "env": "TutorProgressEnv",
        "status": "running",
        "endpoints": ["/tasks", "/reset", "/step", "/grader", "/baseline"]
    }


@app.get("/tasks")
def get_tasks():
    return {
        "num_tasks": len(ALL_TASKS),
        "tasks": ALL_TASKS,
        "action_schema": {
            "type": ["tool", "final_answer"],
            "tools": ["extract_concepts", "detect_weakness"]
        }
    }


from fastapi import Request

@app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
    except:
        body = {}

    task_id = body.get("task_id") or request.query_params.get("task_id")

    if not task_id:
        # fallback → pick first task (CRITICAL for validator)
        task = ALL_TASKS[0]
    else:
        task = next(t for t in ALL_TASKS if t["task_id"] == task_id)

    return env.reset(task)


@app.post("/step")
def step(action: dict):
    return env.step(Action(**action))


@app.get("/state")
def state():
    return env.state()


@app.get("/grader")
def grader(output: str, task_id: str):
    task = next(t for t in ALL_TASKS if t["task_id"] == task_id)
    from grader import grade
    return {"score": grade(output, task["expected"], constraints=task.get("constraints"))}

@app.get("/baseline")
def baseline():
    results = {}

    for task in ALL_TASKS:
        score = run_agent(env, task)
        results[task["task_id"]] = score

    return results
