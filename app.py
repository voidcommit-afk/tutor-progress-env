import sys
import os
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, Request, HTTPException
import json

from env import TutorEnv
from schemas import Action, Observation
from baseline import run_agent

app = FastAPI()

tasks_easy = json.load(open("tasks/easy.json"))
tasks_medium = json.load(open("tasks/medium.json"))
tasks_hard = json.load(open("tasks/hard.json"))

ALL_TASKS = tasks_easy + tasks_medium + tasks_hard
env = TutorEnv(ALL_TASKS)


def _find_task(task_id: str):
    for task in ALL_TASKS:
        if task["task_id"] == task_id:
            return task
    return None


@app.get("/")
def root():
    return {
        "env": "TutorProgressEnv",
        "status": "running",
        "endpoints": [
            "/tasks", "/reset", "/step", "/state", "/grader", "/baseline",
            "/health", "/metadata", "/schema", "/mcp"
        ]
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


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "TutorProgressEnv",
        "description": "OpenEnv environment for evaluating AI tutor diagnosis and study-plan quality."
    }


@app.get("/schema")
def schema():
    action_schema = Action.model_json_schema() if hasattr(Action, "model_json_schema") else Action.schema()
    observation_schema = Observation.model_json_schema() if hasattr(Observation, "model_json_schema") else Observation.schema()
    state_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "difficulty": {"type": "string"},
            "chat_history": {"type": "array", "items": {"type": "string"}},
            "constraints": {"type": "object"},
            "expected": {"type": "object"}
        }
    }
    return {"action": action_schema, "observation": observation_schema, "state": state_schema}


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
        task = _find_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"task_id '{task_id}' not found")

    return env.reset(task)


@app.post("/step")
def step(action: dict):
    return env.step(Action(**action))


@app.get("/state")
def state():
    return env.state()


@app.get("/grader")
def grader(output: str, task_id: str):
    task = _find_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"task_id '{task_id}' not found")
    from grader import grade
    return {"score": grade(output, task["expected"], constraints=task.get("constraints"))}


@app.get("/baseline")
def baseline():
    results = {}

    for task in ALL_TASKS:
        score = run_agent(env, task)
        results[task["task_id"]] = score

    return results


@app.post("/mcp")
def mcp(payload: dict):
    req_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params") or {}

    try:
        if method in ("initialize", "mcp.initialize"):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"server": {"name": "TutorProgressEnv", "version": "0.1.0"}}
            }
        if method in ("tools/list", "list_tools"):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {"name": "env.reset", "description": "Reset environment with optional task_id"},
                        {"name": "env.step", "description": "Take a step with action payload"},
                        {"name": "env.state", "description": "Get current environment state"}
                    ]
                }
            }
        if method in ("env.reset", "reset"):
            task_id = params.get("task_id")
            task = ALL_TASKS[0] if not task_id else _find_task(task_id)
            if task is None:
                raise ValueError(f"task_id '{task_id}' not found")
            result_obj = env.reset(task)
            result = result_obj.model_dump() if hasattr(result_obj, "model_dump") else result_obj.dict()
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        if method in ("env.step", "step"):
            action_payload = params.get("action") or {}
            result_obj = env.step(Action(**action_payload))
            result = result_obj.model_dump() if hasattr(result_obj, "model_dump") else result_obj.dict()
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        if method in ("env.state", "state"):
            return {"jsonrpc": "2.0", "id": req_id, "result": env.state()}

        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}
    except Exception as e:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": str(e)}}
