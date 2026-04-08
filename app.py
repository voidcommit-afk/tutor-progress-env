import copy
import json
import logging
import os
import sys
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import ValidationError

sys.path.append(os.path.dirname(__file__))

from baseline import run_agent
from env import TutorEnv
from grader import grade
from schemas import Action, Observation, ResetRequest, StepRequest

logger = logging.getLogger("tutor_env")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(title="TutorProgressEnv", version="0.2.0")

tasks_easy = json.load(open("tasks/easy.json"))
tasks_medium = json.load(open("tasks/medium.json"))
tasks_hard = json.load(open("tasks/hard.json"))
ALL_TASKS = tasks_easy + tasks_medium + tasks_hard


def _find_task(task_id: str):
    for task in ALL_TASKS:
        if task["task_id"] == task_id:
            return task
    return None


class EnvRegistry:
    def __init__(self, tasks):
        self.tasks = copy.deepcopy(tasks)
        self.sessions = {}

    def get(self, session_id: str) -> TutorEnv:
        if session_id not in self.sessions:
            self.sessions[session_id] = TutorEnv(copy.deepcopy(self.tasks))
        return self.sessions[session_id]

    def reset(self, session_id: str, task: dict, seed: Optional[int] = None, stochastic: bool = False):
        env = self.get(session_id)
        return env.reset(task, session_id=session_id, seed=seed, stochastic=stochastic)

    def step(self, session_id: str, action: Action):
        env = self.get(session_id)
        result = env.step(action)
        result.observation.session_id = session_id
        result.info["session_id"] = session_id
        return result

    def state(self, session_id: str):
        return self.get(session_id).state()


registry = EnvRegistry(ALL_TASKS)


@app.get("/")
def root():
    return {
        "env": "TutorProgressEnv",
        "status": "running",
        "endpoints": [
            "/tasks",
            "/session/new",
            "/reset",
            "/step",
            "/state",
            "/grader",
            "/baseline",
            "/health",
            "/metadata",
            "/schema",
            "/mcp",
        ],
    }


@app.get("/tasks")
def get_tasks():
    return {
        "num_tasks": len(ALL_TASKS),
        "tasks": ALL_TASKS,
        "action_schema": {
            "type": ["tool", "final_answer"],
            "tools": ["extract_concepts", "detect_weakness"],
        },
    }


@app.get("/session/new")
def new_session():
    session_id = str(uuid.uuid4())
    registry.get(session_id)
    return {"session_id": session_id}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "TutorProgressEnv",
        "description": "OpenEnv environment for evaluating AI tutor diagnosis and study-plan quality.",
        "version": "0.2.0",
        "supports_session_isolation": True,
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
            "step_count": {"type": "integer"},
            "episode_done": {"type": "boolean"},
            "features": {"type": "object"},
        },
    }
    return {"action": action_schema, "observation": observation_schema, "state": state_schema}


@app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    try:
        if body and not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Invalid reset payload: expected JSON object.")
        payload = ResetRequest(**body) if body else ResetRequest()
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid reset payload: {e.errors()}")
    task_id = payload.task_id or request.query_params.get("task_id")
    session_id = payload.session_id or request.query_params.get("session_id") or "default"

    if not task_id:
        task = ALL_TASKS[0]
    else:
        task = _find_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"task_id '{task_id}' not found")

    obs = registry.reset(session_id=session_id, task=task, seed=payload.seed, stochastic=payload.stochastic)
    logger.info("reset session=%s task_id=%s stochastic=%s", session_id, task["task_id"], payload.stochastic)
    return obs


@app.post("/step")
def step(payload: StepRequest):
    session_id = payload.session_id or "default"
    action = payload.to_action()
    try:
        result = registry.step(session_id, action)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid action payload: {e.errors()}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    logger.info("step session=%s action_type=%s done=%s reward=%.3f", session_id, action.type, result.done, result.reward)
    return result


@app.get("/state")
def state(session_id: str = Query(default="default")):
    return registry.state(session_id)


@app.get("/grader")
def grader(output: str, task_id: str):
    task = _find_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"task_id '{task_id}' not found")
    return {"score": grade(output, task["expected"], constraints=task.get("constraints"))}


@app.get("/baseline")
def baseline():
    results = {}
    env = TutorEnv(copy.deepcopy(ALL_TASKS), seed=123)
    for task in ALL_TASKS:
        score = run_agent(env, task)
        results[task["task_id"]] = score
    return results


@app.post("/mcp")
def mcp(payload: dict):
    req_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params") or {}
    session_id = params.get("session_id", "default")

    try:
        if method in ("initialize", "mcp.initialize"):
            return {"jsonrpc": "2.0", "id": req_id, "result": {"server": {"name": "TutorProgressEnv", "version": "0.2.0"}}}
        if method in ("tools/list", "list_tools"):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {"name": "env.reset", "description": "Reset environment with optional task_id/session_id"},
                        {"name": "env.step", "description": "Take a step using action payload"},
                        {"name": "env.state", "description": "Get current environment state for a session"},
                    ]
                },
            }
        if method in ("env.reset", "reset"):
            task_id = params.get("task_id")
            task = ALL_TASKS[0] if not task_id else _find_task(task_id)
            if task is None:
                raise ValueError(f"task_id '{task_id}' not found")
            obs = registry.reset(
                session_id=session_id,
                task=task,
                seed=params.get("seed"),
                stochastic=bool(params.get("stochastic", False)),
            )
            result = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        if method in ("env.step", "step"):
            action_payload = params.get("action") or {}
            result_obj = registry.step(session_id, Action(**action_payload))
            result = result_obj.model_dump() if hasattr(result_obj, "model_dump") else result_obj.dict()
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        if method in ("env.state", "state"):
            return {"jsonrpc": "2.0", "id": req_id, "result": registry.state(session_id)}
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}
    except Exception as e:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": str(e)}}
