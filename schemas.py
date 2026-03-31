from pydantic import BaseModel
from typing import List, Optional


class Observation(BaseModel):
    task_id: str
    difficulty: str
    chat_history: List[str]
    constraints: Optional[dict] = None
    step_count: int


class Action(BaseModel):
    type: str   # "tool" | "final_answer"
    content: str = ""
    tool_name: Optional[str] = None


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict