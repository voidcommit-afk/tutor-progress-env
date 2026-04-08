from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class Observation(BaseModel):
    task_id: str
    difficulty: str
    chat_history: List[str]
    constraints: Optional[dict] = None
    step_count: int
    features: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None


class Action(BaseModel):
    type: str   # "tool" | "final_answer"
    content: str = ""
    tool_name: Optional[str] = None


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    seed: Optional[int] = None
    stochastic: bool = False


class StepRequest(BaseModel):
    session_id: Optional[str] = None
    action: Optional[Action] = None
    type: Optional[str] = None
    content: str = ""
    tool_name: Optional[str] = None

    def to_action(self) -> Action:
        if self.action is not None:
            return self.action
        return Action(type=self.type or "", content=self.content, tool_name=self.tool_name)
