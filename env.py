import copy
import random
from typing import Optional

from schemas import Observation, Action, StepResult
from reward import compute_reward
from tools import extract_concepts, detect_weakness


class TutorEnv:
    def __init__(self, tasks, seed: Optional[int] = None, stochastic: bool = False):
        self.tasks = copy.deepcopy(tasks)
        self.current = None
        self.current_chat_history = []
        self.step_count = 0
        self.tool_output = None
        self.episode_done = False
        self.last_action_type = None
        self.stochastic = stochastic
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_steps = 4

    def _build_chat_history(self, chat_history):
        if not self.stochastic:
            return list(chat_history)

        noise_candidates = [
            "Reminder: Focus on understanding, not rote memorization.",
            "Distractor: Student also mentioned sleep issues before exams.",
            "Hint: Time budgeting is often the main bottleneck.",
        ]
        history = list(chat_history)
        if self.rng.random() < 0.4:
            history.append(self.rng.choice(noise_candidates))
        return history

    def _extract_features(self):
        constraints = (self.current or {}).get("constraints") or {}
        text = " ".join(self.current_chat_history).lower()
        return {
            "message_count": len(self.current_chat_history),
            "token_count": len(text.split()),
            "has_constraints": bool(constraints),
            "exam_in_days": constraints.get("exam_in_days"),
            "has_time_budget": bool(constraints.get("time_per_day")),
            "mentions_exam": ("exam" in text),
            "mentions_time_pressure": ("time" in text or "timed" in text),
        }

    def _observation(self, session_id: Optional[str] = None):
        return Observation(
            task_id=self.current["task_id"],
            difficulty=self.current["difficulty"],
            chat_history=list(self.current_chat_history),
            constraints=self.current.get("constraints"),
            step_count=self.step_count,
            features=self._extract_features(),
            session_id=session_id,
        )

    def reset(self, task, session_id: Optional[str] = None, seed: Optional[int] = None, stochastic: Optional[bool] = None):
        self.current = copy.deepcopy(task)
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)
        if stochastic is not None:
            self.stochastic = stochastic
        self.current_chat_history = self._build_chat_history(self.current["chat_history"])
        self.step_count = 0
        self.tool_output = None
        self.episode_done = False
        self.last_action_type = None
        return self._observation(session_id=session_id)

    def step(self, action: Action):
        if self.current is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        if self.episode_done:
            raise ValueError("Episode already finished. Call reset() before calling step() again.")
        if self.step_count >= self.max_steps:
            self.episode_done = True
            raise ValueError("Maximum step limit reached. Call reset() to start a new episode.")
        if action.type not in {"tool", "final_answer"}:
            raise ValueError(f"Invalid action type: {action.type}")
        if action.type == "tool" and not action.tool_name:
            raise ValueError("tool_name is required when type='tool'.")
        if action.type == "final_answer" and not (action.content or "").strip():
            raise ValueError("content is required when type='final_answer'.")

        self.step_count += 1
        self.last_action_type = action.type

        # --- TOOL STEP ---
        if action.type == "tool":
            if action.tool_name == "extract_concepts":
                self.tool_output = extract_concepts(self.current_chat_history)

            elif action.tool_name == "detect_weakness":
                self.tool_output = detect_weakness(self.current_chat_history)

            else:
                raise ValueError(f"Unknown tool: {action.tool_name}")

            # append tool output to observation
            self.current_chat_history = list(self.current_chat_history) + [f"[tool:{action.tool_name}] {self.tool_output}"]
            obs = self._observation()

            return StepResult(
                observation=obs,
                reward=0.08,
                done=False,
                info={
                    "tool_output": self.tool_output,
                    "action_valid": True,
                    "step_budget_remaining": self.max_steps - self.step_count,
                },
            )

        # --- FINAL STEP ---
        elif action.type == "final_answer":
            output = action.content

            result = compute_reward(
                output,
                self.current["expected"],
                constraints=self.current.get("constraints"),
                tool_output=self.tool_output,
                step_count=self.step_count,
            )
            self.episode_done = True

            return StepResult(
                observation=self._observation(),
                reward=result["score"],
                done=True,
                info=result["breakdown"],
            )

    def state(self):
        if self.current is None:
            return None
        return {
            "task_id": self.current["task_id"],
            "difficulty": self.current["difficulty"],
            "step_count": self.step_count,
            "episode_done": self.episode_done,
            "last_action_type": self.last_action_type,
            "seed": self.seed,
            "stochastic": self.stochastic,
            "features": self._extract_features(),
        }
