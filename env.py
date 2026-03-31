from schemas import Observation, Action, StepResult
from reward import compute_reward
from tools import extract_concepts, detect_weakness


class TutorEnv:
    def __init__(self, tasks):
        self.tasks = tasks
        self.current = None
        self.step_count = 0
        self.tool_output = None

    def reset(self, task):
        self.current = task
        self.step_count = 0
        self.tool_output = None

        return Observation(
            task_id=task["task_id"],
            difficulty=task["difficulty"],
            chat_history=task["chat_history"],
            constraints=task.get("constraints"),
            step_count=self.step_count
        )

    def step(self, action: Action):
        if self.current is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        self.step_count += 1

        # --- TOOL STEP ---
        if action.type == "tool":
            if action.tool_name == "extract_concepts":
                self.tool_output = extract_concepts(self.current["chat_history"])

            elif action.tool_name == "detect_weakness":
                self.tool_output = detect_weakness(self.current["chat_history"])

            else:
                raise ValueError(f"Unknown tool: {action.tool_name}")

            # append tool output to observation
            obs = Observation(
                task_id=self.current["task_id"],
                difficulty=self.current["difficulty"],
                chat_history=self.current["chat_history"] + [str(self.tool_output)],
                constraints=self.current.get("constraints"),
                step_count=self.step_count
            )

            return StepResult(
                observation=obs,
                reward=0.1,  # small reward for useful interaction
                done=False,
                info={"tool_output": self.tool_output}
            )

        # --- FINAL STEP ---
        elif action.type == "final_answer":
            output = action.content

            result = compute_reward(output, self.current["expected"])

            obs = Observation(
                task_id=self.current["task_id"],
                difficulty=self.current["difficulty"],
                chat_history=self.current["chat_history"],
                constraints=self.current.get("constraints"),
                step_count=self.step_count
            )

            return StepResult(
                observation=obs,
                reward=result["score"],
                done=True,
                info=result["breakdown"]
            )

        else:
            raise ValueError(f"Invalid action type: {action.type}")

    def state(self):
        return self.current