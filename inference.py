import json
import os
from env import TutorEnv
from schemas import Action


def load_tasks():
    tasks = []
    for file in ["tasks/easy.json", "tasks/medium.json", "tasks/hard.json"]:
        with open(file) as f:
            tasks.extend(json.load(f))
    return tasks


def _mock_output(task, constraints):
    expected = task.get("expected", {})
    summary_terms = expected.get("summary_points", []) or expected.get("concepts", [])
    diagnosis_terms = expected.get("weaknesses", []) or expected.get("issues", [])
    plan_terms = expected.get("plan_features", []) or []
    must_terms = expected.get("must_include", []) or []

    summary = "Summary: " + (", ".join(summary_terms) if summary_terms else "student needs help")
    diagnosis = "Diagnosis: " + (", ".join(diagnosis_terms) if diagnosis_terms else "learning gap")
    plan = "Plan: " + (", ".join(plan_terms + must_terms) if (plan_terms or must_terms) else "practice and review")
    constraints_line = "Constraints: " + (json.dumps(constraints) if constraints else "none")
    return "\n".join([summary, diagnosis, plan, constraints_line])


def main():
    tasks = load_tasks()
    env = TutorEnv(tasks)

    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    api_key = os.getenv("OPENAI_API_KEY")
    mock_inference = os.getenv("MOCK_INFERENCE", "").lower() in {"1", "true", "yes", "on"}
    missing = [k for k, v in {
        "API_BASE_URL": api_base_url,
        "MODEL_NAME": model_name,
        "OPENAI_API_KEY": api_key,
    }.items() if not v]

    use_api = (not mock_inference) and (len(missing) == 0)
    if missing and not mock_inference:
        print(f"[WARN] Missing env vars {missing}; falling back to MOCK_INFERENCE mode.")

    client = None
    if use_api:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=api_base_url)
        except Exception as e:
            print(f"[WARN] Failed to initialize OpenAI client ({e}); using MOCK_INFERENCE mode.")
            use_api = False

    results = {}

    for task in tasks:
        state = env.reset(task)

        constraints = task.get("constraints") or {}
        constraints_text = ""
        if constraints:
            constraints_text = f"\nConstraints: {json.dumps(constraints)}"

        prompt = (
            "You are an AI tutor evaluator. Read the student chat and produce a concise response with:\n"
            "Summary, Diagnosis, Plan, Constraints. Keep it short, actionable, and mention time/days if given.\n"
            f"Chat: {state.chat_history}{constraints_text}"
        )

        output = None
        if use_api and client is not None:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "Respond in four labeled lines: Summary:, Diagnosis:, Plan:, Constraints:."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=256,
                )
                output = completion.choices[0].message.content.strip()
            except Exception as e:
                print(f"[WARN] API inference failed on {task['task_id']} ({e}); using mock output.")

        if not output:
            output = _mock_output(task, constraints)

        action = Action(type="final_answer", content=output)
        res = env.step(action)
        score = res.reward
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
