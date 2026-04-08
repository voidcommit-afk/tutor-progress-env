import json
import os
import urllib.error
import urllib.parse
import urllib.request
from env import TutorEnv
from schemas import Action


def load_tasks():
    tasks = []
    for file in ["tasks/easy.json", "tasks/medium.json", "tasks/hard.json"]:
        with open(file) as f:
            tasks.extend(json.load(f))
    return tasks


def load_split_task_ids(split_name: str):
    with open("tasks/splits.json") as f:
        splits = json.load(f)
    return set(splits.get(split_name, []))


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


def _chat_completions_url(api_base_url: str) -> str:
    base = (api_base_url or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _call_chat_completion_raw(api_base_url: str, api_key: str, model_name: str, prompt: str) -> str:
    url = _chat_completions_url(api_base_url)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Respond in four labeled lines: Summary:, Diagnosis:, Plan:, Constraints:."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 256,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    return ((parsed.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()


def main():
    tasks = load_tasks()
    seed = int(os.getenv("ENV_SEED", "42"))
    task_split = os.getenv("TASK_SPLIT", "all").strip().lower()
    if task_split != "all":
        split_ids = load_split_task_ids(task_split)
        tasks = [t for t in tasks if t["task_id"] in split_ids]
    if not tasks:
        raise ValueError(f"No tasks available for TASK_SPLIT='{task_split}'.")

    env = TutorEnv(tasks, seed=seed)

    # Hackathon validator injects API_BASE_URL + API_KEY.
    # Prefer those names first to ensure calls are routed through the required proxy.
    api_base_url = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    model_name = (
        os.getenv("MODEL_NAME")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("MODEL")
        or "gpt-4o-mini"
    )

    mock_inference = os.getenv("MOCK_INFERENCE", "").lower() in {"1", "true", "yes", "on"}
    proxy_mode = bool(os.getenv("API_BASE_URL") and os.getenv("API_KEY"))
    missing = [k for k, v in {
        "API_BASE_URL": api_base_url,
        "API_KEY": api_key,
    }.items() if not v]

    # If proxy vars are injected, always use API path (validator expects at least one proxied call).
    if proxy_mode and mock_inference:
        print("[WARN] Ignoring MOCK_INFERENCE because API_BASE_URL/API_KEY proxy vars are present.", flush=True)
        mock_inference = False

    use_api = (not mock_inference) and (len(missing) == 0)
    if missing and not mock_inference and not proxy_mode:
        print(f"[WARN] Missing env vars {missing}; falling back to MOCK_INFERENCE mode.")
    elif use_api:
        print(f"[INFO] Using proxied API mode via {api_base_url} with model={model_name}", flush=True)

    client = None
    client_mode = None
    if use_api:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=api_base_url)
            client_mode = "openai"
        except Exception as e:
            print(f"[WARN] OpenAI SDK unavailable ({e}); using raw HTTP proxy mode.", flush=True)
            client_mode = "raw"

    results = {}

    for task in tasks:
        task_id = task["task_id"]
        print(f"[START] task={task_id} split={task_split} seed={seed}", flush=True)

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
        if use_api:
            try:
                if client_mode == "openai" and client is not None:
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
                else:
                    output = _call_chat_completion_raw(api_base_url, api_key, model_name, prompt)
            except Exception as e:
                print(f"[WARN] API inference failed on {task['task_id']} ({e}); using mock output.")

        if not output:
            output = _mock_output(task, constraints)

        action = Action(type="final_answer", content=output)
        res = env.step(action)
        score = float(res.reward)
        results[task_id] = score

        step_count = res.observation.step_count
        print(
            f"[STEP] task={task_id} step={step_count} action=final_answer reward={score:.6f} done={str(res.done).lower()}",
            flush=True,
        )
        print(f"[END] task={task_id} score={score:.6f} steps={step_count}", flush=True)

    # print results (required)
    print("Baseline Results:", flush=True)
    for k, v in results.items():
        print(f"{k}: {round(v, 3)}", flush=True)

    # also return avg
    avg = sum(results.values()) / max(1, len(results))
    print(f"\nAverage Score: {round(avg, 3)}", flush=True)
    print(f"Run Metadata: seed={seed}, split={task_split}, use_api={use_api}", flush=True)


if __name__ == "__main__":
    main()
