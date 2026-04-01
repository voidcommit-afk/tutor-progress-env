---
title: TutorProgressEnv
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

## **TutorProgressEnv — Evaluating AI Tutors on Real Student Data**

### Problem

AI tutors exist, but there is no standardized way to evaluate whether they:

* truly understand student learning gaps
* correctly diagnose weaknesses
* generate actionable, time-constrained study plans

This environment simulates real academic support workflows to benchmark such capabilities.

---

### Environment Overview

This OpenEnv environment models student–tutor interactions using:

* Chat history (student queries)
* Learning context (difficulty, constraints)
* Structured evaluation criteria

Agents interact via:

```text
reset() → observation  
step(action) → (observation, reward, done, info)  
state() → current state
```

---

### Action / Observation Schemas

**Observation**

```json
{
  "task_id": "string",
  "difficulty": "easy|medium|hard",
  "chat_history": ["..."],
  "constraints": { "exam_in_days": 5, "time_per_day": "2 hours" },
  "step_count": 0
}
```

**Action**

```json
{
  "type": "tool|final_answer",
  "content": "string",
  "tool_name": "extract_concepts|detect_weakness (optional)"
}
```

---

### Tasks

Three difficulty levels:

* **Easy** → Summarization of student understanding
* **Medium** → Weakness & pattern detection
* **Hard** → Constrained study plan generation

Hard tasks include:

* time limits (e.g., 2 hrs/day)
* exam deadlines
* prioritization requirements

---

### Action Space

* `tool` → use helper functions

  * `extract_concepts`
  * `detect_weakness`

* `final_answer` → produce final response

---

### Reward Design

Dense reward signal (0–1):

* Concept / weakness / issue detection
* Study plan quality
* Structural correctness
* Verbosity penalty

Additionally:

* Intermediate reward for tool usage
* Final reward based on multi-factor grading

---

### Grading

Deterministic scoring:

```text
score = weighted(keyword overlap + structure + plan features)
```

Breakdown includes:

* concepts
* weaknesses
* planning quality
* structure

---

### Baseline

The baseline inference script uses the OpenAI client to generate a deterministic response (temperature=0) for every task.

Example performance:

```text
easy:   ~0.6–0.8  
medium: ~0.5–0.7  
hard:   ~0.4–0.6  
```

---

### Setup

Required environment variables for inference:

```text
API_BASE_URL   # API endpoint for the LLM provider (required by checklist)
MODEL_NAME     # Model identifier
OPENAI_API_KEY # OpenAI-compatible API key
HF_TOKEN       # Hugging Face token for Space deployment
```

Optional mock mode (no external API calls):

```text
MOCK_INFERENCE=1
```

Run baseline inference:

```bash
python inference.py
```

Run OpenEnv validation:

```bash
openenv validate
```

---

### Hugging Face Space Deployment

1. Build and run locally:

```bash
docker build -t tutor-progress-env .
docker run -p 7860:7860 tutor-progress-env
```

2. Create a Docker Space on Hugging Face and set:
   - `HF_TOKEN`
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `OPENAI_API_KEY`

---

### Why This Matters

Unlike toy environments, this setup:

* models real-world educational workflows
* provides interpretable reward signals
* supports evaluation of reasoning + planning

This makes it suitable for benchmarking AI tutors in edtech systems.

This environment provides interpretable reward decomposition, enabling analysis of agent behavior beyond final scores.
