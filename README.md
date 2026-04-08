---
title: TutorProgressEnv
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

## TutorProgressEnv

OpenEnv environment to evaluate AI tutor quality on:
- student gap diagnosis
- weakness identification
- constrained study-plan generation

The environment is designed for robust hackathon submission behavior: fail-safe inference, required health/metadata/schema endpoints, deterministic seeding, and test/CI coverage.

## Environment API

Core:
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`

Validation/runtime support:
- `GET /health`
- `GET /metadata`
- `GET /schema`
- `POST /mcp`
- `GET /session/new` (session isolation for concurrent runs)

## State and Action

Observation includes:
- `task_id`, `difficulty`, `chat_history`, `constraints`, `step_count`
- `features` (structured diagnostics)
- `session_id`

Action:
- `type`: `tool` or `final_answer`
- `tool_name`: `extract_concepts` or `detect_weakness` (required when `type=tool`)
- `content`: final response text (required when `type=final_answer`)

## Reward Design (v2)

Reward is clipped to `[0, 1]` and combines:
- coverage of expected concepts/weaknesses/issues/plan-features
- must-include terms
- labeled structure quality (`Summary/Diagnosis/Plan/Constraints`)
- constraint adherence (`exam_in_days`, `time_per_day`)
- semantic proxy overlap
- tool-use/step-efficiency bonuses
- anti-gaming penalties:
  - repetition/keyword-stuffing penalty
  - contradiction penalty
  - brevity/verbosity penalties

## Reliability and Reproducibility

- `inference.py` never fail-fast on missing provider vars.
- Falls back to mock inference when provider config/API is unavailable.
- Optional split evaluation via `TASK_SPLIT=train|validation|all`.
- Deterministic execution via `ENV_SEED`.
- Episode guard prevents stepping after `done=True`.

## Task Splits

`tasks/splits.json` defines:
- `train`
- `validation`

Use this for consistent benchmark reporting.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e .[dev]
```

## Submission-safe Env Config

### Option A (most reliable): Mock mode

```bash
export MOCK_INFERENCE=1
export ENV_SEED=42
```

### Option B: Real provider (OpenAI-compatible, e.g. OpenAI/Groq)

```bash
export API_BASE_URL=<provider_base_url>
export MODEL_NAME=<chat_model_name>
export API_KEY=<provider_api_key>
export ENV_SEED=42
```

Compatibility fallback also supported:
- `OPENAI_API_KEY` (if `API_KEY` is not set)

Example Groq-compatible base URL:
- `https://api.groq.com/openai/v1`

### HF deployment token (for push/deploy workflows)

```bash
export HF_TOKEN=<your_hf_token>
```

## Run

```bash
python inference.py
python evaluate.py
```

## Validate

```bash
openenv validate --json --verbose
pytest -q
```

## Docker

```bash
docker build -t tutor-progress-env .
docker run -p 7860:7860 tutor-progress-env
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs:
- compile checks
- pytest
- `openenv validate`
- inference smoke tests in mock mode

## Round 1 Checklist

- [ ] `openenv validate --json --verbose` passes
- [ ] `python inference.py` exits 0 with `MOCK_INFERENCE=1`
- [ ] `python inference.py` exits 0 with provider env vars set
- [ ] `python evaluate.py` produces train/validation report
- [ ] HF Space secrets configured (`MOCK_INFERENCE` or provider vars)
