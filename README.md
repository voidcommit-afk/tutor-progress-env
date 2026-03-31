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

A simple 2-step agent:

1. Uses tool (`extract_concepts`)
2. Produces structured answer

Example performance:

```text
easy:   ~0.6–0.8  
medium: ~0.5–0.7  
hard:   ~0.4–0.6  
```

---

### Why This Matters

Unlike toy environments, this setup:

* models real-world educational workflows
* provides interpretable reward signals
* supports evaluation of reasoning + planning

This makes it suitable for benchmarking AI tutors in edtech systems.

This environment provides interpretable reward decomposition, enabling analysis of agent behavior beyond final scores.