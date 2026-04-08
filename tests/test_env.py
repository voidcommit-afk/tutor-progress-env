import json

import pytest

from env import TutorEnv
from schemas import Action


def _tasks():
    tasks = []
    for f in ["tasks/easy.json", "tasks/medium.json", "tasks/hard.json"]:
        with open(f) as fh:
            tasks.extend(json.load(fh))
    return tasks


def test_step_before_reset_raises():
    env = TutorEnv(_tasks())
    with pytest.raises(ValueError):
        env.step(Action(type="final_answer", content="Summary: x"))


def test_done_guard_blocks_extra_step():
    tasks = _tasks()
    env = TutorEnv(tasks)
    env.reset(tasks[0])
    env.step(Action(type="final_answer", content="Summary: x\nDiagnosis: y\nPlan: z\nConstraints: none"))
    with pytest.raises(ValueError):
        env.step(Action(type="final_answer", content="again"))


def test_reset_with_seed_is_reproducible_for_stochastic_mode():
    task = _tasks()[0]
    env_a = TutorEnv(_tasks())
    env_b = TutorEnv(_tasks())
    obs_a = env_a.reset(task, seed=123, stochastic=True)
    obs_b = env_b.reset(task, seed=123, stochastic=True)
    assert obs_a.chat_history == obs_b.chat_history
