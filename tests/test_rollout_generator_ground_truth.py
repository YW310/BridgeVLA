import sys
from multiprocessing import Value
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "bridgevla" / "libs" / "YARR"))

# open3d is only used by RolloutGenerator's optional point-cloud visualization.
# Keep this rollout-accounting regression test runnable in lightweight envs.
try:
    import open3d  # noqa: F401
except ModuleNotFoundError:
    sys.modules["open3d"] = SimpleNamespace()

from yarr.utils.rollout_generator import RolloutGenerator  # noqa: E402
from yarr.utils.stat_accumulator import SimpleAccumulator  # noqa: E402
from yarr.utils.transition import Transition  # noqa: E402


class _Agent:
    def reset(self):
        pass

    def act(self, *args, **kwargs):
        raise AssertionError("The policy must not run during ground-truth replay.")


class _GroundTruthEnv:
    _lang_goal = "test goal"
    active_task_id = 0

    def __init__(self):
        self.retry_attempts = []

    def reset_to_demo(self, seed, retry_attempt=0):
        self.retry_attempts.append(retry_attempt)
        return {"state": np.asarray([seed], dtype=np.float32)}

    def get_ground_truth_action(self, seed):
        return [np.asarray([seed], dtype=np.float32)]

    def step(self, act_result):
        return Transition(
            observation={"state": np.asarray([1], dtype=np.float32)},
            reward=0.0,
            terminal=False,
            info={},
        )


def _failed_ground_truth_rollout(seed):
    generator = RolloutGenerator(env_device="cpu")
    return list(generator.generator(
        Value("i", 0),
        _GroundTruthEnv(),
        _Agent(),
        episode_length=50,
        timesteps=1,
        eval=True,
        eval_demo_seed=seed,
        record_enabled=False,
        replay_ground_truth=True,
    ))


def test_exhausted_ground_truth_actions_close_failed_episode():
    rollout = _failed_ground_truth_rollout(0)

    assert len(rollout) == 1
    assert rollout[-1].reward == 0.0
    assert rollout[-1].terminal is True
    assert rollout[-1].timeout is True


def test_failed_ground_truth_episodes_are_included_in_final_score():
    accumulator = SimpleAccumulator()
    for seed in (0, 1):
        for transition in _failed_ground_truth_rollout(seed):
            accumulator.step(transition, eval=True)

    summaries = {summary.name: summary.value for summary in accumulator.pop()}
    assert summaries["eval_envs/return"] == pytest.approx(0.0)
    assert summaries["eval_envs/length"] == pytest.approx(1.0)


def test_empty_ground_truth_action_sequence_has_clear_error():
    env = _GroundTruthEnv()
    env.get_ground_truth_action = lambda seed: []
    generator = RolloutGenerator(env_device="cpu")

    with pytest.raises(ValueError, match="at least one expert action"):
        list(generator.generator(
            Value("i", 0), env, _Agent(), episode_length=50,
            timesteps=1, eval=True, replay_ground_truth=True,
        ))


def test_ground_truth_retry_resets_same_demo_as_retry_attempt():
    env = _GroundTruthEnv()
    generator = RolloutGenerator(env_device="cpu")

    rollout = list(generator.generator(
        Value("i", 0), env, _Agent(), episode_length=50,
        timesteps=1, eval=True, eval_demo_seed=7,
        replay_ground_truth=True, ground_truth_attempt=2,
    ))

    assert len(rollout) == 1
    assert env.retry_attempts == [2]
