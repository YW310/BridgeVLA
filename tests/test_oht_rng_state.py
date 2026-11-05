"""RNG capture/restore must reproduce every stream a resumed training step consumes."""
import random

import numpy as np
import pytest
import torch

from finetune.OHT.rng_state import (capture_rng_state, gather_rng_states, restore_rank_rng_state,
                                    restore_rng_state)


def draw():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_restore_replays_python_numpy_and_torch_streams():
    random.seed(1), np.random.seed(2), torch.manual_seed(3)
    state = capture_rng_state()
    first = draw()
    assert draw() != first
    restore_rng_state(state)
    assert draw() == first


def test_single_process_gather_and_rank_restore_round_trip():
    random.seed(4), np.random.seed(5), torch.manual_seed(6)
    states = gather_rng_states(1)
    assert len(states) == 1
    expected = draw()
    assert restore_rank_rng_state(states, 0, 1)
    assert draw() == expected


def test_legacy_checkpoints_without_rng_states_are_reported_not_restored():
    assert restore_rank_rng_state(None, 0, 1) is False
    assert restore_rank_rng_state([], 0, 1) is False


def test_rank_count_mismatch_is_an_error():
    with pytest.raises(ValueError, match='cover 1 ranks'):
        restore_rank_rng_state([capture_rng_state()], 0, 2)


def test_gather_uses_the_distributed_collective_with_one_slot_per_rank():
    class FakeDist:
        def all_gather_object(self, output, value):
            for index in range(len(output)):
                output[index] = dict(value, rank=index)

    states = gather_rng_states(3, FakeDist())
    assert [state['rank'] for state in states] == [0, 1, 2]
