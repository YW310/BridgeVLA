"""Capture and restore the per-rank Python, NumPy and PyTorch RNG streams for exact resumes."""
import random

import numpy as np
import torch


def capture_rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Checkpoint CUDA RNG state was saved with a different number of visible GPUs')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def gather_rng_states(world_size, distributed=None):
    """Return the RNG state of every rank (indexed by rank) on all ranks; call collectively."""
    local = capture_rng_state()
    if world_size == 1 or distributed is None:
        return [local]
    states = [None] * world_size
    distributed.all_gather_object(states, local)
    return states


def restore_rank_rng_state(states, rank, world_size):
    """Restore this rank's stream; returns False for checkpoints written before RNG capture."""
    if not states:
        return False
    if len(states) != world_size:
        raise ValueError(f'Checkpoint RNG states cover {len(states)} ranks but the run has {world_size}')
    restore_rng_state(states[rank])
    return True
