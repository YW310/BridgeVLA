"""Episode reports over validated Parquet/video-derived single-key and V4 caches."""
import json

import numpy as np
import pytest

from tests.test_oht_migration import replay_fixture, FakeAgent  # noqa: F401
from tests.test_oht_chunk_replay import chunk_fixture  # noqa: F401
from finetune.OHT.data.chunk_dataset import OHTChunkDataset
from finetune.OHT.data.dataset import OHTDataset
from finetune.OHT.data.replay import load_contract
from finetune.OHT.eval import open_loop
from finetune.OHT.runtime.policy import Policy


def test_full_single_key_episode_covers_index_and_keeps_old_means(replay_fixture):
    data = OHTDataset(replay_fixture.replay, 'val')
    first = data.samples[0]
    task, episode = first['task'], first['episode_index']
    policy = Policy(FakeAgent(), load_contract(replay_fixture.replay), 'cpu', 'baseline')
    result = open_loop(policy, replay_fixture.replay, tasks=[task], episode_indices=[episode])
    expected = [row for row in data.samples if row['task'] == task and row['episode_index'] == episode]
    report = result['per_episode'][f'{task}/{episode:06d}']
    assert report['complete'] and report['evaluated_samples'] == len(expected)
    assert result['episode_coverage']['complete_episodes'] == 1
    assert [row['frame'] for row in result['predictions']] == [row['frame'] for row in expected]
    assert result['all_samples']['translation_m_mean'] == pytest.approx(np.mean([
        row['translation_m'] for row in result['predictions']]))
    json.dumps(result, allow_nan=False)


def test_full_v4_episode_timeline_matches_real_valid_source_frames(chunk_fixture):
    data = OHTChunkDataset(chunk_fixture.root, 'val')
    task, episode = data.samples[0]['task'], data.samples[0]['episode_index']

    class PolicyFixture:
        contract = data.contract
        chunk_settings = dict(horizon=data.contract['chunk']['horizon'], execute_horizon=2)

        def reset(self):
            pass

        def act(self, sample, goal, frame):
            # Known-error CPU fixture, not a deployable policy or measured score.
            actions = sample['action_chunk'].copy()
            actions[:, 0] += .01
            actions[:, 3:7] *= -1  # q and -q must have identical SO(3) errors.
            actions[~sample['action_valid'], 0] += 100
            return dict(actions=actions, key_action=sample['key_action'])

    result = open_loop(PolicyFixture(), chunk_fixture.root, tasks=[task], episode_indices=[episode])
    selected = [j for j, row in enumerate(data.samples)
                if row['task'] == task and row['episode_index'] == episode]
    frames = {int(frame) for j in selected for frame in data[j]['action_frames'][data[j]['action_valid']]}
    report = result['per_episode'][f'{task}/{episode:06d}']
    assert report['complete']
    assert {row['target_frame'] for row in report['timeline_predictions']} == frames
    assert report['timeline_actions']['samples'] == len(frames)
    assert report['all_actions']['statistics']['translation_m']['max'] == pytest.approx(.01)
    assert report['all_actions']['statistics']['rotation_degrees']['max'] == 0
    assert report['key_goals']['samples'] == sum(bool(data[j]['key_valid']) for j in selected)
    partial = open_loop(PolicyFixture(), chunk_fixture.root, limit=1, tasks=[task], episode_indices=[episode])
    assert not partial['episode_coverage']['all_selected_episodes_complete']
    assert partial['per_episode'][f'{task}/{episode:06d}']['expected_samples'] == len(selected)
