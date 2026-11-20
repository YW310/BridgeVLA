import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_oht_trajectory_replay import trajectory, source_args, replay_simulator, command  # noqa: F401
from finetune.OHT import isaaclab_client as runner
from finetune.OHT.runtime.chunk_replay import ChunkReplay


def contract_for(monkeypatch, source, tmp_path):
    from finetune.OHT.data import chunk_dataset
    folder=tmp_path/'bound-v4'
    folder.mkdir()
    (folder/'contract.json').write_text(json.dumps(dict(schema='oht_bridgevla_v4')))
    contract=dict(schema='oht_bridgevla_v4',sha256='bound-fixture',
        chunk=dict(horizon=4,frame_stride=1,action_dt=1/60),
        source_data_configs={f'{source.task}/lerobot_dataset':dict(data_config=source.config)})
    monkeypatch.setattr(chunk_dataset,'load_chunk_contract',lambda path:contract)
    # chunk_replay imports the strict reader directly.
    monkeypatch.setattr('finetune.OHT.runtime.chunk_replay.load_chunk_contract',lambda path:contract)
    return folder


def test_chunk_source_uses_future_raw_ticks_and_marks_short_tail(trajectory,tmp_path,monkeypatch):
    folder=contract_for(monkeypatch,trajectory,tmp_path)
    source=ChunkReplay(source_args(trajectory,replay_contract=str(folder),replay_data_config=None,
        chunk_execute_horizon=4))
    assert source.length==2 and source.raw_transitions==6
    assert source.chunk_spec['horizon']==4
    first,meta=source.action(0)
    tail,last=source.action(1)
    np.testing.assert_allclose(first[:,:3],trajectory.poses[1:5],atol=1e-7)
    np.testing.assert_allclose(tail[:2,:3],trajectory.poses[5:7],atol=1e-7)
    np.testing.assert_array_equal(tail[2:],np.tile(tail[1],(2,1)))
    assert meta['valid_execute_horizon']==4 and last['valid_execute_horizon']==2
    assert (last['source_frame'],last['target_frame'])==(4,6)
    assert last['chunk_target_frames']==[5,6,6,6]
    with pytest.raises(IndexError):source.action(2)
    source.validate_runtime(SimpleNamespace(step_dt=1/60),source_args(trajectory,
        execution_mode='chunk',max_steps=2,max_substeps=1))


def test_native_chunk_replay_consumes_each_tick_once_including_tail(trajectory,tmp_path,monkeypatch):
    folder=contract_for(monkeypatch,trajectory,tmp_path)
    env,state=replay_simulator(monkeypatch,trajectory,position_scale=.01)
    output=tmp_path/'chunks'
    options=command(trajectory,output,'--no-visualize','--replay-targets','chunk',
        '--chunk-execute-horizon','4','--log-interval','0')
    index=options.index('--replay-data-config')
    del options[index:index+2]
    options.extend(['--replay-contract',str(folder)])
    assert runner.main(options)==0
    run=json.loads((output/'run.json').read_text())
    row=json.loads((output/'episode_0000.json').read_text())
    assert run['execution_mode']=='chunk' and run['replay']['server_queries'] is False
    assert run['chunk_spec']['horizon']==4 and run['chunk_execute_horizon']==4
    assert row['stop_reason']=='trajectory_end' and row['steps']==2
    assert env.common_step_counter==6 and len(env.writes)==1
    assert [r['source_frame'] for r in row['actions']]==[0,4]
    assert [r['target_frame'] for r in row['actions']]==[4,6]
    assert [r['execution']['requested_actions'] for r in row['actions']]==[4,2]
    ticks=[t for r in row['actions'] for t in r['execution']['tracking_samples']]
    assert len(ticks)==6
    np.testing.assert_allclose([t['target_pose_xyzw'][:3] for t in ticks],trajectory.poses[1:],atol=1e-7)
    assert all(r['action_source']=='gt_chunk' for r in row['actions'])
    assert row['actions'][1]['timestamp']==pytest.approx(4/60)
    assert env.closed and state.app_closed


def test_prefix_override_cannot_exceed_nominal_horizon():
    from tests.test_oht_chunk_pipeline import fake_bridge
    from finetune.OHT.runtime.chunk_executor import ChunkExecutor
    from finetune.OHT.runtime.chunk_protocol import chunk_spec
    bridge,_=fake_bridge()
    spec=chunk_spec(dict(horizon=4,execute_horizon=2),dict(chunk=dict(action_dt=.01)))
    executor=ChunkExecutor(bridge,spec)
    for invalid in (0,True,spec['execute_horizon']+1):
        with pytest.raises(ValueError,match='Valid chunk prefix'):
            executor.execute(np.empty((spec['horizon'],8)),{},execute_horizon=invalid)


def test_open_gripper_diagnostic_changes_only_commands_not_tcp_or_initial_fingers(trajectory,tmp_path,monkeypatch):
    folder=contract_for(monkeypatch,trajectory,tmp_path)
    options=source_args(trajectory,replay_contract=str(folder),replay_data_config=None,chunk_execute_horizon=4)
    recorded=ChunkReplay(options)
    options.replay_gripper_mode='open'
    opened=ChunkReplay(options)
    assert opened.metadata['gripper_mode']=='open'
    np.testing.assert_array_equal(opened.targets[:,:7],recorded.targets[:,:7])
    np.testing.assert_array_equal(opened.fingers,recorded.fingers)
    np.testing.assert_array_equal(opened.action(0)[0][:,-1],1.)
    assert np.any(recorded.targets[:,-1]==0)
