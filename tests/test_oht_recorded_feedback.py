import numpy as np
import pytest

from tools.eval_oht_recorded_feedback import run_feedback


class IncrementPolicy:
    def __init__(self):
        self.calls=[]

    def reset(self):
        pass

    def act(self, observation, goal, step):
        self.calls.append({k:np.asarray(v).copy() for k,v in observation.items()})
        actions=np.tile(np.r_[observation['tcp_pose'],1.],(4,1)).astype(np.float32)
        actions[:,:3]+=np.arange(1,5)[:,None]*np.array([.01,0.,0.])
        return dict(actions=actions)


def example(length=11,frames=(0,4,8)):
    poses=np.tile([0.,0.,0.,0.,0.,0.,1.],(length,1)).astype(np.float32)
    poses[:,0]=np.arange(length)*.02
    obs=[dict(frame=f,goal='test',observation=dict(tcp_pose=poses[f].copy(),
        joint_positions=np.arange(6,dtype=np.float32)+f,
        gripper_open01=np.asarray(.25,dtype=np.float32),
        wrist_rgb=np.full((3,2,2),f,dtype=np.uint8))) for f in frames]
    return poses,np.arange(length)/60.,obs


def test_feedback_uses_absolute_endpoint_and_keeps_recorded_observations():
    poses,times,obs=example()
    policy=IncrementPolicy()
    result,rows=run_feedback(policy,iter(obs),poses,times,execute_horizon=4)
    assert result['complete']
    assert result['evaluated_future_ticks']==10
    assert [r['target_frame'] for r in rows]==[4,8,10]
    assert [r['executed_ticks'] for r in rows]==[4,4,2]
    # Second oracle sees recorded x=.08, feedback sees last returned absolute x=.04.
    assert policy.calls[2]['tcp_pose'][0]==pytest.approx(.08)
    assert policy.calls[3]['tcp_pose'][0]==pytest.approx(.04)
    # Third feedback receives .08, not .04+.08 or recorded .16.
    assert policy.calls[5]['tcp_pose'][0]==pytest.approx(.08)
    for i in range(0,len(policy.calls),2):
        for key in ('joint_positions','gripper_open01','wrist_rgb'):
            np.testing.assert_array_equal(policy.calls[i][key],policy.calls[i+1][key])
    np.testing.assert_array_equal(obs[1]['observation']['tcp_pose'],poses[4])
    assert rows[0]['target_timestamp']==pytest.approx(4/60)
    assert rows[-1]['feedback']['position_m'][-1]==pytest.approx(.10)


def test_timing_gap_is_reported_instead_of_feeding_h4_to_frame10():
    poses,times,obs=example(length=15,frames=(0,10))
    result,rows=run_feedback(IncrementPolicy(),iter(obs),poses,times,execute_horizon=4)
    assert not result['complete']
    assert len(rows)==1
    assert 'timeline gap' in result['stop']['message']


def test_feedback_feeds_rotation_as_well_as_position():
    class TurnPolicy(IncrementPolicy):
        def act(self,observation,goal,step):
            result=super().act(observation,goal,step)
            result['actions'][:,3:7]=[0.,0.,1.,0.]
            return result
    poses,times,obs=example()
    policy=TurnPolicy()
    result,rows=run_feedback(policy,iter(obs),poses,times,execute_horizon=4)
    assert result['complete']
    np.testing.assert_array_equal(policy.calls[3]['tcp_pose'][3:],[0,0,1,0])
    assert rows[1]['input_rotation_inconsistency_degrees']==pytest.approx(180.)
