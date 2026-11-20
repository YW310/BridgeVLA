"""Recorded RGB-D/joints with recursive TCP feedback; no simulator or actuation.

This deliberately combines recorded images/joints/gripper with a predicted TCP.
It measures sensitivity to that inconsistent input, not physical closed-loop success.
"""
import argparse
import json
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.OHT.data.geometry import rotation_error


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(count=0, mean=None, p95=None, max=None, final=None)
    return dict(count=len(values), mean=float(values.mean()), p95=float(np.quantile(values, .95)),
                max=float(values.max()), final=float(values[-1]))


def pose_errors(prediction, target):
    return dict(position_m=np.linalg.norm(prediction[:, :3] - target[:, :3], axis=-1).tolist(),
                rotation_degrees=np.rad2deg(rotation_error(prediction[:, 3:7], target[:, 3:7])).tolist())


def run_feedback(policy, observations, poses, timestamps, *, execute_horizon, writer=None,
                 seed_for_frame=None, progress=None):
    """Copy only recursive TCP; advance exactly H raw ticks (shorter final tail).

    The regression policy is stateless across calls. A frame-specific identical
    seed before the paired calls also matches any point sampling randomness.
    """
    if type(execute_horizon) is not int or execute_horizon < 1:
        raise ValueError('execute_horizon must be a positive integer')
    poses, timestamps = np.asarray(poses), np.asarray(timestamps)
    if poses.shape != (len(timestamps), 7) or not np.all(np.diff(timestamps) > 0):
        raise ValueError('Expected aligned poses and strictly increasing timestamps')
    rows, recursive, expected_frame = [], None, None
    policy.reset()
    stopped = None
    try:
        for item in observations:
            frame, observation = item['frame'], item['observation']
            if expected_frame is not None and frame != expected_frame:
                raise ValueError(f'Feedback timeline gap: expected frame {expected_frame}, got {frame}')
            if not 0 <= frame < len(poses) - 1:
                raise ValueError('Observation frame has no future target')
            count = min(execute_horizon, len(poses) - 1 - frame)
            if recursive is None:
                recursive = np.asarray(observation['tcp_pose'], dtype=np.float32).copy()
            feedback_observation = dict(observation, tcp_pose=recursive.copy())
            target = poses[frame+1:frame+count+1]
            started = perf_counter()
            if seed_for_frame:
                seed_for_frame(frame)
            oracle = np.asarray(policy.act(observation, item['goal'], frame)['actions'], dtype=np.float32)
            if seed_for_frame:
                seed_for_frame(frame)
            feedback = np.asarray(policy.act(feedback_observation, item['goal'], frame)['actions'], dtype=np.float32)
            for actions in (oracle, feedback):
                if actions.ndim != 2 or actions.shape[1] != 8 or len(actions) < count or not np.isfinite(actions).all():
                    raise ValueError('Invalid or too-short action chunk')
                if not np.allclose(np.linalg.norm(actions[:, 3:7], axis=1), 1., atol=1e-4):
                    raise ValueError('Predicted quaternions must be unit xyzw')
            row = dict(frame=frame, timestamp=float(timestamps[frame]), target_frame=frame+count,
                target_timestamp=float(timestamps[frame+count]), executed_ticks=count,
                input_tcp_xyzw=recursive.tolist(), recorded_tcp_xyzw=poses[frame].tolist(),
                recorded_joints=np.asarray(observation['joint_positions']).tolist(),
                recorded_gripper_open01=float(observation['gripper_open01']),
                oracle_actions=oracle.tolist(), feedback_actions=feedback.tolist(),
                target_tcp_xyzw=target.tolist(), oracle=pose_errors(oracle[:count, :7], target),
                feedback=pose_errors(feedback[:count, :7], target),
                input_position_inconsistency_m=float(np.linalg.norm(recursive[:3]-poses[frame,:3])),
                input_rotation_inconsistency_degrees=float(np.rad2deg(rotation_error(recursive[3:7], poses[frame,3:7]))),
                paired_latency_seconds=perf_counter()-started)
            rows.append(row)
            if writer:
                writer.write(json.dumps(row, allow_nan=False)+'\n')
                writer.flush()
            recursive = feedback[count-1, :7].copy()  # absolute pose, never add it as a delta
            expected_frame = frame+count
            if progress:
                progress(row, len(rows))
    except Exception as error:
        stopped = dict(type=type(error).__name__, message=str(error), next_frame=expected_frame)
    complete = bool(rows and rows[-1]['target_frame'] == len(poses)-1 and stopped is None)
    summary = dict(kind='recorded_observation_recursive_tcp_diagnostic', complete=complete,
        stop=stopped, observations=len(rows), evaluated_future_ticks=sum(r['executed_ticks'] for r in rows),
        execute_horizon=execute_horizon, results={})
    for name in ('oracle', 'feedback'):
        summary['results'][name] = {metric:distribution([value for row in rows for value in row[name][metric]])
                                    for metric in ('position_m','rotation_degrees')}
    summary['first_endpoint_over_threshold']={str(threshold):next((dict(frame=r['target_frame'],
        timestamp=r['target_timestamp'], position_error_m=r['feedback']['position_m'][-1])
        for r in rows if r['feedback']['position_m'][-1] > threshold), None) for threshold in (.01,.05,.1,.25)}
    return summary, rows


def raw_observations(raw_root, record, columns, data, poses, timestamps, horizon, parity_samples, parity):
    from finetune.OHT.data.actions import gripper_states, low_dim
    from finetune.OHT.data.observation import camera_observation, validate_observation
    from finetune.OHT.data.video import EpisodeVideos, metric_depth
    from finetune.OHT.data.common import GOALS
    measured, binary = gripper_states(columns['observation.state'], data['gripper'])
    videos = EpisodeVideos(raw_root/record['dataset'], data.get('video_timestamp_tolerance', 1/120+.0001),
                           alignment=data['video_alignment'], expected_frames=len(poses))
    try:
        for frame in range(0,len(poses)-1,horizon):
            observation = dict(tcp_pose=poses[frame].astype(np.float32),
                joint_positions=np.asarray(columns['observation.state'][frame][:6],dtype=np.float32),
                gripper_open01=np.asarray(measured[frame],dtype=np.float32),
                low_dim_state=low_dim(measured[frame],binary_state=binary[frame]))
            for camera in data['cameras']:
                rgb=videos.read(columns[f'observation.images.{camera}'][frame],frame=frame)
                depth_config=data['cameras'][camera].get('depth',data['depth'])
                depth=metric_depth(raw_root,record,columns,camera,frame,videos,depth_config)
                observation.update(camera_observation(camera,rgb,depth,
                    columns[f'observation.{camera}_extrinsic'][frame],data))
            validate_observation(observation,data)
            if frame in parity_samples:
                with np.load(parity_samples[frame],allow_pickle=False) as old:
                    for key,value in observation.items():
                        if key not in old or value.shape != old[key].shape or not np.allclose(value,old[key],atol=1e-6,rtol=1e-6,equal_nan=True):
                            raise ValueError(f'Raw/cached observation mismatch at frame={frame}, field={key}')
                parity.append(frame)
                print(f'Raw/cached observation parity passed: frame={frame}',flush=True)
            yield dict(frame=frame,observation=observation,goal=GOALS[record['task']])
        videos.validate_lengths()
    finally:
        videos.close()


def save_plot(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not rows:
        return
    times=np.array([r['target_timestamp'] for r in rows])
    target=np.array([r['target_tcp_xyzw'][-1] for r in rows])
    oracle=np.array([r['oracle_actions'][r['executed_ticks']-1] for r in rows])
    feedback=np.array([r['feedback_actions'][r['executed_ticks']-1] for r in rows])
    fig,axes=plt.subplots(5,1,figsize=(13,13),sharex=True)
    for j,axis in enumerate(axes[:3]):
        axis.plot(times,target[:,j],color='black',label='Recorded GT TCP')
        axis.plot(times,oracle[:,j],color='tab:blue',ls='--',label='GT TCP input each time')
        axis.plot(times,feedback[:,j],color='tab:orange',label='Recursive predicted TCP input')
        axis.set_ylabel('XYZ'[j]+' (m)');axis.grid(alpha=.25)
    axes[0].legend(fontsize=9)
    for name,color in (('oracle','tab:blue'),('feedback','tab:orange')):
        axes[3].plot(times,np.array([r[name]['position_m'][-1] for r in rows])*1000,color=color,label=name)
        axes[4].plot(times,[r[name]['rotation_degrees'][-1] for r in rows],color=color,label=name)
    axes[3].set_ylabel('Endpoint position error (mm)')
    axes[4].set_ylabel('Endpoint rotation error (degrees)')
    axes[4].set_xlabel('Recorded episode time (s)')
    for axis in axes[3:]:
        axis.grid(alpha=.25);axis.legend(fontsize=9)
    fig.suptitle('Offline recursive TCP diagnostic: recorded RGB-D + joints + gripper in BOTH branches\n'
        'Only TCP pose is fed back; observations do not respond to actions; no physics or task success test',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(output/'trajectory.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','replay','raw-root','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--episode-index',type=int,default=31)
    parser.add_argument('--task',default='disassemble_left')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Use a new output directory: {args.output}')
    from finetune.OHT.data.chunk_dataset import load_chunk_contract
    from finetune.OHT.data.common import read_jsonl, file_digest
    from finetune.OHT.data.reader import read_episode, validate_episode
    from finetune.OHT.data.actions import world_tcp_poses
    from finetune.OHT.runtime.loading import load_policy
    contract=load_chunk_contract(args.replay)
    selected=[m for m in read_jsonl(args.replay/'samples.jsonl') if m['task']==args.task and m['episode_index']==args.episode_index]
    if not selected:
        raise ValueError('Episode not found in bound replay')
    first=selected[0]
    record=dict(task=args.task,episode_index=args.episode_index,dataset=first['data_profile'],path=first['diagnostic_objects']['path'])
    expected=first['diagnostic_objects']['parquet_sha256']
    if file_digest(args.raw_root/record['path']) != expected:
        raise ValueError('Raw Parquet checksum differs from training replay')
    columns=read_episode(args.raw_root,record)
    validate_episode(columns,record)
    data=contract['source_data_configs'][first['data_profile']]['data_config']
    poses=world_tcp_poses(columns,data['link_to_tcp'],data['ee_quaternion_order']).astype(np.float32)
    timestamps=np.asarray(columns['timestamp'],dtype=np.float64)
    if not np.allclose(np.diff(timestamps),contract['chunk']['action_dt'],atol=1e-5,rtol=.0001) or contract['chunk']['frame_stride']!=1:
        raise ValueError('Diagnostic requires dense raw action ticks matching the checkpoint')
    args.output.mkdir(parents=True)
    class Progress:
        def update(self,stage,**details):
            print(stage,json.dumps(details),flush=True)
    policy=load_policy(str(args.checkpoint),args.device,progress=Progress())
    if policy.contract['sha256'] != contract['sha256']:
        raise ValueError('Checkpoint and replay contract differ')
    if not policy.chunk_settings or policy.chunk_settings['decoder_type']!='regression':
        raise ValueError('Paired diagnostic currently requires a stateless regression chunk policy')
    horizon=policy.chunk_settings['execute_horizon']
    parity_candidates=[m for m in selected if m['frame']%horizon==0]
    check_frames={parity_candidates[i]['frame'] for i in (0,len(parity_candidates)//2,len(parity_candidates)-1)}
    parity_samples={m['frame']:args.replay/m['observation'] for m in selected if m['frame'] in check_frames}
    parity=[]
    source=raw_observations(args.raw_root,record,columns,data,poses,timestamps,horizon,parity_samples,parity)
    import torch
    def seed(frame):
        random.seed(args.seed+frame);np.random.seed(args.seed+frame);torch.manual_seed(args.seed+frame)
    started=perf_counter()
    def progress(row,count):
        if count==1 or count%25==0 or row['target_frame']==len(poses)-1:
            print(f'feedback {count}/{int(np.ceil((len(poses)-1)/horizon))} '
                f'frame={row["frame"]} t={row["timestamp"]:.3f}s '
                f'oracle_end_mm={row["oracle"]["position_m"][-1]*1000:.3f} '
                f'feedback_end_mm={row["feedback"]["position_m"][-1]*1000:.3f} '
                f'elapsed={perf_counter()-started:.1f}s',flush=True)
    with (args.output/'steps.jsonl').open('w') as writer:
        result,rows=run_feedback(policy,source,poses,timestamps,execute_horizon=horizon,
            writer=writer,seed_for_frame=seed,progress=progress)
    source.close()
    result.update(checkpoint=str(args.checkpoint.resolve()),contract_sha256=contract['sha256'],
        replay=str(args.replay.resolve()),raw_root=str(args.raw_root.resolve()),raw_parquet_sha256=expected,
        episode_index=args.episode_index,task=args.task,split=first['split'],source_episode_frames=len(poses),
        action_dt=contract['chunk']['action_dt'],horizon=policy.chunk_settings['horizon'],seed=args.seed,
        raw_cached_parity_checked_frames=parity,
        feedback_fields=['tcp_pose'],recorded_fields=['RGB','depth','point_cloud','camera_intrinsics','camera_extrinsics','joint_positions','gripper_open01','low_dim_state'],
        limitations=['Recorded observations are unaffected by predictions.',
            'Recursive TCP becomes inconsistent with recorded robot joints, wrist-camera pose, RGB-D and gripper.',
            'No kinematics, dynamics, contacts, execution lag or task success is evaluated.',
            'Each branch advances H=4 raw frames, including observations absent from the stride-10 training cache.',
            'Reported future action errors use each target raw tick once, not overlapping forecasts.'])
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    save_plot(args.output,rows)
    print('hash_checkpoint',flush=True)
    result['checkpoint_sha256']=file_digest(args.checkpoint)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Recorded TCP feedback diagnostic</title><style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style><h1>Recorded observations + recursive TCP feedback</h1><p>Both branches use recorded RGB-D, camera poses, joints and gripper. Orange feeds back predicted TCP only. The resulting robot state and imagery are inconsistent. This is not a simulator rollout or a task success test.</p><img src="trajectory.png"><p><a href="result.json">Summary</a> / <a href="steps.jsonl">Per-step predictions</a></p>')
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['complete'] else 1


if __name__=='__main__':
    raise SystemExit(main())
