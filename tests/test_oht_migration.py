"""CPU integration tests using real Parquet, timestamped MP4 and metric depth."""
import copy
import json
import threading
from http.server import HTTPServer
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from finetune.OHT.data.common import TASKS, CAMERAS, read_jsonl, digest, file_digest, write_json
from finetune.OHT.data.audit import audit, split_records
from finetune.OHT.data.replay import build, load_contract
from finetune.OHT.data.actions import gripper_states, target_labels
from finetune.OHT.data.geometry import backproject, tcp_pose
from finetune.OHT.data.video import VideoReader, decode_depth
from finetune.OHT.data.dataset import OHTDataset, collate
from finetune.OHT.data.role_teacher import build_teacher
from finetune.OHT.data.role_cache import role_fields, create_cache, RoleCache
from finetune.OHT.runtime.predicted_wrapper import PredictedObjectWrapper
from finetune.OHT.runtime.policy import Policy
from finetune.OHT.runtime.protocol import SCHEMA, validate_response, action_response
from finetune.OHT.runtime.executor import relative_eef
from finetune.OHT.runtime.transport import Client, pack_observation, unpack_observation
from finetune.OHT.server import Session, handler
from finetune.OHT.eval import open_loop, closed_loop
from finetune.OHT.config import load
from finetune.OHT.model import load_weights


def video(path, n=7, size=16):
    import av
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=60)
    stream.width, stream.height, stream.pix_fmt = size, size, "yuv420p"
    for i in range(n):
        pixels = np.full((size, size, 3), 20 + i * 25, np.uint8)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.fixture(scope="module")
def replay_fixture(tmp_path_factory):
    import pyarrow as pa
    import pyarrow.parquet as pq
    base = tmp_path_factory.mktemp("oht")
    root = base / "raw"
    config = dict(scene_bounds=[-2,-2,-2,2,2,2], link_to_tcp=np.eye(4).tolist(),
                  image_size=[8,8], rotation_classes=72, camera_quaternion_order="wxyz",
                  ee_quaternion_order="xyzw", gripper=dict(source="config", open=-.006, close=-.040),
                  depth=dict(encoding="metric", kind="z", path_pattern="{task}/depth/{episode:06d}/{frame:06d}.npy"),
                  keypoints=dict(max_translation=.03, max_rotation_degrees=8,max_frames=3),
                  cameras={camera:dict(intrinsics=[[8,0,8],[0,8,8],[0,0,1]],
                                      optical_to_sensor=np.eye(4).tolist()) for camera in CAMERAS})
    n=7
    for task in TASKS:
        for ep in range(3):
            dataset=root/task/"lerobot_dataset"
            data_path=dataset/"data"/"chunk-000"/f"episode_{ep:06d}.parquet"
            data_path.parent.mkdir(parents=True,exist_ok=True)
            movie=dataset/f"rgb_{ep}.mp4"
            video(movie)
            depth_path=root/task/"depth"/f"{ep:06d}"
            depth_path.mkdir(parents=True)
            for frame in range(n):
                depth=np.full((16,16),1,np.float32)
                depth[0,0]=np.nan
                np.save(depth_path/f"{frame:06d}.npy",depth)
            states=[[float(ep),0,0,0,0,0,-.006 if i<2 or i>=5 else -.040] for i in range(n)]
            columns=dict(frame_index=list(range(n)),episode_index=[ep]*n,
                         timestamp=[i/60 for i in range(n)],instruction_id=[0,0,1,1,1,2,2],
                         instruction=["privileged phase text"]*n)
            columns.update({
                "next.done":[False]*(n-1)+[True],"next.success":[False]*(n-1)+[True],
                "action":[[900]*6+[1 if i==2 else -1 if i==5 else 0] for i in range(n)],
                "observation.state":states,
                "observation.gripper_joints":[[-.003,-.003] if s[-1]==-.006 else [-.020,-.020] for s in states],
                "observation.ee_pos_world":[[.1+ep*.01+i*.02,0,.5] for i in range(n)],
                "observation.ee_quat_world":[[0,0,0,1]]*n,
                "observation.objects_pos":[[0]*12]*n,
                "observation.objects_quat":[[0,0,0,1]*4]*n})
            refs=[dict(Path=movie.name, Timestamp=[i/60]) for i in range(n)]
            for camera in CAMERAS:
                columns[f"observation.images.{camera}"]=refs
                columns[f"observation.depth.{camera}"]=refs
                columns[f"observation.{camera}_extrinsic"]=[[0,0,0,1,0,0,0]]*n
            pq.write_table(pa.table(columns), data_path)
    manifest=base/"audit.json"
    report=audit(root,manifest, fractions=(1/3,1/3,1/3))
    replay=base/"replay"
    count=build(root,manifest,config,replay,sample_stride=2)
    return SimpleNamespace(base=base,root=root,config=config,manifest=manifest,
                           report=report,replay=replay,count=count)


def test_real_parquet_video_metric_depth_to_agent_batch(replay_fixture):
    f=replay_fixture
    assert f.report["valid_episodes"]==12
    assert f.report["split_counts"]=={"train":4,"val":4,"test":4}
    rows=read_jsonl(f.replay/"samples.jsonl")
    assert len(rows)==f.count and all(r["target_frame"]>r["frame"] for r in rows)
    assignments={}
    for row in rows:
        assert assignments.setdefault(row["group"],row["split"])==row["split"]
    data=OHTDataset(f.replay,"train")
    assert data.validate_all()==len(data)
    row=data[0]
    assert row["global_left_rgb"].shape==(3,8,8)
    assert row["global_left_rgb"].dtype==np.uint8
    assert np.isnan(row["global_left_point_cloud"][:,0,0]).all()
    np.testing.assert_allclose(row["global_left_point_cloud"][:,4,4], [0,0,1])
    assert data.contract["data_config"]["camera_quaternion_order"] == "wxyz"
    assert "privileged" not in row["goal"]
    assert not any("objects_pos" in key or "oracle" in key or "predicted" in key for key in row)
    # Original commanded EE deltas were deliberately nonsense.
    assert np.max(np.abs(row["action"][:3]))<1
    batch=collate([row,row])
    assert batch["low_dim_state"].shape==(2,1,4)
    assert batch["rot_grip_action_indicies"].shape==(2,1,4)
    assert batch["ignore_collisions"].shape==(2,1,1)
    assert batch["lang_goal"][0][0][0]==row["goal"]


def test_native_gray12_parquet_to_replay_with_previews(replay_fixture, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from finetune.OHT.data.common import read_config
    from finetune.OHT.data.point_filter import point_cloud_mask
    from test_oht_depth_video import write_gray12_video

    root = tmp_path / "raw"
    dataset = root / TASKS[0] / "lerobot_dataset"
    parquet = dataset / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    source = replay_fixture.root / TASKS[0] / "lerobot_dataset/data/chunk-000/episode_000000.parquet"
    columns = pq.read_table(source).to_pydict()
    n = len(columns["timestamp"])
    video(dataset / "rgb.mp4", n=n, size=64)
    samples = [np.full((64, 64), 1000 + frame * 20, dtype=np.uint16) for frame in range(n)]
    for sample in samples:
        sample[0, :2] = [0, 4095]
    write_gray12_video(dataset / "depth.mp4", samples)
    config = read_config(Path(__file__).resolve().parents[1] / "finetune/OHT/configs/dataset.yaml")
    # This synthetic movie contains millimetres, not the exporter quantization.
    config.update(intrinsics_source="config", ee_quaternion_order="xyzw",
                  gripper=dict(source="config", open=-.006, close=-.040))
    config["depth"] = dict(encoding="scaled_integer", pixel_format="gray12le", scale=.001,
                           kind="ray", invalid_values=[0, 4095], limits=[.001, 4.094])
    config["image_size"] = [8, 8]
    for camera in CAMERAS:
        columns[f"observation.images.{camera}"] = [dict(Path="rgb.mp4", Timestamp=[i/60]) for i in range(n)]
        columns[f"observation.depth.{camera}"] = [dict(Path="depth.mp4", Timestamp=[i/60]) for i in range(n)]
        columns[f"observation.{camera}_extrinsic"] = [[.4, 0, 1.6, 1, 0, 0, 0]] * n
        config["cameras"][camera]["intrinsics"] = [[32,0,32],[0,32,32],[0,0,1]]
    pq.write_table(pa.table(columns), parquet)
    manifest = tmp_path / "audit.json"
    assert audit(root, manifest)["valid_episodes"] == 1
    output, previews = tmp_path / "replay", tmp_path / "previews"
    count = build(root, manifest, config, output, sample_stride=2,
                  visualize_every=1, visualize_output_dir=previews)
    data = OHTDataset(output, "train")
    assert data.validate_all() == count > 0
    assert len(list(previews.rglob("*.png"))) == count
    row = data[0]
    np.testing.assert_allclose(row["wrist_depth"][0, 1:, :], 1.)
    assert np.isnan(row["wrist_point_cloud"][:, 0, 0]).all()
    raw_points = backproject(row["wrist_depth"][0], row["wrist_camera_intrinsics"],
                             row["wrist_camera_extrinsics"], kind="ray", limits=config["depth"]["limits"])
    expected_valid = point_cloud_mask(raw_points, config["point_cloud_filter"])
    np.testing.assert_array_equal(np.isfinite(row["wrist_point_cloud"]).all(axis=0), expected_valid)
    np.testing.assert_allclose(row["wrist_point_cloud"].transpose(1, 2, 0)[expected_valid],
                               raw_points[expected_valid], atol=1e-6)
    assert expected_valid.any() and not expected_valid[1:].all()


def test_frame_index_replay_removes_rotation_ghost_from_shifted_video_references(replay_fixture, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tests.test_oht_depth_video import write_gray12_video

    root = tmp_path / "raw"
    dataset = root / TASKS[0] / "lerobot_dataset"
    parquet = dataset / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    source = replay_fixture.root / TASKS[0] / "lerobot_dataset/data/chunk-000/episode_000000.parquet"
    columns = pq.read_table(source).to_pydict()
    n = len(columns["timestamp"])
    video(dataset / "rgb.mp4", n=n, size=64)
    config = copy.deepcopy(replay_fixture.config)
    config.update(video_alignment="frame_index", camera_extrinsic_direction="camera_to_world")
    config["cameras"] = {camera: dict(intrinsics=[[80, 0, 32], [0, 80, 32], [0, 0, 1]],
                                     optical_to_sensor=np.diag([1, -1, -1, 1]).tolist())
                         for camera in ("global_left", "wrist")}
    config["depth"] = dict(encoding="scaled_integer", pixel_format="gray12le", scale=.001,
                           kind="z", invalid_values=[0], limits=[.001, 4.094])
    for camera in CAMERAS:
        columns[f"observation.images.{camera}"] = [dict(Path="rgb.mp4", Timestamp=[i / 60]) for i in range(n)]
        columns[f"observation.depth.{camera}"] = columns[f"observation.images.{camera}"]
    v, u = np.indices((64, 64))
    rays = np.stack(((u - 32) / 80, (v - 32) / 80, np.ones_like(u)), axis=-1)
    for camera in config["cameras"]:
        samples, poses = [], []
        for frame in range(n):
            sensor_rotation = Rotation.from_euler("xyz", [6 * frame, 3 * frame, 4 * frame] if camera == "wrist" else [0, 0, 0], degrees=True)
            origin = np.array([.15 if camera == "wrist" else 0, 0, 1.4])
            optical_R = sensor_rotation.as_matrix() @ np.diag([1, -1, -1])
            depth = (.4 - origin[2]) / (rays @ optical_R.T)[..., 2]
            assert (depth > 0).all() and (depth < 4.094).all()
            samples.append(np.rint(depth * 1000).astype(np.uint16))
            poses.append(np.r_[origin, sensor_rotation.as_quat()[[3, 0, 1, 2]]].tolist())
        write_gray12_video(dataset / f"{camera}.mp4", samples)
        columns[f"observation.images.{camera}"] = [dict(Path="rgb.mp4", Timestamp=[i / 60]) for i in range(n)]
        # Reproduce a reference timestamp offset. The reference converter pairs
        # decoded depth i with pose i; PTS lookup pairs depth i+1 with pose i.
        columns[f"observation.depth.{camera}"] = [dict(Path=f"{camera}.mp4", Timestamp=[(i + int(camera == 'wrist')) / 60]) for i in range(n)]
        columns[f"observation.{camera}_extrinsic"] = poses
    pq.write_table(pa.table(columns), parquet)
    manifest = tmp_path / "audit.json"
    assert audit(root, manifest)["valid_episodes"] == 1
    corrected = tmp_path / "frame-index"
    build(root, manifest, config, corrected, sample_stride=2)
    row = read_jsonl(corrected / "samples.jsonl")[0]
    with np.load(corrected / row["observation"]) as observation:
        for camera in config["cameras"]:
            cloud = observation[f"{camera}_point_cloud"].reshape(3, -1).T
            np.testing.assert_allclose(cloud[:, 2], .4, atol=.0006)
    config["video_alignment"] = "timestamp"
    previous = tmp_path / "timestamp"
    build(root, manifest, config, previous, sample_stride=2)
    old_row = read_jsonl(previous / "samples.jsonl")[0]
    assert old_row["labels"] == row["labels"]  # The fix never rotates action GT.
    with np.load(previous / old_row["observation"]) as observation:
        cloud = observation["wrist_point_cloud"].reshape(3, -1).T
        _, _, vh = np.linalg.svd(cloud - cloud.mean(axis=0), full_matrices=False)
        assert np.arccos(abs(vh[-1, 2])) > np.deg2rad(5)


def test_frame_count_failure_never_marks_replay_complete(replay_fixture, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "raw"
    dataset = root / TASKS[0] / "lerobot_dataset"
    parquet = dataset / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    source = replay_fixture.root / TASKS[0] / "lerobot_dataset/data/chunk-000/episode_000000.parquet"
    columns = pq.read_table(source).to_pydict()
    video(dataset / "short.mp4", n=3)
    for camera in CAMERAS:
        columns[f"observation.images.{camera}"] = [dict(Path="short.mp4", Timestamp=[i / 60]) for i in range(7)]
        columns[f"observation.depth.{camera}"] = columns[f"observation.images.{camera}"]
    pq.write_table(pa.table(columns), parquet)
    manifest = tmp_path / "audit.json"
    audit(root, manifest)
    config = copy.deepcopy(replay_fixture.config)
    config["video_alignment"] = "frame_index"
    output = tmp_path / "bad-replay"
    with pytest.raises(ValueError, match="frame count mismatch"):
        build(root, manifest, config, output)
    assert not (output / "complete.json").exists()


def test_missing_camera_order_fails_before_build_creates_output(replay_fixture, tmp_path):
    config = copy.deepcopy(replay_fixture.config)
    config.pop("camera_quaternion_order")
    output = tmp_path / "invalid-replay"
    with pytest.raises(ValueError, match="camera_quaternion_order explicitly"):
        build(replay_fixture.root, replay_fixture.manifest, config, output)
    assert not output.exists()


def test_legacy_cache_requires_camera_order_even_with_valid_checksums(replay_fixture, tmp_path):
    import shutil
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for name in ("contract.json", "complete.json", "samples.jsonl"):
        shutil.copy2(replay_fixture.replay / name, legacy / name)
    contract = load_contract(legacy)
    original_hash = contract.pop("sha256")
    contract["data_config"].pop("camera_quaternion_order")
    contract["sha256"] = digest(contract)
    assert contract["sha256"] != original_hash
    complete = json.loads((legacy / "complete.json").read_text())
    complete["contract_sha256"] = contract["sha256"]
    write_json(legacy / "contract.json", contract)
    write_json(legacy / "complete.json", complete)
    with pytest.raises(ValueError, match="Legacy v423 caches"):
        load_contract(legacy)
    with pytest.raises(ValueError, match="camera_quaternion_order explicitly"):
        OHTDataset(legacy, "train")


def test_audit_splits_keep_duplicate_trajectories_and_scene_groups():
    records=[dict(task=TASKS[0],episode_index=i,trajectory_hash=f"t{i}") for i in range(100)]
    split=split_records(records,seed=11)
    assert [sum(r["split"]==s for r in split) for s in ("train","val","test")]==[80,10,10]
    records[1]["trajectory_hash"]=records[0]["trajectory_hash"]
    split=split_records(records,groups={f"{TASKS[0]}/1":"shared",f"{TASKS[0]}/2":"shared"})
    assert split[0]["split"]==split[1]["split"]==split[2]["split"]
    assert split==split_records(records,groups={f"{TASKS[0]}/1":"shared",f"{TASKS[0]}/2":"shared"})


def test_unknown_calibration_refuses_replay_before_writing(replay_fixture,tmp_path):
    f=replay_fixture
    for section in ("scene_bounds","link_to_tcp"):
        config=copy.deepcopy(f.config); config[section]=None
        with pytest.raises(ValueError):
            build(f.root,f.manifest,config,tmp_path/section)
        assert not (tmp_path/section).exists()
    with pytest.raises(ValueError,match="Unknown depth"):
        decode_depth(np.zeros((2,2,3),np.uint8),{"encoding":None})
    with pytest.raises(ValueError,match="floating"):
        decode_depth(np.zeros((2,2,3),np.uint8),{"encoding":"metric"})


def test_geometry_camera_axes_tcp_rotation_and_relative_controller():
    K=[[1,0,0],[0,1,0],[0,0,1]]
    world=np.eye(4); world[:3,:3]=Rotation.from_euler("z",90,degrees=True).as_matrix()
    world[:3,3]=[1,2,3]
    points=backproject(np.ones((2,2)),K,world)
    np.testing.assert_allclose(points[0,1],[1,3,4],atol=1e-6)
    ray=backproject(np.ones((2,2)),K,np.eye(4),kind="ray")
    np.testing.assert_allclose(np.linalg.norm(ray,axis=-1),1,atol=1e-6)
    link=np.eye(4); link[0,3]=.1
    q=Rotation.from_euler("z",90,degrees=True).as_quat()
    pose=tcp_pose([1,2,3],q,link)
    np.testing.assert_allclose(pose[:3],[1,2.1,3],atol=1e-6)
    target=pose.copy(); target[0]+=.2
    command=relative_eef(pose,target,np.eye(4),"body","body",.05,.1)
    np.testing.assert_allclose(command[:3],[0,-.05,0],atol=1e-6)
    np.testing.assert_allclose(command[3:],0,atol=1e-6)


def test_observed_gripper_states_and_action_boundaries():
    states=np.zeros((5,7)); states[:,6]=[-.006,-.040,-.040,-.006,-.006]
    measured,desired=gripper_states(states, dict(open=-.006, close=-.040))
    np.testing.assert_array_equal(desired,[1,0,0,1,1])
    np.testing.assert_allclose(measured,[1,0,0,1,1])
    labels=target_labels([0,0,0,0,0,0,1],1,[-1,-1,-1,1,1,1])
    np.testing.assert_array_equal(labels["rot_grip_action_indicies"],[36,36,36,1])
    with pytest.raises(ValueError,match="outside"):
        target_labels([1,0,0,0,0,0,1],1,[-1,-1,-1,1,1,1])


def test_video_pts_forward_backward_and_unmatched_timestamp(replay_fixture):
    path=replay_fixture.root/TASKS[0]/"lerobot_dataset"/"rgb_0.mp4"
    reader=VideoReader(path)
    try:
        first=reader.read(0).mean()
        # Repeated references near the same PTS must retain the earlier bracket.
        assert abs(reader.read(.001).mean()-first)<1
        assert abs(reader.read(.002).mean()-first)<1
        assert abs(reader.read(.002).mean()-first)<1
        last=reader.read(6/60).mean()
        assert last>first+120
        assert abs(reader.read(0).mean()-first)<1
        with pytest.raises(ValueError,match="unmatched"):
            reader.read(10)
    finally:
        reader.close()


def teacher_annotations(f,path):
    annotations=[]
    mask=np.ones((8,8),bool)
    np.savez(path.parent/"masks.npz",**{c:mask for c in CAMERAS})
    for row in read_jsonl(f.replay/"samples.jsonl"):
        annotations.append(dict(id=row["id"],
                                target=dict(present=True,known=True,source="visible_surface",mask_path="masks.npz"),
                                reference=dict(present=False,known=True,source="none")))
    path.write_text("".join(json.dumps(r)+"\n" for r in annotations),encoding="utf-8")
    return annotations


def test_teacher_role_queries_and_prediction_namespaces(replay_fixture,tmp_path):
    f=replay_fixture
    annotations=tmp_path/"annotations.jsonl"
    teacher_annotations(f,annotations)
    teacher=tmp_path/"teacher"
    manifest=build_teacher(f.replay,annotations,teacher,point_count=8)
    assert manifest["provenance"]["mask_sha256"]
    data=OHTDataset(f.replay,"train","role_queries",teacher,8)
    row=data[0]
    assert row["oracle_target_object_valid"]
    assert not row["oracle_reference_object_valid"]
    assert not row["oracle_reference_present"]
    batch=collate([row,row])
    assert batch["oracle_role_present_known"].shape==(2,2)
    assert batch["oracle_target_object_points"].shape==(2,1,8,3)
    with pytest.raises(ValueError,match="Baseline"):
        OHTDataset(f.replay,"train","baseline",teacher,8)
    with pytest.raises(ValueError,match="kind"):
        OHTDataset(f.replay,"train","predicted_external",teacher,8)
    predicted=tmp_path/"predicted"
    fields=role_fields("predicted",np.zeros((2,8,3)),np.array([True,False]),
                       np.array([True,False]),confidence=[.9,.8])
    create_cache(predicted,"predicted",load_contract(f.replay)["sha256"],8,
                 {"model_sha256":"a"*64,"training_split":"train"},
                 ((r["id"],fields) for r in read_jsonl(f.replay/"samples.jsonl")))
    pred=OHTDataset(f.replay,"train","predicted_external",predicted,8)[0]
    assert "predicted_target_confidence" in pred and not any(k.startswith("oracle") for k in pred)


def test_role_null_visibility_and_unknown_are_distinct():
    pts=np.zeros((2,8,3))
    fields=role_fields("teacher",pts,np.array([False,False]),np.array([True,False]),
                       known=np.array([True,True]))
    assert fields["oracle_target_present"] and not fields["oracle_target_object_valid"]
    assert not fields["oracle_reference_present"] and fields["oracle_role_present_known"][1]
    with pytest.raises(ValueError,match="Absent"):
        role_fields("teacher",pts,np.array([True,False]),np.array([False,False]),
                    known=np.array([True,True]))
    with pytest.raises(ValueError,match="confidence"):
        role_fields("predicted",pts,np.array([False,False]),np.array([True,False]),
                    confidence=[np.nan,.8])


class FakeAgent:
    def __init__(self):
        self.resets=0
        self.inputs=[]
    def reset(self):
        self.resets+=1
    def act(self,step,batch,**kwargs):
        self.inputs.append(batch)
        return np.array([.2,0,.5,0,0,0,1,1],np.float32)


def test_prediction_wrapper_and_policy_drop_gt_labels(replay_fixture):
    f=replay_fixture
    row=OHTDataset(f.replay,"train")[0]
    row.update(instruction_id=99,held_obj="secret",objects_pos=np.ones(12),
               oracle_target_object_points=np.ones((8,3)))
    calls=[]
    def predictor(observation,goal):
        calls.append(observation)
        assert not any(k.startswith(("oracle","predicted")) for k in observation)
        assert not set(observation)&{"action","held_obj","instruction_id","objects_pos","goal"}
        assert goal==row["goal"]
        return role_fields("predicted",np.zeros((2,8,3)),np.array([True,False]),
                           np.array([True,False]),confidence=[.9,.8])
    wrapper=PredictedObjectWrapper(predictor,CAMERAS,8)
    agent=FakeAgent()
    policy=Policy(agent,load_contract(f.replay),"cpu","predicted_external",wrapper)
    result=policy.act(row,row["goal"],0)
    assert result.shape==(8,) and calls
    assert "predicted_target_object_points" in agent.inputs[0]
    assert not any(k.startswith("oracle") for k in agent.inputs[0])
    policy.reset()
    assert agent.resets==1
    internal=Policy(FakeAgent(),policy.contract,"cpu","role_queries")
    internal.act(row,row["goal"],0)
    assert not any(k.startswith(("oracle","predicted")) for k in internal.agent.inputs[0])


def observation_only(f):
    data=OHTDataset(f.replay,"train")
    row=data[0]
    with np.load(f.replay/data.samples[0]["observation"],allow_pickle=False) as source:
        return {k:source[k] for k in source.files},row["goal"]


def test_http_roundtrip_stale_rejection_and_episode_reset(replay_fixture):
    f=replay_fixture
    observation,goal=observation_only(f)
    contract=load_contract(f.replay)
    agent=FakeAgent()
    policy=Policy(agent,contract,"cpu","baseline")
    session=Session(policy)
    server=HTTPServer(("127.0.0.1",0),handler(session))
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    client=Client(f"http://127.0.0.1:{server.server_port}",contract["sha256"])
    try:
        action=client.act(observation,goal,"case-1",0,0.)
        assert action.shape==(8,) and agent.resets==1
        client.act(observation,goal,"case-1",1,.1)
        import urllib.error
        with pytest.raises(urllib.error.HTTPError) as exc:
            client.act(observation,goal,"case-1",1,.1)
        assert exc.value.code==400
        client.act(observation,goal,"case-2",0,0.)
        assert agent.resets==2
    finally:
        server.shutdown(); thread.join(); server.server_close()
    packed=pack_observation(observation)
    unpacked=unpack_observation(packed)
    np.testing.assert_allclose(unpacked["global_left_point_cloud"],observation["global_left_point_cloud"],equal_nan=True)
    packed["global_left_rgb"]["shape"]=[999999999,999999999]
    with pytest.raises(ValueError,match="size limit"):
        unpack_observation(packed)
    request=dict(schema=SCHEMA,episode="x",step=0,timestamp=0)
    response=action_response(request,action); response["step"]=1
    with pytest.raises(ValueError,match="Stale"):
        validate_response(request,response)


def test_open_loop_and_closed_loop_failures_remain_in_denominator(replay_fixture):
    f=replay_fixture
    policy=Policy(FakeAgent(),load_contract(f.replay),"cpu","baseline")
    report=open_loop(policy,f.replay,"val",limit=3)
    assert report["kind"]=="open_loop_errors" and report["all_samples"]["samples"]==3
    class Environment:
        def reset(self,case):
            self.fail=case["id"]=="bad"
            return dict(observation={},goal="goal",timestamp=0)
        def step(self,action):
            if self.fail:
                raise TimeoutError("executor")
            return dict(done=True,success=True)
    client=SimpleNamespace(act=lambda *args:np.zeros(8))
    cases=[dict(id="ok",task="T",seed=0),dict(id="bad",task="T",seed=1)]
    result=closed_loop(client,Environment(),cases,2)
    assert result["attempted"]==2 and result["successes"]==1 and result["success_rate"]==.5
    assert "TimeoutError" in result["cases"][1]["error"]


def test_cache_tampering_is_rejected(replay_fixture,tmp_path):
    import shutil
    copy_path=tmp_path/"replay"
    shutil.copytree(replay_fixture.replay,copy_path)
    index=copy_path/"samples.jsonl"
    index.write_text(index.read_text(encoding="utf-8")+"{}\n",encoding="utf-8")
    with pytest.raises(ValueError,match="index changed"):
        load_contract(copy_path)


def test_configs_and_base_checkpoint_initialization_accept_only_object_modules():
    import torch
    from torch import nn
    configs=Path(__file__).resolve().parents[1]/"finetune"/"OHT"/"configs"
    for name in ("baseline","role_queries","predicted_external"):
        assert load(configs/f"{name}.yaml")["mode"]==name
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.base=nn.Linear(3,3)
            self.oracle_prior_feature_adapter1=nn.Linear(3,3)
            self.object_slot_predictor1=nn.Linear(3,3)
    net=Network()
    agent=SimpleNamespace(_net_mod=net)
    base={k:v.clone() for k,v in net.state_dict().items() if k.startswith("base")}
    load_weights(agent,{"model_state":base},initialize=True)
    bad=dict(base); bad["base.weight"]=torch.zeros(2,3)
    with pytest.raises(ValueError,match="Incompatible"):
        load_weights(agent,{"model_state":bad},initialize=True)


def test_production_agent_updates_from_oht_batch_and_masks_collision_loss(replay_fixture):
    """Use the real optimizer/loss/point preprocessing with a tiny renderer substitute."""
    import torch
    from torch import nn
    from finetune.OHT import bootstrap
    from bridgevla.models.bridgevla_agent import RVTAgent
    class Network(nn.Module):
        num_img, img_size = 3, 8
        def __init__(self):
            super().__init__()
            self.trans=nn.Parameter(torch.zeros(3,8,8))
            self.feat=nn.Parameter(torch.zeros(72*3+4))
        def forward(self,pc,img_feat,**kwargs):
            assert all(torch.isfinite(points).all() for points in pc)
            assert kwargs["language_goal"][0][0][0]
            size=len(pc)
            coarse=dict(trans=self.trans[None].expand(size,-1,-1,-1),
                        feat=self.feat[None].expand(size,-1))
            return dict(**coarse,mvt2=dict(coarse))
        def get_pt_loc_on_img(self,points,**kwargs):
            return torch.full((len(points),1,3,2),3.5)
    f=replay_fixture
    data=OHTDataset(f.replay,"train")
    batch=collate([data[0],data[1]])
    network=Network()
    agent=RVTAgent(network,72,True,True,scene_bounds=f.config["scene_bounds"],
                   cameras=list(CAMERAS),optimizer_type="adam",collision_loss_weight=0,
                   transform_augmentation=False,place_with_mean=False,add_rgc_loss=True)
    agent.build(training=True,device=torch.device("cpu"))
    before=network.feat.detach().clone()
    # Exercise accumulation: the first microbatch must not change parameters.
    first=agent.update(batch,loss_scale=.5,reset_gradients=True,step_optimizer=False)
    torch.testing.assert_close(before,network.feat)
    second=agent.update(batch,loss_scale=.5,reset_gradients=False,step_optimizer=True)
    assert first["collision_loss"]==second["collision_loss"]==0
    assert np.isfinite(second["total_loss"])
    assert not torch.equal(before,network.feat)
    torch.testing.assert_close(before[-2:],network.feat[-2:])


def test_task_sampler_resume_and_extended_budget_keep_the_same_draw_stream():
    from finetune.OHT.data.sampling import TaskBalancedSampler
    samples=[{"task":"a"}]*20+[{"task":"b"}]
    full=list(TaskBalancedSampler(samples,1000,seed=7))
    assert list(TaskBalancedSampler(samples,1000,seed=7,start=123))==full[123:]
    assert list(TaskBalancedSampler(samples,2000,seed=7))[:1000]==full
    counts=[samples[index]["task"] for index in full]
    assert 400<counts.count("b")<600


def create_test_predictor():
    """Synthetic predictor used only to exercise the CLI/plugin contract."""
    def predict(observation,goal):
        assert "low_dim_state" in observation and isinstance(goal,str)
        return role_fields("predicted",np.zeros((2,8,3),np.float32),
                           np.array([True,False]),np.array([True,False]),
                           confidence=[.9,.8])
    return predict


def test_cli_audit_validate_prediction_and_training_preflight(replay_fixture,tmp_path):
    import yaml
    from finetune.OHT.cli import main
    from finetune.OHT.train import main as train_main
    f=replay_fixture
    assert main(["audit","--root",str(f.root),"--output",str(tmp_path/"audit.json")])==0
    assert main(["validate","--replay",str(f.replay)])==0
    provenance=tmp_path/"provenance.yaml"
    provenance.write_text(yaml.safe_dump(dict(model_sha256="b"*64,training_split="train")),encoding="utf-8")
    cache=tmp_path/"prediction"
    assert main(["predict","--replay",str(f.replay),"--predictor",
                 "tests.test_oht_migration:create_test_predictor",
                 "--provenance",str(provenance),"--output",str(cache),"--point-count","8"])==0
    assert main(["validate","--replay",str(f.replay),"--mode","predicted_external",
                 "--role-cache",str(cache),"--point-count","8"])==0
    config=Path(__file__).resolve().parents[1]/"finetune/OHT/configs/baseline.yaml"
    assert train_main(["--config",str(config),"--replay",str(f.replay),
                       "--output",str(tmp_path/"unused"),"--validate-only"])==0
    assert not (tmp_path/"unused").exists()


@pytest.mark.parametrize("mode",["baseline","role_queries","predicted_external"])
def test_model_factory_matches_existing_mvt_signature_and_real_agent(mode,replay_fixture,monkeypatch):
    import inspect
    import torch
    from torch import nn
    from finetune.OHT import bootstrap
    from bridgevla.mvt import mvt
    from finetune.OHT.model import build_agent
    from bridgevla.models.bridgevla_agent import RVTAgent
    signature=inspect.signature(mvt.MVT.__init__)
    captured={}
    class SmallMVT(nn.Module):
        def __init__(self,**kwargs):
            super().__init__()
            signature.bind(None,**kwargs)  # All actual constructor arguments must match.
            captured.update(kwargs)
            self.mvt1=nn.Linear(3,3)
            self.mvt1.enable_efficient_paligemma_forward=lambda:None
            self.mvt1.enable_gradient_checkpointing=lambda:None
            if kwargs["oracle_prior_adapter_rank"]:
                self.oracle_prior_feature_adapter1=nn.Linear(3,3)
            if kwargs["object_slots_enabled"]:
                self.object_slot_predictor1=nn.Linear(3,3)
    monkeypatch.setattr(mvt,"MVT",SmallMVT)
    config=load(Path(__file__).resolve().parents[1]/f"finetune/OHT/configs/{mode}.yaml")
    if mode!="baseline":
        config["adapter_only"]=True
    agent=build_agent(config,load_contract(replay_fixture.replay),torch.device("cpu"))
    assert isinstance(agent,RVTAgent) and agent.collision_loss_weight==0
    assert captured["use_point_renderer"] is True
    assert captured["object_slots_enabled"]==(mode=="role_queries")
    assert captured["oracle_prior_relation"]==(mode!="baseline")
    assert captured["object_conditioning_inherit_coarse_roles"]==(mode=="role_queries")
    if mode!="baseline":
        assert not agent._net_mod.mvt1.weight.requires_grad
        assert agent._net_mod.oracle_prior_feature_adapter1.weight.requires_grad


def test_replay_contract_binds_actual_observation_hashes_and_sampling(replay_fixture,tmp_path):
    f=replay_fixture
    contract=load_contract(f.replay)
    assert contract["index_sha256"]==file_digest(f.replay/"samples.jsonl")
    assert contract["sample_stride"]==2
    other=tmp_path/"other-stride"
    build(f.root,f.manifest,f.config,other,sample_stride=3)
    assert load_contract(other)["sha256"]!=contract["sha256"]
    for row in f.report["episodes"]:
        assert "\\" not in row["path"] and "\\" not in row["metadata"]
