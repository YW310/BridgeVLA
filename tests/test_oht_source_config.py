"""Exporter metadata and reference-converter numeric regression tests."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from finetune.OHT.data.actions import gripper_step, gripper_states, low_dim, world_tcp_poses
from finetune.OHT.data.common import file_digest, read_jsonl
from finetune.OHT.data.observation import validate_data_config
from finetune.OHT.data.replay import build, load_contract
from finetune.OHT.data.source_config import resolve_dataset_config, sample_data_config
from finetune.OHT.data.video import decode_depth
from finetune.OHT.data.visualization import projection_status, project_world
from tests.test_oht_migration import replay_fixture


CONFIG = Path(__file__).resolve().parents[1] / "finetune/OHT/configs/dataset.yaml"


@pytest.fixture
def exporter(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    # Separate, documented quantized-writer profile. Do not let a reference
    # converter's log fallback determine the default v423 dataset encoding.
    config["depth"] = dict(encoding="quantized", pixel_format="gray12le", metadata=True,
                           depth_min=.01, depth_max=10., shift=3.5, use_log=True,
                           qmax=4095, kind="ray", invalid_values=[0],
                           path_pattern=None, limits=[.001, 10.])
    intrinsics = {camera.upper(): [[200 + i * 10, 0, 320], [0, 210 + i * 10, 240], [0, 0, 1]]
                  for i, camera in enumerate(config["cameras"])}
    (meta / "camera_intrinsics.json").write_text(json.dumps(intrinsics), encoding="utf-8")
    (meta / "stats.json").write_text(json.dumps({"observation.state": {
        "min": [0] * 6 + [-.04], "max": [1] * 6 + [-.006]}}), encoding="utf-8")
    features = {f"observation.depth.{camera}": {"info": {
        "video.depth_min": .01, "video.depth_max": 10., "video.shift": 3.5,
        "video.use_log": True, "video.qmax": 4095, "video.pix_fmt": "gray12le"}}
                for camera in config["cameras"]}
    features["observation.depth.local_left"]["info"]["video.use_log"] = "false"
    (meta / "info.json").write_text(json.dumps({"features": features}), encoding="utf-8")
    return tmp_path, config, intrinsics


def test_metadata_resolves_per_camera_K_depth_and_dataset_global_gripper(exporter):
    root, config, intrinsics = exporter
    before = deepcopy(config)
    resolved, sources = resolve_dataset_config(root, config)
    assert config == before
    validate_data_config(resolved, resolved=True)
    assert resolved["intrinsics_source"] == "config"
    assert resolved["gripper"]["source"] == "config"
    assert resolved["gripper"]["open"] == -.04 and resolved["gripper"]["close"] == -.006
    for camera, calibration in resolved["cameras"].items():
        assert calibration["intrinsics"] == intrinsics[camera.upper()]
        assert calibration["depth"]["use_log"] == (camera != "local_left")
        assert calibration["depth"]["invalid_values"] == [0]
    for relative, fingerprint in sources.items():
        assert fingerprint == file_digest(root / relative)
    assert set(sources) == {"meta/camera_intrinsics.json", "meta/info.json", "meta/stats.json"}


def test_metadata_changes_are_bound_to_contract_material(exporter):
    root, config, _ = exporter
    first, hashes = resolve_dataset_config(root, config)
    path = root / "meta/camera_intrinsics.json"
    value = json.loads(path.read_text())
    value["LOCAL_LEFT"][0][0] = 650
    path.write_text(json.dumps(value), encoding="utf-8")
    second, next_hashes = resolve_dataset_config(root, config)
    assert first != second and hashes != next_hashes


def test_missing_metadata_fails_instead_of_reusing_document_focal_lengths(exporter):
    root, config, _ = exporter
    (root / "meta/camera_intrinsics.json").unlink()
    with pytest.raises(FileNotFoundError):
        resolve_dataset_config(root, config)


def test_rgb_only_info_does_not_silently_apply_converter_log_defaults(exporter):
    root, config, _ = exporter
    (root / "meta/info.json").write_text(json.dumps({"features": {
        "observation.images.wrist": {"info": {"pix_fmt": "yuv420p", "video_codec": "avc"}}
    }}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing depth quantization metadata") as error:
        resolve_dataset_config(root, config)
    assert "depth.metadata=reference" in str(error.value) and "--config" in str(error.value)


@pytest.mark.parametrize("key", ["depth_min", "depth_max", "shift", "use_log", "qmax", "pix_fmt"])
def test_partial_depth_metadata_cannot_mix_with_converter_defaults(exporter, key):
    root, config, _ = exporter
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    del info["features"]["observation.depth.wrist"]["info"]["video." + key]
    path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match=f"wrist: missing depth quantization metadata.*{key}"):
        resolve_dataset_config(root, config)


def test_complete_plain_depth_metadata_is_supported(exporter):
    root, config, _ = exporter
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    for feature in info["features"].values():
        feature["info"] = {key.removeprefix("video."): value for key, value in feature["info"].items()}
    path.write_text(json.dumps(info), encoding="utf-8")
    resolved, _ = resolve_dataset_config(root, config)
    assert resolved["cameras"]["wrist"]["depth"]["use_log"] is True
    assert resolved["cameras"]["local_left"]["depth"]["use_log"] is False


def test_verified_explicit_quantization_is_still_supported_without_metadata(exporter):
    root, config, _ = exporter
    (root / "meta/info.json").write_text('{"features": {}}', encoding="utf-8")
    config["depth"]["metadata"] = False
    resolved, _ = resolve_dataset_config(root, config)
    spec = resolved["depth"]
    raw = np.array([[0, 1, 512, 1024, 2048, 4094, 4095]], dtype=np.uint16)
    actual = decode_depth(raw, spec)
    # Independent golden values for the supplied converter's log inverse.
    expected = [[np.nan, .01115482449, .6538738760, 1.4158598797,
                 3.3848086488, 9.9955598282, 10.]]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, equal_nan=True)
    assert np.isfinite(actual[0, -1])  # qmax is a valid far-plane sample.
    reference_mm = np.rint(actual[:, 1:] * 1000).astype(np.uint16)
    np.testing.assert_array_equal(reference_mm, [[11, 654, 1416, 3385, 9996, 10000]])


def test_rgb_only_info_with_embedded_K_and_measured_stats_uses_reference_defaults(exporter):
    root, _, intrinsics = exporter
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    (root / "meta/camera_intrinsics.json").unlink()
    (root / "meta/stats.json").unlink()
    info = dict(camera_intrinsics=intrinsics, features={
        f"observation.images.{camera}": {"info": dict(fps=60, pix_fmt="yuv420p", video_codec="avc")}
        for camera in config["cameras"]}, features_stats={"observation.state": dict(
            min=[0] * 6 + [-.04008600115776062], max=[1] * 6 + [-.005636283196508884])})
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    before = deepcopy(config)
    resolved, sources = resolve_dataset_config(root, config)
    assert config == before
    assert sources == {"meta/info.json": file_digest(root / "meta/info.json")}
    for camera, calibration in resolved["cameras"].items():
        assert calibration["intrinsics"] == intrinsics[camera.upper()]
        assert calibration["depth"]["use_log"] is True
        assert calibration["depth"]["round_to_mm"] is True
    assert resolved["gripper"]["open"] == info["features_stats"]["observation.state"]["min"][-1]
    np.testing.assert_allclose(decode_depth(np.array([[897]], np.uint16), resolved["depth"]), [[1.215]])


def test_reference_metadata_uses_camera_overrides_and_field_fallbacks(exporter):
    root, config, _ = exporter
    config["depth"] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["depth"]
    path = root / "meta/info.json"
    path.write_text(json.dumps({"features": {
        "observation.depth.wrist": {"info": {"video.use_log": False, "use_log": True, "depth_max": 8.}},
        "observation.depth.local_left": {"info": {"video.depth_min": .02, "video.use_log": None}},
        "observation.images.global_left": {"info": {"depth_max": 99., "pix_fmt": "yuv420p"}},
    }}), encoding="utf-8")
    before = deepcopy(config)
    resolved, sources = resolve_dataset_config(root, config)
    assert config == before
    wrist = resolved["cameras"]["wrist"]["depth"]
    assert wrist["use_log"] is False and wrist["depth_max"] == 8.
    assert wrist["depth_min"] == .01 and wrist["shift"] == 3.5 and wrist["qmax"] == 4095
    local = resolved["cameras"]["local_left"]["depth"]
    assert local["depth_min"] == .02 and local["use_log"] is True
    global_depth = resolved["cameras"]["global_left"]["depth"]
    assert global_depth["depth_max"] == 10. and global_depth["pixel_format"] == "gray12le"
    assert sources["meta/info.json"] == file_digest(path)


def test_reference_metadata_falls_back_when_info_file_is_absent(exporter):
    root, config, _ = exporter
    config["depth"] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["depth"]
    (root / "meta/info.json").unlink()
    resolved, sources = resolve_dataset_config(root, config)
    assert "meta/info.json" not in sources
    for calibration in resolved["cameras"].values():
        spec = calibration["depth"]
        np.testing.assert_allclose(decode_depth(np.array([[897]], np.uint16), spec), [[1.215]])


@pytest.mark.parametrize("key,value", [("qmax", 0), ("use_log", "guess"), ("pix_fmt", "rgb24")])
def test_reference_metadata_rejects_invalid_present_values(exporter, key, value):
    root, config, _ = exporter
    config["depth"] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["depth"]
    (root / "meta/info.json").write_text(json.dumps({"features": {
        "observation.depth.wrist": {"info": {"video." + key: value}}
    }}), encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_dataset_config(root, config)


def test_intrinsics_sidecar_has_priority_over_embedded_info(exporter):
    root, config, intrinsics = exporter
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info["camera_intrinsics"] = {camera: np.eye(3).tolist() for camera in config["cameras"]}
    path.write_text(json.dumps(info), encoding="utf-8")
    resolved, _ = resolve_dataset_config(root, config)
    for camera, calibration in resolved["cameras"].items():
        assert calibration["intrinsics"] == intrinsics[camera.upper()]


def test_malformed_intrinsics_sidecar_is_not_hidden_by_info_fallback(exporter):
    root, config, intrinsics = exporter
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info["camera_intrinsics"] = intrinsics
    path.write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/camera_intrinsics.json").write_text('{"WRIST": null}', encoding="utf-8")
    with pytest.raises(ValueError, match="K in meta/camera_intrinsics.json"):
        resolve_dataset_config(root, config)


@pytest.mark.parametrize("use_log", [True, False])
def test_quantized_writer_inverse_roundtrip(exporter, use_log):
    _, config, _ = exporter
    spec = dict(config["depth"], use_log=use_log)
    depth = np.array([[.1, .5, 1., 2., 8., 10.]])
    near, far, shift = spec["depth_min"], spec["depth_max"], spec["shift"]
    if use_log:
        norm = (np.log(depth + shift) - np.log(near + shift)) / (np.log(far + shift) - np.log(near + shift))
    else:
        norm = (depth - near) / (far - near)
    raw = np.rint(norm * spec["qmax"]).astype(np.uint16)
    actual = decode_depth(raw, spec)
    np.testing.assert_allclose(actual, depth, atol=.0023)


@pytest.mark.parametrize("field,value", [("qmax", 0), ("qmax", 4095.5), ("shift", -1),
                                        ("depth_min", 10), ("depth_max", float("inf")),
                                        ("use_log", "false")])
def test_bad_quantization_rejected(exporter, field, value):
    _, config, _ = exporter
    config["depth"][field] = value
    with pytest.raises(ValueError):
        decode_depth(np.ones((2, 2), np.uint16), config["depth"])


def test_quantized_out_of_range_is_not_silently_clipped(exporter):
    _, config, _ = exporter
    with pytest.raises(ValueError, match="outside"):
        decode_depth(np.array([[4096]], dtype=np.uint16), config["depth"])


@pytest.mark.parametrize("order", ["xyzw", "wxyz"])
def test_ee_order_controls_rotation_and_offset_before_policy_xyzw_conversion(order):
    rotation = Rotation.from_euler("xyz", [23, -41, 67], degrees=True)
    raw = rotation.as_quat()
    if order == "wxyz":
        raw = raw[[3, 0, 1, 2]]
    transform = np.eye(4)
    transform[:3, 3] = [.01, -.02, .12]
    columns = {"observation.ee_pos_world": [[.5, .2, 1.4]], "observation.ee_quat_world": [raw]}
    actual = world_tcp_poses(columns, transform, order)[0]
    np.testing.assert_allclose(actual[:3], rotation.apply(transform[:3, 3]) + columns["observation.ee_pos_world"][0])
    np.testing.assert_allclose(Rotation.from_quat(actual[3:]).as_matrix(), rotation.as_matrix())


@pytest.mark.parametrize("open_raw,close_raw", [(-.04, -.006), (.3, -.2)])
def test_gripper_endpoints_direction_static_episode_and_policy_low_dim(open_raw, close_raw):
    states = np.zeros((3, 7))
    states[:, 6] = [open_raw, (open_raw + close_raw) / 2, close_raw]
    measured, desired = gripper_states(states, dict(open=open_raw, close=close_raw))
    np.testing.assert_allclose(measured, [1, .5, 0])
    np.testing.assert_array_equal(desired, [1, 0, 0])
    np.testing.assert_allclose(low_dim(1, [-.02, -.02]), [1, .04, .04, 0])
    np.testing.assert_allclose(low_dim(0, [-.003, -.003]), [0, 0, 0, 0])
    states[:, 6] = close_raw
    measured, desired = gripper_states(states, dict(open=open_raw, close=close_raw))
    assert not measured.any() and not desired.any()  # No episode-local normalization.


def test_replay_v2_profiles_and_current_tcp_do_not_leak_into_policy(replay_fixture):
    from finetune.OHT.data.dataset import OHTDataset
    f = replay_fixture
    contract = load_contract(f.replay)
    assert contract["schema"] == "oht_bridgevla_v2"
    row = read_jsonl(f.replay / "samples.jsonl")[0]
    assert row["data_profile"] in contract["source_data_configs"]
    assert len(row["current_tcp"]) == 7
    assert sample_data_config(contract, row)["ee_quaternion_order"] == "xyzw"
    policy_sample = OHTDataset(f.replay, row["split"])[0]
    assert "current_tcp" not in policy_sample and "data_profile" not in policy_sample


def test_bad_source_metadata_fails_before_replay_directory_creation(exporter, replay_fixture, tmp_path):
    root, config, _ = exporter
    output = tmp_path / "invalid"
    with pytest.raises(FileNotFoundError):
        build(replay_fixture.root, replay_fixture.manifest, config, output)
    assert not output.exists()


def test_projection_reports_outside_behind_and_missing_without_clamping():
    K = np.array([[2., 0, 1], [0, 2., 1], [0, 0, 1]])
    pose = [3, 0, 1]
    assert "outside image" in projection_status(pose, K, np.eye(4), (3, 4))
    assert "uv=(7.0,1.0)" in projection_status(pose, K, np.eye(4), (3, 4))
    assert "behind camera" in projection_status([0, 0, -1], K, np.eye(4), (3, 4))
    assert "in view" in projection_status([0, 0, 1], K, np.eye(4), (3, 4))
    assert projection_status(None, K, np.eye(4), (3, 4)) == "unavailable"
    xy, valid = project_world(pose, K, np.eye(4), (3, 4))
    assert not valid[0] and xy[0, 0] == 7


def test_reference_gripper_hysteresis_diff_and_online_step_agree():
    # Close on a measured contraction even if an object prevents full closure;
    # hold through small noise; reopen only on measured expansion/strong open.
    measured = np.array([1., .8, .79, .78, .45, .46, .65, .66, .95])
    states = np.zeros((len(measured), 7))
    states[:, 6] = -measured  # explicit source endpoints: open=-1, close=0
    fraction, binary = gripper_states(states, dict(open=-1, close=0))
    np.testing.assert_array_equal(binary, [1, 0, 0, 0, 0, 0, 1, 1, 1])
    previous_fraction, previous_state = None, None
    for index, value in enumerate(fraction):
        previous_state = gripper_step(float(value), previous_fraction, previous_state)
        assert previous_state == binary[index]
        assert low_dim(value, binary_state=previous_state)[0] == binary[index]
        previous_fraction = float(value)


def test_raw_action_all_components_are_diagnostic_only(replay_fixture, monkeypatch, tmp_path):
    from finetune.OHT.data import replay as replay_module
    f = replay_fixture
    original_read = replay_module.read_episode
    def corrupted_commands(*args):
        columns = original_read(*args)
        n = len(columns["timestamp"])
        columns["action"] = (np.arange(n * 7).reshape(n, 7) * 1000. + .321).tolist()
        return columns
    monkeypatch.setattr(replay_module, "read_episode", corrupted_commands)
    output = tmp_path / "untrusted-action"
    assert build(f.root, f.manifest, f.config, output, sample_stride=2) == f.count
    original = read_jsonl(f.replay / "samples.jsonl")
    rebuilt = read_jsonl(output / "samples.jsonl")
    assert rebuilt == original
    for row in rebuilt:
        with np.load(output / row["observation"]) as actual, np.load(f.replay / row["observation"]) as expected:
            for key in actual.files:
                np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("order", [None, "", "auto"])
def test_missing_or_inferred_ee_order_is_rejected(exporter, order):
    _, config, _ = exporter
    config["ee_quaternion_order"] = order
    with pytest.raises(ValueError, match="ee_quaternion_order explicitly"):
        validate_data_config(config)


def test_v1_contract_rejected_even_with_valid_checksums(replay_fixture, tmp_path):
    from finetune.OHT.data.common import digest, write_json
    import shutil
    root = tmp_path / "v1"
    root.mkdir()
    for name in ("contract.json", "complete.json", "samples.jsonl"):
        shutil.copy2(replay_fixture.replay / name, root / name)
    contract = load_contract(root)
    contract["schema"] = "oht_bridgevla_v1"
    contract.pop("sha256")
    contract["sha256"] = digest(contract)
    complete = json.loads((root / "complete.json").read_text())
    complete["contract_sha256"] = contract["sha256"]
    write_json(root / "contract.json", contract)
    write_json(root / "complete.json", complete)
    with pytest.raises(ValueError, match="rebuild v1 caches"):
        load_contract(root)


@pytest.mark.parametrize("encoding", ["quantized", "scaled_integer", "reference"])
def test_native_depth_to_replay_and_teacher_preview(exporter, replay_fixture, tmp_path, encoding):
    import shutil
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tests.test_oht_depth_video import write_gray12_video
    from tests.test_oht_migration import video
    from finetune.OHT.data.audit import audit
    from finetune.OHT.data.common import TASKS
    from finetune.OHT.data.dataset import OHTDataset
    from finetune.OHT.data.role_teacher import build_teacher
    from finetune.OHT.data import visualization

    metadata_root, config, intrinsics = exporter
    root = tmp_path / "native-raw"
    dataset = root / TASKS[0] / "lerobot_dataset"
    path = dataset / "data/chunk-000/episode_000000.parquet"
    path.parent.mkdir(parents=True)
    shutil.copytree(metadata_root / "meta", dataset / "meta")
    K = {camera: [[30 + i, 0, 32], [0, 32 + i, 32], [0, 0, 1]]
         for i, camera in enumerate(config["cameras"])}
    (dataset / "meta/camera_intrinsics.json").write_text(json.dumps(K), encoding="utf-8")
    source = replay_fixture.root / TASKS[0] / "lerobot_dataset/data/chunk-000/episode_000000.parquet"
    columns = pq.read_table(source).to_pydict()
    n = len(columns["timestamp"])
    video(dataset / "rgb.mp4", n=n, size=64)
    raw = np.full((64, 64), 1024, np.uint16)
    raw[0, 0], raw[0, 8] = 0, 4095
    raw[0, 16], raw[0, 24] = 897, 4094
    write_gray12_video(dataset / "depth.mp4", [raw] * n)
    rotation = Rotation.from_euler("xyz", [17, -29, 63], degrees=True)
    columns["observation.ee_quat_world"] = [rotation.as_quat()[[3, 0, 1, 2]].tolist()] * n
    # Explicitly contradictory raw command: must not affect measured labels.
    columns["action"] = [[777] * 7] * n
    for camera in config["cameras"]:
        columns[f"observation.images.{camera}"] = [dict(Path="rgb.mp4", Timestamp=[i/60]) for i in range(n)]
        columns[f"observation.depth.{camera}"] = [dict(Path="depth.mp4", Timestamp=[i/60]) for i in range(n)]
        # Keep the synthetic near surface inside the world ROI after reference
        # dequantization (q=1024 -> 1.416m instead of 1.024m).
        height = 2.2 if encoding == "reference" else 1.6
        columns[f"observation.{camera}_extrinsic"] = [[.4, 0, height, 1, 0, 0, 0]] * n
    pq.write_table(pa.table(columns), path)
    manifest, output = tmp_path / "native-audit.json", tmp_path / "native-replay"
    assert audit(root, manifest)["valid_episodes"] == 1
    config["image_size"] = [8, 8]
    if encoding in ("scaled_integer", "reference"):
        # Same metadata layout as the user's export: RGB-only features, K
        # embedded in info.json, no writer quantization fields or sidecars.
        config["depth"] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["depth"]
        if encoding == "scaled_integer":
            config["depth"] = dict(encoding="scaled_integer", pixel_format="gray12le", metadata=False,
                                   scale=.001, kind="z", invalid_values=[0, 4095], limits=[.001, 4.094])
        (dataset / "meta/camera_intrinsics.json").unlink()
        (dataset / "meta/info.json").write_text(json.dumps(dict(camera_intrinsics=K, features={
            f"observation.images.{camera}": {"info": dict(pix_fmt="yuv420p", video_codec="avc")}
            for camera in config["cameras"]})), encoding="utf-8")
    count = build(root, manifest, config, output, 2, visualize_every=1)
    contract = load_contract(output)
    data = OHTDataset(output, "train")
    assert data.validate_all() == count
    first = data[0]
    assert first["low_dim_state"][0] == 0  # stats close endpoint, not raw action
    expected_depth = decode_depth(raw, config["depth"])[::8, ::8]
    np.testing.assert_allclose(first["wrist_depth"][0], expected_depth, equal_nan=True)
    if encoding == "quantized":
        # local_left has linear quantization metadata, other cameras have log.
        linear = dict(config["depth"], use_log=False)
        np.testing.assert_allclose(first["local_left_depth"][0], decode_depth(raw, linear)[::8, ::8], equal_nan=True)
    elif encoding == "scaled_integer":
        assert first["wrist_depth"][0, 0, 2] == np.float32(.897)
        assert first["wrist_depth"][0, 0, 3] == np.float32(4.094)
        assert np.isnan(first["wrist_depth"][0, 0, [0, 1]]).all()
        for camera in config["cameras"]:
            np.testing.assert_allclose(first[f"{camera}_depth"][0], expected_depth, equal_nan=True)
    else:
        # Raw codes are dequantized, qmax is valid metric depth, and the point
        # cloud independently applies the reference's 3m maximum Z-depth.
        assert first["wrist_depth"][0, 0, 2] == np.float32(1.215)
        assert first["wrist_depth"][0, 0, 1] == np.float32(10.)
        assert np.isnan(first["wrist_point_cloud"][:, 0, 1]).all()
        for camera in config["cameras"]:
            np.testing.assert_allclose(first[f"{camera}_depth"][0], expected_depth, equal_nan=True)
            profile = sample_data_config(contract, read_jsonl(output / "samples.jsonl")[0])
            assert profile["cameras"][camera]["depth"]["round_to_mm"] is True
    for camera in config["cameras"]:
        expected_K = np.asarray(K[camera], float)
        expected_K[:2] /= 8
        np.testing.assert_allclose(first[f"{camera}_camera_intrinsics"], expected_K)
    np.testing.assert_allclose(Rotation.from_quat(first["gripper_pose"][3:]).as_matrix(), rotation.as_matrix(), atol=1e-6)
    assert len(list((output / "visualizations").rglob("*.png"))) == count
    row = read_jsonl(output / "samples.jsonl")[0]
    annotation = {"id": row["id"], "target": dict(present=False, known=True, source="none"),
                  "reference": dict(present=False, known=True, source="none")}
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(json.dumps(annotation) + "\n", encoding="utf-8")
    # Validate that teacher previews also receive the cached current TCP.
    original = visualization.save_preview
    received = []
    def capture(*args, **kwargs):
        received.append(kwargs["current_tcp"])
        return original(*args, **kwargs)
    from unittest.mock import patch
    with patch.object(visualization, "save_preview", capture):
        build_teacher(output, annotations, tmp_path / "native-teacher", 8, visualize_every=1)
    np.testing.assert_allclose(received, [row["current_tcp"]])
