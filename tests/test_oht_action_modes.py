"""Four physical action modes through real OHT batches, agents and HTTP."""
import copy
import json
import threading
from http.server import HTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from scipy.spatial.transform import Rotation

from tests.test_oht_migration import replay_fixture  # noqa: F401
from finetune.OHT import bootstrap  # noqa: F401
from finetune.OHT.action_modes import (
    MODES, action_options, resolve_action, selected_target, delta_eef,
    apply_eef_delta, action_descriptor, action_size,
)
from finetune.OHT.config import load
from finetune.OHT.data.dataset import OHTDataset, collate
from finetune.OHT.data.replay import load_contract
from finetune.OHT.data.common import read_jsonl, file_digest, write_json
from finetune.OHT.data.role_cache import create_cache, role_fields
from finetune.OHT.model import build_agent, load_weights, model_state
from finetune.OHT.runtime.policy import Policy
from finetune.OHT.runtime.predicted_wrapper import PredictedObjectWrapper, policy_observation
from finetune.OHT.runtime.protocol import (
    SCHEMA, ACTION_SCHEMA, action_response, validate_response, protocol_schema,
)
from finetune.OHT.runtime.transport import Client
from finetune.OHT.runtime.executor import execution_target
from finetune.OHT.server import Session, handler
from finetune.OHT.eval import open_loop


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("translation", ["world", "base", "body"])
@pytest.mark.parametrize("rotation", ["world", "base", "body"])
def test_relative_eef_round_trip_all_coordinate_frames(translation, rotation):
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("xyz", [30, 20, 70], degrees=True).as_matrix()
    base[:3, 3] = [4, 5, 6]
    config = action_options(dict(mode="relative_eef", translation_frame=translation,
                                 rotation_frame=rotation, world_from_base=base.tolist()))
    first = np.r_[[.1, -.2, .3], Rotation.from_euler("xyz", [20, -30, 80], degrees=True).as_quat()]
    target = np.r_[[.4, .2, .6], Rotation.from_euler("xyz", [-40, 10, 25], degrees=True).as_quat()]
    delta = delta_eef(first, target, config)
    recovered = apply_eef_delta(first, delta, config)
    np.testing.assert_allclose(recovered[:3], target[:3], atol=1e-6)
    assert (Rotation.from_quat(recovered[3:]).inv() * Rotation.from_quat(target[3:])).magnitude() < 1e-6
    np.testing.assert_allclose(execution_target(np.r_[delta, 1], config, current_tcp=first), np.r_[recovered, 1])


@pytest.mark.parametrize("mode", MODES)
def test_same_real_replay_selects_all_four_targets(replay_fixture, mode):
    data = OHTDataset(replay_fixture.replay, "train", action_config={"mode": mode})
    sample = data[0]
    meta = data.samples[0]
    assert sample["action"].shape == (action_size({"mode": mode}),)
    assert sample["joint_positions"].shape == (6,) and sample["tcp_pose"].shape == (7,)
    assert sample["joint_action"].dtype == np.float32
    expected = selected_target(meta["labels"], sample, {"mode": mode})
    np.testing.assert_allclose(sample["action"], expected)
    if mode == "joint":
        np.testing.assert_allclose(sample["action"], sample["joint_action"])
    if mode == "relative_joint":
        np.testing.assert_allclose(sample["action"][:6], sample["joint_action"][:6] - sample["joint_positions"])
    assert collate([sample, sample])["action"].shape == (2, 1, action_size({"mode": mode}))


def test_joint_targets_use_measurements_and_only_explicit_continuous_joints_wrap():
    labels = dict(action=[0, 0, 0, 0, 0, 0, 1, 1],
                  joint_action=[-np.pi+.1, .8, .7, .6, .5, .4, 1])
    obs = dict(joint_positions=np.array([np.pi-.1, .1, .2, .3, .4, .5]))
    ordinary = selected_target(labels, obs, {"mode": "relative_joint"})
    wrapped = selected_target(labels, obs, {"mode": "relative_joint", "joint_wrap": [0]})
    assert ordinary[0] < -6
    np.testing.assert_allclose(wrapped[:2], [.2, .7], atol=1e-6)
    absolute = execution_target(wrapped, {"mode": "relative_joint"}, current_joints=obs["joint_positions"])
    np.testing.assert_allclose(absolute[1:6], labels["joint_action"][1:6])
    with pytest.raises(ValueError, match="limits"):
        selected_target(labels, obs, {"mode": "joint", "joint_limits": [[-.1, .1]] * 6})


def test_config_inheritance_changes_actions_without_changing_assistance(tmp_path):
    for object_mode in ("baseline", "role_queries", "predicted_external"):
        for mode in MODES:
            path = tmp_path / f"{object_mode}-{mode}.yaml"
            path.write_text(f"extends: {ROOT.as_posix()}/finetune/OHT/configs/{object_mode}.yaml\naction_mode: {mode}\n", encoding="utf-8")
            config = load(path)
            assert config["mode"] == object_mode and resolve_action(config)["mode"] == mode
    for name in ("joint", "relative_joint", "relative_eef"):
        assert resolve_action(load(ROOT/f"finetune/OHT/configs/{name}.yaml"))["mode"] == name
    for values in ({"mode": "unknown"}, {"mode": "relative_eef", "translation_frame": "base"},
                   {"mode": "joint", "joint_names": ["x"]*6}, {"joint_wrap": [7]},
                   {"joint_scale": 0}, {"hidden_dim": 0}):
        with pytest.raises(ValueError):
            action_options(values)


class TinyMVT(nn.Module):
    """Only renderer/VLM stand in; the action head, agent and optimizer are real."""
    def __init__(self, **kwargs):
        super().__init__()
        self.stage_two = kwargs["stage_two"]
        self.num_img, self.img_size = 3, 8
        self.mvt1 = nn.Module()
        self.mvt1.vlm_dim = 4
        self.mvt1.features = nn.Parameter(torch.randn(24))
        self.mvt1.enable_efficient_paligemma_forward = lambda: None
        self.mvt1.enable_gradient_checkpointing = lambda: None
        self.trans = nn.Parameter(torch.randn(3, 8, 8))
        self.feat = nn.Parameter(torch.randn(220))
        self.internal = kwargs["object_slots_enabled"]
        self.object_conditioning_supervise_mixed_role_maps = False
        if kwargs["oracle_prior_adapter_rank"]:
            self.oracle_prior_feature_adapter1 = nn.Linear(24, 24)
        if self.internal:
            self.object_slot_predictor1 = nn.Linear(24, 3)

    def forward(self, pc, **kwargs):
        batch = len(pc)
        features = self.mvt1.features[None].expand(batch, -1)
        prediction = None
        if self.internal:
            prediction = self.object_slot_predictor1(features)
            # Prediction conditions the action; teacher only enters its loss.
            features = features + prediction[:, :2].mean(dim=1, keepdim=True)
        if hasattr(self, "oracle_prior_feature_adapter1"):
            features = self.oracle_prior_feature_adapter1(features)
        stage = dict(trans=self.trans[None].expand(batch, -1, -1, -1),
                     feat=self.feat[None].expand(batch, -1))
        if getattr(self.mvt1, "oht_return_action_features", False):
            stage["oht_action_features"] = features
        if prediction is not None:
            prior = prediction[:, None, :2, None, None].sigmoid().expand(batch, 3, 2, 2, 2)
            teacher = torch.zeros_like(prior)
            teacher[:, :, 0, 0, 0] = 1
            stage.update(object_slot_predictor_type="role_queries", object_slot_prior=prior,
                         object_slot_target_prior=teacher,
                         object_slot_target_valid=torch.ones(batch, 2).bool(),
                         object_slot_reference_null_probability=prediction[:, 2].sigmoid(),
                         object_slot_confidence=prediction[:, :2].sigmoid(),
                         object_slot_valid=torch.ones(batch, 2).bool())
        if self.stage_two:
            return dict(stage, mvt2=dict(stage, object_slot_roles_inherited=True) if self.internal else dict(stage))
        return stage

    def get_pt_loc_on_img(self, points, **kwargs):
        return torch.full((len(points), 1, 3, 2), 3.5)

    def get_wpt(self, output, *args, **kwargs):
        return torch.zeros(len(output["trans"]), 3)


@pytest.fixture
def patch_mvt(monkeypatch):
    from bridgevla.mvt import mvt
    monkeypatch.setattr(mvt, "MVT", TinyMVT)


@pytest.fixture(scope="module")
def caches(replay_fixture, tmp_path_factory):
    root = tmp_path_factory.mktemp("action-role-caches")
    contract = load_contract(replay_fixture.replay)
    ids = [row["id"] for row in read_jsonl(replay_fixture.replay/"samples.jsonl")]
    pts = np.ones((2, 8, 3), np.float32) * .2
    teacher = role_fields("teacher", pts, np.array([True, False]), np.array([True, False]),
                          known=np.array([True, True]))
    prediction = role_fields("predicted", pts, np.array([True, False]), np.array([True, False]),
                             confidence=[.9, .9])
    create_cache(root/"teacher", "teacher", contract["sha256"], 8, {"test": True}, ((key, teacher) for key in ids))
    create_cache(root/"predicted", "predicted", contract["sha256"], 8, {"test": True}, ((key, prediction) for key in ids))
    return root, prediction


@pytest.mark.parametrize("object_mode", ["baseline", "role_queries", "predicted_external"])
@pytest.mark.parametrize("mode", MODES)
def test_all_modes_train_actual_agent_and_serve_prediction_only(replay_fixture, caches, patch_mvt, object_mode, mode):
    config = load(ROOT/f"finetune/OHT/configs/{object_mode}.yaml")
    config["action_mode"] = mode
    config["point_count"] = 8
    config["action"]["hidden_dim"] = 16
    options = resolve_action(config)
    root, prediction = caches
    cache = None if object_mode == "baseline" else root/("teacher" if object_mode == "role_queries" else "predicted")
    data = OHTDataset(replay_fixture.replay, "train", object_mode, cache, 8, options)
    batch = collate([data[0], data[1]])
    agent = build_agent(config, data.contract, torch.device("cpu"))
    before = {key: value.detach().clone() for key, value in model_state(agent).items()}
    metrics = agent.update(batch, loss_scale=.5, reset_gradients=True, step_optimizer=False)
    assert all(torch.equal(value, model_state(agent)[key]) for key, value in before.items())
    metrics = agent.update(batch, loss_scale=.5, reset_gradients=False, step_optimizer=True)
    assert np.isfinite(metrics["total_loss"])
    if mode != "eef":
        assert metrics["vector_action_loss"] >= 0
        assert metrics["eef_aux_translation_loss"] > 0
        assert any(not torch.equal(value, model_state(agent)[key]) for key, value in before.items() if "oht_action_head" in key)
    if object_mode == "role_queries":
        assert metrics["object_slot_mask_loss"] > 0
        assert agent._net_mod.object_slot_predictor1.weight.grad.abs().sum() > 0
    agent.eval()
    wrapper = None
    observed = []
    if object_mode == "predicted_external":
        def predictor(current, goal):
            observed.append(set(current))
            assert not any(key.startswith("oracle") or key == "action" for key in current)
            return prediction
        wrapper = PredictedObjectWrapper(predictor, data.contract["data_config"]["cameras"], 8)
    policy = Policy(agent, data.contract, "cpu", object_mode, wrapper)
    # Input deliberately includes training labels/teachers; policy must drop them.
    value = policy.act(data[0], data[0]["goal"], 0)
    assert value.shape == (action_size(options),) and np.isfinite(value).all()
    assert value[-1] in (0, 1)
    if observed:
        assert not any("joint_action" in keys for keys in observed)
    report = open_loop(policy, replay_fixture.replay, "val", limit=2)
    assert report["action_spec"]["mode"] == mode
    metric = "joint_mae_radians_mean" if "joint" in mode else "translation_m_mean"
    assert np.isfinite(report["all_samples"][metric])


def test_role_diversity_config_reaches_actual_total_loss_and_logs(replay_fixture, caches, monkeypatch):
    from bridgevla.mvt import mvt

    class VisibleRolesMVT(TinyMVT):
        def forward(self, pc, **kwargs):
            out = super().forward(pc, **kwargs)
            teacher = out['object_slot_target_prior'].clone()
            teacher[:, :, 1, 1, 1] = 1
            raw = torch.logit(out['object_slot_prior'])
            for stage in (out, out['mvt2']):
                stage.update(object_slot_target_prior=teacher, object_role_visible_support=teacher.bool())
            out['object_slot_mask_logits'] = raw
            out['mvt2'].update(object_slot_masks_refined=True,
                               object_refine_mask_logits=raw * 1.2,
                               object_refine_pixel_valid=torch.ones_like(teacher[:, :, 0]).bool())
            return out

    monkeypatch.setattr(mvt, 'MVT', VisibleRolesMVT)
    config = load(ROOT / 'finetune/OHT/configs/role_queries.yaml')
    config['point_count'] = 8
    config['role_loss'].update(map=0., reference_null=0., refine_map=0.,
                               diversity=.3, refine_diversity=.7)
    data = OHTDataset(replay_fixture.replay, 'train', 'role_queries', caches[0] / 'teacher', 8)
    batch = collate([data[0], data[1]])
    batch['oracle_reference_object_valid'].fill_(True)
    batch['oracle_reference_present'].fill_(True)
    agent = build_agent(config, data.contract, torch.device('cpu'))
    metrics = agent.update(batch, step_optimizer=False)
    assert metrics['object_slot_diversity_loss'] > 0
    assert metrics['object_refine_diversity_loss'] > 0
    assert metrics['object_slot_diversity_supervised_pixels'] == 12
    assert metrics['object_refine_diversity_supervised_pixels'] == 12
    assert metrics['object_slot_diversity_loss_weight'] == .3
    assert metrics['object_refine_diversity_loss_weight'] == .7
    expected = (.3 * metrics['object_slot_diversity_loss']
                + .7 * metrics['object_refine_diversity_loss'])
    assert metrics['total_loss'] - metrics['action_total_loss'] == pytest.approx(expected, abs=2e-6)
    assert agent._net_mod.object_slot_predictor1.weight.grad.abs().sum() > 0
    agent.object_slot_diversity_loss_weight = agent.object_refine_diversity_loss_weight = 0.
    disabled = agent.update(batch, step_optimizer=False)
    assert disabled['total_loss'] == pytest.approx(disabled['action_total_loss'], abs=2e-6)


@pytest.mark.parametrize("mode", MODES)
def test_http_action_negotiation_and_physical_payload(replay_fixture, mode):
    options = action_options({"mode": mode})
    sample = OHTDataset(replay_fixture.replay, "train", action_config=options)[0]
    observation = policy_observation(sample, replay_fixture.config["cameras"], options)
    class Agent:
        oht_action_config = options
        def reset(self):
            pass
        def act(self, *args, **kwargs):
            return sample["action"]
    policy = Policy(Agent(), load_contract(replay_fixture.replay), "cpu", "baseline")
    server = HTTPServer(("127.0.0.1", 0), handler(Session(policy)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = Client(f"http://127.0.0.1:{server.server_port}", policy.contract["sha256"], action_config=options)
        np.testing.assert_allclose(client.act(observation, sample["goal"], "x", 0, 0), sample["action"])
        import urllib.error
        wrong = Client(f"http://127.0.0.1:{server.server_port}", policy.contract["sha256"],
                       action_config={"mode": "joint" if mode != "joint" else "eef"})
        with pytest.raises(urllib.error.HTTPError) as error:
            wrong.act(observation, sample["goal"], "wrong", 0, 0)
        assert error.value.code == 400
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("include_spec", [False, True])
def test_legacy_eef_protocol_accepts_optional_matching_action_spec(include_spec):
    options = action_options()
    request = dict(schema=SCHEMA, episode="eef", step=0, timestamp=0)
    if include_spec:
        request["action_spec"] = action_descriptor(options)
    action = np.array([.1, .2, .3, 0, 0, 0, 1, 1], dtype=np.float32)
    response = action_response(request, action, options)
    assert response["frame"] == "world"
    assert response["quaternion_order"] == "xyzw"
    assert response["reference_point"] == "tcp"
    np.testing.assert_allclose(validate_response(request, response), action)


@pytest.mark.parametrize("spec", [
    pytest.param(None, id="null"),
    pytest.param({}, id="incomplete"),
    pytest.param(action_descriptor({"mode": "joint"}), id="wrong-mode"),
    pytest.param(dict(action_descriptor(None), frame="base"), id="wrong-frame"),
    pytest.param(dict(action_descriptor(None), quaternion_order="wxyz"), id="wrong-quaternion-order"),
    pytest.param(dict(action_descriptor(None), units="degrees"), id="wrong-units"),
])
def test_legacy_eef_protocol_rejects_incompatible_action_spec(spec):
    request = dict(schema=SCHEMA, action_spec=spec, episode="eef", step=0, timestamp=0)
    with pytest.raises(ValueError, match="Legacy EEF protocol"):
        action_response(request, [0, 0, 0, 0, 0, 0, 1, 1])


def test_protocol_rejects_wrong_vector_semantics():
    options = action_options({"mode": "relative_eef"})
    request = dict(schema=ACTION_SCHEMA, action_spec=action_descriptor(options), episode="x", step=0, timestamp=0)
    response = action_response(request, np.zeros(7))
    assert validate_response(request, response) == [0.] * 7
    response["action_spec"] = dict(response["action_spec"], rotation_representation="euler")
    with pytest.raises(ValueError, match="mode/units"):
        validate_response(request, response)
    with pytest.raises(ValueError):
        action_response(dict(request, action_spec=dict(request["action_spec"], units="degrees")), np.zeros(7))


def test_legacy_replay_keeps_eef_but_vector_modes_require_current_states(replay_fixture, tmp_path):
    import shutil
    replay = tmp_path/"legacy"
    shutil.copytree(replay_fixture.replay, replay)
    rows = read_jsonl(replay/"samples.jsonl")
    for row in rows:
        path = replay/row["observation"]
        with np.load(path) as archive:
            values = {key: archive[key] for key in archive.files if key not in ("tcp_pose", "joint_positions")}
        np.savez_compressed(path, **values)
        row["observation_sha256"] = file_digest(path)
        row["labels"].pop("joint_action")
    (replay/"samples.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    from finetune.OHT.data.common import digest
    contract = load_contract(replay_fixture.replay)
    contract["index_sha256"] = file_digest(replay/"samples.jsonl")
    contract.pop("sha256")
    contract["sha256"] = digest(contract)
    write_json(replay/"contract.json", contract)
    write_json(replay/"complete.json",dict(contract_sha256=contract["sha256"],index_sha256=contract["index_sha256"],samples=len(rows)))
    assert OHTDataset(replay)[0]["action"].shape == (8,)
    for mode in MODES[1:]:
        with pytest.raises(ValueError, match="rebuild replay"):
            OHTDataset(replay, action_config={"mode": mode})[0]


def test_checkpoint_rejects_semantic_switch_but_explicit_init_rebuilds_head(replay_fixture, patch_mvt):
    config = load(ROOT/"finetune/OHT/configs/joint.yaml")
    config["action"]["hidden_dim"] = 16
    contract = load_contract(replay_fixture.replay)
    agent = build_agent(config, contract, torch.device("cpu"))
    checkpoint = dict(model_state=model_state(agent), config=copy.deepcopy(config))
    same = build_agent(config, contract, torch.device("cpu"))
    load_weights(same, checkpoint)
    config["action_mode"] = "relative_joint"
    changed = build_agent(config, contract, torch.device("cpu"))
    before = changed._net_mod.oht_action_head[0].weight.detach().clone()
    with pytest.raises(ValueError, match="action configuration"):
        load_weights(changed, checkpoint)
    load_weights(changed, checkpoint, initialize=True)
    torch.testing.assert_close(before, changed._net_mod.oht_action_head[0].weight)
    torch.testing.assert_close(changed._net_mod.mvt1.features, agent._net_mod.mvt1.features)


def test_default_eef_resume_signature_remains_compatible(replay_fixture):
    from finetune.OHT.model import resume_fingerprint
    from finetune.OHT.data.common import digest
    config = load(ROOT/"finetune/OHT/configs/baseline.yaml")
    old = {key: value for key, value in config.items() if key not in ("action", "action_mode", "train_visualization")}
    settings = {key: value for key, value in old.items() if key not in ("optimizer_steps", "checkpoint_interval", "num_workers")}
    contract = load_contract(replay_fixture.replay)
    expected = digest(dict(config=settings, replay=contract["sha256"], cache=None, world_size=1))
    assert resume_fingerprint(config, contract, None, 1) == expected
    config["action_mode"] = "joint"
    assert resume_fingerprint(config, contract, None, 1) != expected


def test_vector_agent_act_result_and_head_limits(replay_fixture, patch_mvt):
    config = load(ROOT/"finetune/OHT/configs/relative_joint.yaml")
    config["action"]["joint_limits"] = [[-3., 3.]] * 6
    agent = build_agent(config, load_contract(replay_fixture.replay), torch.device("cpu"))
    agent.eval()
    sample = OHTDataset(replay_fixture.replay, action_config=resolve_action(config))[0]
    batch = collate([sample])
    batch["language_goal"] = batch["lang_goal"]
    result = agent.act(0, batch, return_gembench_action=False)
    assert result.action.shape == (7,)
    absolute = execution_target(result.action, resolve_action(config), current_joints=sample["joint_positions"])
    assert (absolute[:6] >= -3).all() and (absolute[:6] <= 3).all()
    with pytest.raises(ValueError, match="trained agent"):
        Policy(agent, load_contract(replay_fixture.replay), "cpu", "baseline", action_config={"mode": "joint"})


def test_vector_gradient_reduction_completes_on_successive_ddp_steps(replay_fixture, patch_mvt, tmp_path):
    import datetime
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU Gloo backend unavailable")
    # FileStore avoids depending on TCPStore/libuv availability on Windows.
    dist.init_process_group("gloo", rank=0, world_size=1,
                            init_method=(tmp_path/"rendezvous").as_uri(),
                            timeout=datetime.timedelta(seconds=10))
    try:
        config = load(ROOT/"finetune/OHT/configs/joint.yaml")
        config["action"]["hidden_dim"] = 16
        data = OHTDataset(replay_fixture.replay, action_config=resolve_action(config))
        agent = build_agent(config, data.contract, torch.device("cpu"))
        agent._network = torch.nn.parallel.DistributedDataParallel(
            agent._net_mod, find_unused_parameters=True)
        batch = collate([data[0], data[1]])
        for _ in range(3):
            metrics = agent.update(batch)
            assert np.isfinite(metrics["total_loss"])
    finally:
        dist.destroy_process_group()


def test_real_mvt_single_exposes_shared_action_features_without_teacher_leakage():
    from unittest import mock
    from tests.test_object_conditioning_forward import ObjectConditioningForwardTest
    from bridgevla.mvt import mvt_single
    module, predictor, adapter, options, _ = ObjectConditioningForwardTest()._small_policy()
    module.oht_return_action_features = True
    module.eval()
    with mock.patch.object(mvt_single, "select_feat_from_hm", ObjectConditioningForwardTest._sample, create=True):
        first = module(**options, object_slot_target_heatmap=torch.zeros(1, 3, 2, 16, 16))
        second = module(**options, object_slot_target_heatmap=torch.ones(1, 3, 2, 16, 16))
    assert first["oht_action_features"].shape == (1, module.num_img*module.vlm_dim*2)
    torch.testing.assert_close(first["oht_action_features"], second["oht_action_features"])
    first["oht_action_features"].square().mean().backward()
    assert module.hidden.grad.abs().sum() > 0
    assert predictor.feature_reduce.weight.grad.abs().sum() > 0


def test_action_defaults_do_not_share_mutable_joint_configuration():
    config = action_options()
    config["joint_wrap"].append(0)
    assert action_options()["joint_wrap"] == []


@pytest.mark.parametrize("preset", ["baseline", "smoke", "relative_eef", "joint", "relative_joint",
                                    "role_queries", "predicted_external", "oht_full_8x40"])
def test_oht_presets_load_pretrained_convex_upsampling_head(preset):
    from bridgevla.mvt.raft_utils import ConvexUpSample
    config = load(ROOT/f"finetune/OHT/configs/{preset}.yaml")["mvt"]
    # Use production channel dimensions on meta to check the exact checkpoint
    # layout without allocating the multi-gigabyte model or downloading weights.
    with torch.device("meta"):
        pretrained = ConvexUpSample(in_dim=2048, out_dim=1, up_ratio=14)
        head = ConvexUpSample(in_dim=2048, out_dim=1, up_ratio=config.get("img_patch_size", 11))
    result = head.load_state_dict(pretrained.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert head.net_mask[2].weight.shape == (1764, 4096, 1, 1)
    # Also run the real upsampling operation at reduced channel width. The
    # 16x16 PaliGemma image tokens must align with the configured rendered image.
    small = ConvexUpSample(in_dim=2, out_dim=1, up_ratio=head.up_ratio)
    with torch.no_grad():
        output = small(torch.zeros(1, 2, 16, 16))
    assert output.shape == (1, 1, config.get("img_size", 220), config.get("img_size", 220))


def test_new_oht_config_matches_current_vlm_local_feature_width():
    from unittest import mock
    from tests.test_object_conditioning_forward import ObjectConditioningForwardTest
    from bridgevla.mvt import mvt_single
    config = load(ROOT/"finetune/OHT/configs/baseline.yaml")
    module, _, _, options, _ = ObjectConditioningForwardTest()._small_policy()
    module.vlm_dim, module.rot_ver = 2048, 0
    module.hidden = nn.Parameter(torch.randn(1, 770, 2048))
    module.up0 = nn.Conv2d(2048, 1, 1)
    # Exercise the production feature extraction and classifier with real VLM
    # width. A legacy final_dim=64 produces a matrix shape error here.
    module.feat_fc = nn.Linear(module.num_img*(module.vlm_dim+config["mvt"]["final_dim"]), 220)
    module._forward_action_heads = lambda feature, rotation, batch: mvt_single.MVT._forward_action_heads(module, feature, rotation, batch)
    options.update(object_slot_predictor=None, oracle_feature_adapter=None,
                   object_conditioning_shared_action_features=False, object_conditioning_use_context=False)
    module.eval()
    with mock.patch.object(mvt_single, "select_feat_from_hm", ObjectConditioningForwardTest._sample, create=True):
        out = module(**options)
    assert out["feat"].shape == (1, 220) and torch.isfinite(out["feat"]).all()
