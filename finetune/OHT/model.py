"""Construct the existing BridgeVLA architecture for OHT modes."""
from . import bootstrap  # noqa: F401
from .data.common import digest


def build_agent(config, contract, device, pretrain_path=None, training=True, distributed=False):
    import torch
    from bridgevla.mvt.config import get_cfg_defaults
    from bridgevla.mvt.mvt import MVT
    from bridgevla.models.bridgevla_agent import RVTAgent
    data = contract["data_config"]
    classes = int(data.get("rotation_classes", 72))
    mvt = get_cfg_defaults()
    mvt.merge_from_other_cfg(type(mvt)(config.get("mvt", {})))
    if mvt.num_rot != classes:
        raise ValueError("Replay rotation classes and mvt.num_rot differ")
    mvt.feat_dim = classes * 3 + 4
    mode = config["mode"]
    assisted = mode != "baseline"
    internal = mode == "role_queries"
    object_config = config.get("object", {})
    kwargs = dict(
        oracle_prior_adapter_rank=object_config.get("rank", 16) if assisted else 0,
        oracle_prior_relation=assisted, oracle_relation_gated_adapter=assisted,
        oracle_relation_anchor_rank=object_config.get("anchor_rank", 16) if assisted else 0,
        object_conditioning_shared_action_features=assisted,
        object_conditioning_use_context=assisted,
        object_conditioning_inherit_coarse_roles=internal,
        object_conditioning_preserve_role_tokens=internal,
        object_slots_enabled=internal,
        object_slot_predictor_type="role_queries" if internal else "slots",
        object_slot_num_slots=2 if internal else 6,
        object_slot_dim=object_config.get("slot_dim", 128),
        object_slot_point_samples=object_config.get("point_samples", 128),
        object_slot_confidence_threshold=object_config.get("confidence_threshold", .25),
    )
    backbone = MVT(renderer_device=str(device), load_pretrain=bool(pretrain_path),
                   pretrain_path=pretrain_path, **kwargs, **dict(mvt))
    if config.get("efficient_paligemma_forward"):
        backbone.mvt1.enable_efficient_paligemma_forward()
    if training and config.get("gradient_checkpointing"):
        backbone.mvt1.enable_gradient_checkpointing()
    prefix_layers = config.get("freeze_gemma_prefix_layers", 0)
    for name, parameter in backbone.named_parameters():
        if config.get("freeze_vision_tower") and "vision_tower" in name:
            parameter.requires_grad = False
        if config.get("freeze_multimodal_projector") and "multi_modal_projector" in name:
            parameter.requires_grad = False
        if "language_model" in name and ".layers." in name:
            layer = name.split(".layers.", 1)[1].split(".", 1)[0]
            if layer.isdigit() and int(layer) < prefix_layers:
                parameter.requires_grad = False
        if "lm_head" in name or "embed_tokens" in name:
            parameter.requires_grad = False
    if config.get("adapter_only"):
        if not assisted:
            raise ValueError("adapter_only requires object assistance")
        for name, parameter in backbone.named_parameters():
            parameter.requires_grad = "oracle_prior_feature_adapter" in name or "object_slot_predictor" in name
        if not any(p.requires_grad for p in backbone.parameters()):
            raise ValueError("No object module parameters selected")
    backbone = backbone.to(device)
    if distributed:
        backbone = torch.nn.parallel.DistributedDataParallel(
            backbone, device_ids=[device.index], find_unused_parameters=True)
    model_mode = {"baseline": "none", "role_queries": "o2_internal_slots",
                  "predicted_external": "o2_predicted_relation"}[mode]
    options = dict(config["agent"])
    role_loss = config.get("role_loss", {})
    options.update(num_rotation_classes=classes, stage_two=mvt.stage_two, rot_ver=mvt.rot_ver,
                   scene_bounds=data["scene_bounds"], cameras=list(data["cameras"]),
                   image_resolution=data["image_size"], lr=config["learning_rate"],
                   object_prior_mode=model_mode, oracle_prior_mode="none",
                   oracle_prior_relation=assisted, oracle_prior_strict=True,
                   object_prediction_confidence_threshold=object_config.get("confidence_threshold", .25),
                   object_slot_mask_loss_weight=role_loss.get("map", 1),
                   object_slot_null_loss_weight=role_loss.get("reference_null", .25),
                   object_slot_diversity_loss_weight=0)
    agent = RVTAgent(network=backbone, **options)
    agent.build(training=training, device=device)
    backbone.train(training)
    return agent


def model_state(agent):
    return agent._net_mod.state_dict()


def load_weights(agent, checkpoint, initialize=False):
    import torch
    if isinstance(checkpoint, (str, bytes)) or hasattr(checkpoint, "__fspath__"):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if not initialize:
        agent._net_mod.load_state_dict(state, strict=True)
        return checkpoint
    current = agent._net_mod.state_dict()
    allowed = ("oracle_prior_feature_adapter", "object_slot_predictor")
    unexpected = [key for key in state if key not in current and not any(part in key for part in allowed)]
    missing = [key for key in current if key not in state and not any(part in key for part in allowed)]
    incompatible = [key for key in current if key in state and current[key].shape != state[key].shape
                    and not any(part in key for part in allowed)]
    if unexpected or missing or incompatible:
        raise ValueError(f"Incompatible initialization: missing={missing[:5]}, unexpected={unexpected[:5]}, shapes={incompatible[:5]}")
    compatible = {key: value for key, value in state.items()
                  if key in current and current[key].shape == value.shape}
    agent._net_mod.load_state_dict(compatible, strict=False)
    return checkpoint


def resume_fingerprint(config, contract, cache_hash, world_size):
    # Budget increases and logging changes do not alter optimizer continuity.
    settings = {k: v for k, v in config.items() if k not in (
        "optimizer_steps", "checkpoint_interval", "num_workers")}
    return digest(dict(config=settings, replay=contract["sha256"], cache=cache_hash, world_size=world_size))
