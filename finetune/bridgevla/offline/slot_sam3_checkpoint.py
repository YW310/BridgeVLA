"""Frozen policy loading for snapshot experiments, without a simulator import."""
from pathlib import Path


CONDITIONING_FLAGS = (
    'shared_action_features', 'use_context', 'inherit_coarse_roles',
    'preserve_role_tokens', 'refine_role_masks', 'refine_local_query',
    'refine_heatmap_residual', 'soft_coarse_queries', 'soft_coarse_null_renorm',
)


def load_profile(path):
    from bridgevla.config import get_cfg_defaults
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(path))
    if (not cfg.rvt.enable_slot_guided_sam3
            or cfg.rvt.slot_guided_sam3_execution_mode != 'offline_experiment'):
        raise ValueError('use the enabled offline_experiment profile, not the disabled legacy draft')
    if (not cfg.object_slots.enabled or cfg.object_slots.predictor_type != 'role_queries'
            or cfg.oracle_relation_anchor_rank <= 0 or not cfg.oracle_relation_gated_adapter
            or not cfg.object_conditioning.shared_action_features
            or not cfg.object_conditioning.preserve_role_tokens):
        raise ValueError('offline SAM3 requires the existing role-query / role-conditioned Anchor architecture')
    return cfg


def load_saved_configs(checkpoint, profile, exp_cfg_path=None, mvt_cfg_path=None):
    from bridgevla.config import get_cfg_defaults
    from bridgevla.mvt.config import get_cfg_defaults as mvt_defaults
    folder = Path(checkpoint).resolve().parent
    exp_path = Path(exp_cfg_path) if exp_cfg_path else folder / 'exp_cfg.yaml'
    mvt_path = Path(mvt_cfg_path) if mvt_cfg_path else folder / 'mvt_cfg.yaml'
    exp, mvt = get_cfg_defaults(), mvt_defaults()
    exp.merge_from_file(str(exp_path))
    mvt.merge_from_file(str(mvt_path))
    # Saved architecture is authoritative. A runtime profile cannot turn an
    # unrelated checkpoint into a role-query policy by changing config flags.
    for section, keys in (
        ('object_slots', ('enabled', 'predictor_type', 'num_slots', 'slot_dim',
                          'decoder_layers', 'num_heads', 'point_samples')),
        ('object_conditioning', CONDITIONING_FLAGS + ('refine_mask_detach',)),
    ):
        for key in keys:
            if getattr(getattr(exp, section), key) != getattr(getattr(profile, section), key):
                raise ValueError(f'saved checkpoint configuration differs from profile: {section}.{key}')
    for key in ('oracle_prior_adapter_rank', 'oracle_relation_gated_adapter',
                'oracle_relation_anchor_rank', 'oracle_adapter_translation_only'):
        if getattr(exp, key) != getattr(profile, key):
            raise ValueError(f'saved checkpoint configuration differs from profile: {key}')
    if exp.rvt.object_prior_mode != 'o2_internal_slots' or not exp.rvt.oracle_prior_relation:
        raise ValueError('saved checkpoint must use the internal relation Slot path')
    exp.rvt.enable_slot_guided_sam3 = True
    exp.rvt.slot_guided_sam3_execution_mode = 'offline_experiment'
    # Match eval.load_agent's historical single-stage normalization choice.
    if not mvt.stage_two:
        exp.rvt.place_with_mean = True
    return exp, mvt, exp_path.resolve(), mvt_path.resolve()


def validate_checkpoint_metadata(checkpoint, cfg):
    if str(checkpoint.get('object_predictor_type', 'slots')) != cfg.object_slots.predictor_type:
        raise ValueError('checkpoint predictor metadata does not match its saved configuration')
    stored = checkpoint.get('object_conditioning', {})
    if not isinstance(stored, dict):
        raise ValueError('invalid checkpoint object_conditioning metadata')
    for key in CONDITIONING_FLAGS:
        if bool(stored.get(key, False)) != bool(getattr(cfg.object_conditioning, key)):
            raise ValueError(f'checkpoint routing metadata differs: {key}')
    if cfg.object_conditioning.refine_heatmap_residual:
        from bridgevla.models.refine_role_masks import HEATMAP_FUSION_VERSION
        if stored.get('refine_heatmap_fusion') != HEATMAP_FUSION_VERSION:
            raise ValueError('checkpoint refine heatmap fusion version differs')


def model_arguments(exp, mvt):
    """The existing eval MVT arguments, read from the saved architecture."""
    model_args = dict(mvt)
    for key in ('oracle_prior_adapter_rank', 'oracle_relation_gated_adapter',
                'oracle_adapter_translation_only', 'oracle_relation_anchor_rank'):
        model_args[key] = getattr(exp, key)
    for key in CONDITIONING_FLAGS + ('refine_mask_detach',):
        model_args['object_conditioning_' + key] = getattr(exp.object_conditioning, key)
    model_args['object_conditioning_supervise_mixed_role_maps'] = exp.object_conditioning.supervise_mixed_role_maps
    model_args['oracle_prior_relation'] = exp.rvt.oracle_prior_relation
    for key, field in (
        ('enabled', 'enabled'), ('predictor_type', 'predictor_type'),
        ('num_slots', 'num_slots'), ('slot_dim', 'dim'), ('decoder_layers', 'decoder_layers'),
        ('num_heads', 'num_heads'), ('point_samples', 'point_samples'),
    ):
        model_args['object_slots_enabled' if key == 'enabled' else 'object_slot_' + field] = getattr(exp.object_slots, key)
    model_args['object_slot_confidence_threshold'] = exp.rvt.object_prediction_confidence_threshold
    return model_args


def load_frozen_agent(checkpoint, profile, *, device='cuda:0', exp_cfg_path=None,
                      mvt_cfg_path=None, log_dir=''):
    import torch
    from bridgevla.data.observations import DEFAULT_CAMERAS, DEFAULT_SCENE_BOUNDS
    from bridgevla.models.bridgevla_agent import RVTAgent
    from bridgevla.models.slot_guided_geometry import freeze_for_geometry_inference, sha256_file
    from bridgevla.mvt.mvt import MVT
    from bridgevla.utils.rvt_utils import load_agent as load_state

    exp, mvt, exp_path, mvt_path = load_saved_configs(
        checkpoint, profile, exp_cfg_path, mvt_cfg_path)
    checkpoint_sha = sha256_file(checkpoint)
    model_args = model_arguments(exp, mvt)
    network = MVT(renderer_device=device, **model_args).to(device)
    agent = RVTAgent(network=network, image_resolution=[128, 128],
                     stage_two=mvt.stage_two, rot_ver=mvt.rot_ver,
                     scene_bounds=DEFAULT_SCENE_BOUNDS, cameras=DEFAULT_CAMERAS,
                     log_dir=str(log_dir), **exp.peract, **exp.rvt)
    agent.build(training=False, device=torch.device(device))
    load_state(str(checkpoint), agent, strict=True,
               checkpoint_validator=lambda ckpt: validate_checkpoint_metadata(ckpt, exp))
    if sha256_file(checkpoint) != checkpoint_sha:
        raise ValueError('BridgeVLA checkpoint file changed while loading')
    freeze_for_geometry_inference(network)
    agent.eval()
    return agent, {'exp_cfg_path': str(exp_path), 'mvt_cfg_path': str(mvt_path),
                   'bridge_checkpoint_sha256': checkpoint_sha}
