"""Inspect current Coarse/Refine role maps and original-camera prompts, no SAM3."""

from datetime import datetime, timezone
import importlib
import html
import json
from pathlib import Path

import numpy as np

from bridgevla.offline.slot_sam3_perception import TaskPerception
from bridgevla.online.slot_sam3_perception import OnlineTaskPerception


def _stage_images(root, stage, value, arrays):
    """Keep role probabilities distinct from action heatmaps and camera frames."""
    from PIL import Image, ImageDraw

    if value.get('rgb_range') != 'zero_to_one':
        raise ValueError('probe virtual RGB must have the declared zero_to_one range')
    rgb = np.asarray(value['rendered_rgb'])
    prior = np.asarray(value['role_prior'])
    if (rgb.ndim != 4 or rgb.shape[1] != 3 or prior.ndim != 4 or prior.shape[:2] != (len(rgb), 2)
            or not np.isfinite(rgb).all() or not np.isfinite(prior).all()
            or rgb.min() < -1e-5 or rgb.max() > 1. + 1e-5
            or prior.min() < -1e-5 or prior.max() > 1. + 1e-5):
        raise ValueError('probe expects finite RGB [V,3,H,W] and role probabilities [V,2,h,w]')
    arrays[f'{stage}_role_prior'] = prior.copy()
    arrays[f'{stage}_rendered_rgb'] = rgb.copy()
    role_state = {}
    for key in ('confidence', 'valid', 'role_token_valid', 'reference_null_probability'):
        if key in value:
            arrays[f'{stage}_{key}'] = np.asarray(value[key]).copy()
            role_state[key] = arrays[f'{stage}_{key}'].tolist()
    logits = value.get('translation_logits')
    if logits is not None:
        logits = np.asarray(logits)
        if logits.shape != (len(rgb), *rgb.shape[-2:]) or not np.isfinite(logits).all():
            raise ValueError('translation logits must match their own stage virtual RGB')
        arrays[f'{stage}_translation_logits'] = logits.copy()
    directory = root / stage
    directory.mkdir()
    paths = []
    for index, raw in enumerate(rgb):
        pixels = (np.clip(raw.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        height, width = pixels.shape[:2]
        panels = [Image.fromarray(pixels)]
        for role, color in enumerate(((255, 80, 40), (40, 150, 255))):
            probability = Image.fromarray(prior[index, role].astype(np.float32), mode='F').resize(
                (width, height), Image.Resampling.BILINEAR)
            # Fixed absolute probability scale; never amplify a weak role map.
            alpha = np.clip(np.asarray(probability), 0, 1)[..., None] * .85
            panels.append(Image.fromarray(np.clip(pixels * (1 - alpha) + np.array(color) * alpha, 0, 255).astype(np.uint8)))
        if logits is not None:
            probability = np.exp(logits[index].astype(np.float64) - logits[index].max())
            probability /= probability.sum()
            relative = probability / probability.max()
            alpha = relative[..., None] * .85
            panels.append(Image.fromarray(np.clip(pixels * (1 - alpha) + np.array([50, 255, 80]) * alpha, 0, 255).astype(np.uint8)))
        canvas = Image.new('RGB', (len(panels) * width, max(height, 80) + 58), (25, 25, 25))
        for column, panel in enumerate(panels):
            canvas.paste(panel, (column * width, 0))
        draw = ImageDraw.Draw(canvas)
        y = max(height, 80)
        draw.text((3, y), f'{stage} virtual view {index}: RGB | Target | Reference | action (if available)', fill='white')
        draw.text((3, y + 14), 'Roles: absolute probability; action: relative spatial softmax. Not an instance mask.', fill='white')
        draw.text((3, y + 28), f"inherited={value.get('roles_inherited', False)} masks_refined={value.get('masks_refined', False)} "
                  f"relocalized={value.get('relocalized', False)}", fill='white')
        draw.text((3, y + 42), f"confidence={role_state.get('confidence')} geometry_valid={role_state.get('valid')} "
                  f"semantic_valid={role_state.get('role_token_valid')} ref_null={role_state.get('reference_null_probability')}", fill='white')
        filename = directory / f'view_{index:02d}.png'
        canvas.save(filename)
        paths.append(str(filename))
    return dict(images=paths, role_prior_shape=list(prior.shape), rendered_rgb_shape=list(rgb.shape), **role_state,
                coordinate_frame=value['coordinate_frame'], roles_inherited=value.get('roles_inherited', False),
                masks_refined=value.get('masks_refined', False), relocalized=value.get('relocalized', False),
                role_map_statistics=_role_map_statistics(prior))


def _role_map_statistics(prior):
    """Describe response strength and T/R overlap, without claiming GT accuracy."""
    result = {}
    for index, role in enumerate(('target', 'reference')):
        values = prior[:, index].astype(np.float64)
        result[role] = dict(
            peak_probability=float(values.max()), mean_probability=float(values.mean()),
            fraction_ge_025=float((values >= .25).mean()),
            peak_view_row_col=[int(x) for x in np.unravel_index(values.argmax(), values.shape)])
    target, reference = prior[:, 0].astype(np.float64), prior[:, 1].astype(np.float64)
    union = np.maximum(target, reference).sum()
    result['target_reference_soft_overlap'] = float(np.minimum(target, reference).sum() / union) if union > 0 else None
    return result


class SlotPromptProbe(TaskPerception):
    """No SAM backend, candidate masks, fusion, or action conditioning is run."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_indices = {}

    def reset(self):
        self._last_indices.clear()

    def inspect(self, snapshot, coarse_packet, *, instruction, task_name, checkpoint_path,
                bridge_git_sha, diagnostics_dir, expected_snapshot_digest=None):
        api = self._load_api()
        receipt = self.snapshot_receipt(snapshot)
        if expected_snapshot_digest is not None and receipt['snapshot_input_sha256'] != expected_snapshot_digest:
            raise ValueError('current probe snapshot does not match the captured policy frame')
        episode, index = receipt['episode_id'], receipt['source_frame_index']
        if index <= self._last_indices.get(episode, -1):
            raise ValueError('prompt probe requires a fresh current source frame')
        self._last_indices[episode] = index
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError('loaded BridgeVLA checkpoint provenance is required')
        if diagnostics_dir is None:
            raise ValueError('prompt probe needs a diagnostics directory')
        root = Path(diagnostics_dir).resolve()
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise FileExistsError('prompt probe requires a new empty per-frame directory')
        export = OnlineTaskPerception._export(coarse_packet, snapshot)
        visualization = coarse_packet.get('visualization')
        if not isinstance(visualization, dict):
            raise ValueError('prompt probe requires detached coarse visualization arrays')
        guidance = api.slot_guidance_producers.coarse_slot_guidance(
            snapshot, export['slot_output'], point_valid=export['point_valid'], transform=export['transform'],
            instruction=instruction, task_name=task_name, source_checkpoint=str(checkpoint),
            source_git_sha=bridge_git_sha, created_at=datetime.now(timezone.utc).isoformat(),
            point_support_source=export['point_support_source'],
            **{key: export[key] for key in ('reference_exists', 'role_visibility') if key in export})
        root.mkdir(parents=True, exist_ok=True)
        arrays = {f'slot_{key}': np.asarray(value).copy() for key, value in export['slot_output'].items()
                  if value is not None}
        arrays.update({f'coarse_original_{key}': np.asarray(value).copy()
                       for key, value in coarse_packet.get('coarse_original_slot_output', {}).items()
                       if value is not None})
        if 'coarse_original_point_valid' in coarse_packet:
            arrays['coarse_original_point_valid'] = np.asarray(coarse_packet['coarse_original_point_valid']).copy()
        arrays['point_valid'] = np.asarray(export['point_valid']).copy()
        arrays['matrix_world_from_source'] = np.asarray(export['transform']['matrix_world_from_source']).copy()
        stages = {'coarse': _stage_images(root, 'coarse', visualization, arrays)}
        if 'refine' in visualization:
            refine = visualization['refine']
            stages['refine'] = _stage_images(root, 'refine', refine, arrays)
            center = np.asarray(refine['crop_center_coarse'])
            scale = float(refine['crop_scale'])
            if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0:
                raise ValueError('Refine crop metadata must be finite and have positive scale')
            matrix = arrays['matrix_world_from_source']
            stages['refine'].update(crop_center_coarse=center.tolist(), crop_scale=scale,
                                    crop_center_world=(matrix[:3, :3] @ center + matrix[:3, 3]).tolist())
            arrays['refine_crop_center_coarse'] = center.copy()
            reverse_crop = np.eye(4)
            reverse_crop[:3, :3] /= scale
            reverse_crop[:3, 3] = center
            arrays['matrix_world_from_refine_cube'] = matrix @ reverse_crop
            stages['refine']['matrix_world_from_refine_cube'] = arrays['matrix_world_from_refine_cube'].tolist()
        np.savez_compressed(root / 'slot_arrays.npz', **arrays)
        api.slot_guidance.save_guidance(root / 'guidance.json', guidance.metadata,
            target_points_world=guidance.target_points_world, reference_points_world=guidance.reference_points_world)
        adapter = importlib.import_module('slot_prompt_adapter')
        cameras, prompts = {}, {}
        for name, frame in snapshot.cameras.items():
            current = adapter.build_prompts(guidance, frame.camera, frame.rgb, frame.depth_m, self._config.adapter)
            cameras[name] = {'roles': {
                role: dict(prompt=prompt.as_dict(), candidates=[], selected_index=None,
                           status=guidance.metadata[role]['status'], failure_reason=prompt.failure_reason)
                for role, prompt in current.items()}}
            prompts[name] = {role: prompt.as_dict() for role, prompt in current.items()}
        if api.slot_guided_io.snapshot_input_digest(snapshot) != receipt['snapshot_input_sha256']:
            raise ValueError('probe snapshot was modified during inspection')
        report = dict(schema_version=1, artifact_type='bridgevla_slot_prompt_probe', stage='prompts',
                      prompt_mode=self.mode,
                      sam3_executed=False, fusion_executed=False, action_source='baseline_C',
                      **receipt, stages=stages, cameras=cameras,
                      roles={role: {key: guidance.metadata[role].get(key) for key in
                                    ('status', 'semantic_valid', 'confidence', 'center_world', 'point_count')}
                             for role in ('target', 'reference')},
                      reference_null_probability=guidance.metadata['reference_null_probability'])
        report['guidance_selection'] = coarse_packet.get('guidance_selection', dict(requested_stage='coarse'))
        api.slot_guided_io.save_inference_artifact(root / 'prompt_report.json', report, snapshot)
        # A static contact sheet is convenient over VS Code Remote; it runs no
        # server, browser commands or simulator control code.
        images = [Path(filename).relative_to(root).as_posix()
                  for value in stages.values() for filename in value['images']]
        images += [f'{name}.diagnostic.png' for name in snapshot.cameras]
        (root / 'index.html').write_text(
            '<!doctype html><meta charset="utf-8"><title>Slot prompt probe</title>'
            '<style>body{background:#181818;color:#eee;font:16px sans-serif}img{max-width:100%}figure{margin:20px 0}</style>'
            '<h1>Slot prompt probe: baseline action, SAM3 not run</h1>'
            f'<p>{html.escape(instruction)}</p><p>Virtual Coarse/Refine views have separate crop coordinates. '
            'Original-camera images show boxes/points only, with no instance masks.</p>'
            + '<pre>' + html.escape(json.dumps(report['guidance_selection'], indent=2)) + '</pre>'
            + ''.join(f'<figure><figcaption>{html.escape(name)}</figcaption><img src="{html.escape(name, quote=True)}"></figure>'
                      for name in images), encoding='utf-8')
        print(f'[SlotPromptProbe] source_frame={index}; index={root / "index.html"}', flush=True)
        return dict(stage='prompts', sam3_executed=False, action_source='baseline_C',
                    index_path=str(root / 'index.html'),
                    report_path=str(root / 'prompt_report.json'), slot_arrays_path=str(root / 'slot_arrays.npz'),
                    guidance_selection=report['guidance_selection'],
                    stages=stages, roles=report['roles'], reference_null_probability=report['reference_null_probability'],
                    prompts=prompts)
