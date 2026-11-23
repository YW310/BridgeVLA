"""Prompt-only inspection uses real TASK geometry but never SAM3 or fusion."""

import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from test_online_slot_sam3_perception import scene, refined_packet
from bridgevla.online.slot_prompt_probe import SlotPromptProbe


def packet(scene):
    value = copy.deepcopy(scene['packet'])
    visualization = dict(rendered_rgb=np.full((1, 3, 32, 32), .3, dtype=np.float32),
                         role_prior=np.full((1, 2, 4, 4), .2, dtype=np.float32),
                         translation_logits=np.zeros((1, 32, 32)), rgb_range='zero_to_one',
                         coordinate_frame='bridgevla_coarse_virtual_views')
    visualization['role_prior'][:, 0] = .8
    visualization['refine'] = dict(
        rendered_rgb=np.full((1, 3, 32, 32), .7, dtype=np.float32),
        role_prior=np.full((1, 2, 8, 8), .4, dtype=np.float32),
        translation_logits=np.zeros((1, 32, 32)), rgb_range='zero_to_one',
        coordinate_frame='bridgevla_refine_virtual_views', crop_center_coarse=np.array([.2, -.1, .8]),
        crop_scale=4., roles_inherited=True, masks_refined=False,
        valid=np.array([False, False]), role_token_valid=np.array([True, True]))
    value['visualization'] = visualization
    return value


def inspect(scene, tmp_path, value=None, **kwargs):
    client = SlotPromptProbe(scene['repo'], config_path='configs/slot_guided_offline.json')
    options = dict(instruction='move the left block next to the right block', task_name='memory',
                   checkpoint_path=scene['checkpoint'], bridge_git_sha='a' * 40,
                   diagnostics_dir=tmp_path / 'probe',
                   expected_snapshot_digest=scene['client'].snapshot_receipt(scene['snapshot'])['snapshot_input_sha256'])
    options.update(kwargs)
    return client, options, client.inspect(scene['snapshot'], packet(scene) if value is None else value, **options)


def test_real_guidance_prompts_and_stage_images_without_sam_or_fusion(scene, tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail('prompts stage must not run SAM3, candidate inference, or geometry fusion')
    monkeypatch.setattr(scene['api'].slot_guided_sam3, 'Sam3Backend', forbidden)
    monkeypatch.setattr(scene['api'].slot_guided_sam3, 'run_camera', forbidden)
    monkeypatch.setattr(scene['api'].slot_geometry_fusion, 'fuse_geometry', forbidden)
    original_open = Path.open
    def checked_open(path, *a, **kw):
        if path.resolve() == scene['checkpoint'].resolve():
            pytest.fail('probe must not re-read checkpoint contents per frame')
        return original_open(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', checked_open)
    _, _, report = inspect(scene, tmp_path)
    assert report['stage'] == 'prompts' and report['sam3_executed'] is False
    assert report['action_source'] == 'baseline_C'
    for camera in ('front', 'wrist'):
        assert report['prompts'][camera]['target']['prompt_valid']
        assert (tmp_path / 'probe' / f'{camera}.diagnostic.png').is_file()
    saved = json.loads(Path(report['report_path']).read_text(encoding='utf-8'))
    assert saved['sam3_executed'] is saved['fusion_executed'] is False
    assert saved['source_frame_index'] == 150
    assert saved['stages']['refine']['roles_inherited'] is True
    assert saved['stages']['refine']['valid'] == [False, False]
    assert saved['stages']['refine']['role_token_valid'] == [True, True]
    coarse_stats = saved['stages']['coarse']['role_map_statistics']
    assert coarse_stats['target']['peak_probability'] == pytest.approx(.8)
    assert coarse_stats['reference']['peak_probability'] == pytest.approx(.2)
    assert coarse_stats['reference']['fraction_ge_025'] == 0.
    assert coarse_stats['target_reference_soft_overlap'] == pytest.approx(.25)
    assert saved['stages']['refine']['role_map_statistics']['target_reference_soft_overlap'] == 1.
    assert not (tmp_path / 'probe' / 'object_geometry.json').exists()
    for stage, original in (('coarse', 76), ('refine', 178)):
        image = np.array(Image.open(report['stages'][stage]['images'][0]))
        assert image[0, 0, 0] == original  # Each heatmap uses its own stage RGB.
    with np.load(report['slot_arrays_path'], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['coarse_role_prior'], packet(scene)['visualization']['role_prior'])
        assert arrays['refine_role_prior'].shape == (1, 2, 8, 8)
    np.testing.assert_allclose(report['stages']['refine']['crop_center_world'], [.2, -.1, .8])
    transform = np.asarray(report['stages']['refine']['matrix_world_from_refine_cube'])
    np.testing.assert_allclose(transform[:3, 3], [.2, -.1, .8])
    np.testing.assert_allclose(transform[:3, :3], np.eye(3) / 4.)
    assert 'SAM3 not run' in Path(report['index_path']).read_text(encoding='utf-8')


def test_probe_keeps_reference_null_and_reports_invalid_prompts(scene, tmp_path):
    value = packet(scene)
    value['slot_output']['reference_null_probability'] = np.array(.99)
    value['slot_output']['role_token_valid'][1] = False
    value['slot_output']['valid'][1] = False
    _, _, report = inspect(scene, tmp_path, value)
    assert report['roles']['reference']['status'] == 'NULL'
    assert all(row['reference']['failure_reason'] == 'NULL' for row in report['prompts'].values())
    assert report['roles']['target']['status'] == 'VISIBLE'


def test_zero_role_maps_have_no_defined_overlap(scene, tmp_path):
    value = packet(scene)
    value['visualization']['role_prior'][:] = 0
    _, _, report = inspect(scene, tmp_path, value)
    stats = report['stages']['coarse']['role_map_statistics']
    assert stats['target']['peak_probability'] == 0.
    assert stats['target_reference_soft_overlap'] is None


def test_probe_refuses_stale_mutated_or_reused_artifacts(scene, tmp_path):
    client, options, _ = inspect(scene, tmp_path)
    with pytest.raises(ValueError, match='fresh'):
        client.inspect(scene['snapshot'], packet(scene), **options)
    client.reset()
    with pytest.raises(FileExistsError):
        client.inspect(scene['snapshot'], packet(scene), **options)
    client.reset()
    options['expected_snapshot_digest'] = 'f' * 64
    with pytest.raises(ValueError, match='captured policy frame'):
        client.inspect(scene['snapshot'], packet(scene), **options)


def test_probe_requires_actual_maps_instead_of_fabricating_masks(scene, tmp_path):
    value = packet(scene)
    del value['visualization']
    with pytest.raises(ValueError, match='visualization arrays'):
        inspect(scene, tmp_path, value)


def test_probe_records_refine_selection_and_uses_refined_points_for_camera_prompts(scene, tmp_path):
    value = refined_packet(scene)
    value['visualization'] = packet(scene)['visualization']
    _, _, result = inspect(scene, tmp_path, value)
    saved = json.loads(Path(result['report_path']).read_text(encoding='utf-8'))
    assert saved['guidance_selection']['roles']['target']['selected_stage'] == 'refine'
    assert result['guidance_selection'] == saved['guidance_selection']
    assert all(row['target']['prompt_valid'] for row in result['prompts'].values())
    assert 'selected_stage' in Path(result['index_path']).read_text(encoding='utf-8')
