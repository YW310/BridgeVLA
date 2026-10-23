"""Semantic IDs -> explicit current-frame role masks; never policy observations."""
import json
import sys
from pathlib import Path
import numpy as np

from .common import digest, file_digest, inside, read_config, read_jsonl, write_json
from .reader import read_episode
from .source_config import sample_data_config
from .video import EpisodeVideos


def role_spec(value):
    if value == 'unknown':
        return dict(source='unknown', present=False, known=False), None
    if (not isinstance(value, list)
            or any(type(v) is not int or not 0 <= v <= 6 for v in value)
            or len(set(value)) != len(value)):
        raise ValueError('Roles require distinct semantic IDs 0..6, [] or unknown')
    return dict(source='visible_surface' if value else 'none', present=bool(value), known=True), value


def role_mask(ids, classes, size):
    ids = np.asarray(ids)
    if ids.dtype != np.uint8 or ids.ndim != 2 or not np.isin(ids, [0, 1, 2, 3, 4, 5, 6, 255]).all():
        raise ValueError('Segmentation must be native uint8 HxW with exact IDs 0..6/255')
    h, w = size
    if h < 1 or w < 1 or ids.shape[0] % h or ids.shape[1] % w:
        raise ValueError('Segmentation resolution must divide exactly like cached RGB-D')
    # Match camera_observation's integer-stride sampling; no interpolation.
    return np.isin(ids[::ids.shape[0] // h, ::ids.shape[1] // w], classes)


def export_roles(root, manifest, replay, mapping, output):
    root, replay, output = Path(root), Path(replay), Path(output)
    schema = json.loads((replay / 'contract.json').read_text(encoding='utf-8'))['schema']
    if schema == 'oht_bridgevla_v4':
        from .chunk_dataset import load_chunk_contract
        contract = load_chunk_contract(replay)
    else:
        from .replay import load_contract
        contract = load_contract(replay)
    audit = json.loads(Path(manifest).read_text(encoding='utf-8'))
    value = dict(audit)
    checksum = value.pop('manifest_sha256', None)
    if (checksum != digest(value) or checksum != contract.get('audit_manifest_sha256')
            or audit.get('schema') != 'oht_audit_v1'):
        raise ValueError('Segmentation source audit does not match replay; rebuild from the new acquisition')
    rules = read_config(mapping)
    if rules.get('schema') != 'oht_seg_roles_v1':
        raise ValueError('Unsupported segmentation role mapping schema')
    for task in rules['tasks'].values():
        for rule in [task['default'], *task.get('phases', {}).values()]:
            for role in ('target', 'reference'):
                role_spec(rule[role])
    records = {(r['task'], r['episode_index'], r['dataset']): r for r in audit['episodes']}
    rows = read_jsonl(replay / 'samples.jsonl')
    groups = {}
    for row in rows:
        if row['task'] not in rules['tasks']:
            raise ValueError(f"Missing explicit task role mapping: {row['task']}")
        groups.setdefault((row['task'], row['episode_index'], row['data_profile']), []).append(row)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    hashes, annotations = {}, []
    output.mkdir(parents=True)
    # Publish annotations only after all selected streams have passed validation.
    for identity, samples in groups.items():
        record = records[identity]
        print(f"[OHT seg roles] {record['task']}/{record['episode_index']:06d} samples={len(samples)}", file=sys.stderr, flush=True)
        parquet = inside(root, record['path'])
        if file_digest(parquet) != record['parquet_sha256']:
            raise ValueError('Segmentation source Parquet changed after audit')
        columns = read_episode(root, record)
        config = sample_data_config(contract, samples[0])
        videos = EpisodeVideos(inside(root, record['dataset']), 1 / 120 + 1e-4,
                               alignment='frame_index', expected_frames=len(columns['timestamp']))
        try:
            for row in sorted(samples, key=lambda r: r['frame']):
                frame = row['frame']
                if (columns['frame_index'][frame] != frame
                        or columns['episode_index'][frame] != row['episode_index']
                        or not np.isclose(columns['timestamp'][frame], row['timestamp'], rtol=0, atol=1e-9)):
                    raise ValueError('Segmentation frame identity does not match replay')
                observation = inside(replay, row['observation'])
                if file_digest(observation) != row['observation_sha256']:
                    raise ValueError('Replay observation changed')
                task = rules['tasks'][row['task']]
                phase = int(columns['instruction_id'][frame])
                rule = task.get('phases', {}).get(phase, task['default'])
                annotation = dict(id=row['id'], instruction_id=phase)
                masks = {role: {} for role in ('target', 'reference')}
                for role in masks:
                    annotation[role], _ = role_spec(rule[role])
                with np.load(observation, allow_pickle=False) as cached:
                    for camera in config['cameras']:
                        reference = columns[f'observation.seg.{camera}'][frame]
                        path = inside(videos.dataset, reference['Path'])
                        if path.as_posix() not in hashes:
                            hashes[path.as_posix()] = file_digest(path)
                        ids = videos.read(reference, frame=frame, format=None, expected_format='gray')
                        # Seg and RGB must be spatially and temporally aligned at source resolution.
                        rgb_ref = columns[f'observation.images.{camera}'][frame]
                        if not np.allclose(reference['Timestamp'], rgb_ref['Timestamp'], rtol=0, atol=1e-9):
                            raise ValueError('Segmentation/RGB reference timestamp mismatch')
                        rgb = videos.read(rgb_ref, frame=frame)
                        if ids.shape != rgb.shape[:2]:
                            raise ValueError('Segmentation/RGB source resolution mismatch')
                        size = cached[f'{camera}_rgb'].shape[1:]
                        h, w = size
                        if (ids.shape[0] % h or ids.shape[1] % w
                                or not np.array_equal(rgb[::ids.shape[0] // h, ::ids.shape[1] // w].transpose(2, 0, 1),
                                                      cached[f'{camera}_rgb'])):
                            raise ValueError('Source RGB frame/sampling differs from replay; segmentation cannot align')
                        for role in masks:
                            _, classes = role_spec(rule[role])
                            mask = role_mask(ids, classes or [], size)
                            if classes:
                                masks[role][camera] = mask
                for role, arrays in masks.items():
                    if annotation[role]['source'] == 'visible_surface':
                        relative = f'masks/{row["id"]}/{role}.npz'
                        path = inside(output, relative)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(path, **arrays)
                        annotation[role]['mask_path'] = relative
                annotations.append(annotation)
            videos.validate_lengths()
        finally:
            videos.close()
    path = output / 'annotations.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in annotations), encoding='utf-8')
    write_json(output / 'provenance.json', dict(schema='oht_seg_roles_v1',
        replay_contract_sha256=contract['sha256'], audit_manifest_sha256=checksum,
        mapping_sha256=file_digest(mapping), mapping=rules, video_sha256=hashes,
        annotation_sha256=file_digest(path), alignment='frame_index', labels_only=True))
    return dict(samples=len(annotations), annotations=str(path))
