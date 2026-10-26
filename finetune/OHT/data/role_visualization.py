"""Read segmentation role masks for diagnostics without changing replay tensors."""
import json
from pathlib import Path
import numpy as np

from .common import file_digest, inside, read_jsonl
from .role_teacher import sample_points


class SegmentationRoleAnnotations:
    def __init__(self, annotations, contract, sample_ids):
        self.path = Path(annotations).resolve()
        self.base = self.path.parent
        self.provenance = json.loads((self.base / 'provenance.json').read_text(encoding='utf-8'))
        self.checksum = file_digest(self.path)
        if (self.provenance.get('schema') != 'oht_seg_roles_v1'
                or self.provenance.get('replay_contract_sha256') != contract['sha256']
                or self.provenance.get('annotation_sha256') != self.checksum):
            raise ValueError('Role annotations do not match replay/provenance')
        rows = read_jsonl(self.path)
        self.rows = {row['id']: row for row in rows}
        if len(rows) != len(self.rows) or set(self.rows) - set(sample_ids):
            raise ValueError('Duplicate or unknown role annotation sample IDs')
        self.hashes = {}

    def read(self, sample, observation, config):
        if sample['id'] not in self.rows:
            raise ValueError(f"Missing role annotation for preview: {sample['id']}")
        annotation = self.rows[sample['id']]
        masks, points, valid = {}, [], []
        for index, role in enumerate(('target', 'reference')):
            spec = annotation[role]
            if type(spec.get('present')) is not bool or type(spec.get('known')) is not bool:
                raise ValueError('Role preview requires boolean present/known')
            cloud = np.empty((0, 3), np.float32)
            source = spec.get('source')
            if source == 'visible_surface':
                if not spec['present'] or not spec['known']:
                    raise ValueError('Role surface requires known present role')
                path = inside(self.base, spec['mask_path'])
                checksum = file_digest(path)
                expected = self.provenance.get('mask_sha256', {}).get(spec['mask_path'])
                if expected is not None and checksum != expected:
                    raise ValueError(f'Role mask changed: {path}')
                self.hashes[spec['mask_path']] = checksum
                masks[role], clouds = {}, []
                with np.load(path, allow_pickle=False) as data:
                    if set(data.files) != set(config['cameras']):
                        raise ValueError('Role mask cameras do not match replay')
                    for camera in config['cameras']:
                        mask = data[camera]
                        xyz = observation[f'{camera}_point_cloud'].transpose(1, 2, 0)
                        if mask.dtype != np.bool_ or mask.shape != xyz.shape[:2]:
                            raise ValueError('Role masks must be bool at cached RGB-D resolution')
                        masks[role][camera] = mask.copy()
                        clouds.append(xyz[mask])
                cloud = np.concatenate(clouds)
            elif source not in ('none', 'unknown') or (source == 'none' and spec['present']):
                raise ValueError('Role preview supports visible_surface/none/unknown')
            sampled, usable = sample_points(cloud, 512, index)
            points.append(sampled)
            valid.append(usable)
        return dict(role_specs=annotation, role_masks=masks, role_points=np.asarray(points),
                    role_valid=np.asarray(valid, bool))
