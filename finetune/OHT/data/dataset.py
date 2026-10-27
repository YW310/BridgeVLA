"""A map-style cache reader compatible with RVTAgent.update, without YARR."""
from pathlib import Path
import numpy as np
from .common import read_jsonl, inside, file_digest
from .replay import load_contract
from .role_cache import RoleCache
from .observation import validate_observation
from .source_config import sample_data_config
from ..action_modes import action_options, selected_target, state_fields

MODES = ("baseline", "role_queries", "predicted_external")


class OHTDataset:
    def __init__(self, root, split="train", mode="baseline", role_cache=None, point_count=512, action_config=None):
        if mode not in MODES or split not in ("train", "val", "test"):
            raise ValueError("Unknown mode or split")
        self.root, self.mode = Path(root), mode
        self.action_config = action_options(action_config)
        self.contract = load_contract(root)
        self.samples = [r for r in read_jsonl(self.root / "samples.jsonl") if r["split"] == split]
        if not self.samples:
            raise ValueError(f"No {split} samples in replay")
        self.role_cache = None
        if mode != "baseline":
            if role_cache is None:
                raise ValueError(f"{mode} requires an explicit role cache")
            self.role_cache = RoleCache(role_cache, "teacher" if mode == "role_queries" else "predicted",
                                        self.contract["sha256"], point_count)
            missing = {r["id"] for r in self.samples} - set(self.role_cache.manifest["samples"])
            if missing:
                raise ValueError(f"Role cache missing {len(missing)} samples, e.g. {sorted(missing)[0]}")
        elif role_cache is not None:
            raise ValueError("Baseline cannot load role caches")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        path = inside(self.root, row["observation"])
        if file_digest(path) != row["observation_sha256"]:
            raise ValueError(f"Observation checksum mismatch: {row['id']}")
        with np.load(path, allow_pickle=False) as source:
            value = {name: source[name] for name in source.files}
        validate_observation(value, sample_data_config(self.contract, row))
        for name in state_fields(self.action_config):
            if name not in value:
                raise ValueError(f"{self.action_config['mode']} requires {name}; rebuild replay")
        action = selected_target(row["labels"], value, self.action_config)
        # Labels and observations are separate; no raw object/phase fields reach policy.
        value.update({name: np.asarray(label, dtype=np.float32 if name in ("action", "gripper_pose", "joint_action") else np.int32)
                      for name, label in row["labels"].items()})
        value["action"] = action
        if self.role_cache is not None:
            value.update(self.role_cache.read(row["id"]))
        value.update(goal=row["goal"], task=row["task"], sample_id=row["id"])
        return value

    def validate_all(self, workers=1, progress=None):
        """Check every sample, with bounded concurrency and no retained batches."""
        if type(workers) is not int or workers < 1:
            raise ValueError("Validation workers must be a positive integer")
        total = len(self)
        if progress is not None:
            progress(0, total)
        if workers == 1:
            for index in range(total):
                self[index]
                if progress is not None:
                    progress(index + 1, total)
        elif total:
            from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

            def validate(index):
                # Discard decoded tensors in the worker, not in a result queue.
                self[index]

            indices = iter(range(total))
            with ThreadPoolExecutor(max_workers=min(workers, total)) as pool:
                pending = {pool.submit(validate, next(indices)) for _ in range(min(workers, total))}
                completed = 0
                while pending:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        future.result()
                        completed += 1
                        if progress is not None:
                            progress(completed, total)
                        index = next(indices, None)
                        if index is not None:
                            pending.add(pool.submit(validate, index))
        return total


def collate(samples):
    import torch
    batch = {}
    for key in samples[0]:
        if key in ("goal", "task", "sample_id"):
            continue
        values = np.stack([row[key] for row in samples])
        # Presence known is [B,2]; other replay values carry a single timestep.
        batch[key] = torch.from_numpy(values if key == "oracle_role_present_known" else values[:, None])
    batch["lang_goal"] = [[[row["goal"]]] for row in samples]
    batch["tasks"] = [row["task"] for row in samples]
    if all('sample_id' in row for row in samples):
        batch['sample_ids'] = [row['sample_id'] for row in samples]
    return batch


def to_device(batch, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
