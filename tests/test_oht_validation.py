"""Full replay validation remains strict with progress and concurrent reads."""
import json
import shutil
import threading

import numpy as np
import pytest

from tests.test_oht_migration import replay_fixture  # noqa: F401
from finetune.OHT.cli import main
from finetune.OHT.data.common import digest, file_digest, read_jsonl, write_json
from finetune.OHT.data.dataset import OHTDataset
from finetune.OHT.data.role_cache import create_cache, role_fields


@pytest.mark.parametrize("workers", [1, 3])
def test_full_validation_reports_every_sample(replay_fixture, workers):
    data = OHTDataset(replay_fixture.replay)
    progress = []
    assert data.validate_all(workers, lambda done, total: progress.append((done, total))) == len(data)
    assert progress == [(done, len(data)) for done in range(len(data) + 1)]


def test_parallel_validation_is_bounded_and_actually_concurrent():
    barrier, lock = threading.Barrier(3, timeout=10), threading.Lock()
    active, peak, visited = 0, 0, []

    class ProbeDataset(OHTDataset):
        def __len__(self):
            return 12

        def __getitem__(self, index):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                visited.append(index)
            try:
                if index < 3:
                    barrier.wait()
            finally:
                with lock:
                    active -= 1

    data = ProbeDataset.__new__(ProbeDataset)
    assert data.validate_all(workers=3) == 12
    assert peak == 3
    assert sorted(visited) == list(range(12))


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("corruption", ["hash", "tensor"])
def test_parallel_validation_rejects_corrupt_samples(replay_fixture, tmp_path, workers, corruption):
    replay = tmp_path / "replay"
    shutil.copytree(replay_fixture.replay, replay)
    rows = read_jsonl(replay / "samples.jsonl")
    row = next(row for row in rows if row["split"] == "train")
    path = replay / row["observation"]
    if corruption == "hash":
        with path.open("ab") as stream:
            stream.write(b"changed")
        error = "Observation checksum mismatch"
    else:
        with np.load(path, allow_pickle=False) as source:
            values = {name: source[name] for name in source.files}
        rgb = next(name for name in values if name.endswith("_rgb"))
        values[rgb] = values[rgb].astype(np.float32)
        np.savez_compressed(path, **values)
        row["observation_sha256"] = file_digest(path)
        (replay / "samples.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        contract = json.loads((replay / "contract.json").read_text(encoding="utf-8"))
        complete = json.loads((replay / "complete.json").read_text(encoding="utf-8"))
        contract["index_sha256"] = file_digest(replay / "samples.jsonl")
        contract.pop("sha256")
        contract["sha256"] = digest(contract)
        complete.update(contract_sha256=contract["sha256"], index_sha256=contract["index_sha256"])
        write_json(replay / "contract.json", contract)
        write_json(replay / "complete.json", complete)
        error = "RGB must be uint8"
    with pytest.raises(ValueError, match=error):
        OHTDataset(replay).validate_all(workers=workers)


@pytest.mark.parametrize("mode,kind", [("role_queries", "teacher"), ("predicted_external", "predicted")])
def test_parallel_validation_checks_role_caches(replay_fixture, tmp_path, mode, kind):
    data = OHTDataset(replay_fixture.replay)
    cache = tmp_path / "roles"
    fields = role_fields(kind, np.zeros((2, 8, 3), np.float32),
                         np.array([True, False]), np.array([True, False]),
                         known=np.array([True, True]), confidence=[.9, .8])
    manifest = create_cache(cache, kind, data.contract["sha256"], 8, {"test": True},
                            ((row["id"], fields) for row in data.samples))
    assert OHTDataset(replay_fixture.replay, mode=mode, role_cache=cache, point_count=8).validate_all(3) == len(data)
    path = cache / manifest["samples"][data.samples[0]["id"]]["path"]
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Role cache data changed"):
        OHTDataset(replay_fixture.replay, mode=mode, role_cache=cache, point_count=8).validate_all(3)


def test_cli_progress_is_stderr_only_and_quiet_preserves_full_result(replay_fixture, capsys):
    replay = replay_fixture.replay
    before = {path: file_digest(replay / path) for path in ("contract.json", "samples.jsonl", "complete.json")}
    args = ["validate", "--replay", str(replay)]
    assert main([*args, "--workers", "3"]) == 0
    noisy = capsys.readouterr()
    result = json.loads(noisy.out)
    assert result["valid"] is True
    assert sum(result["samples"].values()) == replay_fixture.count
    for split, total in result["samples"].items():
        assert f"split={split} 0/{total}" in noisy.err
        assert f"split={split} {total}/{total}" in noisy.err
    assert "samples_per_s=" in noisy.err and "eta_s=" in noisy.err
    assert main([*args, "--workers", "1", "--quiet"]) == 0
    quiet = capsys.readouterr()
    assert quiet.err == ""
    assert quiet.out == noisy.out
    assert before == {path: file_digest(replay / path) for path in before}


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_dataset_rejects_invalid_worker_counts(workers):
    with pytest.raises(ValueError, match="positive integer"):
        OHTDataset.__new__(OHTDataset).validate_all(workers)


def test_cli_rejects_invalid_workers_before_loading_replay():
    with pytest.raises(SystemExit) as error:
        main(["validate", "--replay", "nonexistent", "--workers", "0"])
    assert error.value.code == 2
