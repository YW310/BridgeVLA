"""Audit I/O deduplication must not skip per-frame validation or change manifests."""
import copy
import json
from pathlib import Path

import pytest

from tests.test_oht_migration import replay_fixture
from finetune.OHT.cli import main
from finetune.OHT.data.common import CAMERAS, TASKS
from finetune.OHT.data import audit as audit_module
from finetune.OHT.data import reader


def test_video_files_checked_once_per_episode_including_late_new_paths(replay_fixture, monkeypatch):
    root = replay_fixture.root
    record = reader.discover(root)[0]
    expected = reader.inspect_episode(root, record, CAMERAS)
    columns = copy.deepcopy(reader.read_episode(root, record))
    # A stream can switch files on the last frame; every distinct path is checked.
    columns["observation.depth.wrist"][-1]["Path"] = "rgb_1.mp4"
    monkeypatch.setattr(reader, "read_episode", lambda *args: columns)
    dataset = root / record["dataset"]
    resolutions, file_checks = [], []
    original_inside, original_is_file = reader.inside, Path.is_file

    def resolve(base, relative):
        if Path(base) == dataset:
            resolutions.append(relative)
        return original_inside(base, relative)

    def is_file(path):
        if path.parent == dataset and path.suffix == ".mp4":
            file_checks.append(path.name)
        return original_is_file(path)

    monkeypatch.setattr(reader, "inside", resolve)
    monkeypatch.setattr(Path, "is_file", is_file)
    for _ in range(2):
        resolutions.clear()
        file_checks.clear()
        assert reader.inspect_episode(root, record, CAMERAS) == expected
        assert resolutions == file_checks == ["rgb_0.mp4", "rgb_1.mp4"]


@pytest.mark.parametrize("timestamp", [[float("nan")], [float("inf")], [-.01], [0., .1]])
def test_repeated_path_does_not_skip_late_invalid_timestamp(replay_fixture, monkeypatch, timestamp):
    record = reader.discover(replay_fixture.root)[0]
    columns = copy.deepcopy(reader.read_episode(replay_fixture.root, record))
    columns["observation.depth.wrist"][-1]["Timestamp"] = timestamp
    monkeypatch.setattr(reader, "read_episode", lambda *args: columns)
    with pytest.raises(ValueError, match="one nonnegative video timestamp"):
        reader.inspect_episode(replay_fixture.root, record, CAMERAS)


@pytest.mark.parametrize("path,error,match", [
    ("missing.mp4", FileNotFoundError, "missing.mp4"),
    ("../outside.mp4", ValueError, "escapes dataset root"),
    (["rgb_0.mp4"], ValueError, "Invalid video path"),
    ("", ValueError, "Invalid video path"),
])
def test_late_new_path_is_validated(replay_fixture, monkeypatch, path, error, match):
    record = reader.discover(replay_fixture.root)[0]
    columns = copy.deepcopy(reader.read_episode(replay_fixture.root, record))
    columns["observation.depth.wrist"][-1]["Path"] = path
    monkeypatch.setattr(reader, "read_episode", lambda *args: columns)
    with pytest.raises(error, match=match):
        reader.inspect_episode(replay_fixture.root, record, CAMERAS)


def test_existing_output_fails_before_discovery(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"
    output.write_text("keep existing audit", encoding="utf-8")

    def unexpected(*args, **kwargs):
        raise AssertionError("Existing output must fail before reading any episodes")

    monkeypatch.setattr(audit_module, "discover", unexpected)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        audit_module.audit(tmp_path / "nonexistent-raw", output)
    assert output.read_text(encoding="utf-8") == "keep existing audit"


def test_optimized_audit_preserves_entire_manifest_and_digest(replay_fixture, tmp_path):
    result = audit_module.audit(replay_fixture.root, tmp_path / "audit.json", fractions=(1/3, 1/3, 1/3))
    assert result == replay_fixture.report


def test_cli_progress_is_stderr_only_and_quiet_preserves_result(replay_fixture, tmp_path, capsys):
    noisy, quiet = tmp_path / "audit.json", tmp_path / "quiet.json"
    args = ["audit", "--root", str(replay_fixture.root), "--output", str(noisy)]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["valid_episodes"] == 12
    assert "[OHT audit] 0/12 starting" in captured.err
    assert "[OHT audit] 12/12" in captured.err
    assert "episode_s=" in captured.err and "elapsed_s=" in captured.err
    assert "valid=12 invalid=0" in captured.err
    args[-1] = str(quiet)
    assert main([*args, "--quiet"]) == 0
    silent = capsys.readouterr()
    assert silent.err == ""
    assert silent.out == captured.out
    assert noisy.read_bytes() == quiet.read_bytes()


def test_progress_keeps_invalid_episodes_in_report(tmp_path, monkeypatch, capsys):
    records = [dict(task=TASKS[0], episode_index=i) for i in range(2)]
    monkeypatch.setattr(audit_module, "discover", lambda root: records)

    def inspect(root, record, cameras):
        if record["episode_index"] == 1:
            raise ValueError("bad timestamp")
        return dict(frames=2, trajectory_hash="unique")

    monkeypatch.setattr(audit_module, "inspect_episode", inspect)
    result = audit_module.audit(tmp_path, tmp_path / "audit.json", progress=True)
    assert result["valid_episodes"] == result["invalid_episodes"] == 1
    assert result["errors"][0]["error"] == "bad timestamp"
    assert len(result["episodes"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "INVALID" in captured.err and "valid=1 invalid=1" in captured.err
