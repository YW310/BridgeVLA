"""Exercise Bash startup checks and argument/environment forwarding without CUDA."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = shutil.which("bash") or (str(WINDOWS_BASH) if WINDOWS_BASH.is_file() else None)
pytestmark = pytest.mark.skipif(BASH is None, reason="Bash is required to test the launcher")


@pytest.fixture
def launch(tmp_path):
    checkout = tmp_path / "checkout with spaces"
    server = checkout / "finetune/OHT/vla_server.py"
    server.parent.mkdir(parents=True)
    server.write_text("# Mock checkout: the test interpreter records invocations.\n")
    interpreter = tmp_path / "python with spaces"
    interpreter.write_text('''#!/usr/bin/env bash
set -euo pipefail
printf '%s\\0' "$PWD" "$PYTHONPATH" "${CUDA_VISIBLE_DEVICES:-}" "$PYTHONUNBUFFERED" \\
  "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE" "$@" > "$CAPTURE_PATH"
''', encoding="utf-8")
    interpreter.chmod(0o755)
    checkpoint = checkout / "run with spaces" / "model_last.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"mock checkpoint")
    capture = tmp_path / "capture.bin"
    env = os.environ.copy()
    for name in ("GPU_ID", "CUDA_VISIBLE_DEVICES", "OHT_MODEL_PATH", "OHT_HOST", "OHT_PORT",
                 "OHT_DEVICE", "OHT_SERVER_OUTPUT", "OHT_PRETRAIN_PATH", "HF_HUB_OFFLINE",
                 "TRANSFORMERS_OFFLINE"):
        env.pop(name, None)
    env.update(BRIDGEVLA_ROOT=checkout.as_posix(), PYTHON_BIN=interpreter.as_posix(),
               CAPTURE_PATH=capture.as_posix(), PYTHONPATH="stale-project-path")

    def run(*args, variables=None, wrapper=False):
        if capture.exists():
            capture.unlink()
        settings = {**env, **(variables or {})}
        script = ROOT / "finetune/OHT" / ("serve.sh" if wrapper else "vla_server.sh")
        result = subprocess.run([BASH, str(script), *args], cwd=tmp_path, env=settings,
                                capture_output=True, text=True, timeout=10)
        recorded = None
        if capture.exists():
            values = capture.read_bytes().decode("utf-8").split("\0")[:-1]
            recorded = dict(zip(("cwd", "pythonpath", "gpu", "unbuffered", "hf", "transformers"), values[:6]))
            recorded["args"] = values[6:]
        return result, recorded

    return run, checkout, checkpoint


def test_environment_launch_gpu_mapping_paths_and_offline_flags(launch):
    run, checkout, checkpoint = launch
    result, recorded = run(variables=dict(OHT_MODEL_PATH=checkpoint.parent.as_posix(), GPU_ID="1",
                                         OHT_HOST="0.0.0.0", OHT_PORT="8004", HF_HUB_OFFLINE="1"))
    assert result.returncode == 0, result.stderr
    assert recorded["gpu"] == "1" and recorded["unbuffered"] == "1"
    assert recorded["hf"] == recorded["transformers"] == "1"
    assert recorded["pythonpath"] == f'{recorded["cwd"]}:{recorded["cwd"]}/finetune'
    assert "stale-project" not in recorded["pythonpath"]
    assert recorded["args"][:2] == ["-m", "finetune.OHT.vla_server"]
    for option, value in (("--model-path", checkpoint.parent.as_posix()), ("--host", "0.0.0.0"),
                          ("--port", "8004"), ("--device", "cuda:0")):
        assert recorded["args"][recorded["args"].index(option) + 1] == value
    assert (checkout / "logs/oht_vla_server").is_dir()
    assert "CHECKPOINT" in result.stdout and "8004" in result.stdout


def test_cli_overrides_environment_and_wrapper_preserves_unknown_options(launch):
    run, checkout, checkpoint = launch
    extra = ["--checkpoint", checkpoint.as_posix(), "--port=8123", "--host", "127.0.0.2",
             "--output_dir", "logs/override with spaces", "--warmup-index", "3", "--model-id", "my model"]
    result, recorded = run(*extra, variables=dict(OHT_MODEL_PATH="missing", OHT_PORT="invalid",
                                                 OHT_HOST="old", CUDA_VISIBLE_DEVICES="4,5"), wrapper=True)
    assert result.returncode == 0, result.stderr
    assert recorded["args"][-len(extra):] == extra
    assert "--model-path" not in recorded["args"]
    assert "--port" not in recorded["args"]
    assert recorded["gpu"] == "4,5"
    assert recorded["hf"] == recorded["transformers"] == "0"
    assert (checkout / "logs/override with spaces").is_dir()
    assert "8123" in result.stdout and "127.0.0.2" in result.stdout


@pytest.mark.parametrize("kind", ["missing_model", "missing_checkpoint", "empty_run", "port",
                                 "empty_option", "missing_option", "pretrain", "python", "root"])
def test_invalid_settings_fail_before_python_is_started(launch, kind):
    run, checkout, checkpoint = launch
    variables, args = {"OHT_MODEL_PATH": checkpoint.as_posix()}, []
    if kind == "missing_model":
        variables["OHT_MODEL_PATH"] = ""
    elif kind == "missing_checkpoint":
        variables["OHT_MODEL_PATH"] = "absent.pth"
    elif kind == "empty_run":
        folder = checkout / "empty run"
        folder.mkdir()
        variables["OHT_MODEL_PATH"] = folder.as_posix()
    elif kind == "port":
        variables["OHT_PORT"] = "65536"
    elif kind == "empty_option":
        args = ["--model-path="]
    elif kind == "missing_option":
        args = ["--port"]
    elif kind == "pretrain":
        variables["OHT_PRETRAIN_PATH"] = "absent pretrain"
    elif kind == "python":
        variables["PYTHON_BIN"] = "definitely-not-a-python-interpreter"
    elif kind == "root":
        variables["BRIDGEVLA_ROOT"] = "absent checkout"
    result, recorded = run(*args, variables=variables)
    assert result.returncode != 0 and "[ERROR]" in result.stderr
    assert recorded is None


def test_help_needs_no_checkpoint_and_does_not_create_logs(launch):
    run, checkout, _ = launch
    result, recorded = run("--help", wrapper=True)
    assert result.returncode == 0, result.stderr
    assert recorded["args"] == ["-m", "finetune.OHT.vla_server", "--help"]
    assert not (checkout / "logs").exists()


def test_model_and_pretrain_equals_options_do_not_evaluate_shell_text(launch):
    run, checkout, _ = launch
    model = checkout / "model $(touch should-not-exist).pth"
    model.write_bytes(b"mock")
    pretrain = checkout / "pretrain with spaces"
    pretrain.mkdir()
    (pretrain / "model.safetensors.index.json").write_text("{}")
    result, recorded = run(f"--model_path={model.as_posix()}", f"--pretrain-path={pretrain.as_posix()}")
    assert result.returncode == 0, result.stderr
    assert recorded["args"][-2:] == [f"--model_path={model.as_posix()}", f"--pretrain-path={pretrain.as_posix()}"]
    assert not (checkout / "should-not-exist").exists()
