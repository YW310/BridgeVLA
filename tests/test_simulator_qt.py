"""Qt environment repair without importing Qt, cv2, SAM3 or the simulator."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "simulator_qt", ROOT / "finetune/RLBench/utils/simulator_qt.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
prepare = MODULE.prepare_coppeliasim_qt_environment


@pytest.mark.parametrize("suffix", ["", "/"])
def test_repairs_opencv_plugin_and_font_paths_after_import(tmp_path, suffix):
    env = {"COPPELIASIM_ROOT": str(tmp_path),
           "QT_QPA_PLATFORM_PLUGIN_PATH": "/env/site-packages/cv2/qt/plugins" + suffix,
           "QT_QPA_FONTDIR": "/env/site-packages/cv2/qt/fonts",
           "DISPLAY": ":88", "QT_QPA_PLATFORM": "xcb", "LD_LIBRARY_PATH": "/custom"}
    expected = dict(env, QT_QPA_PLATFORM_PLUGIN_PATH=str(tmp_path))
    del expected["QT_QPA_FONTDIR"]
    assert prepare(env, "linux") == {
        "QT_QPA_PLATFORM_PLUGIN_PATH": str(tmp_path), "QT_QPA_FONTDIR": None}
    assert env == expected
    assert prepare(env, "linux") == {}  # subsequent launches are unchanged


def test_preserves_explicit_non_opencv_paths(tmp_path):
    env = {"COPPELIASIM_ROOT": str(tmp_path),
           "QT_QPA_PLATFORM_PLUGIN_PATH": "/custom/qt/plugins",
           "QT_QPA_FONTDIR": "/custom/fonts"}
    before = dict(env)
    assert prepare(env, "linux") == {}
    assert env == before


def test_fills_unset_plugin_path_from_valid_simulator_root(tmp_path):
    env = {"COPPELIASIM_ROOT": str(tmp_path)}
    assert prepare(env, "linux") == {"QT_QPA_PLATFORM_PLUGIN_PATH": str(tmp_path)}


@pytest.mark.parametrize("root_kind", ["missing", "nonexistent", "file"])
def test_polluted_path_requires_real_simulator_directory(tmp_path, root_kind):
    root = tmp_path / "sim"
    if root_kind == "file":
        root.write_text("not a directory")
    env = {"QT_QPA_PLATFORM_PLUGIN_PATH": "/env/cv2/qt/plugins",
           "QT_QPA_FONTDIR": "/env/cv2/qt/fonts"}
    if root_kind != "missing":
        env["COPPELIASIM_ROOT"] = str(root)
    before = dict(env)
    with pytest.raises(RuntimeError, match="Set COPPELIASIM_ROOT"):
        prepare(env, "linux")
    assert env == before  # no partial repair on failure


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_non_linux_is_unchanged(platform):
    env = {"QT_QPA_PLATFORM_PLUGIN_PATH": "/env/cv2/qt/plugins"}
    before = dict(env)
    assert prepare(env, platform) == {}
    assert env == before


def test_unconfigured_environment_remains_unchanged():
    env = {}
    assert prepare(env, "linux") == {}
    assert env == {}


def test_opencv_font_repair_preserves_custom_plugin_path():
    env = {"QT_QPA_PLATFORM_PLUGIN_PATH": "/custom/qt",
           "QT_QPA_FONTDIR": "/env/cv2/qt/fonts"}
    assert prepare(env, "linux") == {"QT_QPA_FONTDIR": None}
    assert env == {"QT_QPA_PLATFORM_PLUGIN_PATH": "/custom/qt"}


@pytest.mark.parametrize("online", [False, True])
def test_actual_eval_repairs_after_model_loading_before_launch(monkeypatch, tmp_path, online):
    tree = ast.parse((ROOT / "finetune/RLBench/eval.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "eval")
    launch_guard = next(n for n in function.body if isinstance(n, ast.If)
                        and isinstance(n.test, ast.Name) and n.test.id == "environment_launched")
    wrapper = next(n for n in function.body if isinstance(n, ast.If)
                   and isinstance(n.test, ast.Name) and n.test.id == "online_slot_sam3"
                   and any(isinstance(c, ast.Name) and c.id == "make_online_policy" for c in ast.walk(n)))
    assert function.body.index(wrapper) < function.body.index(launch_guard)
    monkeypatch.setitem(sys.modules, "utils.simulator_qt", MODULE)
    monkeypatch.setattr(MODULE.sys, "platform", "linux")
    monkeypatch.setenv("COPPELIASIM_ROOT", str(tmp_path))
    # Mimic the late import performed by SAM3 (or the baseline cv2 import).
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", "/env/cv2/qt/plugins")
    monkeypatch.setenv("QT_QPA_FONTDIR", "/env/cv2/qt/fonts")
    events = []
    import os
    def launch():
        assert os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] == str(tmp_path)
        assert "QT_QPA_FONTDIR" not in os.environ
        events.append("launch")
    code = compile(ast.Module(body=[launch_guard], type_ignores=[]), "<eval_launch>", "exec")
    scope = {"environment_launched": True, "online_slot_sam3": online,
             "eval_env": SimpleNamespace(launch=launch)}
    exec(code, scope)
    assert events == ["launch"]
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", "/env/cv2/qt/plugins")
    scope["environment_launched"] = False
    exec(code, scope)
    assert events == ["launch"]
    assert os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] == "/env/cv2/qt/plugins"
