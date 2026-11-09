"""Keep OpenCV's bundled Qt plugins out of CoppeliaSim's Qt runtime."""

import os
from pathlib import Path
import sys


def _opencv_qt_path(value):
    parts = value.replace("\\", "/").rstrip("/").split("/")
    return any(parts[i:i + 2] == ["cv2", "qt"] for i in range(len(parts) - 1))


def prepare_coppeliasim_qt_environment(environ=None, platform=None):
    """Repair Linux OpenCV import side effects immediately before sim launch.

    GUI OpenCV wheels overwrite these variables on import, including when SAM3
    imports cv2 transitively. Follow PyRep's COPPELIASIM_ROOT convention only for
    an unset or OpenCV-owned plugin path; preserve explicit other Qt settings.
    Return the applied changes for logging. No Qt/OpenCV imports are needed.
    """
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if not platform.startswith("linux"):
        return {}

    changes = {}
    plugin_key = "QT_QPA_PLATFORM_PLUGIN_PATH"
    plugin_path = environ.get(plugin_key, "")
    polluted = _opencv_qt_path(plugin_path)
    if not plugin_path or polluted:
        root = environ.get("COPPELIASIM_ROOT", "")
        if root and Path(root).is_dir():
            changes[plugin_key] = root
        elif polluted:
            raise RuntimeError(
                "OpenCV replaced QT_QPA_PLATFORM_PLUGIN_PATH with "
                f"{plugin_path!r}. Set COPPELIASIM_ROOT to the existing "
                "CoppeliaSim installation directory before starting evaluation; "
                "see docs/guides/evaluation.md#rlbench-qt."
            )

    font_key = "QT_QPA_FONTDIR"
    if _opencv_qt_path(environ.get(font_key, "")):
        changes[font_key] = None

    for key, value in changes.items():
        if value is None:
            environ.pop(key, None)
        else:
            environ[key] = value
    return changes
