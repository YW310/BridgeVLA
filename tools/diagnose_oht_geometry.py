"""Inspect one cached OHT frame without rebuilding or registering its geometry."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.OHT.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["diagnose-geometry", *sys.argv[1:]]))
