#!/usr/bin/env python3
"""Run from IsaacLab, or copy this launcher there and pass --bridgevla-root."""
import argparse
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument('--bridgevla-root', help='BridgeVLA repository path; alternatively set BRIDGEVLA_ROOT')
    bootstrap, remaining = parser.parse_known_args(argv)
    selected = bootstrap.bridgevla_root or os.environ.get('BRIDGEVLA_ROOT') or Path(__file__).resolve().parents[1]
    root = Path(selected).expanduser().resolve()
    if not (root / 'finetune' / 'OHT' / 'isaaclab_client.py').is_file():
        parser.error('Specify --bridgevla-root /path/to/BridgeVLA (required when this launcher is copied outside the repo)')
    sys.path.insert(0, str(root))
    from finetune.OHT.isaaclab_client import main as run
    if '-h' in remaining or '--help' in remaining:
        print(parser.format_help())
    return run(remaining)


if __name__ == '__main__':
    raise SystemExit(main())
