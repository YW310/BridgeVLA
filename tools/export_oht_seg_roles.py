"""Convert lossless semantic videos to task-role masks, separate from policy inputs."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.OHT.data.seg_roles import export_roles


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'manifest', 'replay', 'mapping', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(export_roles(**vars(args)))
