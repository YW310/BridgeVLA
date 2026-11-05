"""Check new scene bounds against audited V4 targets without video decoding."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.OHT.data.common import read_config, write_json
from finetune.OHT.data.workspace import check_workspace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--config', default='finetune/OHT/configs/dataset.yaml')
    parser.add_argument('--sample-stride', type=int, default=10)
    parser.add_argument('--chunk-horizon', type=int, default=16)
    parser.add_argument('--action-frame-stride', type=int, default=1)
    parser.add_argument('--margin', type=float, default=.02, help='Suggested workspace expansion margin in metres')
    parser.add_argument('--output', help='New JSON report path; existing reports are not overwritten')
    args = parser.parse_args(argv)
    if args.output and Path(args.output).exists():
        raise FileExistsError(f'Use a new workspace report: {args.output}')
    report = check_workspace(args.root, args.manifest, read_config(args.config), args.sample_stride,
        chunk_config=dict(horizon=args.chunk_horizon, frame_stride=args.action_frame_stride), margin=args.margin,
        progress=True)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
