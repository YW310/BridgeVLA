"""Validate the reanchoring mechanism with a synthetic lag plant, without IsaacLab or a model."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.OHT.control_validation import validate_tracking_mechanism
from finetune.OHT.data.common import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gains', type=float, nargs='+', default=[1., .3, .1],
                        help='Chosen first-order response fractions per control tick; not calibrated gains')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--horizon', type=int, default=16)
    parser.add_argument('--execute-horizon', type=int, default=4)
    parser.add_argument('--action-dt', type=float, default=1 / 60)
    parser.add_argument('--speed-m-s', type=float, default=.01)
    parser.add_argument('--output', default='logs/oht-tracking-mechanism.json')
    options = vars(parser.parse_args(argv))
    output = options.pop('output')
    report = validate_tracking_mechanism(**options)
    write_json(output, report)
    summary = dict(report, cases=[{name: value for name, value in case.items()
                                  if name not in ('reference_x_m', 'target_x_m', 'measured_x_m')}
                                 for case in report['cases']])
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f'Report: {Path(output).resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
