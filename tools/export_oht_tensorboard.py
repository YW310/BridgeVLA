"""Convert an existing OHT metrics.jsonl into a separate TensorBoard run."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.OHT.training_log import TensorBoardLogger


def export(metrics, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new TensorBoard export directory: {output}")
    rows = []
    with Path(metrics).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or type(row.get("step")) is not int or row["step"] < 1:
                raise ValueError(f"Invalid optimizer step at line {line_number}")
            if any(isinstance(value, (int, float)) and not math.isfinite(value) for value in row.values()):
                raise ValueError(f"Nonfinite metric at line {line_number}")
            # Resume can roll back to an older checkpoint. Drop the abandoned
            # branch, matching the live writer's purge_step behavior.
            while rows and rows[-1]["step"] >= row["step"]:
                rows.pop()
            rows.append(row)
    if not rows:
        raise ValueError("No training metrics to export")
    with TensorBoardLogger(output) as logger:
        for row in rows:
            logger.log(row["step"], {key: value for key, value in row.items() if key != "step"})
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="OHT run containing metrics.jsonl")
    parser.add_argument("--output", help="New event directory; default: RUN/tensorboard-from-jsonl")
    args = parser.parse_args(argv)
    output = Path(args.output) if args.output else Path(args.run) / "tensorboard-from-jsonl"
    count = export(Path(args.run) / "metrics.jsonl", output)
    print(json.dumps(dict(steps=count, tensorboard=str(output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
