"""Oracle experiment: current GT prompts -> SAM3 -> trained Relation actions."""
from pathlib import Path
import sys

FINETUNE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINETUNE))
sys.path.insert(0, str(FINETUNE / 'RLBench'))
sys.path.insert(0, str(FINETUNE / 'bridgevla/libs/YARR'))

from bridgevla.online.slot_sam3_runtime import main

if __name__ == '__main__':
    raise SystemExit(main(default_profile=Path(__file__).resolve().parent / 'configs/rlbench_gt_guided_sam3_online.yaml'))
