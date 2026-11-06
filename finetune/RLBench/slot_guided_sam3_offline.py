"""Single-observation Slot -> SAM3 -> frozen action predictions; no control loop."""
from pathlib import Path
import sys

FINETUNE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINETUNE))
# Same local YARR package used by the existing policy, without importing RLBench.
sys.path.insert(0, str(FINETUNE / 'bridgevla/libs/YARR'))
sys.path.insert(0, str(FINETUNE / 'bridgevla/libs/yarr'))

from bridgevla.offline.slot_sam3_runner import main

if __name__ == '__main__':
    raise SystemExit(main())
