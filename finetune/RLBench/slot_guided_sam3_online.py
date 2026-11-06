"""Launch current-observation Slot/SAM3 closed-loop evaluation."""
from pathlib import Path
import sys

FINETUNE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINETUNE))
sys.path.insert(0, str(FINETUNE / 'RLBench'))
sys.path.insert(0, str(FINETUNE / 'bridgevla/libs/YARR'))

from bridgevla.online.slot_sam3_runtime import main

if __name__ == '__main__':
    raise SystemExit(main())
