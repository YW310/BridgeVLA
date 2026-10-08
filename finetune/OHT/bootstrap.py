"""Make the local first-party modules and vendored pure-Python YARR visible."""
import sys
from pathlib import Path

FINETUNE = Path(__file__).resolve().parents[1]
for directory in (FINETUNE, FINETUNE / "bridgevla" / "libs" / "YARR"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
