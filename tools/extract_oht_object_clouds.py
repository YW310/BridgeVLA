"""Export approximate OHT object clouds from explicit centre/size box crops."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.OHT.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["extract-object-clouds", *sys.argv[1:]]))
