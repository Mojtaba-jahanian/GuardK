#!/usr/bin/env python3
"""Run the archived G9 script with its output root relocated; no scientific edits."""
from pathlib import Path
import argparse, os, sys
ROOT = Path(__file__).resolve().parents[1]
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output root; must not exist")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        parser.error("Choose a new output directory to preserve existing results")
    target.mkdir(parents=True)
    source = ROOT / "experiments/original/g9_confirmatory_grid.py"
    code = source.read_text()
    marker = 'ROOT = Path.home() / "GuardK_V6"'
    if code.count(marker) != 1:
        raise RuntimeError("Archived script structure changed; aborting")
    code = code.replace(marker, "ROOT = Path(" + repr(str(target)) + ")", 1)
    sys.path.insert(0, str(ROOT / "src"))
    # This replacement relocates outputs only. All original seeds, methods,
    # data generation, thresholds and evaluation expressions remain unchanged.
    exec(compile(code, str(source), "exec"), {"__name__": "__main__", "__file__": str(source)})
if __name__ == "__main__":
    main()
