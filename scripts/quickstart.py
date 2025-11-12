#!/usr/bin/env python3
"""Minimal quickstart to validate the project skeleton.

- Optionally lists a few POP909 MIDI files if a path is provided
- Runs a tiny toy Viterbi path using a random HMM spec (if numpy is available)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Allow running as a script without installing the package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pop909 import describe


def try_hmm_demo() -> None:
    try:
        from src.models.hmm_baseline import toy_spec, ChordHMM
        import numpy as np  # noqa: F401
    except Exception as e:
        print("[quickstart] Skipping HMM demo (numpy not installed):", e)
        return

    spec = toy_spec()
    model = ChordHMM(spec)
    emissions = [0, 1, 2, 1, 0, 3, 2]
    path = model.viterbi(emissions)
    print("[quickstart] HMM demo path:", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop909", type=str, default=None, help="Path to POP909 root (optional)")
    args = ap.parse_args()

    if args.pop909:
        root = Path(args.pop909)
        if root.exists():
            files = describe(root, limit=5)
            print("[quickstart] Found examples:")
            for f in files:
                print(" -", f)
        else:
            print("[quickstart] Provided POP909 path does not exist:", root)

    try_hmm_demo()
    print("[quickstart] Project skeleton OK.")


if __name__ == "__main__":
    main()
