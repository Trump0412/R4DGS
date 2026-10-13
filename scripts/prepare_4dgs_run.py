#!/usr/bin/env python3
"""Build derived trajectory samples from a frozen standard 4DGS checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from refergaussian.run_identity import validate_query_ready_4dgs_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=32768)
    args = parser.parse_args()
    errors = validate_query_ready_4dgs_run(args.run_dir)
    if errors:
        parser.error("\n".join(errors))
    # Linux is the supported CUDA runtime. Serialize preparation when several
    # query workers first encounter the same scene; never expose partial banks.
    import fcntl
    from refergaussian.entitybank.tube_bank import prepare_scene_bank
    lock_path = args.run_dir / ".scene_bank.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(prepare_scene_bank(args.run_dir, args.num_frames, args.chunk_size))


if __name__ == "__main__":
    main()
