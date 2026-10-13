"""Standard upstream 4DGS checkpoint identity and saved configuration."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any


def read_cfg_args(run_dir: str | Path) -> dict[str, Any]:
    """Parse upstream's Namespace without executing checkpoint text."""
    path = Path(run_dir) / "cfg_args"
    node = ast.parse(path.read_text(encoding="utf-8").strip(), mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "Namespace" or node.args:
        raise ValueError(f"Expected a literal Namespace(...) in {path}")
    if any(item.arg is None for item in node.keywords):
        raise ValueError(f"Expanded arguments are not supported in {path}")
    return {item.arg: ast.literal_eval(item.value) for item in node.keywords}


def latest_iteration_dir(run_dir: str | Path) -> Path:
    root = Path(run_dir) / "point_cloud"
    candidates = [p for p in root.glob("iteration_*") if p.is_dir() and p.name[10:].isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No fine-stage iteration_* directory under {root}")
    return max(candidates, key=lambda p: int(p.name[10:]))


def checkpoint_identity(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    iteration_dir = latest_iteration_dir(run_dir)
    files = [run_dir / "cfg_args", iteration_dir / "point_cloud.ply", iteration_dir / "deformation.pth"]
    hashes = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[path.relative_to(run_dir).as_posix()] = digest.hexdigest()
    return {"iteration": int(iteration_dir.name[10:]), "sha256": hashes}


def saved_model_args(run_dir: str | Path, defaults: dict[str, Any]) -> dict[str, Any]:
    """Defaults fill absent fields only; saved model structure always wins."""
    saved = read_cfg_args(run_dir)
    flags = ("warp_enabled", "temporal_extent_enabled", "temporal_acceleration_enabled",
             "temporal_tube_enabled", "temporal_worldtube_enabled", "spacetime_aware_optimization")
    enabled = [key for key in flags if saved.get(key, False)]
    if enabled:
        raise ValueError("Unsupported custom reconstruction checkpoint: " + ", ".join(enabled)
                         + ". Train a standard scene with scripts/train_4dgs.py.")
    return {**defaults, **saved, "model_path": str(Path(run_dir).resolve())}
