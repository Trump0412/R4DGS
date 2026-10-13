"""Sample standard 4DGS deformation into a versioned, checkpoint-bound bank.

The bank is derived inference data, never an additional learned checkpoint.
There is no dependency on the legacy temporal_params.pth representation.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from refergaussian.checkpoint import checkpoint_identity, latest_iteration_dir, read_cfg_args

BANK_SCHEMA_VERSION = 2
BANK_MODE = "upstream_deformation_samples"
BANK_KEYS = ("time_values", "trajectories", "gate", "displacement", "path_length",
             "motion_score", "spatial_extent", "support_factor", "effective_scale",
             "effective_support", "tube_ratio", "occupancy_mass", "visibility_proxy")


@dataclass
class GaussianState:
    xyz: np.ndarray
    rgb: np.ndarray
    spatial_scale: np.ndarray
    anchor: np.ndarray
    scale: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    opacity: np.ndarray


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    # Standard runs store source/model information in cfg_args. Legacy readers
    # may still call this helper with config.yaml; never require that sidecar.
    if (path.parent / "cfg_args").is_file():
        return read_cfg_args(path.parent)
    if not path.exists():
        return {}
    payload = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            payload[key.strip()] = json.loads(value.strip())
        except ValueError:
            payload[key.strip()] = value.strip()
    return payload


find_latest_iteration_dir = latest_iteration_dir


def _validate_bank_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "entitybank" / "tube_bank.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Missing scene bank; run: python scripts/prepare_4dgs_run.py --run-dir {run_dir}") from exc
    if (meta.get("schema_version") != BANK_SCHEMA_VERSION or meta.get("geometry_mode") != BANK_MODE
            or meta.get("checkpoint_identity") != checkpoint_identity(run_dir)):
        raise ValueError(f"Stale or legacy scene bank in {path.parent}; rerun scripts/prepare_4dgs_run.py")
    return meta


def load_gaussian_state(run_dir: Path) -> tuple[GaussianState, dict[str, Any], int]:
    run_dir = Path(run_dir)
    meta = _validate_bank_manifest(run_dir)
    with np.load(run_dir / "entitybank" / "trajectory_samples.npz", allow_pickle=False) as data:
        state = GaussianState(**{key: np.asarray(data[key], dtype=np.float32) for key in GaussianState.__dataclass_fields__})
    if len(state.xyz) != meta["num_gaussians"]:
        raise ValueError("Scene bank Gaussian count does not match its manifest")
    return state, read_cfg_args(run_dir), int(meta["iteration"])


def load_scene_bank(run_dir: Path, num_frames: int = 64) -> dict[str, np.ndarray]:
    meta = _validate_bank_manifest(Path(run_dir))
    if meta.get("num_frames") != num_frames:
        raise ValueError(f"Scene bank has {meta.get('num_frames')} samples; prepare it with --num-frames {num_frames}")
    with np.load(Path(run_dir) / "entitybank" / "trajectory_samples.npz", allow_pickle=False) as data:
        bank = {key: np.asarray(data[key], dtype=np.float32) for key in BANK_KEYS}
    if bank["trajectories"].shape != (meta["num_gaussians"], num_frames, 3):
        raise ValueError("Invalid scene trajectory shape")
    return bank


def bank_from_samples(state: GaussianState, times: np.ndarray, trajectories: np.ndarray,
                      opacity_logits: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize exact network samples; no polynomial trajectory approximation."""
    times = np.asarray(times, dtype=np.float32)
    trajectories = np.asarray(trajectories, dtype=np.float32)
    n, frames = len(state.xyz), len(times)
    if frames < 3 or np.any(np.diff(times) <= 0):
        raise ValueError("At least three strictly increasing time samples are required")
    if trajectories.shape != (n, frames, 3) or opacity_logits.shape != (n, frames):
        raise ValueError("Deformation sample shape does not match the canonical Gaussian order")
    if not np.isfinite(trajectories).all() or not np.isfinite(opacity_logits).all():
        raise ValueError("Non-finite deformation samples")
    velocity = np.gradient(trajectories, times, axis=1, edge_order=2)
    state.velocity = velocity.mean(axis=1).astype(np.float32)
    state.acceleration = np.gradient(velocity, times, axis=1, edge_order=2).mean(axis=1).astype(np.float32)
    span = float(times[-1] - times[0])
    state.anchor = np.full((n, 1), (times[0] + times[-1]) / 2, dtype=np.float32)
    state.scale = np.full((n, 1), span, dtype=np.float32)
    extent = np.clip(state.spatial_scale.max(axis=1, keepdims=True), 1e-4, None)
    path_length = np.linalg.norm(np.diff(trajectories, axis=1), axis=2).sum(axis=1)
    ratio = path_length[:, None] / extent
    visibility = (1 / (1 + np.exp(-np.clip(opacity_logits, -80, 80)))).mean(axis=1)
    # Upstream 4DGS has no learned lifetime gate. Every Gaussian is defined for
    # all sampled times; deformed opacity is recorded separately as visibility.
    return dict(time_values=times, trajectories=trajectories,
                gate=np.ones((n, frames, 1), dtype=np.float32),
                displacement=trajectories[:, -1] - trajectories[:, 0],
                path_length=path_length, motion_score=path_length,
                spatial_extent=extent, support_factor=np.ones((n, 1), dtype=np.float32),
                effective_scale=state.scale.copy(), effective_support=state.scale.copy(),
                tube_ratio=ratio, occupancy_mass=span * (1 + ratio[:, 0]) * visibility,
                visibility_proxy=visibility)


def prepare_scene_bank(run_dir: Path, num_frames: int = 64, chunk_size: int = 32768) -> Path:
    """Evaluate frozen upstream geometry once, with bounded GPU batch size."""
    from refergaussian.semantics.renderer_geometry import _load_renderer_runtime, _renderer_state_at_time
    if num_frames < 3 or chunk_size < 1:
        raise ValueError("num_frames must be >= 3 and chunk_size must be positive")
    run_dir = Path(run_dir)
    identity = checkpoint_identity(run_dir)
    try:
        load_scene_bank(run_dir, num_frames)
        return run_dir / "entitybank"
    except (OSError, ValueError, KeyError):
        pass
    _, gaussians, _, scene, iteration = _load_renderer_runtime(run_dir)
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
    n = len(xyz)
    if n == 0:
        raise ValueError("Empty Gaussian checkpoint")
    state = GaussianState(xyz=xyz, rgb=gaussians.get_features[:, 0, :].detach().cpu().numpy().astype(np.float32),
                          spatial_scale=gaussians.get_scaling.detach().cpu().numpy().astype(np.float32),
                          opacity=gaussians._opacity.detach().cpu().numpy().reshape(-1).astype(np.float32),
                          anchor=np.empty((n, 1)), scale=np.empty((n, 1)),
                          velocity=np.empty((n, 3)), acceleration=np.empty((n, 3)))
    times = np.linspace(0, 1, num_frames, dtype=np.float32)
    trajectories = np.empty((n, num_frames, 3), dtype=np.float32)
    opacity = np.empty((n, num_frames), dtype=np.float32)
    for frame, time_value in enumerate(times):
        for start in range(0, n, chunk_size):
            ids = np.arange(start, min(start + chunk_size, n), dtype=np.int64)
            positions, _, logits = _renderer_state_at_time(gaussians, float(time_value), ids)
            trajectories[ids, frame] = positions
            opacity[ids, frame] = logits
    bank = bank_from_samples(state, times, trajectories, opacity)
    if checkpoint_identity(run_dir) != identity:
        raise RuntimeError("Checkpoint changed while generating scene bank; retry with a frozen run")
    return save_tube_bank(run_dir, state, bank, iteration)


def summarize_tube_bank(state: GaussianState, bank: dict[str, np.ndarray]) -> dict[str, Any]:
    motion_score = bank["motion_score"]
    displacement_norm = np.linalg.norm(bank["displacement"], axis=1)
    speed = np.linalg.norm(state.velocity, axis=1)
    acceleration = np.linalg.norm(state.acceleration, axis=1)
    return {
        "schema_version": 1,
        "num_gaussians": int(state.xyz.shape[0]),
        "num_frames": int(bank["time_values"].shape[0]),
        "motion_score_mean": float(motion_score.mean()),
        "motion_score_max": float(motion_score.max()),
        "displacement_mean": float(displacement_norm.mean()),
        "displacement_max": float(displacement_norm.max()),
        "speed_mean": float(speed.mean()),
        "speed_max": float(speed.max()),
        "acceleration_mean": float(acceleration.mean()),
        "acceleration_max": float(acceleration.max()),
        "support_factor_mean": float(bank["support_factor"].mean()),
        "support_factor_max": float(bank["support_factor"].max()),
        "effective_support_mean": float(bank["effective_support"].mean()),
        "effective_support_max": float(bank["effective_support"].max()),
        "tube_ratio_mean": float(bank["tube_ratio"].mean()),
        "tube_ratio_max": float(bank["tube_ratio"].max()),
        "occupancy_mean": float(bank["occupancy_mass"].mean()),
        "occupancy_max": float(bank["occupancy_mass"].max()),
        "visibility_mean": float(bank["visibility_proxy"].mean()),
        "visibility_max": float(bank["visibility_proxy"].max()),
    }


def save_tube_bank(run_dir: Path, state: GaussianState, bank: dict[str, np.ndarray],
                   iteration: int, output_dir: Path | None = None) -> Path:
    out_dir = Path(output_dir) if output_dir is not None else Path(run_dir) / "entitybank"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {**bank, **{key: getattr(state, key) for key in GaussianState.__dataclass_fields__}}
    meta = summarize_tube_bank(state, bank)
    meta.update(schema_version=BANK_SCHEMA_VERSION, geometry_mode=BANK_MODE, iteration=iteration,
                checkpoint_identity=checkpoint_identity(run_dir),
                lifetime_gate="unit_support_no_learned_lifetime",
                motion_summary="finite_difference_of_deformation_samples")
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, out_dir / "trajectory_samples.npz")
    finally:
        temporary.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out_dir, suffix=".json", delete=False) as handle:
        json.dump(meta, handle, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, out_dir / "tube_bank.json")
    return out_dir
