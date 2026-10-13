"""Validation helpers for frozen upstream 4DGS inputs."""
from pathlib import Path
from refergaussian.checkpoint import latest_iteration_dir, saved_model_args


def validate_query_ready_4dgs_run(run_dir: str | Path) -> list[str]:
    """Validate the fine-stage iteration actually consumed by inference."""
    run_path = Path(run_dir)
    if not run_path.is_dir():
        return [f"run directory is missing: {run_path}"]
    errors: list[str] = []
    try:
        config = saved_model_args(run_path, {})
        if not config.get("source_path"):
            errors.append("cfg_args is missing source_path")
    except (OSError, ValueError, SyntaxError) as exc:
        errors.append(f"missing or invalid upstream 4DGS arguments: {run_path / 'cfg_args'} ({exc})")
    try:
        iteration_dir = latest_iteration_dir(run_path)
    except FileNotFoundError as exc:
        errors.append(f"missing upstream 4DGS checkpoint: {exc}")
        return errors
    for name in ("point_cloud.ply", "deformation.pth"):
        path = iteration_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty upstream 4DGS checkpoint file: {path}")
    iteration = int(iteration_dir.name[10:])
    render_dir = run_path / "test" / f"ours_{iteration}" / "renders"
    if not any(p.is_file() and p.stat().st_size for p in render_dir.glob("*.png")):
        errors.append(f"missing upstream 4DGS test renders for iteration {iteration}: {render_dir}")
    return errors
