# R4DGS Reproduction Status

Updated: 2026-10-13

## Standard-checkpoint correction

The previous quick start trained an unmodified upstream 4DGS model, but the
query loader still required a legacy `temporal_params.pth`, a prebuilt
trajectory bank, and `config.yaml`. This revision removes that contract mismatch,
derives trajectories from `deformation.pth`, preserves saved network arguments,
and validates matching checkpoint/render iterations. HyperNeRF preparation now
runs in the upstream working directory using the configured environment.

CPU regression tests cover missing standard weights, partial iterations,
saved model configuration, exact sampled trajectory preservation, derivatives,
and checkpoint-bound cache invalidation. GPU scene loading, a fresh espresso
query, and complete 9-query/89-query reruns remain pending. The paper tables and
historical audits below are not validation results for this corrected revision.

Validation completed on 2026-10-13 with Linux / Python 3.10:

- `python -m unittest discover -s tests -v`: 196 tests passed, including isolated
  shell tests for dependency revisions and COLMAP working-directory selection.
- `python scripts/check_release.py`: passed.
- `bash -n` for all `scripts/*.sh`: passed.

The new trajectory sampler is covered with nonlinear synthetic network outputs
and round-trip cache tests. These tests do not load a real CUDA checkpoint or
measure segmentation accuracy. Start GPU validation with a fresh espresso
query, then run the complete four-scene protocol in a new output directory.

## Release Boundary

ReferGaussian consumes frozen standard upstream 4DGS checkpoints. A thin
wrapper is included to train and test-render those inputs with the pinned
upstream code; no custom scene optimizer, checkpoint, weight file, or benchmark
result bundle is included.

The release gate verifies:

- the camera-ready title and DOI;
- the 12-scene/89-query dense R4D registry;
- English-only released questions;
- standard upstream 4DGS input layout;
- pinned Grounded-SAM2 and Qwen snapshots;
- complete-evaluation and zero-target contracts;
- absence of custom reconstruction, ablation, checkpoint, and experiment-output code.

## Reported Results

The README and project page retain accepted-paper referring results and label
them as paper-reported. They do not contain scene-quality metrics.

The latest complete Public dynamic-query audit covers four scenes and nine
queries with 240/240 annotated masks and no query below 10% vIoU. This audit
preserves its source commit, manifest, evaluator, and checkpoint input identity
outside the Git repository.

The executable R4D registry contains 12 scenes and 89 queries, but a complete
fresh 89-query aggregate is not yet promoted in this release. Small category
canaries must not be reported as the full benchmark mean.

## External Reproduction

Source checkout, environment setup, standard upstream 4DGS training, strict
inference, and evaluation are documented in the README. A fresh run still
requires users to provide:

- prepared scene data;
- a compatible upstream 4DGS checkpoint and test renders;
- pinned Grounded-SAM2, Grounding DINO, SAM2, and Qwen model snapshots;
- Public or R4D-Bench-QA annotations.

No checkpoint or model weight is embedded in the repository. Dataset and model
download scripts write manifests beside external assets rather than adding the
assets to Git.

## Frozen Evaluation Identity

The camera-ready R4D-Bench-QA protocol is the released set of 89 English
sentence-level queries over 12 scenes. The Public protocol is the released set
of 9 English time-sensitive queries over 4 scenes. Both use frame-wise Acc and
time-sensitive vIoU as defined in `METRICS.md`.
