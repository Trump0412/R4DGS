# Third-party code, models, and data

The [MIT license](LICENSE) applies to the code authored for this repository.
It does not relicense external repositories, model weights, or datasets fetched
by the setup and download scripts. Retain their licenses and attribution when
redistributing permitted assets.

| Component | Source and terms | Version record |
|---|---|---|
| 4DGaussians | [Upstream repository](https://github.com/hustvl/4DGaussians), [pinned top-level license](https://github.com/hustvl/4DGaussians/blob/843d5ac636c37e4b611242287754f3d4ed150144/LICENSE.md). The top-level file is Apache-2.0; individual source headers and rasterizer/submodule notices must also be reviewed. Some headers retain Graphdeco/Inria non-commercial research wording. | `scripts/bootstrap_external.sh` and upstream submodule commits |
| Grounded-SAM-2, Grounding DINO, SAM2 | [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2), [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), [SAM2](https://github.com/facebookresearch/sam2). Consult each project's code and checkpoint terms. | Bootstrap/setup scripts and downloaded asset manifests |
| Qwen3-VL-8B-Instruct | [Model card and license](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `scripts/download_models.sh` and the resolved snapshot manifest |
| 4D LangSplat annotations | [Original release](https://github.com/zrporz/4DLangSplat) | `scripts/download_4dlangsplat_annotations.sh` |
| R4D-Bench-QA | [Dataset card and third-party data terms](https://huggingface.co/datasets/LiYacheng/r4d-bench-qa#license-and-third-party-data). The card currently declares `license-other`; code-repository MIT terms must not be substituted for dataset terms. | `scripts/download_r4d_bench_qa.sh` and download manifest |
| HyperNeRF / Neural 3D Video imagery | [HyperNeRF](https://github.com/google/hypernerf), [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video) | Original dataset releases and scene provenance |

The repository does not distribute these weights or scene checkpoints. This
inventory documents dependency boundaries; it does not resolve conflicting
third-party notices or grant additional commercial or redistribution rights.
