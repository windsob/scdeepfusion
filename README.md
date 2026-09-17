# scDeepFusion — code release (ifnb flagship pipeline)

End-to-end pipeline for the ICLR submission "scDeepFusion: Auditable Gene-Module
Representation Learning with Typed-Expert Routing for Single-Cell RNA-seq Integration".
This package reproduces the flagship ifnb results reported in the paper: the main
benchmark table, the ablation suite, the pathway-scoring baselines, and the
loss-weight sensitivity analysis.

## What's in the box

```
data/step0_seurat_process.R + create_seurat_processed_h5ad.py
  Raw data -> annotated h5ad (batch = donor via data/ifnb_donor_map.csv)
step1_scanorama_processing.py
step2_add_bbknn_harmony.py       Integration baselines (batch = donor)
step3_add_scVI.py
step4_add_scGPT_finetune.py
step5-0_create_HVGsubset.py      HVG=3000 subset
step5-1_harmony_teacher.py       Harmony teachers at multiple PC counts
step5-2_deepfusion_multitoken.py The main model (typed-expert MoE routing)
step5-3_pathway_mlp_control.py   Control: MLP readout on the same module activity
step6_scib_evaluate.py           Standard scIB v1.1.5, native embedding space
step7_evaluate_embeddings_unsupervised.py  MDF (a priori cell-type marker panel)
pathway_baselines/               AUCell / GSVA / UCell + distilled-readout control
SC_data/                         MSigDB C2 v2026.1 module library + converter
envs/                            Per-environment dependency pins
run_all.sh                       Full flagship pipeline
run_ablation_patch.sh            Loss-term ablations + simulated-dropout runs
perf_utils.py                    Wall-time / peak-memory recorder and reporter
```

## 1. Data preparation

| Artifact | How to obtain |
|---|---|
| ifnb raw data (Kang et al. 2018) | R package `SeuratData` — `InstallData("ifnb")`; step0 loads the public `ifnb` object and maps donors with the provided `data/ifnb_donor_map.csv` |
| scGPT whole-human pretrained checkpoint | Official scGPT release: `best_model.pt`, `vocab.json`, `args.json` — place in `scgpt/` |
| scGPT source code | Official scGPT repository checkout; step4 inserts it into `sys.path` (edit the path at the top of `step4_add_scGPT_finetune.py` if your checkout lives elsewhere) |
| MSigDB C2 v2026.1 (Reactome + CGP) | Already included in `SC_data/` (converted and redundancy-filtered parent set; converter `c2_convert_2026.py` included) |

## 2. Installation

Create one conda environment per stage and install the pinned dependencies:

```bash
conda create -n scib       python=3.8  -y && conda run -n scib       pip install -r envs/scib.txt
conda create -n deepfusion python=3.9  -y && conda run -n deepfusion pip install -r envs/deepfusion.txt
conda create -n seed       python=3.9  -y && conda run -n seed       pip install -r envs/scvi.txt
conda create -n scgpt      python=3.9  -y && conda run -n scgpt      pip install -r envs/scgpt.txt
```

- step0 additionally requires R (>= 4.3) with the packages listed in `envs/R.txt`.
- step4 additionally requires the scGPT source checkout and pretrained checkpoint (table above).
- The pipeline runs on Apple Silicon (MPS) and on NVIDIA GPUs (CUDA); torch builds
  with the appropriate backend are selected automatically by step5-2.

## 3. Reproducing the paper's ifnb results

1. Run step0 in `data/` (requires SeuratData's `ifnb` object and the provided
   `data/ifnb_donor_map.csv`):
   ```bash
   cd data && Rscript step0_seurat_process.R && python create_seurat_processed_h5ad.py && cd ..
   ```
2. Baseline chain (each in its own env): step1 (scib) → step2 (scib) →
   step3 (seed, with `KMP_DUPLICATE_LIB_OK=TRUE` on macOS) → step4 (scgpt).
3. Full flagship pipeline:
   ```bash
   nohup bash run_all.sh &
   ```
   Builds the HVG subset and Harmony teachers, trains the full model and the
   pathway-MLP control (3 seeds each), runs the ablation variants
   (concat / no_module / no_coarse / distill), and evaluates everything
   (scIB + unsupervised MDF) including the pathway-scoring baselines.
4. Additional ablations:
   ```bash
   nohup bash run_ablation_patch.sh &
   ```

Every step logs wall time and peak memory to `enhanced_results/perf_report.jsonl`
(run `python perf_utils.py` for a summary). Model checkpoints are tag-isolated,
so interrupted runs resume per training target.

## 4. Expected outputs

| Stage | Produces (under `enhanced_results/`) |
|---|---|
| step0 | annotated h5ad with `batch` (donor) and `seurat_annotations` labels |
| step1-4 | baseline embeddings in one shared h5ad (scanorama, BBKNN, Harmony, scVI, scGPT) |
| step5-0/5-1 | `step4-4_filter_HVG3000.h5ad`, `step4-5_add_special_harmony.h5ad` |
| step5-2/5-3 | model checkpoints per seed/tag; `X_pred_embedding_*`, `X_fusion_*`, `X_expert_weights_*` in `step5_add_DeepFusion.h5ad` |
| step6 | `integration_metrics.csv` (scIB: NMI/ARI/ASW/IsoF1/ASW_batch/GraphConn) |
| step7 | unsupervised MDF scores per embedding |

## 5. Hardware and runtime (measured on a large-memory Apple Studio workstation)

| Step | Wall time | Peak memory |
|---|---|---|
| scDeepFusion training (one seed) | ~27 min | ~10 GB |
| scVI baseline | ~51 min | ~3.5 GB |
| scGPT fine-tune | ~41 min | ~6.5 GB |
| pathway-MLP control | ~1.5 min | ~2.5 GB |
| module-activity preprocessing | ~4 s (one-time, cached) | ~3 GB |

## 6. Troubleshooting

- **scVI fails with OpenMP/library errors on macOS**: run step3 with
  `KMP_DUPLICATE_LIB_OK=TRUE`.
- **scGPT import errors**: the step expects the scGPT source checkout on
  `sys.path`; adjust the path at the top of `step4_add_scGPT_finetune.py`.
- **Resume after interruption**: re-run the same script — artifact-level
  skipping (data steps) and per-tag checkpoint resume (training) are built in.
- **scVI baseline diverges on large atlases**: that is a known property of the
  baseline under its library-default configuration on atlas-scale data; the
  paper's Tabula Sapiens analysis reports it as-is (no per-method stabilization
  was applied, to keep configurations symmetric).

## Citation and license

MIT License (see `LICENSE`). If you use this code, please cite the paper.
