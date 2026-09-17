# scDeepFusion — code release (ifnb flagship pipeline)

End-to-end pipeline for the ICLR submission "scDeepFusion: Auditable Gene-Module
Representation Learning with Typed-Expert Routing for Single-Cell RNA-seq Integration".
This package reproduces the flagship ifnb results (main benchmark, ablations,
pathway-scoring baselines) reported in the paper.

## Layout

```
data/step0_seurat_process.R + create_seurat_processed_h5ad.py
  Raw data -> annotated h5ad (batch = donor via data/ifnb_donor_map.csv)
step1 scanorama -> step2 bbknn+harmony -> step3 scVI -> step4 scGPT fine-tune
  Integration baselines (all re-run with batch=donor)
step5-0 HVG subset (HVG=3000) -> step5-1 Harmony teacher (PCs=40)
step5-2 scDeepFusion   Typed-expert MoE routing, the main model
step5-3 pathway-MLP    Control: MLP readout on the same module activity
step6 scIB evaluation  Standard scIB v1.1.5, native embedding space
step7 unsupervised     MDF_type (43-gene panel) + MDF_isg (25 ISG) + CS
pathway_baselines/     AUCell / GSVA / UCell baselines + distilled-readout control
SC_data/               MSigDB C2 v2026.1 module library + converter
run_all.sh             Full flagship pipeline (baselines check -> teacher ->
                       model + pathway-MLP (3 seeds each) -> ablations ->
                       evaluation -> pathway baselines)
run_ablation_patch.sh  Loss-term ablations (coarse-only / recon-only / no-denoise)
                       and simulated-dropout robustness runs
```

## Environments

- `scib` (python 3.8): step1, step2, step5-0, step5-1, step6, pb_step6
- `seed` (python 3.9, scvi-tools): step3 — needs `KMP_DUPLICATE_LIB_OK=TRUE` on macOS
- `scgpt`: step4 — scGPT source checkout required (`sys.path` insert at the top of
  step4_add_scGPT_finetune.py); the whole-human pretrained checkpoint
  (`best_model.pt`, `vocab.json`, `args.json` from the official scGPT release)
  goes in `scgpt/`
- `deepfusion`: step5-2, step5-3, step7, pb_step7/8
- `pathway_baselines/.venv` (uv): decoupler 2.1.4 + gseapy for pb_step1

## Reproducing the paper's ifnb results

1. Run step0 in `data/` (requires SeuratData's `ifnb` object and the provided
   `ifnb_donor_map.csv`).
2. Baseline chain: step1 (scib) -> step2 (scib) -> step3 (seed) -> step4 (scgpt).
3. `nohup bash run_all.sh &` — teacher, full model and pathway-MLP control
   (3 seeds each), ablations (concat / no_module / no_coarse / distill),
   scIB + unsupervised evaluation, and the pathway-scoring baselines.
4. `nohup bash run_ablation_patch.sh &` — loss-term ablations and the
   simulated-dropout robustness runs.

Every step logs wall time and peak memory to `enhanced_results/perf_report.jsonl`
(`python perf_utils.py` for a summary table). Model checkpoints are tag-isolated,
so interrupted runs resume per training target.

scGPT baseline note: step4 fine-tunes the official pretrained checkpoint with the
official integration recipe (GEPC/MVC + explicit_zero_prob + ECS + DAB/DSBN),
with epoch-level resume checkpoints and a post-training embedding-quality sanity
check printed to the log.
