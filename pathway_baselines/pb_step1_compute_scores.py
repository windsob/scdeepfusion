#!/usr/bin/env python3
"""
Pathway Baselines step1: compute AUCell / GSVA / UCell pathway activity scores
=============================================================================
Pathway-scoring baseline comparison for the reviewer response: our
fluctuation-based module-activity representation vs mainstream pathway
scoring methods. Same cells (donor pipeline, 13,654 cells), same modules
(6,133 valid C2 modules), same log-normalized expression
(layers['normalized']); only the activity computation changes:

- AUCell: decoupler 2.x (rank-based AUC)
- GSVA:   decoupler 2.x (kernel density + enrichment walk)
- UCell:  custom implementation (Mann-Whitney U on ranks, max_rank=1500
          as in the UCell paper default)

Output: pathway_baselines/results/pb_scores.h5ad
  obsm['X_aucell'] / ['X_gsva'] / ['X_ucell']  (cells x 6133)
  obs: batch / stim / seurat_annotations; layers['normalized'] (for step7 markers)

Env: pathway_baselines/.venv (decoupler + gseapy + scanpy)
"""
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from scipy.sparse import issparse, csr_matrix

import decoupler as dc

ROOT = Path(__file__).parent.parent  # project root
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

MAX_RANK = 1500  # rank cutoff, UCell paper default


def ucell_scores(expr: np.ndarray, gene_sets: list, max_rank: int = MAX_RANK) -> np.ndarray:
    """UCell: per-cell descending-expression ranks; score = 1 - U/(n*max_rank),
    U = sum(signature ranks) - n(n+1)/2. Ranks clipped at max_rank (UCell
    default behavior: genes beyond the cutoff are treated as equally
    unimportant)."""
    n_cells, n_genes = expr.shape
    # ranks: 1 = highest expression; argsort twice = rank
    order = np.argsort(-expr, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, n_genes + 1)[None, :].repeat(n_cells, 0), axis=1)
    ranks = np.minimum(ranks, max_rank).astype(np.float64)
    scores = np.zeros((n_cells, len(gene_sets)), dtype=np.float32)
    for j, idx in enumerate(gene_sets):
        n = len(idx)
        U = ranks[:, idx].sum(axis=1) - n * (n + 1) / 2.0
        scores[:, j] = 1.0 - U / (n * max_rank)
    return scores


def main():
    from perf_utils import PerfRecorder
    t0 = time.perf_counter()
    out_path = OUT / "pb_scores.h5ad"

    # ---- resume: skip only if file exists + all 3 keys present + no NaN ----
    if out_path.exists():
        try:
            prev = anndata.read_h5ad(out_path)
            keys = ['X_aucell', 'X_gsva', 'X_ucell']
            if all(k in prev.obsm for k in keys):
                nan_ok = all(np.isfinite(np.asarray(prev.obsm[k])).all() for k in keys)
                if nan_ok:
                    print(f"♻️ {out_path} exists and is complete (all 3 keys, no NaN); skipping")
                    return
                print("⚠️ Existing file contains NaN; recomputing")
        except Exception as e:
            print(f"⚠️ Failed to read existing file ({e}); recomputing")

    print("Loading preprocessed cache (donor pipeline, 13,654 cells)...")
    adata = anndata.read_h5ad(ROOT / "enhanced_results/DeepFusion/cache/preprocessed_data.h5ad")
    print(f"cells={adata.n_obs} genes={adata.n_vars} layers={list(adata.layers.keys())}")

    # ---- module definitions (the same 6,133 valid modules as the main model) ----
    with open(ROOT / "enhanced_results/DeepFusion/cache/valid_modules_cache.pkl", 'rb') as f:
        valid_modules = pickle.load(f)
    var_names = np.array(adata.var_names)
    module_names = list(valid_modules.keys())
    gene_sets = [np.asarray(valid_modules[m]['gene_indices'], dtype=int) for m in module_names]
    print(f"Modules: {len(module_names)} (identical to main model valid_modules)")

    # ---- expression matrix (log-normalized) ----
    expr = adata.layers['normalized']
    if issparse(expr):
        expr = expr.toarray()
    expr = np.asarray(expr, dtype=np.float32)

    # decoupler net format: source/target
    net = pd.DataFrame(
        [(m, var_names[i]) for m, gs in zip(module_names, gene_sets) for i in gs],
        columns=['source', 'target']
    )
    work = anndata.AnnData(X=csr_matrix(expr), obs=adata.obs[[]].copy(),
                           var=pd.DataFrame(index=adata.var_names))

    results = {}

    # ---- AUCell ----
    print("\n[1/3] AUCell (decoupler)...")
    t = time.perf_counter()
    # decoupler's default tmin=5 drops modules with <5 genes (6133 -> 3414);
    # use tmin=1 to keep all modules for a fair comparison with our module activity
    with PerfRecorder("pb_step1_aucell"):
        dc.mt.aucell(work, net=net, tmin=1, verbose=True)
    key = [k for k in work.obsm.keys() if 'aucell' in k.lower()][-1]
    auc = np.asarray(work.obsm[key])
    print(f"  output key obsm['{key}'] shape={auc.shape} time {(time.perf_counter()-t)/60:.1f} min")
    results['X_aucell'] = auc

    # ---- GSVA ----
    print("\n[2/3] GSVA (decoupler)...")
    t = time.perf_counter()
    with PerfRecorder("pb_step1_gsva"):
        dc.mt.gsva(work, net=net, tmin=1, verbose=True)
    key = [k for k in work.obsm.keys() if 'gsva' in k.lower()][-1]
    gsv = np.asarray(work.obsm[key])
    print(f"  output key obsm['{key}'] shape={gsv.shape} time {(time.perf_counter()-t)/60:.1f} min")
    results['X_gsva'] = gsv

    # ---- UCell (custom implementation) ----
    print("\n[3/3] UCell (custom, max_rank=1500)...")
    t = time.perf_counter()
    with PerfRecorder("pb_step1_ucell"):
        results['X_ucell'] = ucell_scores(expr, gene_sets)
    print(f"  shape={results['X_ucell'].shape} time {(time.perf_counter()-t)/60:.1f} min")

    # ---- assemble output (with obs/layers needed for evaluation) ----
    # NaN cleanup: GSVA's kernel density degenerates on zero-variance
    # (constant) genes, producing NaN; these genes carry no information, so
    # the affected module scores are set to 0 and the count is logged
    for k, v in results.items():
        v = np.asarray(v, dtype=np.float32)
        n_nan = int((~np.isfinite(v)).sum())
        if n_nan:
            print(f"⚠️ {k} contains {n_nan} non-finite values ({(~np.isfinite(v)).mean()*100:.2f}%); set to 0")
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        results[k] = v

    out = anndata.AnnData(
        X=csr_matrix(expr),
        obs=adata.obs[['batch', 'stim', 'seurat_annotations']].copy(),
        var=pd.DataFrame(index=adata.var_names),
    )
    out.layers['normalized'] = csr_matrix(expr)  # for the step7 marker panel
    for k, v in results.items():
        # decoupler output column order matches net source order; verify dims
        assert v.shape == (adata.n_obs, len(module_names)), (k, v.shape)
        out.obsm[k] = v

    out_path = OUT / "pb_scores.h5ad"
    out.write_h5ad(out_path)
    print(f"\n✅ Saved {out_path} | total time {(time.perf_counter()-t0)/60:.1f} min")
    print(f"obsm: {list(out.obsm.keys())}")


if __name__ == '__main__':
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from perf_utils import PerfRecorder
    PerfRecorder.start_now("pb_step1_compute_scores")
    main()
