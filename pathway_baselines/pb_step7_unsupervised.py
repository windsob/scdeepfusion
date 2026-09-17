#!/usr/bin/env python3
"""
Pathway Baselines step7: unsupervised dual-panel evaluation (reuses the
metric functions from the main pipeline step7)
==========================================================================
Loads the main pipeline step7 module via importlib and calls its
compute_mdf / compute_cs, guaranteeing an identical protocol (KMeans in
native space, 5 x n_clusters x 3 seeds, 43-type marker + 25 ISG panels,
expression taken from layers['normalized']).

Env: conda activate deepfusion
Output: pathway_baselines/results/pb_unsupervised_scores.csv
"""
import importlib.util
import sys
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
IN = ROOT / "results/pb_scores.h5ad"
OUT = ROOT / "results/pb_unsupervised_scores.csv"
EMBEDDINGS = ['X_aucell', 'X_gsva', 'X_ucell', 'X_mod', 'X_aucell_mlp', 'X_gsva_mlp', 'X_ucell_mlp', 'X_mod_mlp']

# Reuse metric functions and marker panels from the main pipeline step7
spec = importlib.util.spec_from_file_location(
    "step7", ROOT.parent / "step7_evaluate_embeddings_unsupervised.py")
step7 = importlib.util.module_from_spec(spec)
sys.modules["step7"] = step7
spec.loader.exec_module(step7)


def main():
    print(f"Loading {IN} ...")
    adata = anndata.read_h5ad(IN)
    print(f"cells={adata.n_obs}")

    # marker expression (same as the main pipeline: layers['normalized'])
    expr_source = adata.layers['normalized']
    var_index = {g: i for i, g in enumerate(adata.var_names)}

    def build_panel(genes, name):
        avail = [g for g in genes if g in var_index]
        print(f"{name}: {len(avail)}/{len(genes)} available")
        mat = np.zeros((adata.n_obs, len(avail)))
        for i, g in enumerate(avail):
            col = expr_source[:, var_index[g]]
            mat[:, i] = np.asarray(col.toarray()).flatten() if hasattr(col, 'toarray') else np.asarray(col).flatten()
        return mat

    marker_expr = build_panel(step7.MARKER_GENES, "cell-type panel")
    isg_expr = build_panel(step7.PERTURB_MARKER_GENES, "ISG panel")

    rows = {}
    if OUT.exists():
        prev = pd.read_csv(OUT, index_col=0)
        rows = prev.to_dict('index')
        print(f"♻️ Existing results: {list(rows.keys())}")

    for emb in EMBEDDINGS:
        if emb in rows:
            print(f"Skipping {emb} (already evaluated)"); continue
        if emb not in adata.obsm:
            print(f"⚠️ {emb} not found, skipping"); continue
        print(f"\nEvaluating {emb} ...")
        coords = np.asarray(adata.obsm[emb], dtype=np.float32)
        n_bad = int((~np.isfinite(coords)).sum())
        if n_bad:
            print(f"  ⚠️ {emb} contains {n_bad} non-finite values; set to 0 before evaluation")
            coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        t = time.perf_counter()
        mdf_t = step7.compute_mdf(coords, marker_expr)
        mdf_i = step7.compute_mdf(coords, isg_expr)
        rows[emb] = {'MDF_type': mdf_t, 'MDF_isg': mdf_i}
        print(f"  time {(time.perf_counter()-t)/60:.1f} min | MDF_type={mdf_t:.3f} MDF_isg={mdf_i:.3f}")
        # save after each embedding so a crash does not lose results
        pd.DataFrame(rows).T.rename_axis('Method').to_csv(OUT)

    df = pd.DataFrame(rows).T
    df.index.name = 'Method'
    df.to_csv(OUT)
    print("\n" + df.round(4).to_string())
    print(f"\n✅ Saved {OUT}")


if __name__ == '__main__':
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from perf_utils import PerfRecorder
    PerfRecorder.start_now("pb_step7_unsupervised")
    main()
