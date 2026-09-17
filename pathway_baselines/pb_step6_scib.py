#!/usr/bin/env python3
"""
Pathway Baselines step6: standard scIB evaluation (protocol identical to
the main pipeline)
====================================================================
Evaluates the X_aucell / X_gsva / X_ucell pathway-score representations in
pb_scores.h5ad. Protocol = main pipeline step6: native embedding space +
standard neighbor graph + official cluster_optimal_resolution (scib 1.1.5).

Env: conda activate scib
Output: pathway_baselines/results/pb_scib_metrics.csv
"""
import sys
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scib

ROOT = Path(__file__).parent
IN = ROOT / "results/pb_scores.h5ad"
OUT = ROOT / "results/pb_scib_metrics.csv"

BATCH_KEY = 'batch'
LABEL_KEY = 'seurat_annotations'
EMBEDDINGS = ['X_aucell', 'X_gsva', 'X_ucell', 'X_mod', 'X_aucell_mlp', 'X_gsva_mlp', 'X_ucell_mlp', 'X_mod_mlp']


def evaluate(adata, embed_name):
    """Call path identical to the main pipeline step6, item by item."""
    adata.obsm['X_eval'] = adata.obsm[embed_name]
    sc.pp.neighbors(adata, use_rep=embed_name)
    scib.me.cluster_optimal_resolution(
        adata, label_key=LABEL_KEY, cluster_key='predicted_cluster',
        metric=scib.me.nmi, use_rep=embed_name, verbose=False)
    m = {
        'nmi': scib.me.nmi(adata, cluster_key='predicted_cluster', label_key=LABEL_KEY),
        'ari': scib.me.ari(adata, cluster_key='predicted_cluster', label_key=LABEL_KEY),
        'asw_label': scib.me.silhouette(adata, label_key=LABEL_KEY, embed='X_eval'),
        'asw_batch': scib.me.silhouette_batch(adata, batch_key=BATCH_KEY, label_key=LABEL_KEY,
                                              embed='X_eval', verbose=False),
        'graph_conn': scib.me.graph_connectivity(adata, label_key=LABEL_KEY),
        'isolated_f1': scib.me.isolated_labels_f1(adata, label_key=LABEL_KEY,
                                                  batch_key=BATCH_KEY, embed='X_eval', verbose=False),
    }
    m['Bio'] = np.mean([m['nmi'], m['ari'], m['asw_label'], m['isolated_f1']])
    m['Batch'] = np.mean([m['asw_batch'], m['graph_conn']])
    m['Overall'] = 0.6 * m['Bio'] + 0.4 * m['Batch']
    return m


def main():
    print(f"Loading {IN} ...")
    adata = sc.read_h5ad(IN)
    adata.obs[LABEL_KEY] = adata.obs[LABEL_KEY].astype('category')
    adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].astype('category')

    # ---- resume: reuse existing results for already-evaluated embeddings ----
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
            adata.obsm[emb] = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        t = time.perf_counter()
        rows[emb] = evaluate(adata.copy(), emb)
        print(f"  time {(time.perf_counter()-t)/60:.1f} min | Overall={rows[emb]['Overall']:.4f}")
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
    PerfRecorder.start_now("pb_step6_scib")
    main()
