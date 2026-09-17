#!/usr/bin/env python3
"""
Comprehensive Unsupervised Embedding Evaluation Framework (v2.2)
==========================================================================
Marker-anchored, label-free embedding evaluation.

Evaluation metric:
MDF (Marker Discrimination F): KMeans-based Marker F-score, averaged over
multiple n_clusters AND multiple KMeans seeds. Reported as two separate axes:
- MDF_type: cell-type marker panel - tests cell-type structure.
- MDF_isg: perturbation-response panel (where applicable) - tests whether
  embeddings resolve perturbation state within/across cell types.
The two axes are reported separately and are not aggregated.

The unsupervised evaluation is intentionally marker-anchored only: silhouette
needs no dedicated unsupervised variant here because the scIB suite already
covers silhouette-based metrics with ground-truth anchors (ASW label/batch).

v2.1 (2026-08-11): dual marker panels.
- MDF_type: cell-type markers (unchanged v2 panel) - tests coarse cell-type structure.
- MDF_isg: interferon-stimulated genes (IFN-beta perturbation response) - tests
  whether embeddings resolve STIMULATION STATE within/across cell types.
  Rationale: on ifnb the batch axis IS the biological perturbation; the two-level
  architecture's unique contribution (module layer + attention) is expected to
  live on this axis, which the cell-type panel cannot see.
  Panel = canonical type-I IFN response genes; availability-checked against var
  (19/25 present in ifnb, all well-expressed; missing ones are reported, not
  silently dropped by hand).

v2 protocol changes (addressing post-hoc selection bias):
- Markers: canonical literature markers for ALL 13 annotated ifnb cell types,
  chosen a priori from biology, NOT filtered by expression level / rarity
  (rare types like pDC/Eryth are exactly the fine-grained test cases).
- Marker expression read from layers['normalized'] (v1 used raw counts in X,
  which confounds marker F-stats with library size).
- KMeans runs in the NATIVE embedding space (v1 compressed every method to 2D
  UMAP first and truncated different methods to different dims - both unfair
  and lossy for fine-grained structure). UMAP is now used for plotting only.
- No hand-crafted adjustments: marker F normalized as F/(1+F) (monotonic,
  parameter-free) instead of tanh.
- KMeans averaged over 3 seeds (42/43/44) per n_clusters for stability.

Note: All metrics are computed independently and are fully unsupervised
(no ground-truth cell type labels used). Labels are only used for plot coloring.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sc.settings.set_figure_params(dpi=150, facecolor='white')

# ------------------------- default file logging -----------------------------
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

def _enable_default_log(log_name: str):
    """Default logging: stdout/stderr are teed to logs/<log_name> (overwrite per run)."""
    import sys
    from pathlib import Path
    Path("logs").mkdir(exist_ok=True)
    fh = open(Path("logs") / log_name, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print(f"📜 Logging to logs/{log_name}")

# ----------------------------------------------------------------------------
# v2: marker panel derived from the 13 annotated ifnb cell types.
# Principle: 2-4 canonical literature markers per type, chosen a priori.
# NO dropping of rare-type markers - rarity is the fine-grained test.
# (NCAM1/CD56 is absent from this dataset's var; NK uses KLRF1 instead.)
# ----------------------------------------------------------------------------
CELL_TYPE_MARKERS = {
    'CD14 Mono':    ['CD14', 'LYZ', 'S100A8', 'VCAN'],
    'CD16 Mono':    ['FCGR3A', 'MS4A7', 'LST1'],
    'CD4 Naive T':  ['CD3D', 'CD3E', 'CCR7', 'LTB'],
    'CD4 Memory T': ['IL7R', 'S100A4', 'LGALS1'],
    'CD8 T':        ['CD8A', 'CD8B', 'GZMK'],
    'T activated':  ['CD69', 'IL2RA', 'GZMB'],
    'NK':           ['NKG7', 'GNLY', 'KLRD1', 'KLRF1'],
    'B':            ['CD19', 'MS4A1', 'CD79A', 'CD79B'],
    'B Activated':  ['CD83', 'MIR155HG'],
    'DC':           ['CD1C', 'FCER1A', 'CLEC9A'],
    'pDC':          ['LILRA4', 'CLEC4C', 'IL3RA'],
    'Mk':           ['PPBP', 'PF4', 'GP9', 'ITGA2B'],
    'Eryth':        ['HBA1', 'HBA2', 'HBB'],
}
MARKER_GENES = [g for genes in CELL_TYPE_MARKERS.values() for g in genes]

# v2.1: perturbation (ISG) panel - canonical type-I IFN response genes.
# Chosen a priori from IFN biology, same no-hand-filtering principle as the
# cell-type panel. Verified 2026-08-11: 19/25 present in ifnb var, all expressed.
PERTURB_MARKER_GENES = [
    'ISG15', 'IFI6', 'IFIT1', 'IFIT2', 'IFIT3', 'IFIT5', 'MX1', 'MX2',
    'OAS1', 'OAS2', 'OAS3', 'OASL', 'IFITM1', 'IFITM2', 'IFITM3',
    'STAT1', 'IRF7', 'XAF1', 'LY6E', 'RSAD2', 'HERC5', 'ISG20',
    'IFI27', 'IFI27L1', 'SAMD9L',
]

N_CLUSTERS_LIST = [8, 10, 12, 15, 20]
KMEANS_SEEDS = [42, 43, 44]  # v2: average over seeds for stability

# Embedding method configurations (baselines)
# Baselines are evaluated by default in the standard pipeline; set to False to skip.
BASE_EMBEDDINGS_ENABLED = True  # Evaluate all baseline embeddings
BASE_EMBEDDINGS = [
    'X_Seurat_umap', 'X_scVI', 'X_bbknn',
    'X_harmony_pca', 'X_scanorama', 'X_scGPT'
] if BASE_EMBEDDINGS_ENABLED else []

# DeepFusion family: evaluate only the new embeddings whose obsm keys carry
# one of the tags in CURRENT_TAGS.
CURRENT_TAGS = ['_std']  # Standard pipeline tag
INCLUDE_PATHWAY_MLP = True  # Evaluate pathway_mlp alongside the other embeddings

# Methods whose native output is already 2D (used directly for metrics AND plots)
NATIVE_2D_METHODS = {'X_Seurat_umap', 'X_bbknn'}

# DeepFusion-family patterns to auto-collect from obsm (all versions/ablations/seeds)
FUSION_PATTERNS = ('X_pred_embedding', 'X_fusion', 'X_pathway_mlp')


def compute_mdf(coords, marker_expr, n_clusters_list=None, seeds=None):
    """
    MDF: Marker Discrimination F-score (v2: normalized as F/(1+F), parameter-free).
    Averaged over n_clusters values x KMeans seeds.
    Uses raw computation logic (between_ss / within_ss, without dividing by
    degrees of freedom) - same as v1 for continuity.
    """
    if n_clusters_list is None:
        n_clusters_list = N_CLUSTERS_LIST
    if seeds is None:
        seeds = KMEANS_SEEDS
    n_markers = marker_expr.shape[1]
    all_f_scores = []

    for n_clusters in n_clusters_list:
        if len(coords) < n_clusters * 10:
            continue
        for seed in seeds:
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            labels = kmeans.fit_predict(coords)

            f_scores = []
            for m in range(n_markers):
                marker_values = marker_expr[:, m]
                grand_mean = marker_values.mean()

                between_ss = 0
                within_ss = 0

                for c in range(n_clusters):
                    mask = (labels == c)
                    n_c = mask.sum()
                    if n_c < 2:
                        continue
                    cluster_mean = marker_values[mask].mean()
                    between_ss += n_c * (cluster_mean - grand_mean) ** 2
                    within_ss += ((marker_values[mask] - cluster_mean) ** 2).sum()

                # Raw computation logic: do not divide by degrees of freedom
                f = between_ss / (within_ss + 1e-10) if within_ss > 0 else 0
                f_scores.append(f)

            all_f_scores.append(np.mean(f_scores))

    raw_score = np.mean(all_f_scores) if all_f_scores else 0.0
    # v2: monotonic parameter-free normalization to [0,1)
    return float(raw_score / (1.0 + raw_score))


def evaluate_embedding(coords, marker_expr, isg_expr, embedding_name, n_clusters_list=None):
    """Evaluate a single embedding across multiple dimensions"""
    print(f"\nEvaluating {embedding_name} ...")

    results = {'embedding': embedding_name}

    print("  Computing MDF_type (cell-type marker F)...")
    results['MDF_type'] = compute_mdf(coords, marker_expr, n_clusters_list)

    print("  Computing MDF_isg (perturbation-response marker F)...")
    results['MDF_isg'] = compute_mdf(coords, isg_expr, n_clusters_list)

    # Marker-anchored axes only; no aggregate score is computed.
    return results


def create_single_umap_plot(adata, coords, method_name, seurat_types, output_path):
    """Create a single UMAP plot"""
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_types = sorted(seurat_types.unique())
    type_to_num = {t: i for i, t in enumerate(unique_types)}
    colors = [type_to_num[t] for t in seurat_types]

    ax.scatter(coords[:, 0], coords[:, 1],
              c=colors, cmap='tab20', s=5, alpha=0.7, edgecolors='none')

    # Add cluster labels
    for label in unique_types:
        mask = seurat_types == label
        if mask.sum() < 10:
            continue
        center_x = np.median(coords[mask, 0])
        center_y = np.median(coords[mask, 1])
        ax.text(center_x, center_y, label,
               fontsize=10, ha='center', va='center',
               color='black', weight='bold')

    ax.set_title(f'{method_name}', fontsize=16, weight='bold', pad=10)
    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_quantitative_comparison_barplot(results_df, output_path):
    """Create quantitative comparison bar plot (sorted by MDF_type).

    v2: baselines + ALL DeepFusion-family embeddings (pred/fusion/pathway_mlp
    of every version & ablation & seed) - the full unsupervised comparison.
    """
    plot_df = results_df.copy().sort_values('MDF_type', ascending=True)

    name_map = {
        'X_Seurat_umap': 'Seurat',
        'X_scVI': 'scVI',
        'X_bbknn': 'BBKNN',
        'X_harmony_pca': 'Harmony',
        'X_scanorama': 'Scanorama',
        'X_scGPT': 'scGPT',
    }
    plot_df['display_name'] = plot_df['embedding'].map(
        lambda x: name_map.get(x, x.replace('X_pred_embedding_', 'DF_pred\n')
                                   .replace('X_fusion_', 'DF_fus\n')
                                   .replace('X_pathway_mlp_', 'pathMLP\n'))
    )

    n = len(plot_df)
    fig, ax = plt.subplots(figsize=(max(12, n * 0.75), 6))

    x = np.arange(n)
    width = 0.27

    bars1 = ax.bar(x - width / 2, plot_df['MDF_type'], width, label='MDF_type (cell-type)', color='coral', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, plot_df['MDF_isg'], width, label='MDF_isg (perturbation response)', color='seagreen', edgecolor='black', linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Unsupervised Embedding Evaluation (marker-anchored: MDF_type + MDF_isg)', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['display_name'], fontsize=8, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, min(1.2, plot_df[['MDF_type', 'MDF_isg']].values.max() * 1.25))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Quantitative comparison plot saved to: {output_path}")


def main():
    _enable_default_log("step7_unsupervised.log")  # default logging: tee console output to file
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive unsupervised embedding evaluation (v2)')
    parser.add_argument('--input', type=str,
                       default='enhanced_results/results/step5_add_DeepFusion.h5ad',
                       help='Input h5ad file path (default: DeepFusion output with all embeddings)')
    parser.add_argument('--marker-genes', type=str, nargs='+',
                       default=MARKER_GENES,
                       help='List of marker gene symbols for MDF computation')
    parser.add_argument('--perturb-genes', type=str, nargs='+',
                       default=PERTURB_MARKER_GENES,
                       help='List of perturbation/ISG marker gene symbols for MDF_isg computation')
    parser.add_argument('--n-clusters', type=int, nargs='+',
                       default=N_CLUSTERS_LIST,
                       help='List of n_clusters for KMeans evaluation')
    parser.add_argument('--output-dir', type=str, default='./unsupervised_evaluation_v2.1',
                       help='Output directory')
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Read data
    print("=" * 70)
    print("Comprehensive Unsupervised Evaluation (v2: native-space, unbiased markers)")
    print("=" * 70)
    adata = sc.read_h5ad(args.input)
    print(f"Cells: {adata.n_obs}")
    print(f"Genes: {adata.n_vars}")

    # v2: marker expression from the normalized layer (v1 used raw counts in X)
    if 'normalized' in adata.layers:
        expr_source = adata.layers['normalized']
        print("Marker expression source: layers['normalized']")
    else:
        expr_source = adata.X
        print("⚠️ layers['normalized'] not found, falling back to X (check normalization!)")

    marker_genes = args.marker_genes if args.marker_genes else MARKER_GENES
    perturb_genes = args.perturb_genes if args.perturb_genes else PERTURB_MARKER_GENES

    def build_panel(genes, panel_name):
        available = [g for g in genes if g in adata.var_names]
        missing = [g for g in genes if g not in adata.var_names]
        print(f"{panel_name}: {len(available)} / {len(genes)} genes available")
        if missing:
            print(f"  ⚠️ missing (not in var, skipped): {missing}")
        mat = np.zeros((adata.n_obs, len(available)))
        for i, g in enumerate(available):
            expr = expr_source[:, adata.var_names.get_loc(g)]
            if hasattr(expr, 'toarray'):
                expr = expr.toarray().flatten()
            else:
                expr = np.asarray(expr).flatten()
            mat[:, i] = expr
        return mat

    marker_expr = build_panel(marker_genes, "Cell-type panel")
    isg_expr = build_panel(perturb_genes, "ISG panel")

    # baselines + DeepFusion embeddings matching CURRENT_TAGS
    base_embeddings = BASE_EMBEDDINGS
    fusion_pred_embeddings = [k for k in adata.obsm.keys()
                              if any(p in k for p in FUSION_PATTERNS)
                              and any(tag in k for tag in CURRENT_TAGS)
                              and (INCLUDE_PATHWAY_MLP or 'X_pathway_mlp' not in k)]
    embeddings_to_evaluate = list(dict.fromkeys(base_embeddings + fusion_pred_embeddings))
    embeddings_to_evaluate = [e for e in embeddings_to_evaluate if e in adata.obsm.keys()]

    print(f"\nMethods to evaluate: {len(embeddings_to_evaluate)}")
    for e in embeddings_to_evaluate:
        print(f"  - {e} ({adata.obsm[e].shape[1]}D native)")
    print(f"Evaluation metrics: MDF_type, MDF_isg, CS (native space, seeds {KMEANS_SEEDS})")

    # Evaluate
    print("\n" + "=" * 70)
    print("Starting evaluation...")
    print("=" * 70)

    results_list = []
    n_clusters_list = args.n_clusters if args.n_clusters else N_CLUSTERS_LIST

    for method_name in embeddings_to_evaluate:
        embedding = np.asarray(adata.obsm[method_name])
        # v2: metrics in NATIVE embedding space (no UMAP compression, no dim truncation)
        result = evaluate_embedding(embedding, marker_expr, isg_expr, method_name, n_clusters_list)
        results_list.append(result)

    # Create score table
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('MDF_type', ascending=False)

    print("\n" + "=" * 70)
    print("Evaluation Result Score Table")
    print("=" * 70)
    print(results_df.to_string(index=False))

    results_df.to_csv(f'{args.output_dir}/unsupervised_scores.csv', index=False)
    print(f"\nScore table saved to: {args.output_dir}/unsupervised_scores.csv")

    # Show best method
    best_method = results_df.iloc[0]['embedding']
    print(f"\nBest method: {best_method}")
    for metric in ['MDF_type', 'MDF_isg']:
        print(f"  {metric}: {results_df.iloc[0][metric]:.4f}")

    # Generate visualizations (UMAP for plotting only)
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)

    for method_name in embeddings_to_evaluate:
        embedding = np.asarray(adata.obsm[method_name])
        if method_name in NATIVE_2D_METHODS or embedding.shape[1] == 2:
            coords = embedding
        else:
            print(f"  Computing plot UMAP for {method_name}...")
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                               random_state=42, n_jobs=1)
            coords = reducer.fit_transform(embedding)

        display_name = method_name.replace('X_', '').replace(' ', '_')
        safe_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in display_name)

        output_path = f'{args.output_dir}/umap_{safe_name}.png'
        create_single_umap_plot(
            adata, coords, method_name.replace('X_', '').replace('_', ' '),
            adata.obs['seurat_annotations'], output_path
        )
        print(f"    {method_name} -> umap_{safe_name}.png")

    create_quantitative_comparison_barplot(
        results_df,
        f'{args.output_dir}/quantitative_comparison.png'
    )

    print("\n" + "=" * 70)
    print("Evaluation completed!")
    print("=" * 70)


if __name__ == '__main__':
    from perf_utils import PerfRecorder
    PerfRecorder.start_now("step7_unsupervised")
    main()
