import scanpy as sc
import scib
import numpy as np
import pandas as pd
import os
import umap
import hnswlib
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from perf_utils import PerfRecorder
PerfRecorder.start_now("step6_scib")
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

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

_enable_default_log("step6_scib.log")

# Load data
adata = sc.read_h5ad("enhanced_results/results/step5_add_DeepFusion.h5ad")
batch_key = 'batch'
label_key = 'seurat_annotations'

# -------------------------- Core modifications --------------------------
# ----------------------- Evaluation scope configuration -----------------------
# Baseline methods (scVI/Harmony/scGPT/Scanorama/BBKNN/Seurat) are evaluated
# by default in the standard pipeline. Set to False to skip them.
BASE_EMBEDDINGS_ENABLED = True  # Evaluate all baseline embeddings
base_embeddings = [
    'X_scVI',
    'X_harmony_pca',
    'X_scGPT',
    'X_scanorama',
    'X_bbknn',
    'X_Seurat_umap'
] if BASE_EMBEDDINGS_ENABLED else []

# DeepFusion family: evaluate only the new embeddings (X_fusion*/X_pred_embedding*)
# whose obsm keys carry one of the tags in CURRENT_TAGS.
# pathway_mlp embeddings are included and filtered by the same tags.
CURRENT_TAGS = ['_std']  # Standard pipeline tags: *_std / *_std_s43 / *_std_s44 / pathmlp_std*
INCLUDE_PATHWAY_MLP = True  # Evaluate pathway_mlp alongside the other embeddings

# Incremental evaluation: embeddings already in the CSV are kept; entries in
# FORCE_REEVALUATE are re-evaluated (e.g. after a baseline is re-trained).
FORCE_REEVALUATE = []

fusion_pred_embeddings = [
    key for key in adata.obsm.keys()
    if ('X_fusion' in key or 'X_pred_embedding' in key
        or (INCLUDE_PATHWAY_MLP and 'X_pathway_mlp' in key))
    and any(tag in key for tag in CURRENT_TAGS)
]

# 3. Merge and deduplicate fields (keep base fields first), filtering out non-existent ones in adata.obsm
embeddings_to_evaluate = list(dict.fromkeys(base_embeddings + fusion_pred_embeddings))  # Deduplicate
embeddings_to_evaluate = [embed for embed in embeddings_to_evaluate if embed in adata.obsm.keys()]  # Filter out non-existent fields

# Print detected embeddings for debugging
print(f"Detected embeddings to evaluate: {embeddings_to_evaluate}")
# -------------------------------------------------------------------

# Data preprocessing: handle missing label values
adata.obs[label_key] = adata.obs[label_key].astype('category')
adata.obs[batch_key] = adata.obs[batch_key].astype('category')
original_cell_count = adata.n_obs
print(f"Original cluster values: {adata.obs[label_key].unique()}")

missing_strings = ["na", "NA", "N/A", "n/a", "unknown", "missing", "unassigned", "", " "]
for na_str in missing_strings:
    adata.obs[label_key] = adata.obs[label_key].replace(na_str, np.nan)

adata.obs[label_key] = adata.obs[label_key].replace(r'^\s*$', np.nan, regex=True)
adata = adata[~adata.obs[label_key].isna()].copy()

print(f"Original cells: {original_cell_count}")
print(f"After removing missing labels: {adata.n_obs}")

if adata.n_obs == original_cell_count:
    print("Warning: No missing labels found - check if cluster contains truly missing values")

def run_umap(embedding, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        metric='euclidean'
    )
    return reducer.fit_transform(embedding)

def build_knn_graph(embedding, n_neighbors=15):
    try:
        if hasattr(embedding, 'toarray'):
            embedding = embedding.toarray()
        elif hasattr(embedding, 'values'):
            embedding = embedding.values
            
        p = hnswlib.Index(space='l2', dim=embedding.shape[1])
        p.init_index(max_elements=embedding.shape[0], ef_construction=200, M=16)
        p.add_items(embedding)
        knn_indices, knn_distances = p.knn_query(embedding, k=n_neighbors)
        
        n_cells = embedding.shape[0]
        indices = knn_indices.flatten()
        indptr = np.arange(0, n_cells * n_neighbors + 1, n_neighbors)
        distances = knn_distances.flatten()
        
        return csr_matrix((distances, indices, indptr), shape=(n_cells, n_cells))
    except Exception as e:
        print(f"KNN graph failed: {str(e)}")
        return None

def evaluate_integration(adata_tmp, embed_name):
    """Standard scIB evaluation protocol:
    - ASW / isolated F1 are computed in the NATIVE embedding space.
      UMAP is for visualization only.
    - graph_connectivity uses the standard scanpy neighbor graph.
    Note: values are NOT comparable to results from a legacy UMAP-based protocol.
    """
    try:
        embedding = adata_tmp.obsm[embed_name]
        adata_tmp.obsm['X_umap'] = run_umap(embedding)   # visualization only
        adata_tmp.obsm['X_eval'] = embedding             # metrics on native space

        # standard neighbor graph (also feeds clustering + graph_connectivity)
        sc.pp.neighbors(adata_tmp, use_rep=embed_name)

        metrics = {}
        # Official scIB 1.1.5 protocol: leiden clustering at NMI-optimal resolution.
        # Note: cluster_optimal_resolution writes the optimal clustering into obs and
        # returns None in scib 1.1.5; nmi/ari are computed on that clustering after.
        scib.me.cluster_optimal_resolution(
            adata_tmp, label_key=label_key, cluster_key='predicted_cluster',
            metric=scib.me.nmi, use_rep=embed_name, verbose=False)
        metrics['nmi'] = scib.me.nmi(adata_tmp, cluster_key='predicted_cluster', label_key=label_key)
        metrics['ari'] = scib.me.ari(adata_tmp, cluster_key='predicted_cluster', label_key=label_key)
        metrics['asw_label'] = scib.me.silhouette(adata_tmp, label_key=label_key, embed='X_eval')
        metrics['asw_batch'] = scib.me.silhouette_batch(adata_tmp, batch_key=batch_key, label_key=label_key, embed='X_eval')
        metrics['graph_conn'] = scib.me.graph_connectivity(adata_tmp, label_key=label_key)
        metrics['isolated_f1'] = scib.me.isolated_labels_f1(adata_tmp, label_key=label_key, batch_key=batch_key, embed='X_eval')
        return metrics
    except Exception as e:
        print(f"Evaluation failed for {embed_name}: {str(e)}")
        return {k: np.nan for k in ['nmi', 'ari', 'asw_label', 'asw_batch', 'graph_conn', 'isolated_f1']}

def plot_umap_for_embedding(adata, embedding_name, coords):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    batch_codes = adata.obs[batch_key].astype('category').cat.codes
    axs[0].scatter(
        coords[:, 0], 
        coords[:, 1],
        c=batch_codes,
        cmap='tab20',
        s=5,
        alpha=0.7
    )
    axs[0].set_title(f'Batch ({embedding_name})')
    axs[0].set_xlabel('UMAP1')
    axs[0].set_ylabel('UMAP2')

    label_codes = adata.obs[label_key].astype('category').cat.codes
    axs[1].scatter(
        coords[:, 0], 
        coords[:, 1],
        c=label_codes,
        cmap='tab20',
        s=5,
        alpha=0.7
    )
    axs[1].set_title(f'True Labels ({embedding_name})')
    axs[1].set_xlabel('UMAP1')
    axs[1].set_ylabel('UMAP2')

    plt.tight_layout()
    return fig

umap_coords_dict = {}

# Load previous results for incremental evaluation (rows are preserved)
prev_df = None
if os.path.exists("enhanced_results/results/integration_metrics.csv"):
    prev_df = pd.read_csv("enhanced_results/results/integration_metrics.csv", index_col=0)
    print(f"Recycling {len(prev_df)} existing result rows (incremental evaluation)")

results = {}
for embed in embeddings_to_evaluate:
    if prev_df is not None and embed in prev_df.index and embed not in FORCE_REEVALUATE:
        print(f"\nSkipping {embed} (already in results table)")
        results[embed] = prev_df.loc[embed].to_dict()
        continue
    if embed in FORCE_REEVALUATE and prev_df is not None and embed in prev_df.index:
        print(f"\nForce re-evaluating: {embed}")
    print(f"\nEvaluating {embed}...")
    adata_tmp = adata.copy()
    results[embed] = evaluate_integration(adata_tmp, embed)
    
    embedding = adata_tmp.obsm[embed]
    umap_coords = run_umap(embedding)
    umap_coords_dict[embed] = umap_coords
    
    fig = plot_umap_for_embedding(adata_tmp, embed, umap_coords)
    fig.savefig(f'enhanced_results/results/umap_{embed}.png', dpi=300, bbox_inches='tight')
    plt.close(fig) 

# Organize results and save
results_df = pd.DataFrame(results).T
results_df.index.name = 'Method'
results_df.to_csv("enhanced_results/results/integration_metrics.csv")
print("\nResults:")
print(results_df)

# Find the best method and plot its UMAP
# Filter out cases where asw_label is NaN
valid_results = results_df.dropna(subset=['asw_label'])
if not valid_results.empty:
    best_method = valid_results['asw_label'].idxmax()
    print(f"\nBest method: {best_method}")

    if best_method in umap_coords_dict:
        adata.obsm['X_umap'] = umap_coords_dict[best_method]
        
        fig = plot_umap_for_embedding(adata, best_method, umap_coords_dict[best_method])
        fig.savefig('enhanced_results/results/best_integration_results.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        print(f"Warning: Could not find UMAP coordinates for best method {best_method}")
else:
    print("Warning: No valid ASW label metrics found, cannot determine best method")