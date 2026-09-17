import scanpy as sc
import numpy as np
import anndata
import sys
import scipy
import pandas as pd
import warnings
from scipy.sparse import issparse, lil_matrix, csr_matrix
import os
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

# -------------------------- Global configuration (Nature Methods standard) --------------------------
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['xtick.major.width'] = 0.8
mpl.rcParams['ytick.major.width'] = 0.8
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['grid.alpha'] = 0.1

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sc.settings.verbosity = 0
sc.settings.logfile = sys.stdout

def standard_preprocess(adata):
    """Standard preprocessing to complete the raw layer: normalization + log1p + HVG filtering"""
    adata.X = adata.layers["raw"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3", subset=True)
    sc.pp.scale(adata, zero_center=True, max_value=10)
    return adata

# Refactor run_harmony: preserve original logic (single-batch check)
def run_harmony(adata, batch_key, n_pcs):
    original_cell_count = adata.n_obs
    original_batches = adata.obs[batch_key].unique()
    original_batch_count = len(original_batches)
    
    if original_batch_count == 1:
        print(f"⚠️  Only 1 batch detected ({original_batches[0]}), skipping Harmony (no batch effect to correct)")
        adata_temp = adata.copy()
        if 'X_pca' not in adata_temp.obsm or adata_temp.obsm['X_pca'].shape[1] < n_pcs:
            print(f"Calculating PCA with {n_pcs} components (single batch fallback)")
            sc.pp.pca(adata_temp, n_comps=n_pcs, svd_solver='arpack')
        harmony_pca_full = adata_temp.obsm['X_pca'].copy()
        if harmony_pca_full.shape[1] < n_pcs:
            pad = np.zeros((original_cell_count, n_pcs - harmony_pca_full.shape[1]))
            harmony_pca_full = np.hstack([harmony_pca_full, pad])
        
        report = f"""
        Harmony Integration Report (Single Batch Fallback, n_pcs={n_pcs}):
          Original cells: {original_cell_count} | Batches: {original_batch_count}
          Used PCA directly (no Harmony correction needed)
        """
        print(report)
        return harmony_pca_full
    
    min_batch_size = 10
    batch_counts = adata.obs[batch_key].value_counts()
    small_batches = batch_counts[batch_counts < min_batch_size].index.tolist()
    actual_batch_key = batch_key
    
    if small_batches:
        print(f"Merging {len(small_batches)} small batches into 'merged_small_batches' group")
        adata.obs['temp_merged_batch'] = adata.obs[batch_key].astype(str)
        adata.obs.loc[adata.obs['temp_merged_batch'].isin(small_batches), 'temp_merged_batch'] = "merged_small_batches"
        actual_batch_key = 'temp_merged_batch'
    
    valid_batches = [b for b in adata.obs[actual_batch_key].unique() 
                     if adata.obs[actual_batch_key].value_counts()[b] >= 2]
    adata_temp = adata[adata.obs[actual_batch_key].isin(valid_batches)].copy()
    
    if adata_temp.n_obs == 0:
        raise ValueError("All batches filtered out - no cells available for Harmony")
    
    valid_cell_mask = adata.obs_names.isin(adata_temp.obs_names)
    valid_cell_count = np.sum(valid_cell_mask)
    
    n_batches = len(valid_batches)
    total_cells = adata_temp.n_obs
    min_batch_size = adata_temp.obs[actual_batch_key].value_counts().min()
    
    n_pcs = max(1, n_pcs)
    print(f"Using specified n_pcs: {n_pcs}")
    
    if min_batch_size <= 5 or n_batches <= 5:
        theta = 0.8
    elif min_batch_size <= 10 or n_batches <= 10:
        theta = 0.5
    else:
        theta = 0.2
    
    print(f"Using parameters: n_pcs={n_pcs}, theta={theta}, batches={n_batches}")
    
    if 'X_pca' not in adata_temp.obsm or adata_temp.obsm['X_pca'].shape[1] < n_pcs:
        print(f"Calculating PCA with {n_pcs} components")
        sc.pp.pca(adata_temp, n_comps=n_pcs, svd_solver='arpack')
    
    try:
        import harmonypy
        
        batch_labels = adata_temp.obs[actual_batch_key].astype(str).tolist()
        
        ho = harmonypy.run_harmony(
            adata_temp.obsm['X_pca'],
            pd.DataFrame({batch_key: batch_labels}),
            batch_key,
            theta=theta,
            nclust=min(10, n_batches),
            max_iter_harmony=100,
            verbose=False
        )
        
        harmony_pca = ho.Z_corr.T
        print(f"Generated Harmony PCA embedding: {harmony_pca.shape}")
        
    except Exception as e:
        raise RuntimeError(f"Harmony failed: {str(e)}")
    
    pca_dims = harmony_pca.shape[1]
    harmony_pca_full = np.zeros((original_cell_count, pca_dims))
    
    valid_idx = np.where(valid_cell_mask)[0]
    temp_idx_map = {name: idx for idx, name in enumerate(adata_temp.obs_names)}
    temp_idx = np.array([temp_idx_map[name] for name in adata.obs_names[valid_idx]])
    
    harmony_pca_full[valid_idx] = harmony_pca[temp_idx]
    
    print(f"Mapped Harmony PCA embedding size: {harmony_pca_full.shape}")
    
    report = f"""
    Harmony Integration Report (n_pcs={n_pcs}):
      Original cells: {original_cell_count} | Batches: {original_batch_count}
      Valid cells: {valid_cell_count} | Valid batches: {n_batches}
      Min batch size: {min_batch_size}
      Used parameters: theta={theta}, n_pcs={n_pcs}
    """
    print(report)

    return harmony_pca_full

# -------------------------- Core optimization: Nature Methods-level plotting function --------------------------
def plot_pca_variance_curve(adata_processed, save_path, max_pcs=30):
    """
    Plot PCA variance explained curve conforming to Nature Methods standards (PCs up to 30, legend at center-right)
    :param adata_processed: Preprocessed adata
    :param save_path: Save path (auto-generates PDF+PNG)
    :param max_pcs: Maximum number of PCs to compute (fixed at 30)
    """
    # Compute PCA (to obtain variance explained)
    adata_pca_temp = adata_processed.copy()
    n_comps = min(max_pcs, adata_pca_temp.n_vars - 1, adata_pca_temp.n_obs - 1)
    sc.pp.pca(adata_pca_temp, n_comps=n_comps, svd_solver='arpack')
    
    # Extract variance explained
    variance_ratio = adata_pca_temp.uns['pca']['variance_ratio']
    cumulative_variance = np.cumsum(variance_ratio)
    pc_numbers = np.arange(1, len(variance_ratio)+1)
    
    # Define Nature Methods color scheme
    color_individual = '#2874A6'  # Dark blue (individual variance)
    color_cumulative = '#CB4335'  # Dark red (cumulative variance)
    color_80 = '#2ECC71'          # Light green (80% threshold)
    color_90 = '#F39C12'          # Orange (90% threshold)
    
    # Create figure (single-column width 8cm, height 5cm)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=600)
    
    # Plot curves
    ax.plot(pc_numbers, variance_ratio, 
            color=color_individual, linewidth=1.2, marker='o', markersize=2.5, 
            markerfacecolor='none', markeredgecolor=color_individual, markeredgewidth=0.8,
            label='Individual variance')
    ax.plot(pc_numbers, cumulative_variance, 
            color=color_cumulative, linewidth=1.2, marker='s', markersize=2.5,
            markerfacecolor='none', markeredgecolor=color_cumulative, markeredgewidth=0.8,
            label='Cumulative variance')
    
    # Plot 80%/90% threshold lines
    ax.axhline(y=0.8, color=color_80, linestyle='--', linewidth=0.8, alpha=0.8, label='80% variance')
    ax.axhline(y=0.9, color=color_90, linestyle='--', linewidth=0.8, alpha=0.8, label='90% variance')
    
    # Axis settings (PCs up to 30)
    ax.set_xlabel('Number of Principal Components', fontsize=10, labelpad=5)
    ax.set_ylabel('Explained Variance Ratio', fontsize=10, labelpad=5)
    ax.set_xlim(0, max_pcs + 1)
    ax.set_ylim(0, 1.02)         
    ax.set_xticks(np.arange(0, max_pcs + 1, 5))
    ax.set_yticks(np.arange(0, 1.01, 0.1))       
    
    # Grid lines (very faint)
    ax.grid(True, axis='y', linestyle='-', alpha=0.1)
    
    # Core modification: legend adjusted to center-right (no overlap)
    ax.legend(loc='center right', bbox_to_anchor=(0.98, 0.5), ncol=1, handlelength=1.5)
    
    # Title
    ax.set_title('PCA Variance Explained', fontsize=12, pad=10)
    
    # Adjust layout
    plt.tight_layout(pad=0.5)
    
    # Save (PDF+PNG, 600dpi)
    pdf_path = save_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight', format='pdf')
    fig.savefig(save_path, dpi=600, bbox_inches='tight', format='png')
    plt.close(fig)
    
    # Output key variance values
    print(f"PCA variance curve saved to:\n  - PDF: {pdf_path}\n  - PNG: {save_path}")
    print(f"\nKey variance values (max PC={max_pcs}):")
    if len(cumulative_variance) >= 10:
        print(f"  Top 10 PCs cumulative variance: {cumulative_variance[9]:.3f}")
    if len(cumulative_variance) >= 20:
        print(f"  Top 20 PCs cumulative variance: {cumulative_variance[19]:.3f}")
    if len(cumulative_variance) >= 30:
        print(f"  Top 30 PCs cumulative variance: {cumulative_variance[29]:.3f}")
    else:
        print(f"  Top {len(cumulative_variance)} PCs cumulative variance: {cumulative_variance[-1]:.3f}")

# ------------------------- default file logging -----------------------------
class _Tee:
    """Write to console and log file simultaneously."""
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
    from pathlib import Path
    Path("logs").mkdir(exist_ok=True)
    fh = open(Path("logs") / log_name, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print(f"📜 Logging to logs/{log_name}")


if __name__ == "__main__":
    from perf_utils import PerfRecorder
    PerfRecorder.start_now("step5-1_special_harmony")
    _enable_default_log("step51_harmony.log")  # default logging: tee console output to file
    # Configuration parameters
    input_path = "enhanced_results/results/step4-4_filter_HVG3000.h5ad"
    output_path = "enhanced_results/results/step4-5_add_special_harmony.h5ad"
    step1_processed = False
    adataX_raw_count = True
    batch_key = "batch"

    target_pcs_list = np.arange(20, 42, 10).tolist()
    print(f"Will process the following PCs: {target_pcs_list}")

    # Load data
    print(f"Loading data from {input_path}...")
    adata = sc.read_h5ad(input_path)
    print(f"Original data shape: {adata.shape}")
    print(f"Available layers: {list(adata.layers.keys())}")
    
    # Validate batch column
    if batch_key not in adata.obs:
        raise KeyError(f"Batch key '{batch_key}' not found in adata.obs")
    
    batch_counts = adata.obs[batch_key].value_counts()
    print("\nBatch size statistics:")
    print(batch_counts.describe())
    print(f"\nSmallest batch: {batch_counts.min()} cells")
    print(f"Number of batches: {len(batch_counts)}")
    small_batches = batch_counts[batch_counts < 2].index.tolist()
    print(f"Batches with <2 cells: {len(small_batches)}")
    
    # Preprocessing (raw layer + HVGs)
    adata_processed = adata.copy()
    if step1_processed:
        print("\nUsing raw layer + normalization + log1p + HVG filter")
        adata_processed.X = adata_processed.layers["raw"].copy()
        sc.pp.normalize_total(adata_processed, target_sum=1e4)
        sc.pp.log1p(adata_processed)
        sc.pp.highly_variable_genes(adata_processed, n_top_genes=3000, flavor="seurat_v3", subset=True)
        sc.pp.scale(adata_processed, zero_center=True, max_value=10)
    else:
        if adataX_raw_count:
            print("\nPerforming standard preprocess on RAW counts (normalize+log1p+HVGs)")
            adata_processed = standard_preprocess(adata_processed)

    # Plot variance curve (PCs up to 30)
    pca_curve_path = "enhanced_results/results/pca_variance_curve.png"
    plot_pca_variance_curve(adata_processed, pca_curve_path, max_pcs=30)

    # Run Harmony
    try:
        print("\nStarting Harmony integration for multiple PCs...")
        for target_pcs in target_pcs_list:
            print(f"\n=== Processing Harmony with n_pcs={target_pcs} ===")
            harmony_pca = run_harmony(adata_processed, batch_key, target_pcs)
            
            obsm_key = f'X_special_harmony_{target_pcs}'
            adata.obsm[obsm_key] = harmony_pca
            print(f"Added corrected PCA to adata.obsm['{obsm_key}']: {harmony_pca.shape}")
        
        print("Harmony integration for all PCs completed successfully")
    except Exception as e:
        print(f"Harmony integration failed: {str(e)}")
        raise
    
    # Save results
    adata.write(output_path)
    print(f"Results saved to {output_path}")
    
    # Output embedding information
    print("\nEmbedding keys in adata.obsm:")
    for key in adata.obsm.keys():
        if key.startswith('X_special_harmony'):
            shape = adata.obsm[key].shape
            print(f"{key}: {shape[0]} cells × {shape[1]} dimensions")
    
    print("\nHarmony processing complete")