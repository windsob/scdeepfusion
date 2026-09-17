import scanpy as sc
import numpy as np
import anndata
import sys
import scipy
import pandas as pd
from scipy.sparse import issparse, csr_matrix
import os
import torch
import scvi
from scvi.model import SCVI
import math
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if torch.cuda.is_available() else ""

sc.settings.verbosity = 0
sc.settings.logfile = sys.stdout

def filter_small_batches(adata, batch_key, min_cells=2):
    batch_counts = adata.obs[batch_key].value_counts()
    small_batches = batch_counts[batch_counts < min_cells].index.tolist()
    if small_batches:
        print(f"Filtering out small batches (<{min_cells} cells): {small_batches}")
        mask = ~adata.obs[batch_key].isin(small_batches)
        return adata[mask].copy()
    return adata

def run_scvi(adata, batch_key, use_raw_layer=True):
    original_cell_count = adata.n_obs
    original_batches = adata.obs[batch_key].unique()
    original_batch_count = len(original_batches)
    
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
        raise ValueError("All batches filtered out - no cells available for scVI")
    
    valid_cell_mask = adata.obs_names.isin(adata_temp.obs_names)
    valid_cell_count = np.sum(valid_cell_mask)
    n_batches = len(valid_batches)
    min_batch_size_post = adata_temp.obs[actual_batch_key].value_counts().min()
    total_cells = adata_temp.n_obs
    
    def _calculate_scvi_dimension(n_cells):
        min_dim = 30
        if n_cells <= 1000:
            return min_dim
        dim_value = 4 * math.log10(n_cells + 1000)
        return min(50, max(min_dim, int(round(dim_value))))
    
    scvi_dim = _calculate_scvi_dimension(total_cells)
    print(f"Using scVI latent dimension: {scvi_dim}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if use_raw_layer:
        if 'raw' in adata_temp.layers:
            print("Using raw counts from adata.layers['raw']")
            count_data = adata_temp.layers['raw']
        else:
            warnings.warn("'raw' layer not found, falling back to adata.X")
            count_data = adata_temp.X
    else:
        print("Using adata.X as count data")
        count_data = adata_temp.X
    
    adata_scvi = anndata.AnnData(
        X=count_data,
        obs=adata_temp.obs[[actual_batch_key]].copy(),
        var=adata_temp.var.copy()
    )
    
    if not issparse(adata_scvi.X):
        adata_scvi.X = csr_matrix(adata_scvi.X)
    
    SCVI.setup_anndata(adata_scvi, batch_key=actual_batch_key)
    
    n_latent = scvi_dim
    n_hidden = min(1024, max(128, int(total_cells ** 0.5)))
    n_layers = min(3, max(2, int(math.log(total_cells, 1000)) + 1))
    dropout_rate = 0.1 if total_cells < 10000 else 0.2
    gene_likelihood = "zinb"
    
    print(f"scVI parameters: n_latent={n_latent}, n_hidden={n_hidden}, n_layers={n_layers}, "
          f"dropout_rate={dropout_rate}, gene_likelihood={gene_likelihood}")
    
    model = SCVI(
        adata_scvi,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        gene_likelihood=gene_likelihood,
        use_layer_norm="both",
        use_batch_norm="none"
    )
    
    vae = model
    
    max_epochs = min(500, max(200, total_cells // 100))
    batch_size = min(256, max(32, total_cells // 10))
    
    lr_value = 0.001 if total_cells < 50000 else 0.0005
    vae.module.lr = lr_value
    
    print(f"Training scVI: max_epochs={max_epochs}, batch_size={batch_size}, lr={lr_value}")
    
    trainer_kwargs = {
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "train_size": 0.9,
        "validation_size": 0.1,
        "early_stopping": True,
        "early_stopping_monitor": "elbo_validation",
        "early_stopping_patience": 15,
        "check_val_every_n_epoch": 1
    }
    
    if hasattr(vae, '_trainer_class'):
        trainer_kwargs["accelerator"] = "auto" if torch.cuda.is_available() else "cpu"
    
    vae.train(**trainer_kwargs)
    
    latent = vae.get_latent_representation()
    adata_temp.obsm['X_scVI'] = latent
    print(f"scVI latent dimensions: {latent.shape}")
    
    print("Calculating UMAP embedding for scVI...")
    sc.pp.neighbors(adata_temp, use_rep='X_scVI')
    sc.tl.umap(adata_temp)
    
    latent_dims = latent.shape[1]
    umap_dims = adata_temp.obsm['X_umap'].shape[1]
    
    latent_full = np.zeros((original_cell_count, latent_dims))
    umap_full = np.zeros((original_cell_count, umap_dims))
    
    valid_idx = np.where(valid_cell_mask)[0]
    temp_idx_map = {name: idx for idx, name in enumerate(adata_temp.obs_names)}
    temp_idx = np.array([temp_idx_map[name] for name in adata.obs_names[valid_idx]])
    
    latent_full[valid_idx] = latent[temp_idx]
    umap_full[valid_idx] = adata_temp.obsm['X_umap'][temp_idx]
    
    print(f"Mapped scVI latent embedding size: {latent_full.shape}")
    print(f"Mapped scVI UMAP embedding size: {umap_full.shape}")
    
    report = f"""
    scVI Integration Report:
      Original cells: {original_cell_count} | Batches: {original_batch_count}
      Valid cells: {valid_cell_count} | Valid batches: {n_batches}
      Min batch size: {min_batch_size_post}
      Model parameters: n_latent={n_latent}, n_hidden={n_hidden}, n_layers={n_layers}
      Training: epochs={max_epochs}, batch_size={batch_size}, lr={lr_value}
    """
    print(report)
    
    return latent_full, umap_full

if __name__ == "__main__":
    from perf_utils import PerfRecorder
    PerfRecorder.start_now("step3_scVI")
    # Configuration!!!
    input_path = "enhanced_results/results/step2_add_muti_method.h5ad"
    output_path = "enhanced_results/results/step3_add_scVI.h5ad"
    batch_key = "batch"
    use_raw_layer = True  # If True use layers['raw'], False adata.X
    
    print(f"Loading data from {input_path}...")
    adata = sc.read_h5ad(input_path)
    print(f"Original data shape: {adata.shape}")
    
    if batch_key not in adata.obs:
        raise KeyError(f"Batch key '{batch_key}' not found in adata.obs")
    
    batch_counts = adata.obs[batch_key].value_counts()
    print("\nBatch size statistics:")
    print(batch_counts.describe())
    print(f"\nSmallest batch: {batch_counts.min()} cells")
    print(f"Number of batches: {len(batch_counts)}")
    small_batches = batch_counts[batch_counts < 2].index.tolist()
    print(f"Batches with <2 cells: {len(small_batches)}")
    
    if use_raw_layer:
        if 'raw' in adata.layers:
            print("\nWill use raw counts from adata.layers['raw']")
        else:
            warnings.warn("'raw' layer not found, will fall back to adata.X")
    else:
        print("\nWill use adata.X as count data")
    
    try:
        print("\nStarting scVI integration...")
        print(f"Using {adata.shape[0]} cells for integration")
        
        latent, umap_embedding = run_scvi(adata, batch_key, use_raw_layer=use_raw_layer)
        adata.obsm['X_scVI'] = latent
        adata.obsm['X_scVI_umap'] = umap_embedding
        
        print("Added latent embedding to adata.obsm['X_scVI']")
        print("Added UMAP embedding to adata.obsm['X_scVI_umap']")
        print("scVI integration completed successfully")
    except Exception as e:
        print(f"scVI failed: {str(e)}")
        raise
    
    adata.write(output_path)
    print(f"Results saved to {output_path}")
    
    print("\nEmbedding keys added to adata.obsm:")
    for key in adata.obsm.keys():
        if key.startswith('X_'):
            shape = adata.obsm[key].shape
            print(f"{key}: {shape[0]} cells × {shape[1]} dimensions")
    
    print("\nProcessing complete")