import scanpy as sc
import numpy as np
import anndata
import sys
import scipy
import pandas as pd
from bbknn import bbknn
import warnings
from scipy.sparse import issparse, lil_matrix, csr_matrix
import os
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import umap
from sklearn.neighbors import NearestNeighbors
import igraph as ig
import leidenalg
from sklearn.metrics import silhouette_score

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sc.settings.verbosity = 0
sc.settings.logfile = sys.stdout

def standard_preprocess(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata

def filter_small_batches(adata, batch_key, min_cells=2):
    batch_counts = adata.obs[batch_key].value_counts()
    small_batches = batch_counts[batch_counts < min_cells].index.tolist()
    if small_batches:
        print(f"Filtering out small batches (<{min_cells} cells): {small_batches}")
        mask = ~adata.obs[batch_key].isin(small_batches)
        return adata[mask].copy()
    return adata

def run_bbknn(adata, batch_key):
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
    adata_filt = adata[adata.obs[actual_batch_key].isin(valid_batches)].copy()
    if adata_filt.n_obs == 0:
        raise ValueError("All batches filtered out - no cells available for BBKNN")
    
    min_batch_size_post = adata_filt.obs[actual_batch_key].value_counts().min()
    total_cells = adata_filt.n_obs
    
    params = {
        'neighbors_within_batch': max(3, min(20, min_batch_size_post // 2)),
        'trim': max(3, min_batch_size_post // 3),
        # Exact KNN (cKDTree): deterministic and gives the baseline its best fair
        # configuration; at this scale exact search costs only seconds. The
        # approximate annoy path is non-deterministic (unseeded RNG) and degrades
        # local structure, so it is not used.
        'metric': 'manhattan' if total_cells < 2000 else 'euclidean',
        'approx': False,
        'use_faiss': False
    }
    print(f"Smart parameters: {params}")
    
    if 'X_pca' not in adata_filt.obsm:
        n_pcs = min(50, adata_filt.n_vars-1, adata_filt.n_obs-1)
        sc.pp.pca(adata_filt, n_comps=n_pcs, svd_solver='arpack')
        print(f"Calculated PCA with {n_pcs} components")
    else:
        n_pcs = adata_filt.obsm['X_pca'].shape[1]
    
    param_sets = [
        params,
        {**params, 'metric': 'manhattan', 'approx': False},
        {**params, 'neighbors_within_batch': 3, 'trim': 3}
    ]
    
    adata_temp = adata_filt.copy()
    success = False
    
    for i, param_set in enumerate(param_sets):
        try:
            print(f"Trying parameter set {i+1}: neighbors={param_set['neighbors_within_batch']}, trim={param_set['trim']}")
            
            sc.external.pp.bbknn(
                adata_temp,
                batch_key=actual_batch_key,
                neighbors_within_batch=param_set['neighbors_within_batch'],
                trim=param_set['trim'],
                n_pcs=min(50, n_pcs),
                approx=param_set['approx'],
                use_faiss=param_set['use_faiss'],
                copy=False
            )
            
            if 'neighbors' in adata_temp.uns:
                success = True
                print(f"Parameter set {i+1} succeeded")
                break
                
        except Exception as e:
            print(f"Parameter set {i+1} failed: {str(e)}")
    
    if not success:
        raise RuntimeError("All BBKNN parameter sets failed")
    
    print("Calculating UMAP embedding...")
    sc.tl.umap(adata_temp)
    
    umap_dims = adata_temp.obsm['X_umap'].shape[1]
    embedding = np.zeros((original_cell_count, umap_dims))
    
    valid_mask = adata.obs_names.isin(adata_temp.obs_names)
    valid_idx = np.where(valid_mask)[0]
    temp_idx_map = {name: idx for idx, name in enumerate(adata_temp.obs_names)}
    valid_idx_in_temp = np.array([temp_idx_map[name] for name in adata.obs_names[valid_idx]])
    
    embedding[valid_idx] = adata_temp.obsm['X_umap'][valid_idx_in_temp]
    print(f"Stored BBKNN embedding matrix (UMAP-based): {embedding.shape}")
    
    if 'connectivities' in adata_temp.obsp:
        connectivities = lil_matrix((original_cell_count, original_cell_count))
        sub_connect = adata_temp.obsp['connectivities'][valid_idx_in_temp, :][:, valid_idx_in_temp]
        connectivities[valid_idx[:, None], valid_idx] = sub_connect.toarray()
        adata.obsp['bbknn_connectivities'] = connectivities.tocsr()
        print(f"Stored neighbor graph: {adata.obsp['bbknn_connectivities'].shape}")
    else:
        warnings.warn("BBKNN did not generate neighbor graph")
    
    report = f"""
    BBKNN Integration Report:
      Original cells: {original_cell_count} | Batches: {original_batch_count}
      Valid cells: {adata_temp.n_obs} | Valid batches: {adata_temp.obs[actual_batch_key].nunique()}
      Min batch size: {min_batch_size_post}
      Used parameters: neighbors={param_set['neighbors_within_batch']}, trim={param_set['trim']}
    """
    print(report)
    
    return embedding

def run_harmony(adata, batch_key):
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
        raise ValueError("All batches filtered out - no cells available for Harmony")
    
    valid_cell_mask = adata.obs_names.isin(adata_temp.obs_names)
    valid_cell_count = np.sum(valid_cell_mask)
    
    n_batches = len(valid_batches)
    total_cells = adata_temp.n_obs
    min_batch_size = adata_temp.obs[actual_batch_key].value_counts().min()
    
    n_pcs = min(50, min(adata_temp.n_vars - 1, adata_temp.n_obs - 1))
    n_pcs = max(10, n_pcs)
    
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
            nclust=min(20, n_batches * 2),
            max_iter_harmony=50,
            verbose=False
        )
        
        harmony_pca = ho.Z_corr.T
        
        adata_temp.obsm['X_harmony_pca'] = harmony_pca
        print(f"Generated Harmony PCA embedding: {harmony_pca.shape}")
        
    except Exception as e:
        raise RuntimeError(f"Harmony failed: {str(e)}")
    
    print("Calculating UMAP on Harmony-corrected PCA...")
    sc.pp.neighbors(adata_temp, use_rep='X_harmony_pca')
    sc.tl.umap(adata_temp)
    
    harmony_umap = adata_temp.obsm['X_umap']
    print(f"Generated Harmony UMAP embedding: {harmony_umap.shape}")
    
    pca_dims = harmony_pca.shape[1]
    umap_dims = harmony_umap.shape[1]
    
    harmony_pca_full = np.zeros((original_cell_count, pca_dims))
    harmony_umap_full = np.zeros((original_cell_count, umap_dims))
    
    valid_idx = np.where(valid_cell_mask)[0]
    temp_idx_map = {name: idx for idx, name in enumerate(adata_temp.obs_names)}
    temp_idx = np.array([temp_idx_map[name] for name in adata.obs_names[valid_idx]])
    
    harmony_pca_full[valid_idx] = harmony_pca[temp_idx]
    harmony_umap_full[valid_idx] = harmony_umap[temp_idx]
    
    print(f"Mapped Harmony PCA embedding size: {harmony_pca_full.shape}")
    print(f"Mapped Harmony UMAP embedding size: {harmony_umap_full.shape}")
    
    report = f"""
    Harmony Integration Report:
      Original cells: {original_cell_count} | Batches: {original_batch_count}
      Valid cells: {valid_cell_count} | Valid batches: {n_batches}
      Min batch size: {min_batch_size}
      Used parameters: theta={theta}, n_pcs={n_pcs}
    """
    print(report)
    
    return harmony_pca_full, harmony_umap_full

if __name__ == "__main__":
    from perf_utils import PerfRecorder
    PerfRecorder.start_now("step2_bbknn_harmony")
    #config here!!!!!!
    input_path = "enhanced_results/results/step1_scanorama_processed.h5ad"
    output_path = "enhanced_results/results/step2_add_muti_method.h5ad"
    step1_processed = True
    adataX_raw_count = True
    batch_key = "batch"

    run_bbknn_flag = True
    run_harmony_flag = True
    run_scvi_flag = False

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
    
    adata_processed = adata.copy()
    if step1_processed:
        print("\nCopy normalized data to adata.X")
        adata_processed.X = adata_processed.layers["normalized"].copy()
    else:
        if adataX_raw_count:
            print("\nPerforming normalization and log1p transformation...")
            adata_processed = standard_preprocess(adata_processed)

    method_runner = {
        'bbknn': run_bbknn,
        'harmony': run_harmony
    }
    
    flags = {
        'bbknn': run_bbknn_flag,
        'harmony': run_harmony_flag
    }
    
    for method, flag in flags.items():
        if flag:
            try:
                print(f"\nStarting {method} integration...")
                print(f"Using {adata_processed.shape[0]} cells for integration")
                
                from perf_utils import PerfRecorder
                with PerfRecorder(f"step2_{method}"):
                    if method == 'bbknn':
                        embedding = method_runner[method](adata_processed, batch_key)
                        adata.obsm[f'X_{method}'] = embedding
                        print(f"Added embedding to adata.obsm['X_{method}']")
                    else:
                        emb1, emb2 = method_runner[method](adata_processed, batch_key)

                        if method == 'harmony':
                            adata.obsm[f'X_{method}_pca'] = emb1
                            adata.obsm[f'X_{method}_umap'] = emb2
                            print(f"Added PCA embedding to adata.obsm['X_{method}_pca']")
                            print(f"Added UMAP embedding to adata.obsm['X_{method}_umap']")
                
                print(f"{method} integration completed successfully")
            except Exception as e:
                print(f"{method} failed: {str(e)}")
    
    adata.write(output_path)
    print(f"Results saved to {output_path}")
    
    print("\nEmbedding keys added to adata.obsm:")
    for key in adata.obsm.keys():
        if key.startswith('X_'):
            shape = adata.obsm[key].shape
            print(f"{key}: {shape[0]} cells × {shape[1]} dimensions")
    
    print("\nProcessing complete")