import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from sklearn.model_selection import KFold, train_test_split
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import os
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib
import gc
import traceback
from scipy.sparse import issparse, csr_matrix
from scipy import stats
from typing import Tuple, List, Dict, Optional
import optuna
from tqdm import tqdm
from collections import defaultdict
from tabulate import tabulate
import time
import pickle
import copy
import hashlib
import json
import argparse
from torch.utils.data import Sampler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# -------------------------- Core Configuration Class (Optimization: Support Dynamic Embedding Metric) --------------------------
class Config:
    def __init__(self, 
                 embedding_metric_prefix: str = 'X_special_harmony',
                 memory_mode: str = 'high',
                 auto_alpha: bool = False,
                 enable_caching: bool = True,
                 disable_hyper_search: bool = True):
        self._init_paths()
        self.source_h5ad_path = self.h5ad_path  # fingerprint anchor (main() later rewrites h5ad_path)
        self.device = self._get_device()
        self.embedding_metric_prefix = embedding_metric_prefix  # Changed to prefix matching
        self.memory_mode = memory_mode.lower()
        self.auto_alpha = auto_alpha
        self.enable_caching = enable_caching
        self.disable_hyper_search = disable_hyper_search
        self.fixed_alpha = 0.7
        # ---- fusion architecture ----
        self.fusion_mode = 'moe'  # default architecture: typed-expert routing
        self.num_heads = 8
        self.n_query_tokens = 4
        self.val_fraction = 0.1
        self.norm_version = 'v7_log1p_w999_gc5'  # bump to invalidate all stale caches
        self.suffix_tag = ''  # ablation run tag (set via --suffix_tag)
        self.seed = 42  # set via --seed
        # ---- dual-level objective ----
        self.objective = 'denoise'  # distill | dual_level | denoise (denoise is the default)
        self.w_coarse = 0.5   # Harmony-PC alignment (coarse cell-type anchor)
        # Where and how the coarse (Harmony cosine) loss is attached:
        #   coarse_target 'bio'    -> bio branch only (risk: bio collapses and the
        #       fused output loses its anchor)
        #   coarse_target 'fusion' -> coarse loss on pred/fused head (required)
        #   attn_detach_coarse 1 (default): in the coarse path the ATTENTION OUTPUT
        #       is detached, so coarse gradients flow only through the gene query and
        #       output layers, never into attention weights / module tokens. Attention
        #       is shaped purely by recon + denoise while fused stays anchored.
        #   attn_detach_coarse 0 + coarse_target 'fusion' = no gradient-path isolation.
        self.coarse_target = 'fusion'  # bio | fusion
        self.attn_detach_coarse = 1    # 1 = gradient-path isolation for the coarse loss
        self.w_fine = 0.4     # two-level kNN InfoNCE (dual_level mode only)
        self.w_recon = 0.2    # module activity reconstruction (biological anchor)
        self.w_denoise = 0.3  # masked gene reconstruction (denoise mode)
        self.w_lb = 0.01      # MoE load-balancing aux weight (anti expert-collapse)
        self.size_trust = 0   # learnable per-module trust scale ~ module size (1=on)
        self.min_module_size = 1  # minimum number of present genes per module (filter tiny modules)
        # ---- input-side knowledge integration ----
        self.keep_all_hvg = 1       # default: feed all HVGs into the gene branch
        self.gene_mask_channel = 0  # 1 = dual-channel gene input [fluctuation, detection mask]
        self.gene_gate = 0          # 1 = per-gene sigmoid gate (full-resolution, detach-shielded)
        self.view_dropout = 0.3  # training-time gene-input corruption rate
        self.data_dropout = 0.0  # data-level dropout simulation (0/0.5/0.7/0.9 for robustness tests)
        self.knn_k = 15          # fine positives per cell
        self.coarse_pool = 100   # coarse scaffold candidate pool size
        self.tau = 0.2           # InfoNCE temperature
        self._hyper_ranges = {
            'batch_size': (32, 128),
            'embedding_dim': (256, 512),
            'hidden_dim': (256, 512),
            'dropout_rate': (0.28, 0.48),
            'lr': (2.5e-4, 1.0e-3),
            'weight_decay': (3e-6, 1e-4),
            'alpha_values': [round(x, 2) for x in np.arange(0.30, 0.71, 0.02)]
        }
        self.n_trials = 20
        self.k_folds = 10
        self.num_workers = 0 if sys.platform == 'win32' else 4
        self.epochs = 300
        self.patience = 15  # Early stopping patience
        self.delta = 0.001  # Early stopping minimum improvement threshold
        self.batch_key = "batch"  # Specify batch column name (must match column name in adata.obs)
        self._adjust_parameters()
        self._create_dirs()
        
        # Dynamically set current embedding metric (updated in loop)
        self.current_embedding_metric = None
    
    def _init_paths(self):
        base_dir = Path("enhanced_results")
        self.sc_gene_label_path = Path("SC_data/c2.reactome_cgp.v2026.1.Hs.symbols_filtered_parent.csv")  # default: Reactome+CGP module set with redundancy removed
        self.h5ad_path = Path("enhanced_results/results/step4-5_add_special_harmony.h5ad")
        self.output_path = base_dir / "results/step5_add_DeepFusion.h5ad"
        self.plot_dir = base_dir / "DeepFusion/training_plots"
        self.model_dir = base_dir / "DeepFusion/saved_models" 
        self.results_dir = base_dir / "DeepFusion/results"
        self.analysis_dir = base_dir / "DeepFusion/alpha_analysis"
        self.cache_dir = base_dir / "DeepFusion/cache"
        self.study_storage = self.cache_dir / "hyper_study.db"
        self.study_name = "scRNA_hyper_study"
    
    def _create_dirs(self):
        for d in [self.plot_dir, self.model_dir, self.results_dir, 
                 self.analysis_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _adjust_parameters(self):
        if self.memory_mode == 'low':
            print("⚠️ Low memory mode enabled, adjusting parameters...")
            self._hyper_ranges['batch_size'] = (16, 64)
            self._hyper_ranges['embedding_dim'] = (64, 256)
            self._hyper_ranges['hidden_dim'] = (128, 512)
            self.n_trials = 10
            self.k_folds = 3
            self.num_workers = 0
        else:
            print("✅ High memory mode enabled, using default parameters")
    
    def validate_paths(self):
        required_paths = [
            self.sc_gene_label_path,
            self.h5ad_path
        ]
        missing = [str(p) for p in required_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing critical files: {missing}")
    
    def validate_embedding(self, adata, embedding_metric: str):
        """Validate if the specified embedding is valid (supports dynamic input)"""
        if embedding_metric not in adata.obsm:
            available = list(adata.obsm.keys())
            raise ValueError(f"Embedding matrix {embedding_metric} not found. Available embeddings: {available}")
        emb = adata.obsm[embedding_metric]
        if not isinstance(emb, (np.ndarray, csr_matrix)):
            raise TypeError("Embedding matrix must be a numpy array or sparse matrix")
        if emb.shape[0] != adata.n_obs:
            raise ValueError(f"Embedding matrix row count {emb.shape[0]} doesn't match cell count {adata.n_obs}")
        print(f"✅ Validation passed: Using {embedding_metric} as supervised signal")

    def get_all_special_harmony_embeddings(self, adata) -> List[str]:
        """Get all embedding keys starting with X_special_harmony"""
        harmony_keys = [key for key in adata.obsm.keys() if key.startswith(self.embedding_metric_prefix)]
        # Sort (by numeric suffix, e.g., 3,6,9...)
        def extract_number(key):
            parts = key.split('_')
            if len(parts) >= 3 and parts[-1].isdigit():
                return int(parts[-1])
            return 0  # Keys without numeric suffix come first
        harmony_keys.sort(key=extract_number)
        return harmony_keys

    def get_embedding_suffix(self, embedding_metric: str) -> str:
        """Generate suffix from embedding name (e.g., X_special_harmony_3 → _3, X_special_harmony → empty)"""
        if embedding_metric == self.embedding_metric_prefix:
            return ""
        suffix_part = embedding_metric.replace(self.embedding_metric_prefix, "")
        return suffix_part if suffix_part.startswith('_') else f"_{suffix_part}"

    def get_run_tag(self) -> str:
        """Ablation run tag (empty for the main full-model run)."""
        return getattr(self, 'suffix_tag', '').lstrip('_')

    def get_dir_name(self, embedding_metric: str) -> str:
        """Per-run model/results directory name (tag-separated for ablations)."""
        base = embedding_metric.replace('/', '_')
        tag = self.get_run_tag()
        return f"{base}_{tag}" if tag else base

    def get_output_suffix(self, embedding_metric: str) -> str:
        """obsm key suffix including the ablation tag (e.g. '_20_concat')."""
        suffix = self.get_embedding_suffix(embedding_metric)
        tag = self.get_run_tag()
        return f"{suffix}_{tag}" if tag else suffix

    def validate_params(self, params: dict):
        required_keys = ['embedding_dim', 'hidden_dim', 'dropout', 'batch_size', 'lr', 'weight_decay', 'alpha']
        for key in required_keys:
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")
        
        if not (self.embedding_dim_range[0] <= params['embedding_dim'] <= self.embedding_dim_range[1]):
            raise ValueError(f"embedding_dim {params['embedding_dim']} out of range [{self.embedding_dim_range[0]}, {self.embedding_dim_range[1]}]")
        
        if not (self.hidden_dim_range[0] <= params['hidden_dim'] <= self.hidden_dim_range[1]):
            raise ValueError(f"hidden_dim {params['hidden_dim']} out of range [{self.hidden_dim_range[0]}, {self.hidden_dim_range[1]}]")
        
        if not (self.dropout_range[0] <= params['dropout'] <= self.dropout_range[1]):
            raise ValueError(f"dropout {params['dropout']} out of range [{self.dropout_range[0]}, {self.dropout_range[1]}]")
        
        if not (self.batch_size_range[0] <= params['batch_size'] <= self.batch_size_range[1]):
            raise ValueError(f"batch_size {params['batch_size']} out of range [{self.batch_size_range[0]}, {self.batch_size_range[1]}]")
        
        if not (self.lr_range[0] <= params['lr'] <= self.lr_range[1]):
            raise ValueError(f"lr {params['lr']} out of range [{self.lr_range[0]}, {self.lr_range[1]}]")
        
        if not (self.weight_decay_range[0] <= params['weight_decay'] <= self.weight_decay_range[1]):
            raise ValueError(f"weight_decay {params['weight_decay']} out of range [{self.weight_decay_range[0]}, {self.weight_decay_range[1]}]")
        
        if not (self.auto_alpha or params['alpha'] == self.fixed_alpha):
            if params['alpha'] not in self.alpha_values:
                raise ValueError(f"alpha {params['alpha']} not in allowed values")
        
        print("✅ Parameters validation passed")
    
    @property
    def batch_size_range(self) -> Tuple[int, int]:
        return self._hyper_ranges['batch_size']
    
    @property
    def embedding_dim_range(self) -> Tuple[int, int]:
        return self._hyper_ranges['embedding_dim']
    
    @property
    def hidden_dim_range(self) -> Tuple[int, int]:
        return self._hyper_ranges['hidden_dim']
    
    @property
    def dropout_range(self) -> Tuple[float, float]:
        return self._hyper_ranges['dropout_rate']
    
    @property
    def lr_range(self) -> Tuple[float, float]:
        return self._hyper_ranges['lr']
    
    @property
    def weight_decay_range(self) -> Tuple[float, float]:
        return self._hyper_ranges['weight_decay']
    
    @property
    def alpha_values(self) -> list:
        return self._hyper_ranges['alpha_values']

# -------------------------- Cache fingerprint helpers --------------------------
def compute_data_fingerprint(config: 'Config') -> str:
    """Fingerprint of the source data (h5ad + module CSV + alignment mode).
    Governs preprocessed_data.h5ad."""
    h = hashlib.sha256()
    for p in [config.source_h5ad_path, config.sc_gene_label_path]:
        p = Path(p)
        if p.exists():
            s = p.stat()
            h.update(f"{p.name}:{s.st_size}:{int(s.st_mtime)}".encode())
    h.update(f"keep_all_hvg={getattr(config, 'keep_all_hvg', 0)}".encode())
    return h.hexdigest()[:16]

def compute_module_fingerprint(config: 'Config') -> str:
    """Module-activity cache fingerprint = data fingerprint + normalization version
    + min_module_size (changing the cutoff must invalidate the cache)."""
    h = hashlib.sha256()
    h.update(compute_data_fingerprint(config).encode())
    h.update(config.norm_version.encode())
    h.update(str(getattr(config, 'min_module_size', 1)).encode())
    return h.hexdigest()[:16]

def _fp_valid(fp_file: Path, expected: str) -> bool:
    if not fp_file.exists():
        return False
    try:
        return json.loads(fp_file.read_text()).get('fingerprint') == expected
    except Exception:
        return False

def _fp_write(fp_file: Path, payload: dict):
    fp_file.parent.mkdir(parents=True, exist_ok=True)
    fp_file.write_text(json.dumps(payload, indent=2))

def cache_fingerprint_valid(config: 'Config') -> bool:
    """Data-level cache (preprocessed_data.h5ad) validity."""
    return _fp_valid(Path(config.cache_dir) / "data_fingerprint.json",
                     compute_data_fingerprint(config))

def write_cache_fingerprint(config: 'Config'):
    _fp_write(Path(config.cache_dir) / "data_fingerprint.json",
              {'fingerprint': compute_data_fingerprint(config)})

def module_cache_valid(config: 'Config') -> bool:
    """Module-activity cache validity (binds norm_version)."""
    return _fp_valid(Path(config.cache_dir) / "module_fingerprint.json",
                     compute_module_fingerprint(config))

def write_module_fingerprint(config: 'Config'):
    _fp_write(Path(config.cache_dir) / "module_fingerprint.json",
              {'fingerprint': compute_module_fingerprint(config),
               'norm_version': config.norm_version})

def combat(raw_counts, batch, mod=None, mean_only=False, ref_batch=None):
    """
    Pure Python implementation of ComBat batch correction (adapted for RNA-seq raw counts, no log1p)
    :param raw_counts: Cell × gene raw count matrix (numpy array)
    :param batch: Batch label array (matches cell count)
    :param mod: Covariate matrix (None for no covariates)
    :param mean_only: Whether to correct mean only (False preserves variance)
    :param ref_batch: Reference batch (None uses global mean)
    :return: Corrected count matrix (cell × gene)
    """
    # Transpose to gene × cell (ComBat standard input)
    dat = raw_counts.T.copy()
    batch = np.array(batch)
    batch_levels = np.unique(batch)
    batch_info = [np.where(batch == level)[0] for level in batch_levels]
    n_batch = len(batch_info)
    n_batches = np.array([len(x) for x in batch_info])
    n_array = float(sum(n_batches))

    # 1. Centering (remove gene mean)
    if mod is None:
        design = np.zeros((len(batch), n_batch))
        for i, level in enumerate(batch_levels):
            design[batch == level, i] = 1
    else:
        design = np.column_stack((mod, np.eye(len(batch))[batch]))

    # Calculate gene means
    mu = np.mean(dat, axis=1).reshape(-1, 1)
    dat = dat - mu

    # 2. Fit linear model to calculate batch effects
    beta_hat = np.linalg.lstsq(design, dat.T, rcond=None)[0].T
    sigma_hat = []
    for i, idx in enumerate(batch_info):
        sigma_hat.append(np.var(dat[:, idx] - beta_hat[:, i:(i+1)].dot(design[idx, i:(i+1)].T), axis=1, ddof=1))
    sigma_hat = np.array(sigma_hat).T

    # 3. Bayesian estimation of batch effects
    gamma_star = []
    delta_star = []
    for i in range(dat.shape[0]):
        # Batch effect estimation for each gene
        gamma = []
        delta = []
        for j in range(n_batch):
            if n_batches[j] > 1:
                gamma.append(beta_hat[i, j])
                delta.append(sigma_hat[i, j])
            else:
                gamma.append(0)
                delta.append(1)
        gamma_star.append(gamma)
        delta_star.append(delta)
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)

    # 4. Correct batch effects (fix division by zero error)
    dat_corrected = dat.copy()
    for j, idx in enumerate(batch_info):
        # Handle zero/negative values in delta_star to avoid division by zero
        delta_star_j = delta_star[:, j:j+1].copy()
        delta_star_j[delta_star_j <= 0] = 1e-8  # Replace zero/negative values with small value
        sqrt_delta_j = np.sqrt(delta_star_j)
        
        if ref_batch is not None and j != ref_batch:
            # Correct based on reference batch
            delta_star_ref = delta_star[:, ref_batch:ref_batch+1].copy()
            delta_star_ref[delta_star_ref <= 0] = 1e-8
            sqrt_delta_ref = np.sqrt(delta_star_ref)
            
            dat_corrected[:, idx] = (dat[:, idx] - gamma_star[:, j:j+1]) / sqrt_delta_j * sqrt_delta_ref + gamma_star[:, ref_batch:ref_batch+1]
        else:
            # Correct based on global mean
            dat_corrected[:, idx] = (dat[:, idx] - gamma_star[:, j:j+1]) / sqrt_delta_j if not mean_only else dat[:, idx] - gamma_star[:, j:j+1]

    # 5. Restore gene means and transpose back to cell × gene
    dat_corrected = dat_corrected + mu
    corrected_counts = dat_corrected.T

    # ========== Critical Fix: Strict Data Cleaning ==========
    # 1. Replace NaN/Inf with 0
    corrected_counts = np.nan_to_num(corrected_counts, nan=0.0, posinf=0.0, neginf=0.0)
    # 2. Fix small negative values (raw counts are non-negative)
    corrected_counts[corrected_counts < 0] = 0.0
    # 3. Limit values to float32 valid range (avoid precision overflow)
    max_float32 = np.finfo(np.float32).max
    corrected_counts[corrected_counts > max_float32 * 0.1] = max_float32 * 0.1  # Limit to 10% of float32 max
    # 4. Convert to float32 and validate again
    corrected_counts = corrected_counts.astype(np.float32)
    
    # Validate cleaning results
    if np.isinf(corrected_counts).any():
        print(f"⚠️ Warning: Detected {np.isinf(corrected_counts).sum()} Inf values, forced replacement to 0")
        corrected_counts = np.nan_to_num(corrected_counts, posinf=0.0, neginf=0.0)
    if np.isnan(corrected_counts).any():
        print(f"⚠️ Warning: Detected {np.isnan(corrected_counts).sum()} NaN values, forced replacement to 0")
        corrected_counts = np.nan_to_num(corrected_counts, nan=0.0)
    
    return corrected_counts

def correct_batch_raw_counts(adata, config: Config, ref_batch=None):
    """
    Apply pure Python ComBat batch correction on raw layer (no log1p, preserve raw count characteristics)
    """
    # 1. Check if batch column exists
    if config.batch_key not in adata.obs:
        raise KeyError(f"Batch column {config.batch_key} not found in adata.obs! Please check if batch_key in Config is correct")
    
    # 2. Check number of batches, skip correction for single batch
    batch_labels = adata.obs[config.batch_key].astype("category")
    n_batches = len(batch_labels.cat.categories)
    if n_batches <= 1:
        print(f"📌 Only {n_batches} batch detected, no batch correction needed, copy raw layer to raw_corrected directly")
        adata.layers["raw_corrected"] = adata.layers["raw"].copy()
        return adata
    
    # 3. Extract raw layer and convert to dense matrix
    raw_mat = adata.layers["raw"]
    if issparse(raw_mat):
        raw_mat = raw_mat.toarray()
    raw_mat = raw_mat.astype(np.float32)
    
    # ========== New: Raw Data Pre-cleaning ==========
    raw_mat = np.nan_to_num(raw_mat, nan=0.0, posinf=0.0, neginf=0.0)
    raw_mat[raw_mat < 0] = 0.0
    
    # 4. Prepare batch information
    batch_ids = batch_labels.cat.codes.values  # Encode batches as 0,1,2...
    print(f"📌 Detected {n_batches} batches, starting pure Python ComBat batch removal...")
    
    # 5. Run pure Python ComBat
    corrected_raw = combat(
        raw_counts=raw_mat,
        batch=batch_ids,
        mod=None,
        mean_only=False,
        ref_batch=ref_batch
    )
    
    # ========== New: Post-correction Data Validation ==========
    print(f"\n🔍 Post-correction data validation:")
    print(f"   - Minimum value: {corrected_raw.min():.4f}")
    print(f"   - Maximum value: {corrected_raw.max():.4f}")
    print(f"   - Number of NaNs: {np.isnan(corrected_raw).sum()}")
    print(f"   - Number of Infs: {np.isinf(corrected_raw).sum()}")
    print(f"   - Number of negative values: {np.sum(corrected_raw < 0)}")
    
    # 6. Store in adata and validate
    # Keep same sparse/dense type as original raw layer
    if issparse(adata.layers["raw"]):
        adata.layers["raw_corrected"] = csr_matrix(corrected_raw)
        # Additional validation for sparse matrix data
        if np.isinf(adata.layers["raw_corrected"].data).any():
            adata.layers["raw_corrected"].data = np.nan_to_num(adata.layers["raw_corrected"].data, posinf=0.0, neginf=0.0)
        if np.isnan(adata.layers["raw_corrected"].data).any():
            adata.layers["raw_corrected"].data = np.nan_to_num(adata.layers["raw_corrected"].data, nan=0.0)
    else:
        adata.layers["raw_corrected"] = corrected_raw
    
    print(f"✅ Completed raw layer batch correction: added layers['raw_corrected'] (shape {corrected_raw.shape})")
    
    # 7. Visual validation of batch correction effect (final compatible version)
    try:
        # New version scanpy: supports layer parameter
        sc.tl.pca(adata, layer="raw_corrected", n_comps=20)
        # Rename PCA results for differentiation
        if 'X_pca' in adata.obsm:
            adata.obsm['X_pca_raw_corrected'] = adata.obsm.pop('X_pca')
        if 'pca' in adata.uns:
            adata.uns['pca_raw_corrected'] = adata.uns.pop('pca')
    except TypeError as e:
        if 'layer' in str(e):
            # Old version scanpy: does not support layer parameter, temporarily replace X
            backup_X = adata.X.copy()
            adata.X = adata.layers["raw_corrected"]
            
            # ========== New: Data cleaning before temporary X replacement ==========
            if issparse(adata.X):
                adata.X.data = np.nan_to_num(adata.X.data, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
            
            sc.tl.pca(adata, n_comps=20)
            # Rename PCA results for differentiation
            adata.obsm['X_pca_raw_corrected'] = adata.obsm.pop('X_pca')
            adata.uns['pca_raw_corrected'] = adata.uns.pop('pca')
            adata.X = backup_X
        else:
            raise
    except ValueError as e:
        if 'infinity or a value too large' in str(e):
            print(f"⚠️ PCA calculation failed (value out of range), trying to recalculate with clipped data...")
            # Emergency handling: clip corrected_raw and recalculate PCA
            temp_data = adata.layers["raw_corrected"].toarray() if issparse(adata.layers["raw_corrected"]) else adata.layers["raw_corrected"]
            temp_data = temp_data.clip(0, np.percentile(temp_data, 99.9))  # Clip to 99.9 percentile
            backup_X = adata.X.copy()
            adata.X = temp_data
            sc.tl.pca(adata, n_comps=20)
            adata.obsm['X_pca_raw_corrected'] = adata.obsm.pop('X_pca')
            adata.uns['pca_raw_corrected'] = adata.uns.pop('pca')
            adata.X = backup_X
        else:
            raise

    pca_plot_name = "batch_correction_pca_pure_python.png"
    # Save original X_pca (if exists), temporarily replace with corrected PCA results
    has_original_pca = 'X_pca' in adata.obsm
    original_pca = adata.obsm.get('X_pca', None)
    original_pca_uns = adata.uns.get('pca', None)
    
    # Temporarily replace X_pca with corrected results (adapt to old scanpy version plotting)
    adata.obsm['X_pca'] = adata.obsm['X_pca_raw_corrected']
    adata.uns['pca'] = adata.uns['pca_raw_corrected']
    
    try:
        # Do not pass basis parameter to sc.pl.pca (old version does not support)
        sc.pl.pca(adata, 
                  color=config.batch_key, 
                  title="PCA of Corrected Raw Counts (Pure Python ComBat, No log1p)", 
                  save=f"_{pca_plot_name}", 
                  show=False)
        print(f"📊 Batch correction effect PCA plot saved to: scanpy_plot_dir/pca_{pca_plot_name}")
    except Exception as e:
        print(f"⚠️ PCA plotting failed: {str(e)}, skipping plotting and continuing process")
    finally:
        # Restore original PCA results (if exists)
        if has_original_pca:
            adata.obsm['X_pca'] = original_pca
            adata.uns['pca'] = original_pca_uns
        else:
            # If no original X_pca, delete temporarily created one
            del adata.obsm['X_pca']
            del adata.uns['pca']
    
    return adata

def add_fluctuation_layer(adata: anndata.AnnData,
                          base_layer: str = "raw_corrected",
                          new_layer: str = "fluctuation") -> None:

    expr = adata.layers[base_layer]
    if issparse(expr):
        expr = expr.toarray()

    lib_size = expr.sum(axis=1, keepdims=True)
    lib_size[lib_size == 0] = 1
    norm_expr = (expr / lib_size) * 1e4

    median_expr = np.zeros(norm_expr.shape[1])
    for i in range(norm_expr.shape[1]):
        nz_mask = expr[:, i] > 0
        nz_vals = norm_expr[:, i][nz_mask]
        median_expr[i] = np.median(nz_vals) if len(nz_vals) else 0.0

    eps = 1e-8
    log_expr   = np.log2(norm_expr + eps)
    log_median = np.log2(median_expr + eps)
    fluct = np.abs(log_expr - log_median)

    fluct[(median_expr == 0) | (expr == 0)] = 0.0

    adata.layers[new_layer] = fluct.astype(np.float32)
    print(f"✅ Fluctuation layer '{new_layer}' added (symmetric log2-fold).")

# -------------------------- Data Processor Class --------------------------
class SCDataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self._validate_paths()

    def _validate_paths(self):
        required_paths = [
            self.config.sc_gene_label_path,
            self.config.h5ad_path
        ]
        for p in required_paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"File not found: {p}")       
        
    def load_data(self, reload: bool = False) -> Tuple[anndata.AnnData, Dict]:
        cache_file = Path(self.config.cache_dir) / "preprocessed_data.h5ad"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Cache is valid only if the fingerprint matches (data + module CSV + norm version)
        if (not reload and cache_file.exists() and self.config.enable_caching
                and cache_fingerprint_valid(self.config)):
            return self._load_cached_data(cache_file)
        if cache_file.exists() and not cache_fingerprint_valid(self.config):
            print("⚠️ Cache fingerprint mismatch (stale cache), reprocessing data...")
        return self._process_and_cache_data(cache_file)

    def _load_cached_data(self, cache_path: Path) -> Tuple[anndata.AnnData, Dict]:
        print("♻️ Loading data from cache...")
        adata = anndata.read_h5ad(cache_path)
        self._optimize_memory(adata)
        gene_modules = self._create_gene_modules(adata.var["gene_module"])
        return adata, gene_modules
    
    def _process_and_cache_data(self, cache_path: Path) -> Tuple[anndata.AnnData, Dict]:
        # Runs only on cache miss/rebuild: raw-count batch correction + fluctuation
        # layer + module-activity computation. Recorded separately as the paper's
        # "preprocessing & module-score computation" performance item.
        from perf_utils import PerfRecorder
        with PerfRecorder("step5-2_preprocess_module_activity"):
            print("🔬 Processing raw data...")
            adata = anndata.read_h5ad(self.config.h5ad_path)

            # First batch correction, then calculate fluctuation
            adata = correct_batch_raw_counts(adata, self.config)
            add_fluctuation_layer(adata)

            gene_labels = pd.read_csv(self.config.sc_gene_label_path, index_col=0)
            adata = self._align_data(adata, gene_labels)
            self._preprocess(adata)
            self._optimize_memory(adata, for_caching=True)

            adata.write(cache_path)
            write_cache_fingerprint(self.config)  # bind data cache to data fingerprint
            print(f"✅ Data cached to {cache_path}")
            return adata, self._create_gene_modules(adata.var["gene_module"])

    def _preprocess(self, adata: anndata.AnnData):
        required_layers = ["normalized", "raw", "raw_corrected", "fluctuation"]
        missing = [layer for layer in required_layers if layer not in adata.layers]
        if missing:
            raise KeyError(f"Missing required layers after batch correction: {missing}")
        
        # Validate value ranges of each layer
        norm_data = adata.layers["normalized"]
        raw_data = adata.layers["raw"]
        raw_corrected = adata.layers["raw_corrected"]
        fluct_data = adata.layers["fluctuation"]
        
        norm_min = norm_data.min() if not issparse(norm_data) else norm_data.data.min()
        norm_max = norm_data.max() if not issparse(norm_data) else norm_data.data.max()
        raw_min = raw_data.min() if not issparse(raw_data) else raw_data.data.min()
        raw_max = raw_data.max() if not issparse(raw_data) else raw_data.data.max()
        corr_min = raw_corrected.min() if not issparse(raw_corrected) else raw_corrected.data.min()
        corr_max = raw_corrected.max() if not issparse(raw_corrected) else raw_corrected.data.max()
        fluct_min = fluct_data.min() if not issparse(fluct_data) else fluct_data.data.min()
        fluct_max = fluct_data.max() if not issparse(fluct_data) else fluct_data.data.max()
        
        print(f"✅ Data validated after batch correction: ")
        print(f"   - normalized layer: min={norm_min:.4f}, max={norm_max:.4f}")
        print(f"   - raw layer: min={raw_min}, max={raw_max}")
        print(f"   - raw_corrected layer: min={corr_min:.4f}, max={corr_max:.4f}")
        print(f"   - fluctuation layer (absolute fluctuation): min={fluct_min:.4f}, max={fluct_max:.4f}")
        
    def _align_data(self, adata, gene_labels) -> anndata.AnnData:
        if getattr(self.config, 'keep_all_hvg', 0):
            # Keep ALL genes (restore the ~400 prior-uncovered HVGs);
            # uncovered genes get NaN module label (skipped by _create_gene_modules)
            adata.var["gene_module"] = gene_labels.reindex(adata.var_names).iloc[:, 0]
            n_uncov = adata.var["gene_module"].isna().sum()
            print(f"📊 keep_all_hvg: {adata.shape[1]} genes kept, {n_uncov} prior-uncovered (empty module)")
        else:
            common_genes = adata.var_names.intersection(gene_labels.index)
            adata = adata[:, common_genes].copy()
            adata.var["gene_module"] = gene_labels.loc[common_genes, gene_labels.columns[0]].astype("category")
        print(f"📊 Final data dimensions: {adata.shape}")
        return adata

    def _optimize_memory(self, adata: anndata.AnnData, for_caching: bool = False):
        if self.config.memory_mode == 'low':
            print("🔽 Applying low memory optimizations")
            if issparse(adata.X):
                if not isinstance(adata.X, csr_matrix):
                    adata.X = csr_matrix(adata.X)
                adata.X.data = adata.X.data.astype(np.float32)
            adata.strings_to_categoricals()
            
            if for_caching:
                for layer in adata.layers:
                    if issparse(adata.layers[layer]):
                        adata.layers[layer] = csr_matrix(adata.layers[layer])
                gc.collect()

    @staticmethod
    def _create_gene_modules(gene_labels: pd.Series) -> Dict[str, list]:
        modules = defaultdict(list)
        for gene, labels in gene_labels.items():
            if pd.isna(labels):
                continue
            for module in map(str.strip, str(labels).split(';')):
                module = module.strip()
                if module:
                    module_name = f"Module_{module}"
                    modules[module_name].append(gene)
        
        print(f"Original modules: {len(modules)}, Valid modules: {len([v for v in modules.values() if v])}")
        return {k: v for k, v in modules.items() if v}

# -------------------------- Dataset Class (Supports Dynamic Embedding Metric) --------------------------
class SCNDataset(Dataset):
    def __init__(self, adata: anndata.AnnData, gene_modules: Dict[str, list], config: Config, current_embedding_metric: str):
        self.config = config
        self.current_embedding_metric = current_embedding_metric  # Currently used embedding
        gene_idx = {gene: i for i, gene in enumerate(adata.var_names)}
        self.modules = {}

        for mod_name, genes in gene_modules.items():
            valid_genes = []
            for g in genes:
                if g in gene_idx:
                    valid_genes.append(gene_idx[g])
                else:
                    if config.memory_mode != 'low': 
                        print(f"⚠️ Gene {g} not in dataset, skipping")
            min_size = getattr(config, 'min_module_size', 1)
            if len(valid_genes) >= min_size:
                self.modules[mod_name] = valid_genes
            else:
                print(f"⚠️ Module {mod_name} has {len(valid_genes)} valid genes (< min_module_size={min_size}), filtering")

        print(f"Original modules: {len(gene_modules)} | Valid modules: {len(self.modules)}")
        if not self.modules:
            raise ValueError("No valid gene modules found")

        # Load absolute fluctuation version of fluctuation layer
        if "fluctuation" not in adata.layers:
            raise KeyError("'fluctuation' layer (absolute fluctuation version) not found in AnnData")
        norm_layer = adata.layers["fluctuation"]
        self.norm_data = torch.tensor(
            norm_layer.toarray() if issparse(norm_layer) else norm_layer
        ).float()

        # Load batch-corrected raw layer
        if "raw_corrected" not in adata.layers:
            raise KeyError("'raw_corrected' layer (batch-corrected) not found in AnnData")
        raw_layer = adata.layers["raw_corrected"]
        self.raw_data = torch.tensor(
            raw_layer.toarray() if issparse(raw_layer) else raw_layer
        ).float()

        # Detection mask channel (dropout as explicit missingness).
        # Mask comes from the TRUE raw counts layer (0 = not detected), NOT the
        # ComBat-corrected layer. Module activity computation still uses the
        # single-channel norm_data; the mask only joins the MODEL input.
        self.gene_mask_channel = getattr(config, 'gene_mask_channel', 0)
        if self.gene_mask_channel:
            true_raw = adata.layers["raw"]
            true_raw = true_raw.toarray() if issparse(true_raw) else np.asarray(true_raw)
            self.mask_data = torch.tensor((true_raw > 0).astype(np.float32))
            print(f"🎭 Mask channel on: detection rate = {self.mask_data.mean().item():.3f}")

        print(f"🔍 Data loaded (batch-corrected + absolute fluctuation): ")
        print(f"   - fluctuation layer: min={self.norm_data.min().item():.4f}, max={self.norm_data.max().item():.4f}")
        print(f"   - raw_corrected layer: min={self.raw_data.min().item()}, max={self.raw_data.max().item()}")

        # Check if module matrix cache exists (for resuming after crash)
        # Cache filenames carry the data-dropout tag so corrupted variants
        # never load the clean module matrix
        self.data_dropout = getattr(config, 'data_dropout', 0.0)
        self.dd_tag = f"_dd{self.data_dropout}" if self.data_dropout > 0 else ""
        if self.data_dropout > 0:
            rng = np.random.default_rng(config.seed)
            keep_mask = torch.tensor(rng.random(self.norm_data.shape) >= self.data_dropout).float()
            self.norm_data = self.norm_data * keep_mask
            self.raw_data = self.raw_data * keep_mask
            if getattr(self, 'mask_data', None) is not None:
                self.mask_data = self.mask_data * keep_mask  # dropout also counts as missing
            print(f"🎲 Data-level dropout {self.data_dropout:.0%} applied to fluctuation+raw (seed={config.seed})")

        module_cache_path = Path(self.config.cache_dir) / f"module_matrix_cache{self.dd_tag}.pt"
        stats_path = Path(self.config.analysis_dir) / f"module_statistics{self.dd_tag}.csv"
        valid_modules_cache_path = Path(self.config.cache_dir) / f"valid_modules_cache{self.dd_tag}.pkl"
        
        if (module_cache_path.exists() and stats_path.exists() and valid_modules_cache_path.exists()
                and module_cache_valid(self.config)):  # invalidate stale module caches
            print(f"♻️ Loading cached module activity from {module_cache_path}")
            self.module_matrix = torch.load(module_cache_path, map_location='cpu')
            import pickle
            with open(valid_modules_cache_path, 'rb') as f:
                self.valid_modules = pickle.load(f)
            print(f"   - Module matrix shape: {self.module_matrix.shape}")
            
            # Skip analysis if already done, otherwise do it
            if not (Path(self.config.analysis_dir) / "normalized_activity_distribution.png").exists():
                print("📊 Running activity distribution analysis...")
                self._analyze_activity_distribution()
            else:
                print("✅ Distribution analysis already done, skipping")
            
            # Load embedding
            config.validate_embedding(adata, self.current_embedding_metric)
            emb_matrix = adata.obsm[self.current_embedding_metric]
            self.embedding_data = torch.tensor(
                emb_matrix.toarray() if issparse(emb_matrix) else emb_matrix,
                dtype=torch.float32
            )
            self.embedding_dim = self.embedding_data.shape[1]
            print(f"🔧 Using embedding supervision: {self.current_embedding_metric} (dim={self.embedding_dim})")
            
            # Update adata for compatibility
            adata.obsm["module_activity"] = self.module_matrix.numpy().astype(np.float32)
            adata.uns["gene_modules"] = {
                "module_names": list(self.valid_modules.keys()),
                "gene_mapping": {k: [adata.var_names[i] for i in v["gene_indices"]] 
                               for k, v in self.valid_modules.items()},
                "module_stats_path": str(stats_path)
            }
            if getattr(self.config, 'objective', 'distill') == 'dual_level':
                self._build_knn_graph()
            return

        module_vectors = []
        self.valid_modules = {}
        module_stats = []
        
        for mod_name, mod_genes in tqdm(self.modules.items(), 
                                        desc="Calculating enhanced module activities",
                                        disable=config.memory_mode=='low'):
            valid_gene_indices = [idx for idx in mod_genes if idx < self.raw_data.shape[1]]
            n_genes = len(valid_gene_indices)
            
            if n_genes == 0:
                continue

            norm_mod_expr = self.norm_data[:, valid_gene_indices]
            raw_mod_expr = self.raw_data[:, valid_gene_indices]

            with torch.no_grad():
                # Calculate detection rate based on batch-corrected raw layer
                detected_mask = (raw_mod_expr > 0)
                detected_count = detected_mask.sum(dim=1).float()
                detection_rate = detected_count / (n_genes + 1e-8)

                # Calculate module mean based on absolute fluctuation values
                mod_mean = norm_mod_expr.sum(dim=1) / (n_genes + 1e-8)

                if n_genes == 1:
                    mod_std = torch.zeros_like(mod_mean)
                else:
                    mod_std = norm_mod_expr.std(dim=1, unbiased=True)
                
                mod_cv = torch.zeros_like(mod_mean)
                non_zero_mask = mod_mean > 1e-8
                mod_cv[non_zero_mask] = mod_std[non_zero_mask] / mod_mean[non_zero_mask]

                max_val, _ = norm_mod_expr.max(dim=1)
                min_val, _ = norm_mod_expr.min(dim=1)
                specificity = torch.zeros_like(max_val)
                non_zero_mask = (max_val + min_val) > 1e-8
                specificity[non_zero_mask] = (max_val[non_zero_mask] - min_val[non_zero_mask]) / (max_val[non_zero_mask] + min_val[non_zero_mask] + 1e-8)

                if n_genes > 1:
                    centered = norm_mod_expr - mod_mean.unsqueeze(1)
                    cov_matrix = torch.matmul(centered.transpose(0, 1), centered) / (centered.shape[0] - 1)
                    gene_corr = torch.mean(cov_matrix) / (mod_std.mean() + 1e-8)
                else:
                    gene_corr = torch.tensor(1.0)
                
                # Clamp gene_corr before it is used as an amplification factor.
                # When a module's mean std is ~0, gene_corr explodes and the
                # WHOLE module column is amplified (per-module winsorizing cannot fix
                # a column-wise blow-up). Cap the amplification at 2x.
                gene_corr = torch.clamp(gene_corr, min=0.0, max=5.0)

                # Module activity calculation
                base_activity = mod_mean * detection_rate
                specificity_factor = specificity ** 0.8
                dynamic_factor = 0.7 * specificity_factor + 0.3
                cv_factor = torch.sigmoid(mod_cv * 2)
                activity = base_activity * (1.0 + cv_factor * dynamic_factor)
                
                if gene_corr > 0.6:
                    activity *= (1.0 + 0.2 * gene_corr)
                
                size_factor = torch.ones_like(activity)
                if n_genes < 10:
                    size_factor *= 1.0 + 1.0/(n_genes + 1)
                elif n_genes > 30:  
                    size_factor *= 1.0 - 0.15 * detection_rate.mean()
                
                scaled_activity = activity * size_factor

                # Winsorize per-module outliers at the 99.9th percentile BEFORE log1p.
                # The heuristic amplification factors (e.g. gene_corr on tiny-std modules)
                # produce a handful of extreme values (~0.03% of entries) that would
                # otherwise dominate the MSE reconstruction loss and the token scale.
                cap = torch.quantile(scaled_activity, 0.999)
                scaled_activity = torch.minimum(scaled_activity, cap)
                # log1p normalization; decoder is linear (no Sigmoid), so the reconstruction
                # target is reachable (avoids a value-range mismatch in the recon loss)
                normalized_activity = torch.log1p(scaled_activity.clamp_min(0))
		#normalized_activity = torch.sqrt(scaled_activity + 1e-8)
                #normalized_activity = torch.log1p(scaled_activity + 1e-8)
                
                stat = {
                    "module": mod_name,
                    "n_genes": n_genes,
                    "mean_activity": normalized_activity.mean().item(),
                    "cv": mod_cv.mean().item(),
                    "specificity": specificity.mean().item(),
                }
                module_stats.append(stat)
            
            module_vectors.append(normalized_activity)
            self.valid_modules[mod_name] = {
                "gene_indices": valid_gene_indices,
                "n_genes": n_genes
            }

        self._save_module_stats(module_stats)

        self.module_matrix = torch.stack(module_vectors, dim=1)

        nan_count = torch.isnan(self.module_matrix).sum().item()
        if nan_count > 0:
            print(f"⚠️ Warning: Found {nan_count} NaN values in module matrix")
            self.module_matrix = torch.nan_to_num(self.module_matrix, nan=0.0)

        # Save cache immediately after computation (before analysis that may crash)
        module_cache_path = Path(self.config.cache_dir) / f"module_matrix_cache{self.dd_tag}.pt"
        valid_modules_cache_path = Path(self.config.cache_dir) / f"valid_modules_cache{self.dd_tag}.pkl"
        torch.save(self.module_matrix, module_cache_path)
        write_module_fingerprint(self.config)  # bind module cache to data + norm_version
        import pickle
        with open(valid_modules_cache_path, 'wb') as f:
            pickle.dump(self.valid_modules, f)
        print(f"💾 Module activity cached to {module_cache_path}")

        self._analyze_activity_distribution()
        
        print(f"🔬 Module matrix (absolute fluctuation) shape: {self.module_matrix.shape} | ")
        print(f"   - Min activity: {self.module_matrix.min().item():.4f}")
        print(f"   - Max activity: {self.module_matrix.max().item():.4f}")
        print(f"   - Mean activity: {self.module_matrix.mean().item():.4f}")

        # Validate and load current embedding
        config.validate_embedding(adata, self.current_embedding_metric)
        emb_matrix = adata.obsm[self.current_embedding_metric]
        self.embedding_data = torch.tensor(
            emb_matrix.toarray() if issparse(emb_matrix) else emb_matrix,
            dtype=torch.float32
        )
        self.embedding_dim = self.embedding_data.shape[1]
        print(f"🔧 Using embedding supervision: {self.current_embedding_metric} (dim={self.embedding_dim})")

        adata.obsm["module_activity"] = self.module_matrix.numpy().astype(np.float32)

        adata.uns["gene_modules"] = {
            "module_names": list(self.valid_modules.keys()),
            "gene_mapping": {k: [adata.var_names[i] for i in v["gene_indices"]] 
                           for k, v in self.valid_modules.items()},
            "module_stats_path": str(Path(self.config.analysis_dir) / f"module_statistics{self.dd_tag}.csv")
        }

        if getattr(self.config, 'objective', 'distill') == 'dual_level':
            self._build_knn_graph()

    def _build_knn_graph(self):
        """Two-level kNN positive graph for L_fine.
        Candidate pool = coarse scaffold (Harmony target embedding) neighbors
        -> batch-robust by construction; within the pool, rank by fluctuation
        distance -> fine subtype resolution. The two-level principle is applied
        even to graph construction. Cached per target metric, bound to the data
        fingerprint."""
        k, pool = self.config.knn_k, self.config.coarse_pool
        graph_path = Path(self.config.cache_dir) / f"knn_graph_{self.current_embedding_metric}_k{k}_pool{pool}.npy"
        if graph_path.exists() and cache_fingerprint_valid(self.config):
            pos = np.load(graph_path)
            assert pos.min() >= 0 and pos.max() < pos.shape[0], "cached kNN graph out of range"
            self.pos_sets = torch.tensor(pos, dtype=torch.long)
            print(f"♻️ Loaded two-level kNN graph: {graph_path.name} {tuple(self.pos_sets.shape)}")
            return
        print(f"🔗 Building two-level kNN graph (coarse pool={pool} → fine top-{k})...")
        scaffold = self.embedding_data.numpy()
        nn_coarse = NearestNeighbors(n_neighbors=pool + 1, metric='euclidean').fit(scaffold)
        cand = nn_coarse.kneighbors(return_distance=False)[:, 1:]  # [N, pool], exclude self
        fluct = self.norm_data  # torch [N, G]
        n = fluct.shape[0]
        pos = np.zeros((n, k), dtype=np.int32)
        chunk = 256
        for s in tqdm(range(0, n, chunk), desc="Two-level kNN graph", disable=self.config.memory_mode == 'low'):
            e = min(s + chunk, n)
            c = torch.tensor(cand[s:e], dtype=torch.long)          # [b, pool]
            f_cand = fluct[c]                                      # [b, pool, G]
            d = ((f_cand - fluct[s:e].unsqueeze(1)) ** 2).sum(-1)  # [b, pool]
            top = d.argsort(dim=1)[:, :k]                          # [b, k] pool ranks
            pos[s:e] = np.take_along_axis(c.numpy(), top.numpy(), axis=1)
        np.save(graph_path, pos)
        assert pos.min() >= 0 and pos.max() < n, "kNN graph out of range"
        self.pos_sets = torch.tensor(pos, dtype=torch.long)
        print(f"💾 Two-level kNN graph saved: {graph_path}")

    def _save_module_stats(self, stats: list):
        stats_df = pd.DataFrame(stats)    
        if not stats_df.empty:
            stats_df['module_type'] = 'medium'
            stats_df.loc[stats_df['n_genes'] < 10, 'module_type'] = 'small'
            stats_df.loc[stats_df['n_genes'] > 30, 'module_type'] = 'large'
            stats_path = Path(self.config.analysis_dir) / f"module_statistics{self.dd_tag}.csv"
            stats_df.to_csv(stats_path, index=False)
            print(f"📊 Module statistics (absolute fluctuation version) saved to {stats_path}")
    
    def __len__(self) -> int:
        return self.norm_data.shape[0]

    def __getitem__(self, idx: int) -> dict:
        norm = self.norm_data[idx]
        if getattr(self, 'gene_mask_channel', 0):
            norm = torch.cat([norm, self.mask_data[idx]])  # [fluct, mask] dual channel
        return {
            "norm": norm,
            "modules": self.module_matrix[idx],
            "embedding": self.embedding_data[idx],
            "index": idx
        }
 
    def _analyze_activity_distribution(self):
        """Analyze the distribution of normalized_activity (memory-optimized version)"""
        total_elements = self.module_matrix.numel()
        
        # Memory control: use sampling for large datasets
        SAMPLE_SIZE = 50_000_000  # 50 million elements ≈ 200MB
        
        if total_elements > SAMPLE_SIZE:
            print(f"⚠️  Large dataset ({total_elements/1e9:.2f}B elements), using sampling")
            # Random sampling
            flat_view = self.module_matrix.cpu().numpy().ravel()
            rng = np.random.default_rng(42)
            indices = rng.choice(total_elements, size=SAMPLE_SIZE, replace=False)
            activity_data = flat_view[indices].copy()
            del flat_view
        else:
            activity_data = self.module_matrix.cpu().numpy().flatten()
        
        # Compute statistics
        stats_dict = {
            "Mean": np.mean(activity_data),
            "Median": np.median(activity_data),
            "Standard_Deviation": np.std(activity_data),
            "Min_Value": np.min(activity_data),
            "Max_Value": np.max(activity_data),
            "25th_Percentile": np.percentile(activity_data, 25),
            "75th_Percentile": np.percentile(activity_data, 75),
            "95th_Percentile": np.percentile(activity_data, 95),
            "99th_Percentile": np.percentile(activity_data, 99),
            "Skewness (Right-skew degree)": stats.skew(activity_data),
            "Zero_Value_Ratio": (activity_data == 0).sum() / len(activity_data) * 100,
            "Is_Sampled": total_elements > SAMPLE_SIZE,
            "Sample_Size": len(activity_data)
        }

        print("\n===== normalized_activity Distribution Statistics =====")
        for key, value in stats_dict.items():
            if "Ratio" in key or "Skewness" in key:
                print(f"{key}: {value:.2f}")
            elif key in ["Is_Sampled"]:
                print(f"{key}: {value}")
            elif "Size" in key:
                print(f"{key}: {value/1e6:.1f}M")
            else:
                print(f"{key}: {value:.4f}")
        print("="*70)

        # Plotting also uses sampled data (10 million is sufficient)
        PLOT_SAMPLE = 10_000_000
        if len(activity_data) > PLOT_SAMPLE:
            rng = np.random.default_rng(42)
            plot_idx = rng.choice(len(activity_data), size=PLOT_SAMPLE, replace=False)
            plot_data = activity_data[plot_idx]
        else:
            plot_data = activity_data

        # Plotting (reduce DPI to save memory)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].hist(plot_data, bins=100, color="#2E86AB", alpha=0.7, log=True)
        axes[0].axvline(stats_dict["Median"], color="red", linestyle="--")
        axes[0].axvline(stats_dict["Mean"], color="orange", linestyle="--")
        axes[0].set_title("Distribution Histogram (sampled)")
        
        axes[1].boxplot(plot_data, vert=False, widths=0.7, patch_artist=True,
                        boxprops=dict(facecolor="#A23B72", alpha=0.7))
        axes[1].set_title("Box Plot (sampled)")
        
        sorted_data = np.sort(plot_data)
        cumulative = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        axes[2].plot(sorted_data, cumulative * 100, color="#F18F01", linewidth=2)
        axes[2].set_title("Cumulative Distribution")
        
        save_dir = Path(self.config.analysis_dir)
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "normalized_activity_distribution.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")  # Reduce DPI
        plt.close()
        print(f"\n✅ Distribution plot saved to: {save_path}")

        stats_df = pd.DataFrame([stats_dict])
        stats_save_path = save_dir / "normalized_activity_stats.csv"
        stats_df.to_csv(stats_save_path, index=False, encoding="utf-8")
        print(f"✅ Distribution statistics saved to: {stats_save_path}")
        
        # Cleanup
        del activity_data, plot_data
        gc.collect()


# -------------------------- Multi-Token Cross-Attention Fusion --------------------------
class MultiTokenCrossAttentionFusion(nn.Module):
    """Genuine multi-token gene-module cross-attention.

    Avoids degenerate single-token attention (softmax over a length-1
    key sequence is identically 1, i.e. no attention at all).

    - Module side keeps ONE TOKEN PER GENE MODULE:
        token_i = activity_i * W_act + E_i
      where activity_i is the cell-level module activity scalar and E_i a learned
      module identity embedding -> [B, M, d]. The module axis is preserved.
    - Gene side produces n_query learned query tokens conditioned on the gene
      embedding -> [B, n_q, d].
    - Softmax runs over M (>1) module tokens, so attention weights are meaningful
      per-cell pathway importances and can be visualised / audited.
    Complexity: O(n_q * M * d), linear in the number of modules.
    """
    def __init__(self, embed_dim: int, n_modules: int, num_heads: int = 8,
                 n_query_tokens: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_query_tokens = n_query_tokens
        self.activity_proj = nn.Linear(1, embed_dim)
        self.module_embed = nn.Embedding(n_modules, embed_dim)
        nn.init.normal_(self.module_embed.weight, std=0.02)
        self.query_embed = nn.Parameter(torch.randn(1, n_query_tokens, embed_dim) * 0.02)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.last_attn = None  # [B, heads, n_query, M], detached, for inspection

    def forward(self, gene_emb: torch.Tensor, module_activity: torch.Tensor,
                detach_attn: bool = False) -> torch.Tensor:
        # module_activity: [B, M] raw per-module activity scores
        tokens = self.activity_proj(module_activity.unsqueeze(-1)) + self.module_embed.weight.unsqueeze(0)
        tokens = self.token_norm(tokens)                      # [B, M, d]
        q = gene_emb.unsqueeze(1) + self.query_embed          # [B, n_q, d]
        attn_out, attn_w = self.attn(q, tokens, tokens,
                                     need_weights=True, average_attn_weights=False)
        if detach_attn:
            # Coarse anchor path - attention output is detached so the coarse
            # (Harmony) loss cannot push attention weights toward a smoothed average.
            attn_out = attn_out.detach()
        else:
            self.last_attn = attn_w.detach()                  # [B, heads, n_q, M]
        x = self.norm1(gene_emb.unsqueeze(1) + attn_out)      # residual on gene query
        x = self.norm2(x + self.ffn(x))
        fused = x.mean(dim=1)                                 # pool query tokens -> [B, d]
        return fused

# -------------------------- Gate-Writeback Fusion (attention-as-gating) --------------------------
class GateWritebackFusion(nn.Module):
    """Per-cell, per-module sigmoid gating with FULL-RESOLUTION writeback.

    Single-variable change vs concat: the module activity vector is gated per cell
    BEFORE the module encoder; everything downstream is identical to concat mode.
    Design principles:
      - no multi-token mean-pool: gated activity keeps all M dims (no resolution
        loss);
      - sigmoid instead of softmax: gates are independent (no forced competition
        over a unit budget), so 'select nothing' / 'select many' are expressible;
      - gate path is detach-shielded from the coarse loss exactly like the
        attention path (see BioNet.forward), so gates are shaped by recon+denoise only.
    Gates are exported per cell as X_module_gate{suffix} (interpretability + entropy).
    """
    def __init__(self, embed_dim: int, n_modules: int, gate_dim: int = 64):
        super().__init__()
        self.gate_dim = gate_dim
        self.activity_proj = nn.Linear(1, gate_dim)
        self.module_embed = nn.Embedding(n_modules, gate_dim)
        nn.init.normal_(self.module_embed.weight, std=0.02)
        self.query_proj = nn.Linear(embed_dim, gate_dim)
        self.gate_bias = nn.Parameter(torch.zeros(n_modules))
        self.last_gate = None  # [B, M], detached, for inspection/export

    def forward(self, gene_emb: torch.Tensor, module_activity: torch.Tensor) -> torch.Tensor:
        # module_activity: [B, M] raw per-module activity scores
        tokens = self.activity_proj(module_activity.unsqueeze(-1)) + self.module_embed.weight.unsqueeze(0)  # [B, M, d]
        q = self.query_proj(gene_emb)                                              # [B, d]
        logits = (tokens * q.unsqueeze(1)).sum(-1) / (self.gate_dim ** 0.5) + self.gate_bias  # [B, M]
        w = torch.sigmoid(logits)
        self.last_gate = w.detach()
        return module_activity * w                                                 # [B, M] gated, full resolution

# -------------------------- Gate++ (gene-conditioned scale+shift FiLM) --------------------------
class GateScaleShiftFusion(nn.Module):
    """Gate++ — per-cell FiLM (scale + shift) on the module vector.

    Strict generalization of the pure selection gate:
      module' = gamma(gene_emb) * activity + beta(gene_emb)
    - gamma = 1 + tanh(z): starts at identity (~= concat), learns per-module
      amplification/suppression (the pure gate is the beta=0 special case);
    - beta (zero-initialised): the genuinely new capacity — IMPUTATION.
      The model can inject inferred pathway activity where measured activity
      is zero (dropout). concat and pure gating cannot do this in principle;
      the masked-gene denoise objective directly rewards it.
    Full-resolution writeback; detach-shielded from the coarse loss (see
    BioNet.forward), so gamma/beta are shaped by recon+denoise only.
    Diagnostics: last_gamma / last_beta exported per cell.
    """
    def __init__(self, embed_dim: int, n_modules: int, beta_scale: float = 1.0):
        super().__init__()
        self.n_modules = n_modules
        self.beta_scale = beta_scale
        self.gamma_proj = nn.Linear(embed_dim, n_modules)   # rows ~ module identity
        self.beta_proj = nn.Linear(embed_dim, n_modules)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        self.last_gamma = None  # [B, M], detached
        self.last_beta = None   # [B, M], detached

    def forward(self, gene_emb: torch.Tensor, module_activity: torch.Tensor) -> torch.Tensor:
        gamma = 1.0 + torch.tanh(self.gamma_proj(gene_emb))          # [B, M] in (0, 2)
        beta = torch.tanh(self.beta_proj(gene_emb)) * self.beta_scale  # [B, M] in (-s, s)
        self.last_gamma = gamma.detach()
        self.last_beta = beta.detach()
        return gamma * module_activity + beta

# -------------------------- Typed-Expert Routing (MoE over knowledge sources) --------------------------
class TypedExpertFusion(nn.Module):
    """Per-cell sparse ROUTING over prior-typed experts (not mixing).

    Motivation: the two knowledge lines fail on DIFFERENT cell types, so simply
    MIXING the lines cannot exploit their complementarity; the fusion must
    route per cell.

    - Experts: one per prior-knowledge group (KEGG/REACTOME/WP/BIOCARTA/PID/CGP)
      + one gene-line expert (data-driven knowledge). Each module expert sees
      ONLY its own group's activity slice.
    - Router: input [gene_emb, group-mean activities], top-k sparse softmax
      (k=2 default) over the experts.
    - Router probabilities are detach-shielded from the coarse loss: the router
      is shaped by recon+denoise only; experts/trunk still receive the coarse
      gradient.
    - Switch-style load-balancing aux loss prevents expert collapse.
    Diagnostics: last_router exported per cell as X_expert_weights{suffix}.
    """
    def __init__(self, embed_dim: int, group_idx: torch.LongTensor, n_groups: int, top_k: int = 2,
                 expert_depth: int = 1, shared_expert: bool = False):
        super().__init__()
        self.n_groups = n_groups
        self.top_k = top_k
        self.shared_expert = shared_expert
        self.register_buffer('group_idx', group_idx)  # [M] values 0..n_groups-1
        sizes = torch.bincount(group_idx, minlength=n_groups).tolist()
        def make_expert(s):
            layers = [nn.Linear(s, embed_dim), nn.LayerNorm(embed_dim), nn.LeakyReLU(0.2)]
            if expert_depth >= 2:  # deeper expert option
                layers += [nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.LeakyReLU(0.2)]
            return nn.Sequential(*layers)
        self.experts = nn.ModuleList([make_expert(s) for s in sizes])
        # Shared (generalist) expert - sees ALL modules with a joint 2-layer MLP.
        # Carries cross-group joint features that typed experts structurally cannot
        # express (e.g. combinatorial signatures spanning multiple knowledge sources).
        if shared_expert:
            M = group_idx.shape[0]
            self.shared = nn.Sequential(
                nn.Linear(M, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                nn.Linear(512, embed_dim), nn.LayerNorm(embed_dim), nn.LeakyReLU(0.2))
        n_exp_out = n_groups + 1 + (1 if shared_expert else 0)  # typed + gene (+ shared)
        self.router = nn.Linear(embed_dim + n_groups, n_exp_out)
        self.last_router = None  # [B, n_exp_out] detached
        self.last_lb = None      # load-balancing aux loss (with grad)

    def forward(self, gene_emb: torch.Tensor, module_activity: torch.Tensor,
                detach_router: bool = False) -> torch.Tensor:
        z_list = [expert(module_activity[:, self.group_idx == k])
                  for k, expert in enumerate(self.experts)]
        if self.shared_expert:
            z_list.append(self.shared(module_activity))      # generalist: full joint view
        Z = torch.stack(z_list, dim=1)                                    # [B, K(+1), D]
        group_means = torch.stack(
            [module_activity[:, self.group_idx == k].mean(1) for k in range(self.n_groups)], dim=1)
        logits = self.router(torch.cat([gene_emb, group_means], dim=-1))  # [B, n_exp]
        topv, topi = logits.topk(self.top_k, dim=-1)
        r = torch.zeros_like(logits).scatter(-1, topi, F.softmax(topv, dim=-1))
        if detach_router:
            r = r.detach()
        probs = F.softmax(logits, dim=-1)
        f = (r > 0).float().mean(0)
        self.last_lb = (f * probs.mean(0)).sum() * logits.shape[1]
        self.last_router = r.detach()
        n_typed = Z.shape[1]
        fused = (Z * r[:, :n_typed, None]).sum(1) + r[:, n_typed:] * gene_emb
        return fused                                                          # [B, D]

# -------------------------- Routed-Gate Fusion (typed routing x full-resolution writeback) --------------------------
class RoutedGateFusion(nn.Module):
    """Unify full-resolution gating with typed routing.

    Combining expert outputs by weighted SUM is still a form of mixing, while
    thousands of per-module sigmoid gates can be unstable. This block therefore
    routes at the GROUP level and writes back at full resolution: few, typed,
    stable gates.

    - 8 prior-typed module groups (KEGG/REACTOME/WP/BIOCARTA/PID +
      CGP_UP/CGP_DN/CGP_OTH, perturbation-direction split) + 1 gene expert.
    - Router: [gene_emb, group-mean activities] -> top-k sparse softmax.
    - Writeback: module' = group_weight (scattered router weights) * activity;
      gene_emb scaled by r_gene. Trunk = concat trunk (no output mixing).
    - Router detach-shielded from coarse loss; Switch-style lb aux (anti-collapse).
    Diagnostics: last_router exported per cell as X_expert_weights{suffix}.
    """
    def __init__(self, embed_dim: int, group_idx: torch.LongTensor, n_groups: int, top_k: int = 2):
        super().__init__()
        self.n_groups = n_groups
        self.top_k = top_k
        self.register_buffer('group_idx', group_idx)                    # [M]
        onehot = torch.zeros(n_groups, group_idx.shape[0])
        onehot.scatter_(0, group_idx.unsqueeze(0), 1.0)
        self.register_buffer('group_onehot', onehot)                    # [K, M]
        self.router = nn.Linear(embed_dim + n_groups, n_groups + 1)     # +1 = gene
        self.last_router = None
        self.last_lb = None

    def forward(self, gene_emb: torch.Tensor, module_activity: torch.Tensor,
                detach_router: bool = False) -> torch.Tensor:
        group_means = torch.stack(
            [module_activity[:, self.group_idx == k].mean(1) for k in range(self.n_groups)], dim=1)
        logits = self.router(torch.cat([gene_emb, group_means], dim=-1))  # [B, K+1]
        topv, topi = logits.topk(self.top_k, dim=-1)
        r = torch.zeros_like(logits).scatter(-1, topi, F.softmax(topv, dim=-1))
        if detach_router:
            r = r.detach()
        probs = F.softmax(logits, dim=-1)
        self.last_lb = ((r > 0).float().mean(0) * probs.mean(0)).sum() * (self.n_groups + 1)
        self.last_router = r.detach()
        gw = (r[:, :self.n_groups, None] * self.group_onehot[None]).sum(1)  # [B, M]
        module_gated = module_activity * gw
        return module_gated, r[:, self.n_groups:]                             # ([B,M], [B,1])

# -------------------------- Core Network Model (with ablation switches) --------------------------
class BioNet(nn.Module):
    """DeepFusion model. fusion_mode selects the fusion block:
    - 'cross_attention' : MultiTokenCrossAttentionFusion (full model)
    - 'concat'          : concatenation + MLP          (ablation: w/o cross-attention)
    - 'gated'           : gated residual fusion        (fallback architecture)
    - 'no_module'       : gene encoder only            (ablation: w/o module branch)
    Decoder is LINEAR (no Sigmoid): module activities are log1p-normalised and
    unbounded, so the reconstruction target is now reachable (no value-range
    mismatch).
    """
    def __init__(self,
                 input_dim: int,
                 module_dim: int,
                 embedding_dim: int,
                 params: dict):
        super().__init__()
        num_heads = params.get('num_heads', 8)
        n_query_tokens = params.get('n_query_tokens', 4)
        self.fusion_mode = params.get('fusion_mode', 'moe')
        # detach attention output in the coarse anchor path (1) or not (0)
        self.attn_detach_coarse = params.get('attn_detach_coarse', 1)
        adjusted_embed_dim = params['embedding_dim'] - (params['embedding_dim'] % num_heads)
        self.embedding_dim = adjusted_embed_dim
        self.hidden_dim = params['hidden_dim']
        self.dropout = params['dropout']

        self.gene_encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.LeakyReLU(0.2)
        )

        # Used by concat / gated ablation modes (and reported as X_mod)
        self.module_encoder = nn.Sequential(
            nn.Linear(module_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout)
        )

        # Static size-trust scaling (init = exact identity)
        if params.get('size_trust', 0):
            self.register_buffer('module_log_sizes', params['module_log_sizes'].float())
            self.size_gamma = nn.Parameter(torch.zeros(1))
            self.size_bias = nn.Parameter(torch.zeros(module_dim))

        if self.fusion_mode == 'cross_attention':
            self.cross_fusion = MultiTokenCrossAttentionFusion(
                embed_dim=self.embedding_dim,
                n_modules=module_dim,
                num_heads=num_heads,
                n_query_tokens=n_query_tokens,
                dropout=self.dropout
            )
        elif self.fusion_mode == 'concat':
            self.concat_fusion = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.LeakyReLU(0.2)
            )
        elif self.fusion_mode == 'gate_attn':
            # gate-writeback. Trunk identical to concat; only the gate is new.
            self.gate_fusion = GateWritebackFusion(
                embed_dim=self.embedding_dim,
                n_modules=module_dim,
                gate_dim=params.get('gate_dim', 64)
            )
            self.concat_fusion = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.LeakyReLU(0.2)
            )
        elif self.fusion_mode == 'gate_film':
            # gate++ scale+shift FiLM. Trunk identical to concat; only the
            # conditioning heads are new.
            self.gate_film = GateScaleShiftFusion(
                embed_dim=self.embedding_dim,
                n_modules=module_dim,
                beta_scale=params.get('beta_scale', 1.0)
            )
            self.concat_fusion = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.LeakyReLU(0.2)
            )
        elif self.fusion_mode == 'moe':
            # typed-expert routing. group_idx threaded from data pipeline.
            self.moe_fusion = TypedExpertFusion(
                embed_dim=self.embedding_dim,
                group_idx=params['module_group_idx'],
                n_groups=params['n_module_groups'],
                top_k=params.get('top_k', 2)
            )
        elif self.fusion_mode in ('moe_updn', 'moe_deep'):
            # typed routing + CGP UP/DN expert split (moe_updn) + deeper experts (moe_deep)
            self.moe_fusion = TypedExpertFusion(
                embed_dim=self.embedding_dim,
                group_idx=params['module_group_idx'],
                n_groups=params['n_module_groups'],
                top_k=params.get('top_k', 2),
                expert_depth=params.get('expert_depth', 1)
            )
        elif self.fusion_mode == 'moe_shared':
            # typed experts + shared generalist expert (joint view)
            self.moe_fusion = TypedExpertFusion(
                embed_dim=self.embedding_dim,
                group_idx=params['module_group_idx'],
                n_groups=params['n_module_groups'],
                top_k=params.get('top_k', 2),
                shared_expert=True
            )
        elif self.fusion_mode == 'moe2':
            # routed-gate full-resolution writeback (typed json with UP/DN split)
            self.moe2_fusion = RoutedGateFusion(
                embed_dim=self.embedding_dim,
                group_idx=params['module_group_idx'],
                n_groups=params['n_module_groups'],
                top_k=params.get('top_k', 2)
            )
            self.concat_fusion = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.LeakyReLU(0.2)
            )
        elif self.fusion_mode == 'gated':
            self.gate = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        elif self.fusion_mode != 'no_module':
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        # Shared output block for non-attention modes
        self.out_norm = nn.LayerNorm(self.embedding_dim)
        self.out_ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        )

        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Tanh()
        )

        # Dedicated Harmony-alignment head for the BIO branch (gene encoder).
        # The coarse cosine loss attaches here (coarse_target='bio'), leaving the
        # fused pathway free to be shaped by recon + denoise only. Coarse signal
        # still reaches the fused output indirectly via the residual connection
        # on the gene query inside the fusion block.
        self.bio_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Tanh()
        )

        # Per-gene sigmoid gate (single-gene express lane, mask-aware).
        # Query comes from the module side (mod_emb): pathway state guides which
        # individual genes to trust. Full-resolution writeback on the fluctuation
        # channel only (mask channel stays untouched metadata). Detach-shielded
        # from the coarse loss (anchor path uses w.detach()).
        if params.get('gene_gate', 0):
            n_genes = params['n_genes']
            g_dim = params.get('gate_dim', 64)
            self.gene_gate_embed = nn.Embedding(n_genes, g_dim)
            nn.init.normal_(self.gene_gate_embed.weight, std=0.02)
            self.gene_gate_query = nn.Linear(self.embedding_dim, g_dim)
            self.gene_gate_bias = nn.Parameter(torch.zeros(n_genes))
            self.gene_gate_dim = g_dim
            self.n_genes = n_genes
            self.last_gene_gate = None

        # Linear decoder head: target is log1p-normalised activity (unbounded, >= 0)
        self.decoder = nn.Linear(self.embedding_dim, module_dim)

        # Gene-level decoder for the masked-reconstruction (denoising) objective.
        # Reconstructing a dropped gene's value requires retrieving the modules
        # containing it via cross-attention - module averaging scores poorly on this
        # task, so selective retrieval becomes the optimal strategy.
        self.gene_decoder = nn.Linear(self.embedding_dim, input_dim)

    def forward(self, gene_input: torch.Tensor, module_input: torch.Tensor) -> tuple:
        # Static size-trust scaling of module activities.
        # trust_i = 1 + tanh(gamma*log(n_i) + b_i), init gamma=b=0 -> exact identity.
        # The anchor (coarse-facing) path uses trust.detach(): the coarse gradient
        # never touches the trust parameters (shaped by recon+denoise only).
        if getattr(self, 'size_gamma', None) is not None:
            trust = 1.0 + torch.tanh(self.size_gamma * self.module_log_sizes + self.size_bias)
            module_main = module_input * trust
            module_anchor = module_input * trust.detach()
        else:
            module_main = module_anchor = module_input

        mod_emb = self.module_encoder(module_main)
        mod_emb_anchor = (self.module_encoder(module_anchor)
                          if getattr(self, 'size_gamma', None) is not None else mod_emb)

        # Per-gene sigmoid gate (single-gene express lane). Query from the
        # module side (mod_emb): pathway state guides which genes to trust. Gate
        # applies to the fluctuation channel only; mask channel stays metadata.
        if getattr(self, 'gene_gate_embed', None) is not None:
            q = self.gene_gate_query(mod_emb)                                   # [B, d]
            logits = ((self.gene_gate_embed.weight * q.unsqueeze(1)).sum(-1)
                      / (self.gene_gate_dim ** 0.5) + self.gene_gate_bias)      # [B, G]
            w = torch.sigmoid(logits)
            self.last_gene_gate = w.detach()
            G = self.n_genes
            fluct, rest = gene_input[:, :G], gene_input[:, G:]
            gene_in_main = torch.cat([fluct * w, rest], dim=1) if rest.shape[1] else fluct * w
            gene_in_anchor = (torch.cat([fluct * w.detach(), rest], dim=1)
                              if rest.shape[1] else fluct * w.detach())
        else:
            gene_in_main = gene_in_anchor = gene_input

        gene_emb = self.gene_encoder(gene_in_main)
        gene_emb_anchor = (self.gene_encoder(gene_in_anchor)
                           if getattr(self, 'gene_gate_embed', None) is not None else gene_emb)

        if self.fusion_mode == 'cross_attention':
            fused_emb = self.cross_fusion(gene_emb, module_main)
            if self.attn_detach_coarse:
                # Separate anchor path for the coarse loss - identical values,
                # but attention output is detached (gradient isolated from attention)
                fused_anchor = self.cross_fusion(gene_emb_anchor, module_anchor, detach_attn=True)
            else:
                fused_anchor = fused_emb
        elif self.fusion_mode == 'concat':
            fused_emb = self.concat_fusion(torch.cat([gene_emb, mod_emb], dim=-1))
            fused_anchor = self.concat_fusion(torch.cat([gene_emb_anchor, mod_emb_anchor], dim=-1))
        elif self.fusion_mode == 'gate_attn':
            # Gate the activity, then run the concat trunk. In the coarse
            # anchor path the gate output is detached (same shield as the
            # attention path): gates are shaped by recon+denoise only, never smoothed
            # by the coarse objective.
            gated_activity = self.gate_fusion(gene_emb, module_main)
            fused_emb = self.concat_fusion(
                torch.cat([gene_emb, self.module_encoder(gated_activity)], dim=-1))
            if self.attn_detach_coarse:
                fused_anchor = self.concat_fusion(
                    torch.cat([gene_emb, self.module_encoder(gated_activity.detach())], dim=-1))
            else:
                fused_anchor = fused_emb
        elif self.fusion_mode == 'gate_film':
            # Scale+shift condition the activity, then run the concat trunk.
            # Same detach shield: conditioning heads are shaped by recon+denoise only.
            mod_activity = self.gate_film(gene_emb, module_main)
            fused_emb = self.concat_fusion(
                torch.cat([gene_emb, self.module_encoder(mod_activity)], dim=-1))
            if self.attn_detach_coarse:
                fused_anchor = self.concat_fusion(
                    torch.cat([gene_emb, self.module_encoder(mod_activity.detach())], dim=-1))
            else:
                fused_anchor = fused_emb
        elif self.fusion_mode == 'moe2':
            # Group-gated full-resolution writeback through the concat trunk.
            # Anchor path detaches router weights (router shaped by recon+denoise only).
            module_gated, r_gene = self.moe2_fusion(gene_emb, module_main)
            fused_emb = self.concat_fusion(
                torch.cat([gene_emb * r_gene, self.module_encoder(module_gated)], dim=-1))
            if self.attn_detach_coarse:
                module_gated_a, r_gene_a = self.moe2_fusion(gene_emb, module_anchor, detach_router=True)
                fused_anchor = self.concat_fusion(
                    torch.cat([gene_emb * r_gene_a, self.module_encoder(module_gated_a)], dim=-1))
            else:
                fused_anchor = fused_emb
        elif self.fusion_mode in ('moe', 'moe_updn', 'moe_deep', 'moe_shared'):
            # Expert routing. Anchor path detaches router probs only
            # (experts/trunk still receive the coarse gradient, as for the trunk).
            fused_emb = self.moe_fusion(gene_emb, module_main)
            fused_emb = self.out_norm(fused_emb)
            fused_emb = self.out_norm(fused_emb + self.out_ffn(fused_emb))
            if self.attn_detach_coarse:
                fused_anchor = self.moe_fusion(gene_emb_anchor, module_anchor, detach_router=True)
                fused_anchor = self.out_norm(fused_anchor)
                fused_anchor = self.out_norm(fused_anchor + self.out_ffn(fused_anchor))
            else:
                fused_anchor = fused_emb
        elif self.fusion_mode == 'gated':
            gate = torch.sigmoid(self.gate(torch.cat([gene_emb, mod_emb], dim=-1)))
            fused_emb = self.out_norm(gene_emb + gate * mod_emb)
            fused_emb = self.out_norm(fused_emb + self.out_ffn(fused_emb))
            fused_anchor = fused_emb
        else:  # no_module
            fused_emb = self.out_norm(gene_emb)
            fused_emb = self.out_norm(fused_emb + self.out_ffn(fused_emb))
            fused_anchor = fused_emb

        pred_embedding = self.projection(fused_anchor)  # coarse loss attaches here (trained)
        bio_pred = self.bio_projection(gene_emb)  # coarse anchor for the bio branch
        recon = self.decoder(fused_emb)
        gene_recon = self.gene_decoder(fused_emb)

        return gene_emb, mod_emb, fused_emb, pred_embedding, bio_pred, recon, gene_recon

# -------------------------- Hyperparameter Optimization Class (Supports Dynamic Embedding) --------------------------
class HyperOptimizer:
    def __init__(self, config: Config, current_embedding_metric: str):
        self.config = config
        self.current_embedding_metric = current_embedding_metric  # Current embedding
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        self.alpha_records = []  
        self._init_analysis_dir() 
        
        self.cached_dataset = None
        self.cached_adata = None
        self.cached_gene_modules = None
        
        if self.config.enable_caching:
            print("🚀 Initializing hyperoptimizer with dataset caching...")
            processor = SCDataProcessor(self.config)
            try:
                self.cached_adata, self.cached_gene_modules = processor.load_data()
                self.cached_dataset = SCNDataset(
                    self.cached_adata, 
                    self.cached_gene_modules, 
                    self.config,
                    self.current_embedding_metric  # Pass current embedding
                )
                print(f"✅ Dataset cached ({self.cached_dataset.module_matrix.shape[0]} cells, "
                      f"{self.cached_dataset.module_matrix.shape[1]} modules)")
            except Exception as e:
                print(f"⚠️ Dataset caching failed: {str(e)}")
                traceback.print_exc()
        else:
            print("⚠️ Dataset caching disabled - will reload for each trial")
        
    def _init_analysis_dir(self):
        (Path(self.config.analysis_dir)/"details").mkdir(parents=True, exist_ok=True)

    def objective(self, trial: optuna.Trial) -> float:
        params = {
            'embedding_dim': trial.suggest_int('embedding_dim', *self.config.embedding_dim_range),
            'hidden_dim': trial.suggest_int('hidden_dim', *self.config.hidden_dim_range),
            'dropout': trial.suggest_float('dropout', *self.config.dropout_range),
            'batch_size': trial.suggest_int('batch_size', *self.config.batch_size_range),
            'lr': trial.suggest_float('lr', *self.config.lr_range, log=True),
            'weight_decay': trial.suggest_float('weight_decay', *self.config.weight_decay_range, log=True),
        }
        
        params['alpha'] = trial.suggest_categorical(
            'alpha', 
            self.config.alpha_values if self.config.auto_alpha else [self.config.fixed_alpha]
        )
            
        if self.cached_dataset is not None:
            dataset = self.cached_dataset
            print(f"♻️ Using cached dataset for trial {trial.number}")
        else:
            print(f"♻️ Loading dataset for trial {trial.number}")
            processor = SCDataProcessor(self.config)
            adata, gene_modules = processor.load_data()
            dataset = SCNDataset(adata, gene_modules, self.config, self.current_embedding_metric)
        
        kfold = KFold(n_splits=self.config.k_folds, shuffle=True)
        fold_similarities = []
    
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            model = BioNet(
                input_dim=dataset[0]["norm"].shape[0],
                module_dim=len(dataset.valid_modules),
                embedding_dim=dataset.embedding_dim,
                params=params
            ).to(self.config.device)
        
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )
        
            best_sim = self.train_fold(
                model, optimizer,
                Subset(dataset, train_idx),
                Subset(dataset, val_idx),
                params
            )
            fold_similarities.append(best_sim)        

            self._save_fold_details(trial.number, fold, params, best_sim)

        trial.set_user_attr("full_fold_similarities", fold_similarities)
        mean_sim = np.mean(fold_similarities)
    
        record = {
            'trial': trial.number,
            'mean_similarity': mean_sim,
            'fold_similarities': fold_similarities,
            'alpha': params['alpha'],
            'hidden_dim': params['hidden_dim'],
            'lr': params['lr'],
            **params
        }
        self.alpha_records.append(record)
        
        return mean_sim

    def train_fold(self, model, optimizer, train_set, val_set, params):
        best_sim = 0.0
        train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=params['batch_size'], drop_last=True)

        for epoch in range(self.config.epochs):
            model.train()
            for batch in train_loader:
                gene_input = batch["norm"].to(self.config.device)
                module_input = batch["modules"].to(self.config.device)
                target_embedding = batch["embedding"].to(self.config.device)

                optimizer.zero_grad()
                _, _, _, pred_embedding, _, recon, _ = model(gene_input, module_input)
                
                cos_loss = 1 - F.cosine_similarity(pred_embedding, target_embedding).mean()
                recon_loss = F.mse_loss(recon, module_input)
                loss = params['alpha'] * cos_loss + (1 - params['alpha']) * recon_loss
                
                loss.backward()
                optimizer.step()

            val_sim = self._validate(model, val_loader)
            best_sim = max(best_sim, val_sim)
            
        return best_sim

    def _validate(self, model, val_loader):
        model.eval()
        total_sim = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                gene_input = batch["norm"].to(self.config.device)
                module_input = batch["modules"].to(self.config.device)
                target_embedding = batch["embedding"].to(self.config.device)
                
                _, _, _, pred_embedding, _, _, _ = model(gene_input, module_input)
                total_sim += F.cosine_similarity(pred_embedding, target_embedding).mean().item()
                
        return total_sim / len(val_loader)

    def generate_alpha_analysis(self):
        if not self.alpha_records:
            print("⚠️ No alpha analysis data available")
            return
        
        df = pd.DataFrame(self.alpha_records)
        required_cols = ['alpha', 'mean_similarity', 'hidden_dim', 'lr']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
            return
            
        # Add embedding identifier
        df['embedding_metric'] = self.current_embedding_metric
        df.to_csv(Path(self.config.analysis_dir)/f"alpha_trials_{self.current_embedding_metric.replace('/', '_')}.csv", index=False)
        print(f"✅ Alpha analysis saved to {self.config.analysis_dir}")

    def _save_fold_details(self, trial_num, fold_num, params, similarity):
        detail = {
            'trial': trial_num,
            'fold': fold_num,
            'similarity': similarity,
            'embedding_metric': self.current_embedding_metric,
            **params
        }
        details_dir = Path(self.config.analysis_dir) / "details"
        csv_path = details_dir / f"trial_{trial_num}_fold_{fold_num}_{self.current_embedding_metric.replace('/', '_')}.csv"
        details_dir.mkdir(parents=True, exist_ok=True)
        
        if not csv_path.exists():
            pd.DataFrame([detail]).to_csv(csv_path, index=False)
        else:
            pd.DataFrame([detail]).to_csv(csv_path, mode='a', header=False, index=False)
        
    def optimize(self):
        study_path = self.config.study_storage
        os.makedirs(self.config.cache_dir, exist_ok=True)
        storage_url = f"sqlite:///{study_path}"
        study_name = f"{self.config.study_name}_{self.current_embedding_metric.replace('/', '_')}"

        try:
            if study_path.exists():
                print(f"♻️ Loading existing study: {storage_url} (name: {study_name})")
                try:
                    self.study = optuna.load_study(
                        study_name=study_name,
                        storage=storage_url
                    )
                except KeyError as e:
                    print(f"⚠️ Study format incompatible, creating new: {str(e)}")
                    self.study = optuna.create_study(
                        study_name=study_name,
                        storage=storage_url,
                        direction="maximize",
                        sampler=optuna.samplers.TPESampler(seed=42)
                    )
            else:
                self.study = optuna.create_study(
                    study_name=study_name,
                    storage=storage_url,
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=42)
                )
        except Exception as e:
            print(f"⚠️ Study loading failed: {str(e)}")
            traceback.print_exc()
            self.study = optuna.create_study(direction="maximize")

        print(f"✅ Current study status: {len(self.study.trials)} trials ({len(self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))} completed)")
        remaining = self.config.n_trials - len(self.study.trials)
        if remaining > 0:
            print(f"🔍 Running additional {remaining} trials for {self.current_embedding_metric}")
            try:
                self.study.optimize(self.objective, n_trials=remaining)
            except Exception as e:
                print(f"⚠️ Optimization interrupted: {str(e)}")
                traceback.print_exc()

        self.generate_alpha_analysis()
        print(f"💾 Study data automatically saved to: {storage_url}")

        return self.study.best_params

# -------------------------- Dual-level objective components --------------------------
def info_nce_loss(emb: torch.Tensor, idx: torch.Tensor, pos_sets: torch.Tensor, tau: float) -> torch.Tensor:
    """Multi-positive InfoNCE on the output embedding.
    emb: [B, d] embeddings; idx: [B] global cell indices (same order as emb rows);
    pos_sets: [N, k] global positive sets. Positives = two-level kNN neighbors
    present among idx; negatives = all other in-candidate cells."""
    z = F.normalize(emb, dim=1)
    sim = (z @ z.t()) / tau                       # [B, B]
    sim.fill_diagonal_(float('-inf'))
    pos = pos_sets[idx]                           # [B, k]
    pos_mask = (pos.unsqueeze(1) == idx.view(1, -1, 1)).any(-1)  # [B, B]
    # exclude self-matches defensively: a row whose ONLY positive is itself would
    # produce -inf log-positive and poison the mean
    pos_mask = pos_mask & ~torch.eye(emb.size(0), dtype=torch.bool, device=emb.device)
    valid = pos_mask.any(dim=1)
    if valid.sum() == 0:
        return emb.new_zeros(())
    sim_pos = sim.masked_fill(~pos_mask, float('-inf'))
    loss = -(torch.logsumexp(sim_pos, dim=1) - torch.logsumexp(sim, dim=1))
    # normalize by log(#candidates): random chance ~= 1.0 regardless of candidate
    # count, so train batches and full-val evaluation share the same scale and the
    # loss does not numerically swamp L_coarse / L_recon
    return loss[valid].mean() / np.log(emb.size(0))

class NeighborhoodSampler(Sampler):
    """Graph-aware batch sampler for dual-level training.
    In-batch InfoNCE with random batches would give ~0 positives per anchor
    (k=15 positives among ~14k cells). Each batch instead contains anchors plus
    a few of their (train-restricted) two-level neighbors, guaranteeing positive
    pairs while keeping negatives in-batch."""
    def __init__(self, train_idx, pos_sets: torch.Tensor, batch_size: int = 128,
                 n_pos: int = 3, seed: int = 42):
        # Force OWNING, contiguous int64 copies. Views into torch storage can
        # dangle nondeterministically (observed garbage indices from reused memory).
        self.anchors = np.array(train_idx, dtype=np.int64, copy=True)
        self.pos_sets = np.array(pos_sets.cpu().numpy(), dtype=np.int64, copy=True)
        if self.pos_sets.size and (self.pos_sets.min() < 0 or self.pos_sets.max() >= self.pos_sets.shape[0]):
            raise ValueError("pos_sets contains out-of-range indices")
        self.n_anchor = max(1, batch_size // (n_pos + 1))
        self.n_pos = n_pos
        self.seed = seed
        self.train_mask = np.zeros(self.pos_sets.shape[0], dtype=bool)
        self.train_mask[self.anchors] = True

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        anchors = rng.permutation(self.anchors)
        for i in range(0, len(anchors), self.n_anchor):
            batch = []
            for a in anchors[i:i + self.n_anchor]:
                batch.append(int(a))
                pos = self.pos_sets[a]
                pos = pos[self.train_mask[pos]]
                if len(pos) > 0:
                    batch.extend(rng.choice(pos, size=min(self.n_pos, len(pos)), replace=False).tolist())
            yield batch

    def __len__(self):
        return int(np.ceil(len(self.anchors) / self.n_anchor))

# -------------------------- Early Stopping Class --------------------------
class EarlyStopping:
    def __init__(self, patience=15, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_sim = 0.0
        self.best_epoch = 0

    def __call__(self, val_sim, epoch):
        if self.best_score is None:
            self.best_score = val_sim
            self.best_sim = val_sim
            self.best_epoch = epoch
        elif val_sim < self.best_score + self.delta:
            self.counter += 1
            print(f"⚠️ EarlyStopping counter: {self.counter}/{self.patience} (Best Sim: {self.best_sim:.4f} @ Epoch {self.best_epoch})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_sim
            self.best_sim = val_sim
            self.best_epoch = epoch
            self.counter = 0
        
        return self.early_stop

# -------------------------- Trainer Class (single-split transductive protocol) --------------------------
class Trainer:
    """Training protocol.

    Replaces the legacy 10-fold / fold-1-inference scheme, in which ~90% of all
    cells were inside the final model's training set (evaluation leakage), with
    the standard transductive protocol used by scVI/Harmony: train on all cells,
    hold out a small random split ONLY for early stopping / best-checkpoint
    selection, then embed all cells with the best checkpoint.
    Best checkpoints are stored via copy.deepcopy (fixes the stale-reference bug
    where `best_model = model.state_dict()` kept being updated by later epochs).
    """
    def __init__(self, config: Config, params: dict, current_embedding_metric: str):
        config.validate_params(params)
        self.config = config
        self.params = params
        self.current_embedding_metric = current_embedding_metric
        self.suffix = config.get_output_suffix(current_embedding_metric)  # includes ablation tag
        dir_name = config.get_dir_name(current_embedding_metric)          # tag-separated dirs

        self.model_dir = Path(config.model_dir) / dir_name
        self.results_dir = Path(config.results_dir) / dir_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "best_model.pth"
        self.history_path = self.model_dir / "history.pth"

        processor = SCDataProcessor(config)
        self.adata, self.gene_modules = processor.load_data()
        self.dataset = SCNDataset(self.adata, self.gene_modules, config, current_embedding_metric)

        self.best_state = None
        self.history = None
        self.train_seconds = 0.0
        self.inference_seconds = 0.0

    def _init_model(self):
        params = dict(self.params)
        if params.get('gene_gate', 0):
            params['n_genes'] = self.dataset.norm_data.shape[1]  # single-channel gene count
            print(f"🧬 Gene gate enabled: {params['n_genes']} per-gene gates")
        if params.get('size_trust', 0):
            # per-module log(size) for the trust scale, aligned to valid_modules order
            import math as _math
            sizes = [self.dataset.valid_modules[m]['n_genes'] for m in self.dataset.valid_modules.keys()]
            params['module_log_sizes'] = torch.tensor([_math.log(max(s, 1)) for s in sizes])
            print(f"📏 Size-trust enabled: {len(sizes)} modules, size range [{min(sizes)}, {max(sizes)}]")
        if params.get('fusion_mode') in ('moe', 'moe_shared'):
            # build prior-typed expert grouping aligned to valid_modules order
            import json
            gmap_path = Path(self.config.sc_gene_label_path).parent / 'module_groups_v2026.json'
            with open(gmap_path) as f:
                gmap = json.load(f)
            order = list(self.dataset.valid_modules.keys())
            group_names = sorted({gmap[m] for m in order})
            g2i = {g: i for i, g in enumerate(group_names)}
            params['module_group_idx'] = torch.tensor([g2i[gmap[m]] for m in order], dtype=torch.long)
            params['n_module_groups'] = len(group_names)
            print(f"🧬 MoE experts: {len(group_names)} groups {group_names} + 1 gene expert")
        elif params.get('fusion_mode') in ('moe2', 'moe_updn', 'moe_deep'):
            # typed grouping with CGP UP/DN split
            import json
            gmap_path = Path(self.config.sc_gene_label_path).parent / 'module_groups_v2026_typed.json'
            with open(gmap_path) as f:
                gmap = json.load(f)
            order = list(self.dataset.valid_modules.keys())
            group_names = sorted({gmap[m] for m in order})
            g2i = {g: i for i, g in enumerate(group_names)}
            params['module_group_idx'] = torch.tensor([g2i[gmap[m]] for m in order], dtype=torch.long)
            params['n_module_groups'] = len(group_names)
            print(f"🧬 Routed-gate groups: {len(group_names)} groups {group_names} + 1 gene expert")
            if params.get('fusion_mode') == 'moe_deep':
                params['expert_depth'] = 2
                print("🧬 moe_deep: experts are 2-layer MLPs")
        model = BioNet(
            input_dim=self.dataset[0]["norm"].shape[0],
            module_dim=len(self.dataset.valid_modules),
            embedding_dim=self.dataset.embedding_dim,
            params=params
        ).to(self.config.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.params['lr'],
            weight_decay=self.params['weight_decay']
        )
        return model, optimizer

    def train(self, reload: bool = False):
        print(f"\n=== DeepFusion Training (single-split transductive) for {self.current_embedding_metric} ===")
        print(f"📂 Model save directory: {self.model_dir}")
        print(f"⚙️ EarlyStopping: patience={self.config.patience}, delta={self.config.delta}")
        print(f"🧩 Fusion mode: {self.params.get('fusion_mode', 'moe')} | "
              f"alpha={self.params['alpha']} | coarse_target={getattr(self.config, 'coarse_target', 'bio')} | suffix='{self.suffix}'")

        if not reload and self.model_path.exists() and self.history_path.exists():
            print(f"♻️ Found completed model, loading: {self.model_path}")
            self.load_best_model()
            self.history = torch.load(self.history_path, map_location='cpu', weights_only=False)
            print(f"✅ Loaded | Best Val Sim: {self.history['best_sim']:.4f} @ Epoch {self.history['best_epoch']}")
            return

        n = len(self.dataset)
        indices = np.arange(n)
        train_idx, val_idx = train_test_split(
            indices, test_size=self.config.val_fraction, random_state=self.config.seed, shuffle=True
        )
        print(f"📊 Cells: {n} total | train {len(train_idx)} | val(early-stop only) {len(val_idx)}")

        train_set = Subset(self.dataset, train_idx.tolist())
        val_set = Subset(self.dataset, val_idx.tolist())
        objective = getattr(self.config, 'objective', 'distill')
        if objective == 'dual_level':
            # Graph-aware batches guarantee positive pairs for InfoNCE.
            # num_workers=0: keep sampler single-process (macOS worker pickling
            # bit us twice already; collation is cheap vs the model forward)
            pos_sets_dev = self.dataset.pos_sets.to(self.config.device)
            sampler = NeighborhoodSampler(train_idx, self.dataset.pos_sets,
                                          batch_size=self.params['batch_size'], n_pos=3,
                                          seed=self.config.seed)
            train_loader = DataLoader(self.dataset, batch_sampler=sampler, num_workers=0)
        else:
            train_loader = DataLoader(
                train_set, batch_size=self.params['batch_size'], shuffle=True,
                num_workers=0 if sys.platform == 'win32' else self.config.num_workers,
                pin_memory=torch.cuda.is_available(), drop_last=True
            )
        val_loader = DataLoader(val_set, batch_size=self.params['batch_size'], drop_last=False)

        model, optimizer = self._init_model()
        early_stopping = EarlyStopping(patience=self.config.patience, delta=self.config.delta)

        best_val_sim = -np.inf
        best_monitor = None
        best_state = None
        train_losses, val_similarities, val_fines = [], [], []

        t0 = time.perf_counter()
        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = epoch_cos = epoch_recon = epoch_fine = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch{epoch+1}", leave=False):
                gene_input   = batch["norm"].to(self.config.device)
                module_input = batch["modules"].to(self.config.device)
                target_emb   = batch["embedding"].to(self.config.device)
                batch_idx    = batch["index"].to(self.config.device)

                # Corrupted view of the gene input. Module activities stay clean
                # (robust multi-gene aggregates) - to keep fine neighborhoods correct
                # with a corrupted gene query, the model MUST retrieve module context
                # via cross-attention. This is what gives attention a real job.
                if objective in ('dual_level', 'denoise') and self.config.view_dropout > 0:
                    keep = (torch.rand_like(gene_input) >= self.config.view_dropout).float()
                    gene_in = gene_input * keep
                else:
                    gene_in = gene_input

                optimizer.zero_grad()
                _, _, _, pred_embedding, bio_pred, recon, gene_recon = model(gene_in, module_input)

                # Coarse (Harmony cosine) loss attaches to the BIO branch by
                # default, NOT to the fused/pred head. With coarse_target='bio' the
                # fused pathway is shaped only by recon + denoise, so attention can
                # no longer resort to module-averaging to please the coarse loss.
                coarse_anchor = bio_pred if (
                    objective in ('dual_level', 'denoise')
                    and getattr(self.config, 'coarse_target', 'fusion') == 'bio'
                ) else pred_embedding
                cos_loss   = 1 - F.cosine_similarity(coarse_anchor, target_emb).mean()
                recon_loss = F.mse_loss(recon, module_input)
                if objective == 'dual_level':
                    fine_loss = info_nce_loss(pred_embedding, batch_idx, pos_sets_dev, self.config.tau)
                    loss = (self.config.w_coarse * cos_loss + self.config.w_fine * fine_loss
                            + self.config.w_recon * recon_loss)
                elif objective == 'denoise':
                    # Masked gene reconstruction - only the MASKED entries count.
                    # Averaging all modules cannot reconstruct specific dropped genes,
                    # so selective module retrieval becomes the optimal strategy.
                    masked = (keep == 0)
                    if masked.any():
                        per_entry = (gene_recon - gene_input) ** 2
                        fine_loss = per_entry[masked].mean()
                    else:
                        fine_loss = gene_recon.new_zeros(())
                    loss = (self.config.w_coarse * cos_loss + self.config.w_recon * recon_loss
                            + self.config.w_denoise * fine_loss)
                    # MoE load-balancing aux loss (anti expert-collapse)
                    lb = getattr(model, 'moe_fusion', None)
                    if lb is not None and lb.last_lb is not None:
                        loss = loss + self.config.w_lb * lb.last_lb
                else:
                    fine_loss = torch.zeros(())
                    loss = self.params['alpha'] * cos_loss + (1 - self.params['alpha']) * recon_loss

                loss.backward()
                optimizer.step()

                epoch_loss  += loss.item() * gene_input.size(0)
                epoch_cos   += cos_loss.item() * gene_input.size(0)
                epoch_recon += recon_loss.item() * gene_input.size(0)
                epoch_fine  += fine_loss.item() * gene_input.size(0)

            n_train = len(train_idx)
            epoch_loss /= n_train; epoch_cos /= n_train; epoch_recon /= n_train; epoch_fine /= n_train
            train_losses.append(epoch_loss)

            val_loss, val_sim, val_fine = self._validate(model, val_loader)
            val_similarities.append(val_sim)
            val_fines.append(val_fine)

            print(
                f"Epoch {epoch+1:03d} | Train Loss: {epoch_loss:.4f} | "
                f"Cos: {epoch_cos:.4f} | Recon: {epoch_recon:.4f} | Fine: {epoch_fine:.4f} | "
                f"Val Sim: {val_sim:.4f} | Val Aux: {val_fine:.4f} | Val Obj: {-val_loss if objective != 'distill' else val_sim:.4f}"
            )

            # distill: monitor val cosine sim (higher=better); dual_level: monitor val objective (lower=better)
            monitor = val_sim if objective == 'distill' else -val_loss
            if best_monitor is None or monitor > best_monitor:
                best_monitor = monitor
                best_val_sim = val_sim
                best_state = copy.deepcopy(model.state_dict())  # deep copy (avoid a stale reference to the live model)

            if early_stopping(monitor, epoch + 1):
                print(f"🛑 Early stopping at epoch {epoch+1} (patience={self.config.patience})")
                break

        self.train_seconds = time.perf_counter() - t0

        self.best_state = best_state
        self.history = {
            'train_loss': train_losses,
            'val_sim': val_similarities,
            'val_fine': val_fines,
            'best_sim': max(val_similarities),
            'best_epoch': int(np.argmax(val_similarities)) + 1,
            'train_seconds': self.train_seconds,
            'fusion_mode': self.params.get('fusion_mode', 'moe'),
            'alpha': self.params['alpha'],
            'embedding_metric': self.current_embedding_metric,
        }
        torch.save(self.best_state, self.model_path)
        torch.save(self.history, self.history_path)
        print(f"💾 Model saved: {self.model_path} | Best Val Sim: {best_val_sim:.4f} "
              f"@ Epoch {self.history['best_epoch']} | Train time: {self.train_seconds:.1f}s")

    def load_best_model(self):
        self.best_state = torch.load(self.model_path, map_location=self.config.device, weights_only=False)

    def _validate(self, model, val_loader):
        model.eval()
        objective = getattr(self.config, 'objective', 'distill')
        total_sim = 0.0
        total_cos = 0.0
        total_recon = 0.0
        total_denoise = 0.0
        all_pred, all_idx = [], []
        # deterministic corruption mask for reproducible denoise validation
        gen = torch.Generator(device=self.config.device).manual_seed(self.config.seed)

        with torch.no_grad():
            for batch in val_loader:
                gene_input = batch["norm"].to(self.config.device)
                module_input = batch["modules"].to(self.config.device)
                target_embedding = batch["embedding"].to(self.config.device)

                if objective == 'denoise' and self.config.view_dropout > 0:
                    keep = (torch.rand(gene_input.shape, generator=gen,
                                       device=self.config.device) >= self.config.view_dropout).float()
                    gene_in = gene_input * keep
                else:
                    keep = None
                    gene_in = gene_input

                _, _, _, pred_embedding, bio_pred, recon, gene_recon = model(gene_in, module_input)

                # Cosine loss measured on the coarse anchor (bio branch by
                # default); val_sim keeps tracking pred<->Harmony alignment as a
                # diagnostic of how far the fused head drifts from the teacher.
                coarse_anchor = bio_pred if (
                    objective in ('dual_level', 'denoise')
                    and getattr(self.config, 'coarse_target', 'fusion') == 'bio'
                ) else pred_embedding
                cos_loss = 1 - F.cosine_similarity(coarse_anchor, target_embedding).mean()
                recon_loss = F.mse_loss(recon, module_input)

                total_cos   += cos_loss.item() * gene_input.size(0)
                total_recon += recon_loss.item() * gene_input.size(0)
                total_sim   += F.cosine_similarity(pred_embedding, target_embedding).mean().item()
                if objective == 'denoise' and keep is not None:
                    masked = (keep == 0)
                    if masked.any():
                        total_denoise += ((gene_recon - gene_input) ** 2)[masked].mean().item() * gene_input.size(0)
                all_pred.append(pred_embedding)
                all_idx.append(batch["index"].to(self.config.device))

        n_val = max(len(val_loader.dataset), 1)
        cos_avg = total_cos / n_val
        recon_avg = total_recon / n_val
        val_sim = total_sim / max(len(val_loader), 1)

        if objective == 'dual_level':
            # fine loss computed over ALL val cells at once (stable, dense positives)
            pred = torch.cat(all_pred)
            idx = torch.cat(all_idx)
            val_fine = info_nce_loss(pred, idx, self.dataset.pos_sets.to(self.config.device),
                                     self.config.tau).item()
            val_loss = (self.config.w_coarse * cos_avg + self.config.w_fine * val_fine
                        + self.config.w_recon * recon_avg)
        elif objective == 'denoise':
            val_fine = total_denoise / n_val  # masked gene reconstruction error
            val_loss = (self.config.w_coarse * cos_avg + self.config.w_recon * recon_avg
                        + self.config.w_denoise * val_fine)
        else:
            val_fine = 0.0
            val_loss = self.params['alpha'] * cos_avg + (1 - self.params['alpha']) * recon_avg
        return val_loss, val_sim, val_fine

    def save_training_report(self):
        self._extract_and_save_embeddings()
        self._generate_parameter_table()
        self._validate_model_loading()
        self._report_size_trust()

    def _report_size_trust(self):
        """Report the learned size-trust curve (is bigger really more trusted?)."""
        model, _ = self._init_model()
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        if getattr(model, 'size_gamma', None) is None:
            return
        import pandas as pd
        trust = (1.0 + torch.tanh(model.size_gamma * model.module_log_sizes + model.size_bias)).detach().cpu().numpy()
        sizes = np.exp(model.module_log_sizes.cpu().numpy())
        corr = np.corrcoef(np.log(sizes), trust)[0, 1]
        names = list(self.dataset.valid_modules.keys())
        df = pd.DataFrame({'module': names, 'n_genes': sizes.astype(int), 'trust': trust})
        out = self.results_dir / f"module_size_trust{self.suffix}.csv"
        df.sort_values('trust', ascending=False).to_csv(out, index=False)
        print(f"📏 Size-trust: corr(log size, trust) = {corr:+.3f} | "
              f"trust range [{trust.min():.3f}, {trust.max():.3f}] | saved {out}")

    def _extract_and_save_embeddings(self):
        """Extract and save embeddings (X_fusion is the PRIMARY output)."""
        model, _ = self._init_model()
        if self.best_state is None:
            self.load_best_model()
        model.load_state_dict(self.best_state)

        bio_emb, mod_emb, fused_emb, pred_emb, attn, beta, gg = self._extract_embeddings(model)

        self.adata.obsm[f'X_bio{self.suffix}'] = bio_emb
        self.adata.obsm[f'X_mod{self.suffix}'] = mod_emb
        self.adata.obsm[f'X_fusion{self.suffix}'] = fused_emb            # PRIMARY
        self.adata.obsm[f'X_pred_embedding{self.suffix}'] = pred_emb    # secondary (Harmony-distilled head)

        print(f"✅ Embeddings stored in AnnData with suffix '{self.suffix}':")
        print(f"   - X_bio{self.suffix}")
        print(f"   - X_mod{self.suffix}")
        print(f"   - X_fusion{self.suffix}   <-- primary embedding")
        print(f"   - X_pred_embedding{self.suffix}")

        if attn is not None:
            # attn: [n_cells, M] averaged over heads/queries (attention) or raw gates (gate_attn)
            attn_key = ('X_expert_weights' if self.params.get('fusion_mode') in ('moe', 'moe2', 'moe_updn', 'moe_deep', 'moe_shared')
                        else 'X_module_gate' if self.params.get('fusion_mode') in ('gate_attn', 'gate_film')
                        else 'X_module_attention')
            self.adata.obsm[f'{attn_key}{self.suffix}'] = attn
            module_names = list(self.dataset.valid_modules.keys())
            if attn.shape[1] == len(module_names):
                # module-level weights (attention/gate modes): per-module importance
                importance = pd.DataFrame({
                    'module': module_names,
                    'mean_attention': attn.mean(axis=0),
                    'std_attention': attn.std(axis=0),
                }).sort_values('mean_attention', ascending=False)
            else:
                # moe mode: expert-level importance (K module experts + gene expert)
                n_exp = attn.shape[1]
                try:
                    import json
                    gmap_file = ('module_groups_v2026_typed.json' if self.params.get('fusion_mode') in ('moe2', 'moe_updn', 'moe_deep')
                                 else 'module_groups_v2026.json')
                    gmap_path = Path(self.config.sc_gene_label_path).parent / gmap_file
                    gnames = sorted(set(json.load(open(gmap_path)).values()))
                    if self.params.get('fusion_mode') == 'moe_shared' and len(gnames) == n_exp - 2:
                        expert_names = gnames + ['shared', 'gene']
                    elif len(gnames) == n_exp - 1:
                        expert_names = gnames + ['gene']
                    else:
                        expert_names = [f'expert_{i}' for i in range(n_exp)]
                except Exception:
                    expert_names = [f'expert_{i}' for i in range(n_exp)]
                importance = pd.DataFrame({
                    'expert': expert_names,
                    'mean_weight': attn.mean(axis=0),
                    'std_weight': attn.std(axis=0),
                }).sort_values('mean_weight', ascending=False)
            imp_path = self.results_dir / f"module_importance{self.suffix}.csv"
            importance.to_csv(imp_path, index=False)
            print(f"   - {attn_key}{self.suffix} (per-cell pathway weights, [n_cells, {attn.shape[1]}])")
            print(f"📊 Module importance ranking saved to: {imp_path}")

        if beta is not None:
            # per-cell imputation (shift) values
            self.adata.obsm[f'X_module_beta{self.suffix}'] = beta
            print(f"   - X_module_beta{self.suffix} (per-cell pathway imputation, [n_cells, {beta.shape[1]}])")

        if gg is not None:
            # per-gene gate values
            self.adata.obsm[f'X_gene_gate{self.suffix}'] = gg
            print(f"   - X_gene_gate{self.suffix} (per-cell gene gates, [n_cells, {gg.shape[1]}])")

    def _extract_embeddings(self, model: BioNet) -> tuple:
        model.eval()
        is_attn = getattr(model, 'fusion_mode', None) == 'cross_attention'
        is_gate = getattr(model, 'fusion_mode', None) in ('gate_attn', 'gate_film')
        is_moe = getattr(model, 'fusion_mode', None) in ('moe', 'moe2', 'moe_updn', 'moe_deep', 'moe_shared')
        loader = DataLoader(self.dataset, batch_size=512, shuffle=False)
        bio_embs, mod_embs, fused_embs, pred_embs, attns, betas, ggs = [], [], [], [], [], [], []
        t0 = time.perf_counter()
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Extracting embeddings for {self.current_embedding_metric}", leave=False):
                gene_input = batch["norm"].to(self.config.device)
                module_input = batch["modules"].to(self.config.device)
                g_emb, m_emb, f_emb, p_emb, _, _, _ = model(gene_input, module_input)
                bio_embs.append(g_emb.cpu())
                mod_embs.append(m_emb.cpu())
                fused_embs.append(f_emb.cpu())
                pred_embs.append(p_emb.cpu())
                if is_attn and model.cross_fusion.last_attn is not None:
                    # [B, heads, n_q, M] -> mean over heads and queries -> [B, M]
                    attns.append(model.cross_fusion.last_attn.cpu().mean(dim=(1, 2)))
                elif is_gate and model.fusion_mode == 'gate_attn' and model.gate_fusion.last_gate is not None:
                    # [B, M] sigmoid gates, exported raw (diagnostics normalise on use)
                    attns.append(model.gate_fusion.last_gate.cpu())
                elif is_gate and model.fusion_mode == 'gate_film':
                    attns.append(model.gate_film.last_gamma.cpu())  # [B, M] scale ~ 1
                    betas.append(model.gate_film.last_beta.cpu())   # [B, M] shift ~ 0
                elif is_moe:
                    moe_mod = getattr(model, 'moe_fusion', None) or getattr(model, 'moe2_fusion', None)
                    if moe_mod is not None and moe_mod.last_router is not None:
                        attns.append(moe_mod.last_router.cpu())  # [B, K+1] expert weights
                if getattr(model, 'last_gene_gate', None) is not None:
                    ggs.append(model.last_gene_gate.cpu())       # [B, n_genes]
        self.inference_seconds = time.perf_counter() - t0
        print(f"⏱ Inference on {len(self.dataset)} cells: {self.inference_seconds:.1f}s")

        attn_out = torch.cat(attns).numpy().astype(np.float32) if attns else None
        beta_out = torch.cat(betas).numpy().astype(np.float32) if betas else None
        gg_out = torch.cat(ggs).numpy().astype(np.float32) if ggs else None
        return (
            torch.cat(bio_embs).numpy(),
            torch.cat(mod_embs).numpy(),
            torch.cat(fused_embs).numpy(),
            torch.cat(pred_embs).numpy(),
            attn_out,
            beta_out,
            gg_out
        )

    def _generate_parameter_table(self):
        params = [
            ["Parameter", "Value"],
            ["Version", "DeepFusion (full-HVG + mask channel + per-gene gate, detach-shielded)"],
            ["Fusion Mode", self.params.get('fusion_mode', 'moe')],
            ["Coarse Target", getattr(self.config, 'coarse_target', 'fusion')],
            ["Attn Detach Coarse", getattr(self.config, 'attn_detach_coarse', 1)],
            ["Alpha", self.params['alpha']],
            ["Embedding Dim", self.params['embedding_dim']],
            ["Hidden Dim", self.params['hidden_dim']],
            ["Dropout", self.params['dropout']],
            ["Batch Size", self.params['batch_size']],
            ["Learning Rate", f"{self.params['lr']:.2e}"],
            ["Weight Decay", f"{self.params['weight_decay']:.2e}"],
            ["Num Heads", self.params.get('num_heads', 8)],
            ["Query Tokens", self.params.get('n_query_tokens', 4)],
            ["Epochs (max)", self.config.epochs],
            ["Protocol", "single-split transductive (val for early stopping only)"],
            ["Val Fraction", self.config.val_fraction],
            ["EarlyStopping Patience", self.config.patience],
            ["EarlyStopping Delta", self.config.delta],
            ["Embedding Metric", self.current_embedding_metric],
            ["Output Suffix", self.suffix],
            ["Caching Enabled", self.config.enable_caching],
            ["Activity Normalization", "log1p (decoder: linear, no Sigmoid)"],
            ["Train Time (s)", f"{self.train_seconds:.1f}"],
            ["Inference Time (s)", f"{self.inference_seconds:.1f}"],
        ]

        print(f"\n=== Model Parameters for {self.current_embedding_metric} ===")
        print(tabulate(params, headers="firstrow", tablefmt="grid"))

    def _validate_model_loading(self):
        print(f"\n🔍 Verifying model loading for {self.current_embedding_metric}...")
        try:
            model, _ = self._init_model()
            model.load_state_dict(torch.load(self.model_path, map_location=self.config.device, weights_only=False))
            print("✅ Model loading verified")
            return True
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            return False

    def get_adata(self) -> anndata.AnnData:
        """Return processed adata"""
        return self.adata

# -------------------------- New function: load completed training Embedding results --------------------------
def load_completed_embeddings(config: Config, completed_embeddings: List[str], base_adata: anndata.AnnData) -> anndata.AnnData:
    """
    Load all completed training embedding results, merge into base adata to resolve checkpoint resume losing historical results
    :param config: Configuration object
    :param completed_embeddings: List of completed training embeddings
    :param base_adata: Base adata object
    :return: adata merged with all historically completed embeddings
    """
    final_adata = base_adata.copy()
    print(f"\n========================================")
    print(f"📥 Starting to load historically completed training Embedding results")
    print(f"========================================")

    for embedding_metric in completed_embeddings:
        try:
            suffix = config.get_output_suffix(embedding_metric)  # includes ablation tag
            # Build embedding key names to load
            emb_keys = [
                f'X_bio{suffix}',
                f'X_mod{suffix}',
                f'X_fusion{suffix}',
                f'X_pred_embedding{suffix}'
            ]
            optional_keys = [f'X_module_attention{suffix}', f'X_module_gate{suffix}', f'X_module_beta{suffix}', f'X_expert_weights{suffix}', f'X_gene_gate{suffix}']  # cross_attention / gate / moe / gene-gate modes

            # Check if final output file exists; if so, read saved embedding directly
            if config.output_path.exists():
                temp_adata = anndata.read_h5ad(config.output_path)
                loaded_count = 0
                for key in emb_keys:
                    if key in temp_adata.obsm:
                        final_adata.obsm[key] = temp_adata.obsm[key].copy()
                        loaded_count += 1
                for key in optional_keys:  # attention maps are optional (cross_attention mode only)
                    if key in temp_adata.obsm:
                        final_adata.obsm[key] = temp_adata.obsm[key].copy()
                if loaded_count == len(emb_keys):
                    print(f"✅ Loaded all Embeddings for {embedding_metric} from final output file")
                    continue

            # If final file has no results, re-extract via Trainer (ensure compatibility)
            print(f"ℹ️ Re-extracting Embedding for {embedding_metric} from model file")
            # Read optimal hyperparameters for this embedding
            if config.disable_hyper_search:
                # When hyperparameter search is disabled, use fixed parameters (consistent with training)
                print(f"⚙️ Hyper-parameter search disabled, using fixed params for {embedding_metric}")
                best_params = {
                    'alpha': config.fixed_alpha,
                    'embedding_dim': 256,
                    'hidden_dim': 512,
                    'dropout': 0.3,
                    'batch_size': 128,
                    'lr': 5e-4,
                    'weight_decay': 1e-4,
                    'num_heads': config.num_heads,
                    'n_query_tokens': config.n_query_tokens,
                    'fusion_mode': config.fusion_mode,
                    'attn_detach_coarse': getattr(config, 'attn_detach_coarse', 1),
                    'size_trust': getattr(config, 'size_trust', 0),
                    'gene_gate': getattr(config, 'gene_gate', 0),
                    'top_k': getattr(config, 'top_k', 2),
                }
            else:
                study_name = f"{config.study_name}_{embedding_metric.replace('/', '_')}"
                storage_url = f"sqlite:///{config.study_storage}"
                study = optuna.load_study(study_name=study_name, storage=storage_url)
                best_params = study.best_params

            # Initialize Trainer and extract embedding
            trainer = Trainer(config, best_params, embedding_metric)
            # Extract embedding directly, no retraining needed
            trainer.load_best_model()  # single best model (no fold-based checkpoints)
            trainer._extract_and_save_embeddings()
            processed_adata = trainer.get_adata()

            # Merge into final_adata
            for key in emb_keys:
                if key in processed_adata.obsm:
                    final_adata.obsm[key] = processed_adata.obsm[key].copy()

            print(f"✅ Successfully extracted and merged Embedding for {embedding_metric}")
            # Free memory
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Failed to load historical Embedding for {embedding_metric}: {str(e)}")
            traceback.print_exc()
            continue

    print(f"\n========================================")
    print(f"🎉 Historical Embedding loading completed, all finished results merged")
    print(f"========================================")
    return final_adata

# -------------------------- Check training progress function (preserve original logic) --------------------------
def check_training_progress(config: Config, harmony_embeddings: List[str]) -> Tuple[List[str], List[str]]:
    """
    Scan model storage directory, check which embeddings have completed training, return lists of completed and incomplete embeddings
    :param config: Configuration object
    :param harmony_embeddings: List of all pending X_special_harmony embeddings
    :return: (list of completed embeddings, list of incomplete embeddings)
    """
    completed_embeddings = []
    incomplete_embeddings = []
    
    # Iterate over all pending embeddings
    for embedding_metric in harmony_embeddings:
        # respect the ablation tag so ablation runs are not mistaken for completed
        embedding_model_dir = Path(config.model_dir) / config.get_dir_name(embedding_metric)
        
        # Condition 1: does model directory exist
        if not embedding_model_dir.exists():
            print(f"⚠️ {embedding_metric}: Model directory does not exist, training not started")
            incomplete_embeddings.append(embedding_metric)
            continue
        
        # single-model protocol - check best_model.pth + history.pth
        model_ok = (embedding_model_dir / "best_model.pth").exists()
        history_ok = (embedding_model_dir / "history.pth").exists()

        if model_ok and history_ok:
            print(f"✅ {embedding_metric}: training completed (best_model.pth + history.pth)")
            completed_embeddings.append(embedding_metric)
        else:
            print(f"⚠️ {embedding_metric}: model={model_ok}, history={history_ok}, training incomplete")
            incomplete_embeddings.append(embedding_metric)
    
    # Output progress summary
    print(f"\n========================================")
    print(f"📊 Training progress check results:")
    print(f"   Completed: {len(completed_embeddings)} / Total: {len(harmony_embeddings)}")
    if completed_embeddings:
        print(f"   Completed list: {completed_embeddings}")
    print(f"   Pending list: {incomplete_embeddings}")
    print(f"========================================\n")
    
    return completed_embeddings, incomplete_embeddings

# -------------------------- Main function (fix checkpoint resume, integrate historical results) --------------------------
def main():
    # command-line switches for ablations
    parser = argparse.ArgumentParser(description="DeepFusion multi-token cross-attention")
    parser.add_argument('--fusion_mode', default='cross_attention',
                        choices=['cross_attention', 'concat', 'gated', 'no_module', 'gate_attn', 'gate_film', 'moe', 'moe2', 'moe_updn', 'moe_deep', 'moe_shared'],
                        help='Fusion block: cross_attention (full) | concat / no_module (ablations) | gated (fallback)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Loss weight alpha (cosine alignment). Overrides fixed_alpha, e.g. 1.0 / 0.0 for ablations')
    parser.add_argument('--suffix_tag', default='',
                        help='Extra tag appended to output embedding suffixes (for ablation runs)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (model init + train/val split)')
    parser.add_argument('--targets', nargs='*', default=None,
                        help='Restrict teacher embeddings, e.g. --targets 40 or --targets 20 40')
    # dual-level objective
    parser.add_argument('--objective', default='denoise', choices=['distill', 'dual_level', 'denoise'])
    parser.add_argument('--w_coarse', type=float, default=0.5)
    parser.add_argument('--coarse_target', default='fusion', choices=['bio', 'fusion'],
                        help='Attach coarse (Harmony cosine) loss to the bio branch (bio) '
                             'or the fused/pred head (fusion, default)')
    parser.add_argument('--attn_detach_coarse', type=int, default=1,
                        help='1 = detach attention output in the coarse anchor path '
                             '(gradient isolation, default); 0 = no isolation')
    parser.add_argument('--w_fine', type=float, default=0.4)
    parser.add_argument('--w_recon', type=float, default=0.2)
    parser.add_argument('--w_denoise', type=float, default=0.3)
    parser.add_argument('--size_trust', type=int, default=0,
                        help='1 = learnable per-module trust scale ~ module size (identity-init)')
    parser.add_argument('--min_module_size', type=int, default=1,
                        help='filter modules with fewer present genes than this (default 1 = no filtering)')
    parser.add_argument('--keep_all_hvg', type=int, default=0,
                        help='1 = keep all HVGs (restore prior-uncovered genes)')
    parser.add_argument('--gene_mask_channel', type=int, default=0,
                        help='1 = dual-channel gene input [fluctuation, detection mask]')
    parser.add_argument('--gene_gate', type=int, default=0,
                        help='1 = per-gene sigmoid gate (full-resolution, detach-shielded)')
    parser.add_argument('--top_k', type=int, default=2,
                        help='MoE router sparsity (experts selected per cell; default 2)')
    parser.add_argument('--gene_label_path', type=str, default=None,
                        help='override module CSV path (e.g. REACTOME+CGP filtered set)')
    parser.add_argument('--view_dropout', type=float, default=0.3)
    parser.add_argument('--data_dropout', type=float, default=0.0,
                        help='Data-level dropout simulation applied to fluctuation/raw before module computation')
    parser.add_argument('--knn_k', type=int, default=15)
    parser.add_argument('--coarse_pool', type=int, default=100)
    parser.add_argument('--tau', type=float, default=0.2)
    args = parser.parse_args()

    # Initialize configuration
    config = Config()
    config.fusion_mode = args.fusion_mode
    if args.alpha is not None:
        config.fixed_alpha = args.alpha
    config.suffix_tag = args.suffix_tag
    config.seed = args.seed
    config.objective = args.objective
    config.w_coarse = args.w_coarse
    config.coarse_target = args.coarse_target
    config.attn_detach_coarse = args.attn_detach_coarse
    config.w_fine = args.w_fine
    config.w_recon = args.w_recon
    config.view_dropout = args.view_dropout
    config.w_denoise = args.w_denoise
    config.size_trust = args.size_trust
    config.min_module_size = args.min_module_size
    config.keep_all_hvg = args.keep_all_hvg
    config.gene_mask_channel = args.gene_mask_channel
    config.gene_gate = args.gene_gate
    config.top_k = args.top_k
    if args.gene_label_path:
        config.sc_gene_label_path = Path(args.gene_label_path)
    config.data_dropout = args.data_dropout
    config.knn_k = args.knn_k
    config.coarse_pool = args.coarse_pool
    config.tau = args.tau
    # Re-seed all RNGs for this run (overrides the default seeding in __main__)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    config.validate_paths()

    # Preprocess data (batch correction + fluctuation calculation)
    processor = SCDataProcessor(config)
    adata, _ = processor.load_data()
    new_path = config.cache_dir / "adata_with_fluct.h5ad"
    if not new_path.exists():  # avoid rewriting on every run (deterministic given cache)
        adata.write(new_path)
    config.h5ad_path = new_path

    # Key: Get all embedding lists starting with X_special_harmony
    harmony_embeddings = config.get_all_special_harmony_embeddings(adata)
    n_harmony = len(harmony_embeddings)

    # optional target filter (--targets 40 ...)
    if args.targets:
        wanted = set()
        for t in args.targets:
            t = str(t)
            wanted.add(t if t.startswith('X_') else f"X_special_harmony_{t}")
        harmony_embeddings = [e for e in harmony_embeddings if e in wanted]
        if not harmony_embeddings:
            raise ValueError(f"--targets {args.targets} matched nothing; available: "
                             f"{config.get_all_special_harmony_embeddings(adata)}")
        n_harmony = len(harmony_embeddings)
        print(f"🎯 Target filter active: {harmony_embeddings}")
    
    # -------------------------- Core modification: distinguish completed/incomplete, load historical results --------------------------
    completed_embeddings, incomplete_embeddings = check_training_progress(config, harmony_embeddings)
    
    # Step 1: load all completed training embeddings, merge into final_adata
    # Use the EXISTING output file as the merge base whenever present, so that
    # successive runs (main model, ablations, controls) ACCUMULATE embeddings in
    # step5_add_DeepFusion.h5ad instead of overwriting each other.
    output_path = Path(config.output_path)
    if output_path.exists():
        print(f"♻️ Merging into existing output file: {output_path}")
        base_adata = anndata.read_h5ad(output_path)
    else:
        base_adata = adata
    final_adata = load_completed_embeddings(config, completed_embeddings, base_adata)
    
    # If none incomplete, save and exit directly
    if not incomplete_embeddings:
        print("🎉 All embeddings have completed training, no further execution needed")
        # Optimize memory and save final results
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if issparse(final_adata.X):
            final_adata.X = csr_matrix(final_adata.X)
        final_adata.var["gene_module"] = final_adata.var["gene_module"].astype("category")
        final_adata.write(output_path)
        print(f"✅ Final results saved to: {output_path}")
        
        # Print final embedding list
        print(f"\nFinal output embeddings list:")
        output_embs = [k for k in final_adata.obsm.keys() if k.startswith(('X_bio', 'X_fusion', 'X_pred_embedding'))]
        for emb in sorted(output_embs):
            print(f"   - {emb}")
        return
    
    # Print pending incomplete embedding list
    print(f"\n========================================")
    print(f"📌 Pending incomplete embeddings ({len(incomplete_embeddings)}):")
    for i, emb in enumerate(incomplete_embeddings):
        suffix = config.get_embedding_suffix(emb)
        print(f"   [{i+1}] {emb} → output suffix: '{suffix}'")
    print(f"========================================\n")
    
    if n_harmony == 0:
        raise ValueError("No X_special_harmony embeddings found in adata.obsm!")
    
    # Initialize performance tracking
    perf_data = {
        'total_start': time.perf_counter(),
        'embeddings': {},  # Time consumption for each embedding
        'total': 0
    }
    
    # -------------------------- Only loop over incomplete embeddings, merge into final_adata after training --------------------------
    for idx, embedding_metric in enumerate(incomplete_embeddings):
        # Compute global progress
        global_idx = harmony_embeddings.index(embedding_metric) + 1
        print(f"\n========================================")
        print(f"🔄 Processing embedding [{global_idx}/{n_harmony}] (incomplete queue: {idx+1}/{len(incomplete_embeddings)}): {embedding_metric}")
        print(f"========================================")
        
        # Initialize performance tracking for current embedding
        emb_start = time.perf_counter()
        perf_data['embeddings'][embedding_metric] = {
            'stage1': 0, 'stage2': 0, 'stage3': 0, 'total': 0
        }
        
        try:
            # -------------------------- Stage 1: Hyperparameter Optimization --------------------------
            print(f"\n=== Stage 1: Hyperparameter Tuning for {embedding_metric} ===")
            if config.disable_hyper_search:
                print(f"⚙️ Hyper-parameter search disabled, using fixed params for {embedding_metric}")
                best_params = {
                    'alpha': config.fixed_alpha,
                    'embedding_dim': 256,
                    'hidden_dim': 512,
                    'dropout': 0.3,
                    'batch_size': 128,
                    'lr': 5e-4,
                    'weight_decay': 1e-4,
                    'num_heads': config.num_heads,
                    'n_query_tokens': config.n_query_tokens,
                    'fusion_mode': config.fusion_mode,
                    'attn_detach_coarse': getattr(config, 'attn_detach_coarse', 1),
                    'size_trust': getattr(config, 'size_trust', 0),
                    'gene_gate': getattr(config, 'gene_gate', 0),
                    'top_k': getattr(config, 'top_k', 2),
                }
            else:
                stage1_start = time.perf_counter()
                # Initialize hyperparameter optimizer for current embedding
                hyper_optimizer = HyperOptimizer(config, embedding_metric)
                best_params = hyper_optimizer.optimize()
                stage1_end = time.perf_counter()
                perf_data['embeddings'][embedding_metric]['stage1'] = stage1_end - stage1_start
                print(f"✅ Hyper-tuning completed for {embedding_metric} | Time: {perf_data['embeddings'][embedding_metric]['stage1']:.1f}s")
            
            # -------------------------- Stage 2: Model Training --------------------------
            print(f"\n=== Stage 2: Model Training for {embedding_metric} ===")
            stage2_start = time.perf_counter()
            # Initialize trainer for current embedding
            trainer = Trainer(config, best_params, embedding_metric)
            trainer.train(reload=False)
            stage2_end = time.perf_counter()
            perf_data['embeddings'][embedding_metric]['stage2'] = stage2_end - stage2_start
            print(f"✅ Model training completed for {embedding_metric} | Time: {perf_data['embeddings'][embedding_metric]['stage2']:.1f}s")
            
            # -------------------------- Stage 3: Results Export --------------------------
            print(f"\n=== Stage 3: Results Export for {embedding_metric} ===")
            stage3_start = time.perf_counter()
            trainer.save_training_report()
            # Get processed adata (contains results of current embedding)
            processed_adata = trainer.get_adata()
            # Merge to final adata
            suffix = config.get_output_suffix(embedding_metric)  # includes ablation tag
            emb_keys_to_merge = [f'X_bio{suffix}', f'X_mod{suffix}', f'X_fusion{suffix}', f'X_pred_embedding{suffix}',
                                 f'X_module_attention{suffix}', f'X_module_gate{suffix}', f'X_module_beta{suffix}',
                                 f'X_expert_weights{suffix}', f'X_gene_gate{suffix}']
            for key in emb_keys_to_merge:
                if key in processed_adata.obsm:
                    final_adata.obsm[key] = processed_adata.obsm[key].copy()
            
            stage3_end = time.perf_counter()
            perf_data['embeddings'][embedding_metric]['stage3'] = stage3_end - stage3_start
            perf_data['embeddings'][embedding_metric]['total'] = time.perf_counter() - emb_start
            # true inference-only time measured inside the trainer (export stage also includes I/O)
            perf_data['embeddings'][embedding_metric]['inference_s'] = getattr(trainer, 'inference_seconds', 0.0)
            print(f"✅ Results exported for {embedding_metric} | Time: {perf_data['embeddings'][embedding_metric]['stage3']:.1f}s")
            
            # Clean up memory
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n❌ Error processing {embedding_metric}: {str(e)}")
            traceback.print_exc()
            print(f"⚠️ Continuing with next embedding...")
            continue
    
    # -------------------------- Final Save --------------------------
    print(f"\n========================================")
    print(f"📥 Saving final integrated results")
    print(f"========================================")
    
    # Optimize memory and save final results
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Memory optimization
    if issparse(final_adata.X):
        final_adata.X = csr_matrix(final_adata.X)
    final_adata.var["gene_module"] = final_adata.var["gene_module"].astype("category")
    
    # Save final file
    final_adata.write(output_path)
    print(f"✅ Final results saved to: {output_path}")
    
    # -------------------------- Generate Performance Report --------------------------
    perf_data['total'] = time.perf_counter() - perf_data['total_start']
    print(f"\n========================================")
    print(f"⏱ Performance Report (All Embeddings)")
    print(f"========================================")
    
    # Print time consumption for each embedding
    report_rows = [["Embedding", "Hyper-tuning (s)", "Training (s)", "Export (s)", "Total (s)"]]
    for emb, times in perf_data['embeddings'].items():
        report_rows.append([
            emb,
            f"{times['stage1']:.1f}",
            f"{times['stage2']:.1f}",
            f"{times['stage3']:.1f}",
            f"{times['total']:.1f}"
        ])
    # Total row
    total_stage1 = sum([v['stage1'] for v in perf_data['embeddings'].values()])
    total_stage2 = sum([v['stage2'] for v in perf_data['embeddings'].values()])
    total_stage3 = sum([v['stage3'] for v in perf_data['embeddings'].values()])
    report_rows.append([
        "TOTAL",
        f"{total_stage1:.1f}",
        f"{total_stage2:.1f}",
        f"{total_stage3:.1f}",
        f"{perf_data['total']:.1f}"
    ])
    
    print(tabulate(report_rows, headers="firstrow", tablefmt="github"))
    
    # Save performance report
    perf_report_path = config.results_dir / "performance_report_all_embeddings.pkl"
    with open(perf_report_path, 'wb') as f:
        pickle.dump(perf_data, f)
    print(f"\n📊 Performance data saved to: {perf_report_path}")

    # Fold key figures into the unified perf record (perf_utils, written at exit)
    _perf = globals().get('_PERF_RECORDER')
    if _perf is not None:
        n_cells = int(final_adata.n_obs)
        total_train = sum(v['stage2'] for v in perf_data['embeddings'].values())
        total_infer = sum(v.get('inference_s', 0.0) for v in perf_data['embeddings'].values())
        _perf.extra.update({
            'n_cells': n_cells,
            'n_embeddings': len(perf_data['embeddings']),
            'train_s': round(total_train, 1),
            'inference_s': round(total_infer, 1),
            'inference_cells_per_s': round(n_cells / total_infer, 1) if total_infer > 0 else None,
        })
    
    # Print final output embedding list
    print(f"\n========================================")
    print(f"🎉 All processes completed successfully!")
    print(f"========================================")
    print(f"Final output embeddings in adata.obsm:")
    output_embs = [k for k in final_adata.obsm.keys() if k.startswith(('X_bio', 'X_fusion', 'X_pred_embedding'))]
    for emb in sorted(output_embs):
        print(f"   - {emb}")
    print(f"\nFinal results file: {output_path}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    from perf_utils import PerfRecorder
    _PERF_RECORDER = PerfRecorder.start_now("step5-2_deepfusion")

    # Start main process
    try:
        main()
    except Exception as e:
        print(f"\n❌ Critical error in main process: {str(e)}")
        traceback.print_exc()
        sys.exit(1)