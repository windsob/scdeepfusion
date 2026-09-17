#!/usr/bin/env python3
"""
Step 5-0: Create HVG Subsets

This script reads the input H5AD file and creates HVG (Highly Variable Genes) subsets
with different numbers of top genes (2000, 3000, 5000).

Input: step4_add_scGPT.h5ad

Note: Uses raw layer directly for HVG selection with seurat_v3 flavor (no batch correction).
"""

import scanpy as sc
import anndata
import numpy as np
from scipy.sparse import issparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def create_hvg_subset(adata: anndata.AnnData, n_top_genes: int, batch_key: str = "batch") -> anndata.AnnData:
    """
    Create a HVG subset of adata using raw layer directly.
    Uses seurat_v3 flavor for HVG selection.
    """
    print(f"\n{'='*60}")
    print(f"Creating HVG subset with {n_top_genes} genes...")
    print(f"{'='*60}")
    
    # Check if raw layer exists
    if 'raw' not in adata.layers:
        raise ValueError("'raw' layer not found in adata!")
    
    print("   ✅ Using 'raw' layer directly for HVG selection")
    
    # Temporarily swap X with raw layer for HVG calculation
    # This avoids creating a full copy of adata
    original_X = adata.X
    layer_data = adata.layers['raw']
    adata.X = layer_data if not issparse(layer_data) else layer_data.toarray()
    
    # Ensure n_top_genes doesn't exceed the number of genes
    n_top_genes = min(n_top_genes, adata.n_vars)
    print(f"   Input genes: {adata.n_vars}")
    print(f"   Selecting top {n_top_genes} HVGs")
    
    # HVG selection using seurat_v3 (works directly on raw counts)
    # Task B note: with 8 donor batches, per-batch loess fits can hit numerical
    # singularities (small donor groups with many all-zero genes). Try batch-aware
    # first; fall back to pooled HVG on numerical failure.
    print("   Computing highly variable genes (seurat_v3 flavor)...")
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3', batch_key=batch_key)
    except ValueError as e:
        print(f"   ⚠️ batch-aware HVG failed ({e}); falling back to pooled HVG (batch_key=None)")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3', batch_key=None)
    hvg_mask = adata.var['highly_variable'].values
    
    # Restore original X immediately
    adata.X = original_X
    
    # Create subset with only HVG
    hvg_subset = adata[:, hvg_mask].copy()
    
    print(f"✅  HVG selection completed: {hvg_subset.n_vars} genes selected")
    
    return hvg_subset


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
    import sys
    from pathlib import Path
    Path("logs").mkdir(exist_ok=True)
    fh = open(Path("logs") / log_name, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print(f"📜 Logging to logs/{log_name}")


def print_adata_info(adata: anndata.AnnData, label: str = "AnnData"):
    """Print detailed information about an AnnData object."""
    print(f"\n{'-'*60}")
    print(f"{label} Structure:")
    print(f"{'-'*60}")
    print(f"  • Shape: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"  • Layers: {list(adata.layers.keys())}")
    print(f"  • Obs (cell metadata): {list(adata.obs.columns)}")
    print(f"  • Obsm (cell embeddings): {list(adata.obsm.keys())}")
    print(f"  • Var (gene metadata): {list(adata.var.columns)}")
    print(f"  • Uns (unstructured data): {list(adata.uns.keys())}")
    
    # Print layer shapes
    print(f"\n  Layer details:")
    for layer_name in adata.layers.keys():
        layer_data = adata.layers[layer_name]
        shape = layer_data.shape if hasattr(layer_data, 'shape') else 'N/A'
        sparse_type = type(layer_data).__name__ if issparse(layer_data) else 'dense'
        print(f"    - {layer_name}: {shape} ({sparse_type})")
    
    # Print embedding shapes
    print(f"\n  Embedding details (obsm):")
    for emb_name in adata.obsm.keys():
        emb_shape = adata.obsm[emb_name].shape if hasattr(adata.obsm[emb_name], 'shape') else 'N/A'
        print(f"    - {emb_name}: {emb_shape}")
    
    print(f"{'-'*60}\n")


def main():
    _enable_default_log("step50_hvg.log")  # default logging: tee console output to file
    """Main function to create HVG subsets."""
    
    # Configuration
    input_file = Path("enhanced_results/results/step4_add_scGPT.h5ad")
    output_dir = Path("enhanced_results/results")
    hvg_settings = [3000, 4000]
    batch_key = "batch"  # Column name for batch information
    
    print(f"\n{'#'*70}")
    print(f"# Step 5-0: Create HVG Subsets")
    print(f"{'#'*70}")
    
    # Check input file exists
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        print(f"   Please ensure the file exists before running this script.")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input data
    print(f"\n📥 Loading input file: {input_file}")
    adata = sc.read_h5ad(input_file)
    print(f"✅ Successfully loaded data")
    
    # Print original data structure
    print_adata_info(adata, "Original Data (step4_add_scGPT.h5ad)")
    
    # Create HVG subsets for each setting
    for n_hvg in hvg_settings:
        print(f"\n{'#'*70}")
        print(f"# Processing HVG{n_hvg}")
        print(f"{'#'*70}")
        
        # Create HVG subset (original adata is restored after function call)
        hvg_subset = create_hvg_subset(adata, n_hvg, batch_key=batch_key)
        
        # Print subset structure
        print_adata_info(hvg_subset, f"HVG{n_hvg} Subset")
        
        # Save output file
        output_filename = f"step4-4_filter_HVG{n_hvg}.h5ad"
        output_path = output_dir / output_filename
        
        print(f"💾 Saving HVG{n_hvg} subset to: {output_path}")
        hvg_subset.write(output_path)
        print(f"✅ Successfully saved: {output_filename}")
        
        # Delete subset to free memory before next iteration
        del hvg_subset
    
    # Summary
    print(f"\n{'#'*70}")
    print(f"# Summary: All HVG subsets created successfully!")
    print(f"{'#'*70}")
    print(f"\nGenerated files:")
    for n_hvg in hvg_settings:
        output_filename = f"step4-4_filter_HVG{n_hvg}.h5ad"
        output_path = output_dir / output_filename
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {output_filename} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {output_filename} (not found)")
    print(f"\n{'#'*70}\n")


if __name__ == "__main__":
    from perf_utils import PerfRecorder
    PerfRecorder.start_now("step5-0_hvg_subset")
    main()
