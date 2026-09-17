import scanpy as sc
import pandas as pd
import numpy as np
from scipy.io import mmread
import anndata
import os
import sys
from pathlib import Path

# Unified performance recording (same standard as all steps); run from data/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from perf_utils import PerfRecorder
PerfRecorder.start_now("step0_create_h5ad")

# ==================== 1. Read Counts matrix ====================
print("Reading counts matrix...")
counts = mmread("counts.mtx").T.tocsr()  # Transpose to cells x genes

# Read cell/gene names
with open("cells.txt") as f:
    cells = [line.strip() for line in f]
with open("genes.txt") as f:
    genes = [line.strip() for line in f]

print(f"Counts matrix shape: {counts.shape}")
print(f"Number of cells: {len(cells)}")
print(f"Number of genes: {len(genes)}")

# ==================== 2. Read Metadata ====================
print("Reading metadata...")
meta = pd.read_csv("sobj_seurat_processed_metadata.csv", index_col=0)

# Ensure metadata index is string
meta.index = meta.index.astype(str)

# ==================== 3. Create AnnData ====================
print("Creating AnnData...")

# Ensure cells are also strings
cells = [str(c) for c in cells]

# Find common cells, preserving original cells.txt order
# Fix: do not use set, preserve original order
meta_index_set = set(meta.index)
common_cells = [c for c in cells if c in meta_index_set]
print(f"Common cells: {len(common_cells)}")

# Create cell-to-index mapping
cell_order_dict = {cell: idx for idx, cell in enumerate(cells)}

# Get indices of common cells in the original counts matrix (preserve cells.txt order)
common_indices = [cell_order_dict[cell] for cell in common_cells]

# Subset counts matrix, preserving cells.txt order
counts_subset = counts[common_indices, :]

# Subset metadata, reorder according to cells.txt
meta_subset = meta.loc[common_cells]

# Verify order consistency
assert list(meta_subset.index) == common_cells, "Metadata order does not match cells!"
assert counts_subset.shape[0] == len(common_cells), "Counts row count does not match number of cells!"

# Create AnnData
adata = anndata.AnnData(
    X=counts_subset,
    obs=meta_subset,
    var=pd.DataFrame(index=genes)
)

print(f"AnnData shape: {adata.shape}")

# ==================== 4. Add UMAP ====================
print("Adding UMAP...")
umap = pd.read_csv("sobj_umap_matrix_v5.csv", index_col=0)

# Align UMAP with adata.obs, in the order of adata.obs
umap = umap.loc[adata.obs.index]
adata.obsm["X_Seurat_umap"] = umap.values

print(f"UMAP shape: {umap.shape}")

# ==================== 5. Optional: Add PCA ====================
pca_file = "sobj_pca_matrix_v5.csv"
if os.path.exists(pca_file):
    print("Adding PCA...")
    pca = pd.read_csv(pca_file, index_col=0)
    # Align PCA with adata.obs
    pca = pca.loc[adata.obs.index]
    adata.obsm["X_Seurat_pca"] = pca.values
    print(f"PCA shape: {pca.shape}")

# ==================== 6. Save ====================
print("Saving h5ad...")
adata.write("ifnb_seurat_processed_with_batch.h5ad")

print(f"\n✅ Done!")
print(f"Cells: {adata.n_obs}")
print(f"Genes: {adata.n_vars}")
print(f"Metadata columns: {list(adata.obs.columns)}")
print(f"Obsm keys: {list(adata.obsm.keys())}")

# ==================== 7. Validation ====================
print("\n=== Validation ===")
print(f"First 3 cell IDs: {list(adata.obs.index[:3])}")
print(f"Validation: first 3 adata.obs.index match cells.txt: {list(adata.obs.index[:3]) == cells[:3]}")
