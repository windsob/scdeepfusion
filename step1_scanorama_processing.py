import scanpy as sc
import numpy as np
import pandas as pd
import torch
import warnings
from pathlib import Path
import os
import math
import sys
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import umap.umap_ as umap
from matplotlib.lines import Line2D
import leidenalg
import igraph as ig
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import scipy.sparse
from scipy import stats
import scanorama

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
warnings.filterwarnings('ignore')
sc.settings.set_figure_params(dpi=80, facecolor='white')

class ScanoramaProcessor:
    def __init__(self, h5ad_path, config=None):
        self.h5ad_path = Path(h5ad_path)
        self._adata = None
        self.config = config or self.DefaultConfig()
        
        self.config.base_dir.mkdir(parents=True, exist_ok=True)
        (self.config.base_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.config.base_dir / "results").mkdir(parents=True, exist_ok=True)
        (self.config.base_dir / "meta").mkdir(parents=True, exist_ok=True)
        
    class DefaultConfig:
        def __init__(self):
            self.base_dir = Path("./enhanced_results")
            self.umap_n_neighbors = 15
            self.umap_min_dist = 0.3
            self.batch_column = "batch"
            self.umap_seed = 42
            self.memory_mode = 'high'
            
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            
            def _get_device():
                if torch.cuda.is_available():
                    return torch.device('cuda')
                elif torch.backends.mps.is_available():
                    return torch.device('mps')
                else:
                    return torch.device('cpu')
            self.device = _get_device()

    @property
    def adata(self):
        if self._adata is None:
            print("Make sure the adata.X is raw count!")
            self._adata = sc.read_h5ad(self.h5ad_path)
            self._adata.layers["raw"] = self._adata.X.copy()
            sc.pp.normalize_total(self._adata, target_sum=1e4)
            sc.pp.log1p(self._adata)
            self._adata.layers['normalized'] = self._adata.X.copy()
            self._adata.X = self._adata.layers["raw"].copy()
        return self._adata

    def plot_umap(self, adata, umap_key, color_by, title, save_path, palette_name='tab20', legend=True, size=1):
        umap_emb = adata.obsm[umap_key]
        color_values = adata.obs[color_by]
        
        plt.figure(figsize=(8, 6))
        
        if color_values.dtype.name == 'category':
            color_codes = color_values.cat.codes.values
            unique_labels = color_values.cat.categories
        else:
            if not pd.api.types.is_numeric_dtype(color_values):
                color_values = color_values.astype('category')
                color_codes = color_values.cat.codes.values
                unique_labels = color_values.cat.categories
            else:
                color_codes = color_values.values
                unique_labels = None
        
        is_categorical = unique_labels is not None
        if is_categorical and len(unique_labels) > 50:
            is_categorical = False
        
        if is_categorical:
            n_categories = len(unique_labels)
            cmap = plt.get_cmap(palette_name, n_categories) if n_categories <= 20 else plt.get_cmap('viridis', n_categories)
            scatter = plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=color_codes, cmap=cmap, s=size, alpha=0.6)
            
            if legend and n_categories < 30:
                handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i),
                                markersize=8, label=str(label)[:30]) for i, label in enumerate(unique_labels)]
                plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title=color_by[:15], fontsize='x-small')
        else:
            scatter = plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=color_codes, cmap='viridis', s=size, alpha=0.6)
            plt.colorbar(scatter, label=color_by)
        
        plt.title(title)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def compute_umap(self, embedding, n_neighbors=None, min_dist=None):
        if n_neighbors is None:
            n_neighbors = self.config.umap_n_neighbors
        if min_dist is None:
            min_dist = self.config.umap_min_dist
            
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=self.config.umap_seed)
        return reducer.fit_transform(embedding)

    def leiden_clustering(self, adata, adjacency=None, resolution=0.8, random_state=42, n_neighbors=15):
        if adjacency is None:
            if 'X_scanorama' not in adata.obsm:
                raise ValueError("X_scanorama embeddings not found in adata.obsm")
            
            embeddings = adata.obsm['X_scanorama']
            n_cells = embeddings.shape[0]
            n_neighbors = min(n_neighbors, n_cells - 1)
            
            from scipy.sparse import lil_matrix
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            nn.fit(embeddings)
            distances, indices = nn.kneighbors()
            
            adjacency = lil_matrix((n_cells, n_cells), dtype=np.int8)
            for i in range(n_cells):
                for j in indices[i]:
                    if i != j:
                        adjacency[i, j] = 1
            adjacency = adjacency.tocsr()
        
        sources, targets = adjacency.nonzero()
        g = ig.Graph()
        g.add_vertices(adata.shape[0])
        g.add_edges(list(zip(sources, targets)))
        
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=random_state
        )
        return np.array(partition.membership)

    def _save_metadata_files(self, processed_path):
        try:
            input_filename = self.h5ad_path.stem
            metadata_filename = f"{input_filename}_metadata.csv"
            cluster_filename = "scanorama_cluster.csv"
            
            metadata_path = self.config.base_dir / "meta" / metadata_filename
            self.adata.obs.to_csv(metadata_path)
            
            if "scanorama_optimized_clusters" in self._adata.obs.columns:
                cluster_df = self._adata.obs[["scanorama_optimized_clusters"]].copy()
                cluster_df.index.name = "cell_id"
                cluster_path = self.config.base_dir / "meta" / cluster_filename
                cluster_df.to_csv(cluster_path)
            
        except Exception as e:
            print(f"Error saving metadata files: {str(e)}")
            raise

    def _perform_clustering(self, processed_path):
        try:
            if 'X_scanorama' not in self._adata.obsm:
                raise ValueError("X_scanorama embeddings not found in adata.obsm")
            
            embeddings = self._adata.obsm['X_scanorama']
            n_cells = embeddings.shape[0]
            n_neighbors = min(self.config.umap_n_neighbors, n_cells - 1)
            
            from scipy.sparse import lil_matrix
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            nn.fit(embeddings)
            distances, indices = nn.kneighbors()
            
            adjacency = lil_matrix((n_cells, n_cells), dtype=np.int8)
            for i in range(n_cells):
                for j in indices[i]:
                    if i != j:
                        adjacency[i, j] = 1
            adjacency = adjacency.tocsr()
            
            if 'X_scanorama_umap' not in self._adata.obsm:
                self._adata.obsm['X_scanorama_umap'] = self.compute_umap(embeddings)
            
            n_cells_total = n_cells
            sample_size = max(10000, int(0.1 * n_cells_total))
            sample_size = min(sample_size, n_cells_total)
            np.random.seed(42)
            sample_indices = np.random.choice(n_cells_total, sample_size, replace=False)
            
            best_score = -1
            best_clusters = None
            best_resolution = None
            resolutions = np.arange(0.1, 2.1, 0.1).round(2).tolist()
            
            for res in resolutions:
                clusters = self.leiden_clustering(self._adata, adjacency=adjacency, resolution=res, random_state=42)
                
                if len(np.unique(clusters)) == 1:
                    continue
                
                sampled_embeddings = embeddings[sample_indices]
                sampled_clusters = clusters[sample_indices]
                
                if len(np.unique(sampled_clusters)) < 2:
                    continue
                
                score = silhouette_score(sampled_embeddings, sampled_clusters, metric='euclidean')
                print(f"Resolution {res:.2f} -> {len(np.unique(clusters))} clusters | Silhouette: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_clusters = clusters
                    best_resolution = res
        
            if best_clusters is None:
                raise ValueError("Clustering failed")
            
            print(f"Selected resolution: {best_resolution:.2f} (Silhouette: {best_score:.4f} on sampled cells)")
            self._adata.obs["scanorama_optimized_clusters"] = [str(x) for x in best_clusters]
            self.cell_labels = self._adata.obs["scanorama_optimized_clusters"]
            
            scanorama_cluster_path = self.config.base_dir / "plots/step1_scanorama_clusters.png"
            self.plot_umap(self._adata, 'X_scanorama_umap', "scanorama_optimized_clusters", 
                         "Scanorama Clustering (Optimized)", scanorama_cluster_path)
            
            self._save_metadata_files(processed_path)
            
            self._adata.write(processed_path)
        except Exception as e:
            print(f"Error during clustering: {str(e)}")
            raise
            
    def _run_scanorama(self):
        batch_key = self.config.batch_column
        original_cell_count = self.adata.n_obs
        original_batches = self.adata.obs[batch_key].unique()
        original_batch_count = len(original_batches)
        
        min_batch_size = 10
        batch_counts = self.adata.obs[batch_key].value_counts()
        small_batches = batch_counts[batch_counts < min_batch_size].index.tolist()
        actual_batch_key = batch_key
        
        if small_batches:
            print(f"Merging {len(small_batches)} small batches into 'merged_small_batches' group")
            self.adata.obs['temp_merged_batch'] = self.adata.obs[batch_key].astype(str)
            self.adata.obs.loc[self.adata.obs['temp_merged_batch'].isin(small_batches), 'temp_merged_batch'] = "merged_small_batches"
            actual_batch_key = 'temp_merged_batch'
        
        valid_batches = [b for b in self.adata.obs[actual_batch_key].unique() 
                         if self.adata.obs[actual_batch_key].value_counts()[b] >= 2]
        valid_cell_mask = self.adata.obs[actual_batch_key].isin(valid_batches)
        adata_temp = self.adata[valid_cell_mask].copy()
        
        if adata_temp.n_obs == 0:
            raise ValueError("All batches filtered out - no cells available for Scanorama")
        
        valid_cell_count = np.sum(valid_cell_mask)
        n_batches = len(valid_batches)
        min_batch_size = adata_temp.obs[actual_batch_key].value_counts().min()
        
        print("Preparing batch data for Scanorama integration...")
        adata_list = []
        batch_indicies = []
        
        for batch in valid_batches:
            batch_mask = (adata_temp.obs[actual_batch_key] == batch)
            batch_adata = adata_temp[batch_mask].copy()
            
            # Ensure we're using normalized data for Scanorama
            if scipy.sparse.issparse(batch_adata.layers['normalized']):
                batch_adata.X = batch_adata.layers['normalized'].toarray()
            else:
                batch_adata.X = batch_adata.layers['normalized'].copy()
                
            adata_list.append(batch_adata)
            batch_indicies.append(np.where(batch_mask)[0])
            print(f"Batch '{batch}': {batch_adata.n_obs} cells")
        
        try:
            print(f"Starting Scanorama integration with {len(adata_list)} batches...")
            scanorama.integrate_scanpy(adata_list)
            
            scanorama_emb_list = [ad.obsm['X_scanorama'] for ad in adata_list]
            scanorama_emb = np.concatenate(scanorama_emb_list, axis=0)
            
            scanorama_idx = np.concatenate(batch_indicies)
            adata_temp.obsm['X_scanorama'] = scanorama_emb[np.argsort(scanorama_idx)]
            
            print(f"Scanorama embedding dimensions: {scanorama_emb.shape}")
            
        except Exception as e:
            raise RuntimeError(f"Scanorama failed: {str(e)}")
        
        # Calculate UMAP on Scanorama embeddings
        adata_temp.obsm['X_scanorama_umap'] = self.compute_umap(adata_temp.obsm['X_scanorama'])
        print(f"Generated Scanorama UMAP embedding: {adata_temp.obsm['X_scanorama_umap'].shape}")
        
        # Create full-sized arrays to store results for all cells
        scanorama_emb_full = np.zeros((original_cell_count, scanorama_emb.shape[1]))
        scanorama_umap_full = np.zeros((original_cell_count, 2))
        
        # Create mapping for valid cells
        valid_idx = np.where(valid_cell_mask.values)[0]
        temp_idx_map = {name: idx for idx, name in enumerate(adata_temp.obs_names)}
        temp_idx = np.array([temp_idx_map[name] for name in self.adata.obs_names[valid_idx]])
        
        # Map results back to original positions
        scanorama_emb_full[valid_idx] = adata_temp.obsm['X_scanorama'][temp_idx]
        scanorama_umap_full[valid_idx] = adata_temp.obsm['X_scanorama_umap'][temp_idx]
        
        print(f"Mapped Scanorama embedding size: {scanorama_emb_full.shape}")
        print(f"Mapped Scanorama UMAP embedding size: {scanorama_umap_full.shape}")
        
        report = f"""
        Scanorama Integration Report:
          Original cells: {original_cell_count} | Batches: {original_batch_count}
          Valid cells: {valid_cell_count} | Valid batches: {n_batches}
          Min batch size: {min_batch_size}
        """
        print(report)
        
        return scanorama_emb_full, scanorama_umap_full
    
    def process_with_scanorama(self):
        processed_path = self.config.base_dir / "results/step1_scanorama_processed.h5ad"
        
        if processed_path.exists():
            self._adata = sc.read_h5ad(processed_path)
            if "X_scanorama" in self._adata.obsm and "scanorama_optimized_clusters" in self._adata.obs.columns:
                return processed_path
            elif "X_scanorama" in self._adata.obsm:
                return self._perform_clustering(processed_path)
        
        # Run Scanorama integration
        scanorama_emb_full, scanorama_umap_full = self._run_scanorama()
        
        # Store results in AnnData
        self._adata.obsm["X_scanorama"] = scanorama_emb_full
        self._adata.obsm["X_scanorama_umap"] = scanorama_umap_full
        
        # Reset X to raw counts
        self._adata.X = self._adata.layers["raw"].copy()
        
        # Save processed data
        self._save_metadata_files(processed_path)
        self._adata.write(processed_path)
        
        # Perform clustering
        return self._perform_clustering(processed_path)

if __name__ == "__main__":
    from perf_utils import PerfRecorder
    PerfRecorder.start_now("step1_scanorama")
    h5ad_path = "data/ifnb_seurat_processed_with_batch.h5ad"
    class CustomConfig(ScanoramaProcessor.DefaultConfig):
        def __init__(self):
            super().__init__()
            self.batch_column = "batch"
    
    processor = ScanoramaProcessor(h5ad_path, config=CustomConfig())
    processed_path = processor.process_with_scanorama()
    print(f"Scanorama processing complete. Results saved to: {processed_path}")