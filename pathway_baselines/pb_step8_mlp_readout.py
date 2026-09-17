#!/usr/bin/env python3
"""
Pathway Baselines step8: readout-level comparison (key addition for the
pathway-scoring baseline response)
==========================================================
Raw score matrices and distilled embeddings operate at different levels and
cannot be compared directly. This script feeds four activity matrices
(X_mod = our fluctuation activity / X_aucell / X_gsva / X_ucell) into the
**same PathwayMLP, same teacher (X_special_harmony_40), same training
protocol** (identical to step5-3), producing readout-aligned distilled
embeddings written into pb_scores.h5ad:
  X_mod_mlp / X_aucell_mlp / X_gsva_mlp / X_ucell_mlp
pb_step6 / pb_step7 then evaluate them (the EMBEDDINGS lists already
include these keys; completed items are skipped on resume).

Sanity check: X_mod_mlp should match the frozen pathway_mlp value 0.7702
(protocol reproduction check); comparing all four matrices on the same
footing shows which activity construction is most useful for integration.

Env: conda activate deepfusion (requires torch)
"""
import pickle
import time
from pathlib import Path

import anndata
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent  # project root
PB = Path(__file__).parent
SCORES = PB / "results/pb_scores.h5ad"

SEED = 42
EPOCHS = 300
PATIENCE = 15
DELTA = 0.001


class PathwayMLP(nn.Module):
    """Identical to step5-3: module_dim -> 512 -> 512 -> target_dim."""
    def __init__(self, module_dim, target_dim, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(module_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def train_one(X: np.ndarray, Y: np.ndarray, device) -> np.ndarray:
    """Training protocol identical to step5-3: cosine loss, AdamW, 10% val early stopping, seed=42."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    idx = np.arange(len(X))
    tr, va = train_test_split(idx, test_size=0.1, random_state=SEED, shuffle=True)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
    tr_ld = DataLoader(ds, batch_size=128, sampler=torch.utils.data.SubsetRandomSampler(tr), drop_last=True)
    va_ld = DataLoader(ds, batch_size=128, sampler=torch.utils.data.SubsetRandomSampler(va))

    model = PathwayMLP(X.shape[1], Y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best, best_state, stall = -np.inf, None, 0
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = 1 - F.cosine_similarity(model(xb), yb).mean()
            loss.backward(); opt.step()
        model.eval(); sims = []
        with torch.no_grad():
            for xb, yb in va_ld:
                sims.append(F.cosine_similarity(model(xb.to(device)), yb.to(device)).mean().item())
        sim = float(np.mean(sims))
        if sim > best + DELTA:
            best, best_state, stall = sim, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            stall += 1
            if stall >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        emb = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    print(f"    best val cos={best:.4f} @ ~epoch {epoch+1} | {(time.perf_counter()-t0)/60:.1f} min")
    return emb.astype(np.float32)


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {device}")
    adata = anndata.read_h5ad(SCORES)
    cache = anndata.read_h5ad(ROOT / "enhanced_results/DeepFusion/cache/preprocessed_data.h5ad")
    assert (adata.obs_names == cache.obs_names).all(), "inconsistent cell order!"
    Y = np.asarray(cache.obsm['X_special_harmony_40'], dtype=np.float32)

    # our fluctuation activity (the main model's module input, from cache)
    if 'X_mod' not in adata.obsm:
        mod_mat = torch.load(ROOT / "enhanced_results/DeepFusion/cache/module_matrix_cache.pt",
                             map_location='cpu').numpy().astype(np.float32)
        assert mod_mat.shape[0] == adata.n_obs
        adata.obsm['X_mod'] = mod_mat
        print(f"Added X_mod (our fluctuation activity) {mod_mat.shape}")

    for name in ['X_mod', 'X_aucell', 'X_gsva', 'X_ucell']:
        out_key = f"{name}_mlp"
        if out_key in adata.obsm:
            print(f"Skipping {out_key} (already exists)"); continue
        X = np.asarray(adata.obsm[name], dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0)
        print(f"\nTraining {out_key} (input {X.shape})...")
        adata.obsm[out_key] = train_one(X, Y, device)
        adata.write_h5ad(SCORES)  # save after each one
        print(f"  ✅ {out_key} written and saved")

    print("\nAll done. Next run pb_step6 / pb_step7 (only new keys are evaluated automatically)")


if __name__ == '__main__':
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from perf_utils import PerfRecorder
    PerfRecorder.start_now("pb_step8_mlp_readout")
    main()
