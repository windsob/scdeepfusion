"""
DeepFusion - Pathway-MLP control

Control experiment: "Harmony with fixed pathway activity scores and an MLP".
Trains a plain MLP to map the FIXED module (pathway) activity matrix directly to
the Harmony target embedding with a cosine-similarity loss -- no gene-level branch,
no cross-attention, no learned fusion. This isolates how much of scDeepFusion's
performance comes from the pathway features themselves vs. the architecture.

Output: adata.obsm['X_pathway_mlp{suffix}'] appended to step5_add_DeepFusion.h5ad,
so step6 evaluates it side by side with all other embeddings.

Usage:
    python step5-3_pathway_mlp_control.py            # all X_special_harmony_* targets
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# -------------------------- Load step5-2 as a module (reuse Config/data pipeline) --------------------------
MAIN_PATH = Path(__file__).parent / "step5-2_deepfusion_multitoken.py"
spec = importlib.util.spec_from_file_location("deepfusion_main", MAIN_PATH)
dfm = importlib.util.module_from_spec(spec)
sys.modules["deepfusion_main"] = dfm
spec.loader.exec_module(dfm)


class PathwayMLP(nn.Module):
    """MLP: fixed pathway activity -> Harmony embedding (control, no gene branch)."""
    def __init__(self, module_dim: int, target_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(module_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
            nn.Tanh(),  # match the projection head range of the main model
        )

    def forward(self, x):
        return self.net(x)


def train_pathway_mlp(config, dataset, target_dim: int, lr=5e-4, weight_decay=1e-4,
                      batch_size=128, dropout=0.3, hidden_dim=512):
    """Same training protocol as the main Trainer: train split + held-out val for early stopping."""
    n = len(dataset)
    train_idx, val_idx = train_test_split(np.arange(n), test_size=config.val_fraction,
                                          random_state=config.seed, shuffle=True)
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=batch_size)

    model = PathwayMLP(dataset.module_matrix.shape[1], target_dim, hidden_dim, dropout).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = dfm.EarlyStopping(patience=config.patience, delta=config.delta)

    best_sim, best_state = -np.inf, None
    t0 = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            module_input = batch["modules"].to(config.device)
            target_emb = batch["embedding"].to(config.device)
            optimizer.zero_grad()
            pred = model(module_input)
            loss = 1 - F.cosine_similarity(pred, target_emb).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        sims = []
        with torch.no_grad():
            for batch in val_loader:
                module_input = batch["modules"].to(config.device)
                target_emb = batch["embedding"].to(config.device)
                sims.append(F.cosine_similarity(model(module_input), target_emb).mean().item())
        val_sim = float(np.mean(sims))
        if val_sim > best_sim:
            best_sim = val_sim
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:03d} | Val CosSim: {val_sim:.4f} | Best: {best_sim:.4f}")
        if early_stopping(val_sim, epoch + 1):
            print(f"  🛑 Early stopping at epoch {epoch+1}")
            break

    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state)

    # Inference on all cells
    model.eval()
    full_loader = DataLoader(dataset, batch_size=512, shuffle=False)
    preds = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in full_loader:
            preds.append(model(batch["modules"].to(config.device)).cpu())
    infer_s = time.perf_counter() - t0
    emb = torch.cat(preds).numpy().astype(np.float32)
    print(f"  ✅ Best Val CosSim: {best_sim:.4f} | train {train_s:.1f}s | inference {infer_s:.1f}s")
    return emb, train_s, infer_s


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pathway-MLP control")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--targets', nargs='*', default=None)
    parser.add_argument('--suffix_tag', default='')
    parser.add_argument('--data_dropout', type=float, default=0.0)
    parser.add_argument('--gene_label_path', type=str, default=None,
                        help='override module CSV (fair comparison with filtered-expert runs)')
    parser.add_argument('--keep_all_hvg', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = dfm.Config()
    config.seed = args.seed
    config.suffix_tag = args.suffix_tag
    config.data_dropout = args.data_dropout
    if args.gene_label_path:
        from pathlib import Path as _P
        config.sc_gene_label_path = _P(args.gene_label_path)
    config.keep_all_hvg = args.keep_all_hvg
    config.validate_paths()

    # Load the SAME preprocessed data as the main model (fingerprint-checked cache)
    processor = dfm.SCDataProcessor(config)
    adata, gene_modules = processor.load_data()

    # Output file produced by step5-2 (or create from processed data)
    out_path = Path(config.output_path)
    if out_path.exists():
        final_adata = __import__('anndata').read_h5ad(out_path)
        print(f"♻️ Appending control embeddings to existing {out_path}")
    else:
        final_adata = adata
        print(f"⚠️ {out_path} not found, will create it from processed data")

    harmony_embeddings = config.get_all_special_harmony_embeddings(adata)
    if args.targets:
        wanted = {t if str(t).startswith('X_') else f"X_special_harmony_{t}" for t in args.targets}
        harmony_embeddings = [e for e in harmony_embeddings if e in wanted]
        if not harmony_embeddings:
            raise ValueError(f"--targets {args.targets} matched nothing")
    print(f"📌 Targets: {harmony_embeddings}")

    total_train_s = 0.0
    total_infer_s = 0.0
    n_trained = 0
    for embedding_metric in harmony_embeddings:
        suffix = config.get_output_suffix(embedding_metric)  # includes --suffix_tag
        key = f'X_pathway_mlp{suffix}'
        if key in final_adata.obsm:
            print(f"♻️ {key} already exists, skipping")
            continue

        print(f"\n=== Pathway-MLP control for {embedding_metric} ===")
        dataset = dfm.SCNDataset(adata, gene_modules, config, embedding_metric)
        emb, train_s, infer_s = train_pathway_mlp(config, dataset, target_dim=dataset.embedding_dim)
        final_adata.obsm[key] = emb
        total_train_s += train_s
        total_infer_s += infer_s
        n_trained += 1
        print(f"✅ Stored {key} (shape {emb.shape})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_adata.write(out_path)
    print(f"\n🎉 Control embeddings saved to: {out_path}")

    # Fold key figures into the unified perf record (perf_utils, written at exit)
    _perf = globals().get('_PERF_RECORDER')
    if _perf is not None:
        n_cells = int(final_adata.n_obs)
        _perf.extra.update({
            'n_cells': n_cells,
            'n_embeddings': n_trained,
            'train_s': round(total_train_s, 1),
            'inference_s': round(total_infer_s, 1),
            'inference_cells_per_s': round(n_cells / total_infer_s, 1) if total_infer_s > 0 else None,
        })


if __name__ == "__main__":
    from perf_utils import PerfRecorder
    _PERF_RECORDER = PerfRecorder.start_now("step5-3_pathway_mlp")
    main()
