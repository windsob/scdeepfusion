#!/usr/bin/env python3
"""
Step4 (Task B version): scGPT batch-aware fine-tuning integration baseline
==========================================================================
Fixes reviewer fFsE's criticism (R7): the old step4 ran the pretrained checkpoint
frozen, zero-shot, CLS-only, without batch labels - NOT the batch-integration
procedure from the original scGPT work.

This script implements the OFFICIAL scGPT integration recipe
(~/scGPT/tutorials/Tutorial_Integration.ipynb):
- Fine-tune the whole-human pretrained checkpoint on THIS dataset
- do_dab=True  (domain-adversarial batch classification, gradient reversal)
- domain_spec_batchnorm=True (DSBN, per-donor batch norm)
- use_batch_labels=True, batch = donor (obs['batch'], D-prefixed)
- GEPC/MVC + explicit_zero_prob + ECS (0.8), mask_ratio 0.4, lr 1e-4, 15 epochs
- per-seq-batch sampling (SubsetsBatchSampler): DSBN requires batch-homogeneous
  training batches (this scGPT version's _encode uses batch_labels[0])
- Output embedding X_scGPT = cell embedding (cls) of the fine-tuned model,
  extracted per donor group (DSBN needs homogeneous inference batches too)

Env: conda env `scgpt` (has torchtext); scGPT source at ~/scGPT (PYTHONPATH).
No wandb (plain stdout logging); AMP only on CUDA (MPS runs fp32).
"""

import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import scanpy as sc
import anndata
from scipy.sparse import issparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", message="Using PyTorch native attention instead of flash_attn")
warnings.filterwarnings("ignore", message="flash_attn is not installed")

sys.path.insert(0, str(Path.home() / "scGPT"))  # scGPT source checkout
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.loss import masked_mse_loss, criterion_neg_log_bernoulli
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, load_pretrained

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
    Path("logs").mkdir(exist_ok=True)
    fh = open(Path("logs") / log_name, "w")
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print(f"Logging to logs/{log_name}")

# MPS fix: torch's TransformerEncoder eval-mode fast path uses
# aten::_nested_tensor_from_mask_left_aligned, which is not implemented on MPS.
# Loop the layers manually (same math, gradients unaffected).
from torch.nn import TransformerEncoder as _TorchTransformerEncoder

def _patched_te_forward(self, src, mask=None, src_key_padding_mask=None):
    if src_key_padding_mask is not None:
        src_key_padding_mask = src_key_padding_mask.bool().contiguous()
    output = src
    for mod in self.layers:
        output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
    if self.norm is not None:
        output = self.norm(output)
    return output

_TorchTransformerEncoder.forward = _patched_te_forward

# ----------------------------- config (official integration recipe) -----------------------------
CONFIG = dict(
    seed=42,
    n_hvg=1200,          # architectural constraint of the pretrained model
    n_bins=51,
    mask_ratio=0.4,
    mask_value=-1,
    pad_value=-2,
    epochs=15,
    lr=1e-4,
    batch_size=64,
    dropout=0.2,
    schedule_ratio=0.9,
    dab_weight=1.0,      # domain-adversarial weight (batch correction strength)
    ecs_thres=0.8,
    GEPC=True,           # MVC objective
    DSBN=True,
    explicit_zero_prob=True,
    log_interval=50,
    valid_ratio=0.1,
)
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.9)
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_anndata(adata_full: anndata.AnnData, vocab: GeneVocab, cfg: dict) -> anndata.AnnData:
    """Official Preprocessor on the vocab-covered genes; batch = donor."""
    adata = adata_full.copy()

    # batch labels (donor, D-prefixed strings)
    adata.obs["str_batch"] = adata.obs["batch"].astype(str)
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    num_batch_types = len(set(batch_id_labels))
    print(f"Batch (donor) categories: {num_batch_types} -> "
          f"{sorted(adata.obs['str_batch'].unique())}")
    # Generic diagnostic: print the batch distribution only (no dataset-specific condition columns)
    print("batch distribution:")
    print(adata.obs["str_batch"].value_counts())

    # keep only genes covered by the scGPT vocab
    in_vocab = adata.var_names.isin(vocab.get_stoi() if hasattr(vocab, "get_stoi") else vocab)
    adata = adata[:, in_vocab].copy()
    print(f"Genes in vocab: {adata.n_vars}")

    use_key = "raw" if "raw" in adata.layers else "X"
    print(f"Preprocessor input layer: {use_key}")
    preprocessor = Preprocessor(
        use_key=use_key,
        filter_gene_by_counts=3,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=True,                     # our input layer is raw counts
        result_log1p_key="X_log1p",
        subset_hvg=cfg["n_hvg"],
        hvg_flavor="seurat_v3",         # raw-count data
        binning=cfg["n_bins"],
        result_binned_key="X_binned",
    )
    # 8-donor stratified HVG can hit loess singularities (small donor groups with
    # many all-zero genes). Try batch-aware first (official), fall back to pooled.
    try:
        preprocessor(adata, batch_key="str_batch")
    except ValueError as e:
        print(f"⚠️ batch-aware HVG failed ({e}); falling back to pooled HVG (batch_key=None)")
        adata = adata_full.copy()
        adata.obs["str_batch"] = adata.obs["batch"].astype(str)
        adata.obs["batch_id"] = adata.obs["str_batch"].astype("category").cat.codes.values
        in_vocab = adata.var_names.isin(vocab.get_stoi() if hasattr(vocab, "get_stoi") else vocab)
        adata = adata[:, in_vocab].copy()
        preprocessor(adata, batch_key=None)
    print(f"After preprocessing: {adata.shape}")
    return adata


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_data(tokenized, batch_labels: np.ndarray, cfg: dict):
    """Re-mask values every epoch (as the official tutorial does)."""
    masked_values = random_mask_value(
        tokenized["values"], mask_ratio=cfg["mask_ratio"],
        mask_value=cfg["mask_value"], pad_value=cfg["pad_value"],
    )
    return {
        "gene_ids": tokenized["genes"],
        "values": masked_values,
        "target_values": tokenized["values"],
        "batch_labels": torch.from_numpy(batch_labels).long(),
    }


def prepare_dataloader(data_pt, batch_size: int, shuffle: bool):
    """per-seq-batch sampling: DSBN requires batch-homogeneous batches."""
    dataset = SeqDataset(data_pt)
    subsets = []
    bl = data_pt["batch_labels"].numpy()
    for b in np.unique(bl):
        subsets.append(np.where(bl == b)[0].tolist())
    return DataLoader(
        dataset=dataset,
        batch_sampler=SubsetsBatchSampler(
            subsets, batch_size,
            intra_subset_shuffle=True, inter_subset_shuffle=shuffle, drop_last=False,
        ),
        num_workers=0, pin_memory=False,
    )


def build_model(vocab: GeneVocab, model_dir: Path, num_batch_types: int, cfg: dict, device):
    """Instantiate from the checkpoint's own config (embsize/nlayers etc.), then load weights."""
    model_configs = json.load(open(model_dir / "args.json"))
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],       # 512
        nhead=model_configs["nheads"],          # 8
        d_hid=model_configs["d_hid"],           # 512
        nlayers=model_configs["nlayers"],       # 12 (old zero-shot baseline wrongly used 6!)
        nlayers_cls=model_configs.get("n_layers_cls", 3),
        vocab=vocab,
        dropout=cfg["dropout"],
        pad_token=PAD_TOKEN,
        pad_value=cfg["pad_value"],
        do_mvc=cfg["GEPC"],
        do_dab=True,
        use_batch_labels=True,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=cfg["DSBN"],
        n_input_bins=cfg["n_bins"],
        ecs_threshold=cfg["ecs_thres"],
        explicit_zero_prob=cfg["explicit_zero_prob"],
        use_fast_transformer=False,             # no flash_attn on MPS
        pre_norm=False,
    )
    ckpt = torch.load(model_dir / "best_model.pt", map_location="cpu")
    # Key remap: the checkpoint was trained with flash_attn FlashMHA (Wqkv naming);
    # this MPS-patched scGPT copy uses torch MultiheadAttention (in_proj naming).
    # Same parameter layout [q;k;v], so a pure rename is exact.
    # (This also means the OLD zero-shot baseline silently ran with random QKV weights
    # in all layers - shape-filtered loading dropped every Wqkv key.)
    ckpt = {k.replace("Wqkv.weight", "in_proj_weight").replace("Wqkv.bias", "in_proj_bias"): v
            for k, v in ckpt.items()}
    load_pretrained(model, ckpt, verbose=False)
    n_matched = sum(1 for k, v in model.state_dict().items()
                    if k in ckpt and ckpt[k].shape == v.shape)
    print(f"Pretrained weights: {n_matched}/{len(model.state_dict())} keys matched "
          f"(rest are fine-tune-only modules: dsbn/discriminator/batch_encoder/cls_decoder "
          f"+ decoder.fc.0 & mvc_decoder.W reshaped by batch-emb concat - official behaviour)")
    model.to(device)
    return model


def train_one_epoch(model, loader, optimizer, scaler, criterion, criterion_dab, cfg, device, vocab):
    model.train()
    use_amp = device.type == "cuda"
    total_loss = total_dab = 0.0
    nb = 0
    for batch_data in loader:
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[PAD_TOKEN])

        with torch.cuda.amp.autocast(enabled=use_amp):
            output_dict = model(
                input_gene_ids, input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if cfg["DSBN"] else None,
                MVC=cfg["GEPC"], ECS=cfg["ecs_thres"] > 0,
            )
            masked_positions = input_values.eq(cfg["mask_value"])
            loss = criterion(output_dict["mlm_output"], target_values, masked_positions)
            if cfg["explicit_zero_prob"]:
                loss = loss + criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions)
            if cfg["GEPC"]:
                loss = loss + criterion(output_dict["mvc_output"], target_values, masked_positions)
                if cfg["explicit_zero_prob"]:
                    loss = loss + criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], target_values, masked_positions)
            if cfg["ecs_thres"] > 0:
                loss = loss + 10 * output_dict["loss_ecs"]
            loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            loss = loss + cfg["dab_weight"] * loss_dab

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item(); total_dab += loss_dab.item(); nb += 1
    return total_loss / max(nb, 1), total_dab / max(nb, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, cfg, device, vocab):
    model.eval()
    total_loss = 0.0; nb = 0
    for batch_data in loader:
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[PAD_TOKEN])
        output_dict = model(
            input_gene_ids, input_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=batch_labels if cfg["DSBN"] else None,
            MVC=cfg["GEPC"], ECS=False,
        )
        masked_positions = input_values.eq(cfg["mask_value"])
        loss = criterion(output_dict["mlm_output"], target_values, masked_positions)
        total_loss += loss.item(); nb += 1
    return total_loss / max(nb, 1)


@torch.no_grad()
def extract_embeddings(model, tokenized_all, batch_ids_all, cfg, device, vocab, batch_size=64):
    """Per-donor-group extraction (DSBN uses batch_labels[0] -> homogeneous batches only)."""
    model.eval()
    n = tokenized_all["genes"].shape[0]
    embs = np.zeros((n, 0), dtype=np.float32)
    for b in np.unique(batch_ids_all):
        idx = np.where(batch_ids_all == b)[0]
        ids = tokenized_all["genes"][idx]
        vals = tokenized_all["values"][idx]
        bl = torch.from_numpy(batch_ids_all[idx]).long()
        emb_b = model.encode_batch(
            ids, vals,
            src_key_padding_mask=ids.eq(vocab[PAD_TOKEN]),
            batch_size=batch_size,
            batch_labels=bl.to(device),
            time_step=0,        # position of the <cls> token -> per-cell embedding [n, 512]
            return_np=True,
        )
        if embs.shape[1] == 0:
            embs = np.zeros((n, emb_b.shape[1]), dtype=np.float32)
        embs[idx] = emb_b
        print(f"  extracted donor group {b}: {len(idx)} cells, emb dim {emb_b.shape[1]}")
    return embs


def main():
    _enable_default_log("step4_scgpt_finetune.log")  # keep the per-epoch curve for diagnosis
    cfg = CONFIG
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device} | AMP: {device.type == 'cuda'}")

    h5ad_path = "enhanced_results/results/step3_add_scVI.h5ad"
    out_h5ad = "enhanced_results/results/step4_add_scGPT.h5ad"
    out_npy = "enhanced_results/results/scGPT_cell_embeddings.npy"
    out_model = "enhanced_results/results/scgpt_finetuned_donor.pt"
    model_dir = Path("scgpt")

    # ---- data ----
    print(f"Loading {h5ad_path} ...")
    adata_full = sc.read_h5ad(h5ad_path)
    print(f"Full data: {adata_full.shape}")

    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab[PAD_TOKEN])

    adata = prepare_anndata(adata_full, vocab, cfg)
    num_batch_types = adata.obs["batch_id"].nunique()
    # adata_full is only needed again at the final write-back; free it during
    # training (it is re-read from disk then) to cut several GB of idle memory
    del adata_full
    import gc; gc.collect()

    # tokenize train/valid for training; the all-cell tokenization (needed only
    # for final embedding extraction) is deferred to after training, so the
    # ~2 GB tokenized_all tensors are not resident during training
    all_counts = adata.layers["X_binned"]
    if issparse(all_counts):
        all_counts = all_counts.toarray()
    gene_ids = np.array([vocab[g] for g in adata.var_names], dtype=int)
    batch_ids_all = adata.obs["batch_id"].values.astype(int)

    def _tokenize(row_idx=None):
        mat = all_counts if row_idx is None else all_counts[row_idx]
        return tokenize_and_pad_batch(
            mat, gene_ids, max_len=cfg["n_hvg"] + 1, vocab=vocab,
            pad_token=PAD_TOKEN, pad_value=cfg["pad_value"],
            append_cls=True, include_zero_gene=True, return_pt=True,
        )

    train_idx, valid_idx = train_test_split(
        np.arange(adata.n_obs), test_size=cfg["valid_ratio"], shuffle=True,
        random_state=cfg["seed"],
    )
    tokenized_train = _tokenize(train_idx)
    tokenized_valid = _tokenize(valid_idx)
    bl_train, bl_valid = batch_ids_all[train_idx], batch_ids_all[valid_idx]
    print(f"Train {len(train_idx)} / valid {len(valid_idx)} cells | {num_batch_types} donor batches")

    # ---- model ----
    model = build_model(vocab, model_dir, num_batch_types, cfg, device)
    criterion = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=cfg["schedule_ratio"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ---- resume support: full training state is checkpointed every epoch ----
    resume_path = out_model + ".resume"
    start_epoch = 1
    best_val_loss = float("inf")
    best_state = None
    if os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        best_val_loss = ck["best_val_loss"]
        best_state = ck["best_state"]
        start_epoch = ck["epoch"] + 1
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        print(f"Resuming from epoch {ck['epoch']} (best val {best_val_loss:.4f})", flush=True)

    # ---- fine-tune ----
    t0 = time.perf_counter()
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        train_pt = prepare_data(tokenized_train, bl_train, cfg)
        valid_pt = prepare_data(tokenized_valid, bl_valid, cfg)
        train_loader = prepare_dataloader(train_pt, cfg["batch_size"], shuffle=True)
        valid_loader = prepare_dataloader(valid_pt, cfg["batch_size"], shuffle=False)

        tr_loss, tr_dab = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, criterion_dab, cfg, device, vocab)
        val_loss = evaluate(model, valid_loader, criterion, cfg, device, vocab)
        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        # memory telemetry per epoch: OS high-water, current RSS, MPS allocator
        import resource as _resource
        _hw = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1e9
        try:
            import psutil as _psutil
            _cur = _psutil.Process().memory_info().rss / 1e9
        except Exception:
            _cur = float('nan')
        _mps = (torch.mps.current_allocated_memory() / 1e9
                if device.type == 'mps' else float('nan'))
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} (dab {tr_dab:.4f}) | "
              f"valid mse {val_loss:.4f} | best {best_val_loss:.4f} | "
              f"{time.perf_counter()-t0:.0f}s elapsed | "
              f"mem hw {_hw:.1f}G cur {_cur:.1f}G mps {_mps:.1f}G", flush=True)
        if device.type == 'mps':
            torch.mps.empty_cache()  # return unused cached MPS blocks between epochs
        # epoch-granularity checkpoint so a kill/crash can be resumed
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "best_val_loss": best_val_loss,
                    "best_state": best_state, "torch_rng": torch.get_rng_state(),
                    "numpy_rng": np.random.get_state()}, resume_path)

    # ---- embeddings from the best checkpoint ----
    model.load_state_dict(best_state)
    print("Extracting batch-corrected cell embeddings (per donor group)...")
    tokenized_all = _tokenize()  # all cells, built only now (memory optimization)
    embs = extract_embeddings(model, tokenized_all, batch_ids_all, cfg, device, vocab)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    # ---- sanity guard: label/batch purity of the extracted embedding ----
    # Fine-tuning objectives (masked MSE + adversarial DAB) do not measure
    # biological structure, and best-val selection cannot detect a bio collapse.
    try:
        from sklearn.neighbors import NearestNeighbors
        # label column: dataset.json first (template datasets define label_key there),
        # then a fallback candidate list; warn loudly if nothing matches
        label_col = None
        try:
            from dataset_config import load_dataset_config
            _lk = load_dataset_config().get('label_key')
            if _lk and _lk in adata.obs.columns:
                label_col = _lk
        except Exception:
            pass
        if label_col is None:
            label_col = next((c for c in ('seurat_annotations', 'celltype', 'cell_type')
                              if c in adata.obs.columns), None)
        batch_col = next((c for c in ('batch', 'str_batch')
                          if c in adata.obs.columns), None)
        if label_col is None:
            print(f"WARNING: no label column found for the sanity guard "
                  f"(obs: {list(adata.obs.columns)})", flush=True)
        if label_col is not None:
            labels = adata.obs[label_col].astype(str).values
            nn_idx = NearestNeighbors(n_neighbors=16).fit(embs).kneighbors(return_distance=False)
            lp = float(np.mean([np.mean(labels[nn_idx[i][1:]] == labels[i])
                                for i in range(len(embs))]))
            msg = f"Embedding sanity: 16-NN label purity {lp:.3f}"
            if batch_col is not None:
                batches = adata.obs[batch_col].astype(str).values
                bp = float(np.mean([np.mean(batches[nn_idx[i][1:]] == batches[i])
                                    for i in range(len(embs))]))
                msg += f" | batch purity {bp:.3f}"
            print(msg, flush=True)
            if lp < 0.5:
                print("WARNING: label purity < 0.5 - fine-tune likely collapsed; "
                      "re-run or reduce dab_weight/epochs.", flush=True)
    except Exception as e:
        print(f"sanity check skipped: {e}")

    # ---- write back ----
    # reload the full object from disk (it was freed during training);
    # adata (HVG subset) and adata_full share cell order
    adata_full = sc.read_h5ad(h5ad_path)
    assert (adata.obs_names == adata_full.obs_names).all()
    adata_full.obsm["X_scGPT"] = embs
    adata_full.write_h5ad(out_h5ad)
    np.save(out_npy, embs)
    torch.save(best_state, out_model)
    os.remove(resume_path)  # training completed cleanly; drop the resume checkpoint

    print("\n" + "=" * 60)
    print("scGPT batch-aware fine-tuning completed!")
    print(f"  X_scGPT: {embs.shape} -> {out_h5ad}")
    print(f"  fine-tuned model -> {out_model}")
    print(f"  total time: {(time.perf_counter()-t0)/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    from perf_utils import PerfRecorder
    PerfRecorder.start_now("step4_scGPT_finetune")
    main()
