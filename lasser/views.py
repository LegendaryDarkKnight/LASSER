"""Per-gene views (plan §3): E (expression), R (response), S (structure), optional X.

Every view is computed from training cells / training perturbations / A0 only.
Genes without signal in a view (zero variance, or isolated in A0 for S) are marked
``missing``: their feature row is exactly zero and the encoder replaces it with a
learned "missing" token, so they aren't read as real, identical genes.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .config import GSRConfig
from .data import TrainingCells, build_metacells, group_means, rows_csr

logger = logging.getLogger(__name__)

_EPS = 1e-8


@dataclass
class ViewFeatures:
    x: torch.Tensor        # (num_genes, dim) float32, zero rows where missing
    missing: torch.Tensor  # (num_genes,) bool


def _standardise_columns(Z: np.ndarray, keep: np.ndarray) -> np.ndarray:
    out = np.zeros_like(Z, dtype=np.float32)
    if keep.any():
        mu = Z[keep].mean(0)
        sd = Z[keep].std(0)
        sd[sd < _EPS] = 1.0
        out[keep] = (Z[keep] - mu) / sd
    return out


def profile_features(P: np.ndarray, dim: int, seed: int) -> ViewFeatures:
    """Gene profiles (genes x samples) -> row-standardised -> PCA -> standardised dims."""
    from sklearn.decomposition import PCA

    P = np.asarray(P, dtype=np.float64)
    sd = P.std(1)
    missing = sd < _EPS
    keep = ~missing
    Zr = np.zeros_like(P)
    Zr[keep] = (P[keep] - P[keep].mean(1, keepdims=True)) / sd[keep, None]
    dim = int(min(dim, P.shape[1], max(int(keep.sum()) - 1, 1)))
    Z = np.zeros((P.shape[0], dim))
    if keep.sum() > dim:
        Z[keep] = PCA(n_components=dim, random_state=seed).fit_transform(Zr[keep])
    Z = _standardise_columns(Z, keep)
    return ViewFeatures(torch.from_numpy(Z).float(), torch.from_numpy(missing))


def expression_view(adata, tc: TrainingCells, cfg: GSRConfig, seed: int) -> ViewFeatures:
    X = rows_csr(adata, tc.cell_idx)
    M = build_metacells(X, cfg.n_metacells, cfg.metacell_svd_dim, seed)
    return profile_features(M.T, cfg.e_dim, seed)


def response_profiles(adata, tc: TrainingCells) -> np.ndarray:
    """(genes x training perturbations) mean delta vs mean control, training cells only."""
    X = rows_csr(adata, tc.cell_idx)
    cond = tc.conditions[tc.cell_idx]
    labels, means = group_means(X, cond)
    is_ctrl = labels == "ctrl"
    if not is_ctrl.any():
        raise ValueError("no control cells among training cells")
    delta = means[~is_ctrl] - means[is_ctrl][0]
    return delta.T


def response_view(adata, tc: TrainingCells, cfg: GSRConfig, seed: int) -> ViewFeatures:
    return profile_features(response_profiles(adata, tc), cfg.r_dim, seed)


def undirected_simple(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Both directions, no self-loops, no duplicates (sorted by source)."""
    ei = edge_index.cpu()
    ei = torch.cat([ei, ei.flip(0)], 1)
    ei = ei[:, ei[0] != ei[1]]
    ids = torch.unique(ei[0] * num_nodes + ei[1])
    return torch.stack([ids // num_nodes, ids % num_nodes])


def deepwalk(edge_index: torch.Tensor, num_nodes: int, cfg: GSRConfig, seed: int,
             device: torch.device) -> ViewFeatures:
    """DeepWalk (uniform walks + skip-gram with negative sampling) on A0.

    Written out in plain torch instead of PyG's Node2Vec, whose random walks need
    pyg-lib or torch-cluster, which GEARS doesn't depend on.
    """
    g = torch.Generator().manual_seed(seed)
    ei = undirected_simple(edge_index, num_nodes)
    src, col = ei[0], ei[1]
    deg = torch.bincount(src, minlength=num_nodes)
    rowptr = torch.cat([torch.zeros(1, dtype=torch.long), deg.cumsum(0)])
    missing = deg == 0
    starts = torch.arange(num_nodes).repeat(cfg.s_walks_per_node)
    starts = starts[deg[starts] > 0]
    emb_in = torch.randn(num_nodes, cfg.s_dim, generator=g) * 0.1
    if len(starts) == 0:
        logger.warning("A0 has no edges; structure view is all missing")
        return ViewFeatures(torch.zeros(num_nodes, cfg.s_dim), torch.ones(num_nodes, dtype=torch.bool))

    cur, walks = starts, [starts]
    for _ in range(cfg.s_walk_length - 1):
        off = (torch.rand(len(cur), generator=g) * deg[cur]).long()
        cur = col[rowptr[cur] + off]
        walks.append(cur)
    W = torch.stack(walks, 1)
    a, b = [], []
    for o in range(1, min(cfg.s_window, cfg.s_walk_length - 1) + 1):
        a += [W[:, :-o].reshape(-1), W[:, o:].reshape(-1)]
        b += [W[:, o:].reshape(-1), W[:, :-o].reshape(-1)]
    centers, contexts = torch.cat(a), torch.cat(b)

    emb_in = emb_in.to(device).requires_grad_(True)
    emb_out = (torch.randn(num_nodes, cfg.s_dim, generator=g) * 0.1).to(device).requires_grad_(True)
    opt = torch.optim.Adam([emb_in, emb_out], lr=cfg.s_lr)
    n, bs = len(centers), cfg.s_batch_size
    for epoch in range(cfg.s_epochs):
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            c = centers[idx].to(device)
            x = contexts[idx].to(device)
            neg = torch.randint(0, num_nodes, (len(idx), cfg.s_neg), generator=g).to(device)
            zc = emb_in[c]
            pos = (zc * emb_out[x]).sum(-1)
            negs = torch.bmm(emb_out[neg], zc.unsqueeze(-1)).squeeze(-1)
            loss = -F.logsigmoid(pos).mean() - F.logsigmoid(-negs).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        logger.info("deepwalk epoch %d: loss %.4f over %d pairs", epoch + 1, total / n, n)
    Z = emb_in.detach().cpu().numpy()
    keep = ~missing.numpy()
    return ViewFeatures(torch.from_numpy(_standardise_columns(Z, keep)).float(), missing)


def external_view(path: str, gene_list: List[str], dim: int, seed: int) -> ViewFeatures:
    """Pretrained gene embeddings from a file: .csv/.tsv (gene index) or .pkl/.pt dict gene->vector."""
    import pandas as pd
    from sklearn.decomposition import PCA

    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t" if ext != ".csv" else ",", index_col=0)
        table = {str(k): v for k, v in zip(df.index, df.values)}
    elif ext == ".pt":
        table = {str(k): np.asarray(v) for k, v in torch.load(path, map_location="cpu").items()}
    else:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if hasattr(obj, "index") and hasattr(obj, "values"):
            table = {str(k): v for k, v in zip(obj.index, obj.values)}
        else:
            table = {str(k): np.asarray(v) for k, v in obj.items()}
    d = len(next(iter(table.values())))
    Z = np.zeros((len(gene_list), d))
    missing = np.ones(len(gene_list), dtype=bool)
    for i, gname in enumerate(gene_list):
        if gname in table:
            Z[i] = table[gname]
            missing[i] = False
    keep = ~missing
    if d > dim and keep.sum() > dim:
        Zp = np.zeros((len(gene_list), dim))
        Zp[keep] = PCA(n_components=dim, random_state=seed).fit_transform(Z[keep])
        Z = Zp
    logger.info("view X: %d / %d genes found in %s", int(keep.sum()), len(gene_list), path)
    return ViewFeatures(torch.from_numpy(_standardise_columns(Z, keep)).float(), torch.from_numpy(missing))


def build_views(adata, tc: TrainingCells, a0_edge_index: torch.Tensor, gene_list: List[str],
                cfg: GSRConfig, seed: int, device: torch.device) -> Dict[str, ViewFeatures]:
    views: Dict[str, ViewFeatures] = {}
    n = len(gene_list)
    for v in cfg.views:
        if v == "E":
            views[v] = expression_view(adata, tc, cfg, seed)
        elif v == "R":
            views[v] = response_view(adata, tc, cfg, seed)
        elif v == "S":
            views[v] = deepwalk(a0_edge_index, n, cfg, seed, device)
        elif v == "X":
            views[v] = external_view(cfg.x_path, gene_list, cfg.x_dim, seed)
        logger.info("view %s: dim %d, missing genes %d", v, views[v].x.shape[1], int(views[v].missing.sum()))
    return views
