"""Label-free graph checks (plan §6.1). Returns a plain dict; nothing here reads val/test cells.

Not implemented: external agreement with STRING / TRRUST (needs data files that
aren't part of GEARS; to be added when those files are available).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from .config import GSRConfig
from .data import TrainingCells, rows_csr
from .pretrain import _corr, collapse_metrics
from .utils import graph_stats

logger = logging.getLogger(__name__)


def _undirected_sets(edge_index: np.ndarray, n: int):
    s, d = edge_index
    m = s != d
    ids = np.unique(np.minimum(s[m], d[m]).astype(np.int64) * n + np.maximum(s[m], d[m]))
    nbrs = [set() for _ in range(n)]
    u, v = ids // n, ids % n
    for a, b in zip(u.tolist(), v.tolist()):
        nbrs[a].add(b)
        nbrs[b].add(a)
    return ids, nbrs


def _degree_matched_random(pairs: np.ndarray, deg: np.ndarray, rng: np.random.Generator, n_bins: int = 10) -> np.ndarray:
    """For each pair (u, v), a random partner v' of u from v's degree bin."""
    edges = np.unique(np.quantile(deg, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, deg, side="right") - 1, 0, max(len(edges) - 2, 0))
    members = {b: np.flatnonzero(bins == b) for b in np.unique(bins)}
    out = pairs.copy()
    for b, idx in members.items():
        sel = np.flatnonzero(bins[pairs[:, 1]] == b)
        out[sel, 1] = rng.choice(idx, len(sel))
    out = out[out[:, 0] != out[:, 1]]
    return out


@torch.no_grad()
def graph_checks(adata, tc: TrainingCells, a0_edge_index: torch.Tensor, refined_edge_index: torch.Tensor,
                 edits: pd.DataFrame, embeddings: Dict[str, torch.Tensor], cfg: GSRConfig, seed: int,
                 device: torch.device) -> Dict[str, Any]:
    n = len(adata.var)
    rng = np.random.default_rng(seed)
    a0 = a0_edge_index.cpu().numpy()
    ref = refined_edge_index.cpu().numpy()
    out: Dict[str, Any] = {"a0": graph_stats(a0, n), "refined": graph_stats(ref, n)}

    # H2 hit rate: |r| on held-out half H2 above threshold.
    idx = tc.h2_idx
    if len(idx) > cfg.pos_max_cells:
        idx = np.sort(rng.choice(idx, cfg.pos_max_cells, replace=False))
    X = torch.from_numpy(rows_csr(adata, idx).toarray()).to(device)
    C = _corr(X).abs()
    del X

    def hit(pairs: np.ndarray) -> float:
        if len(pairs) == 0:
            return float("nan")
        p = torch.from_numpy(pairs).to(device)
        return float((C[p[:, 0], p[:, 1]] > cfg.h2_hit_threshold).float().mean())

    added = edits[edits.action.isin(["add", "topup"])][["source", "target"]].to_numpy()
    a0_ids, a0_nbrs = _undirected_sets(a0, n)
    a0_pairs = np.stack([a0_ids // n, a0_ids % n], 1)
    deg_a0 = np.array([len(s) for s in a0_nbrs])
    ref_ids, ref_nbrs = _undirected_sets(ref, n)
    deg_ref = np.array([len(s) for s in ref_nbrs])
    rand = _degree_matched_random(added, deg_ref, rng) if len(added) else added
    out["h2"] = {
        "threshold": cfg.h2_hit_threshold,
        "hit_rate_added": hit(added),
        "hit_rate_added_only": hit(edits[edits.action == "add"][["source", "target"]].to_numpy()),
        "hit_rate_topup": hit(edits[edits.action == "topup"][["source", "target"]].to_numpy()),
        "hit_rate_degree_matched_random": hit(rand),
        "hit_rate_a0": hit(a0_pairs),
        "hit_rate_removed": hit(edits[edits.action == "remove"][["source", "target"]].to_numpy()),
    }
    h = out["h2"]
    h["passes"] = bool(len(added) == 0 or h["hit_rate_added"] > h["hit_rate_degree_matched_random"])
    del C

    # Coverage of low-degree genes (undirected A0 degree < d_min).
    low = deg_a0 < cfg.d_min
    out["coverage"] = {
        "low_degree_genes": int(low.sum()),
        "low_degree_mean_degree_a0": float(deg_a0[low].mean()) if low.any() else float("nan"),
        "low_degree_mean_degree_refined": float(deg_ref[low].mean()) if low.any() else float("nan"),
        "undirected_isolated_a0": int((deg_a0 == 0).sum()),
        "undirected_isolated_refined": int((deg_ref == 0).sum()),
    }

    # Neighbourhood overlap with A0.
    jac = np.array([len(a & b) / len(a | b) if (a or b) else 1.0 for a, b in zip(a0_nbrs, ref_nbrs)])
    out["jaccard_to_a0"] = {"mean": float(jac.mean()), "median": float(np.median(jac)), "min": float(jac.min()),
                            "edge_jaccard": float(len(np.intersect1d(a0_ids, ref_ids)) / max(len(np.union1d(a0_ids, ref_ids)), 1))}

    # Collapse monitors on the pretrained embeddings.
    sample = torch.from_numpy(rng.permutation(n)[:min(cfg.collapse_sample, n)])
    out["collapse"] = {v: collapse_metrics(z.float(), sample) for v, z in embeddings.items()}
    out["collapse_passes"] = bool(all(m["cos_std"] > 0.05 and m["effective_rank"] > 5 for m in out["collapse"].values()))
    out["edits"] = edits.action.value_counts().to_dict()
    logger.info("graph checks: H2 %s | coverage %s | jaccard %.3f | collapse ok %s",
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["h2"].items()},
                out["coverage"], out["jaccard_to_a0"]["mean"], out["collapse_passes"])
    if not (h["passes"] and out["collapse_passes"]):
        logger.warning("graph fails the plan §6.1 checks (H2 passes=%s, collapse passes=%s)",
                       h["passes"], out["collapse_passes"])
    return out
