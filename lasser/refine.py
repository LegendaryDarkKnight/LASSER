"""Build the refined co-expression graph once (plan §5).

Everything here runs under ``torch.no_grad()``. Pairwise scores are computed in
chunks of ``chunk_size`` query genes and only the top ``cand_topk`` candidates per
gene are kept, so no N x N tensor is ever held.

Graph format follows GEARS's co-expression graph: directed edges source -> target,
where the sources of a target are its neighbours (SGConv aggregates at the target).
A0 edges are kept in GEARS's order and direction; edits only remove A0 edges or add
new ones (new edges are added in both directions).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import GSRConfig
from .pretrain import in_sorted, pair_ids

logger = logging.getLogger(__name__)


@dataclass
class RefinedGraph:
    edge_index: torch.Tensor   # (2, M) long, CPU
    edge_weight: torch.Tensor  # (M,) float32, CPU
    edits: pd.DataFrame        # source, target, action (remove/add/topup), score
    stats: Dict[str, float]


def _norm_embs(embs: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {v: F.normalize(z.float().to(device), dim=-1) for v, z in embs.items()}


@torch.no_grad()
def _cos_range(zn: torch.Tensor, chunk: int) -> Tuple[float, float]:
    """Global min / max of off-diagonal cosines (for GSR's min-max normalisation)."""
    n = zn.shape[0]
    lo, hi = float("inf"), float("-inf")
    for s in range(0, n, chunk):
        S = zn[s:s + chunk] @ zn.T
        r = torch.arange(s, min(s + chunk, n), device=S.device)
        S[r - s, r] = float("nan")
        lo = min(lo, float(torch.nan_to_num(S, nan=float("inf")).min()))
        hi = max(hi, float(torch.nan_to_num(S, nan=float("-inf")).max()))
    return lo, hi


class EdgeScorer:
    """E_ij = sum_v beta_v * minmax_v(cos(z_i^v, z_j^v))."""

    def __init__(self, embs: Dict[str, torch.Tensor], beta: Dict[str, float], chunk: int, device):
        self.zn = _norm_embs(embs, device)
        total = sum(beta[v] for v in self.zn)
        self.beta = {v: beta[v] / total for v in self.zn}
        self.range = {v: _cos_range(z, chunk) for v, z in self.zn.items()}
        self.chunk = chunk
        self.n = next(iter(self.zn.values())).shape[0]

    def _scale(self, v: str, c: torch.Tensor) -> torch.Tensor:
        lo, hi = self.range[v]
        return (c - lo) / max(hi - lo, 1e-12)

    def pairs(self, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return sum(self.beta[v] * self._scale(v, (z[u] * z[w]).sum(-1)) for v, z in self.zn.items())

    def rows(self, s: int, e: int) -> torch.Tensor:
        return sum(self.beta[v] * self._scale(v, z[s:e] @ z.T) for v, z in self.zn.items())

    def topk(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k candidates per gene (self excluded): (n, k) indices and scores."""
        idx, val = [], []
        for s in range(0, self.n, self.chunk):
            e = min(s + self.chunk, self.n)
            S = self.rows(s, e)
            r = torch.arange(s, e, device=S.device)
            S[r - s, r] = float("-inf")
            v, i = S.topk(k, dim=1)
            idx.append(i)
            val.append(v)
        return torch.cat(idx), torch.cat(val)


def _rank_within(groups: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Rank of each element within its group, ascending by key (vectorised)."""
    if groups.numel() == 0:
        return groups.clone()
    order = torch.argsort(keys, stable=True)
    order = order[torch.argsort(groups[order], stable=True)]
    g = groups[order]
    start = torch.ones_like(g, dtype=torch.bool)
    start[1:] = g[1:] != g[:-1]
    first = torch.cummax(torch.where(start, torch.arange(len(g), device=g.device), torch.zeros_like(g)), 0).values
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(g), device=g.device) - first
    return rank


def _quantile_map(x: torch.Tensor, ref_scores: torch.Tensor, ref_weights: torch.Tensor) -> torch.Tensor:
    """Map scores to the A0 weight distribution: CDF over A0 edge scores -> A0 weight quantile."""
    if ref_scores.numel() == 0:
        return torch.ones_like(x)
    rs = torch.sort(ref_scores).values
    q = torch.searchsorted(rs, x.contiguous(), right=True).float() / rs.numel()
    rw = torch.sort(ref_weights.float()).values
    return rw[torch.round(q.clamp(0, 1) * (rw.numel() - 1)).long()]


@torch.no_grad()
def refine_graph(embs: Dict[str, torch.Tensor], a0_edge_index: torch.Tensor, a0_edge_weight: torch.Tensor,
                 cfg: GSRConfig, k_gears: int, device) -> RefinedGraph:
    n = next(iter(embs.values())).shape[0]
    scorer = EdgeScorer(embs, cfg.beta, cfg.chunk_size, device)
    src = a0_edge_index[0].to(device)
    dst = a0_edge_index[1].to(device)
    w0 = a0_edge_weight.float().to(device)
    loop = src == dst
    score_a0 = scorer.pairs(src, dst)

    # 1. Remove: for each target gene, the fraction r_minus of its non-loop A0 edges with lowest E.
    keep = torch.ones(len(src), dtype=torch.bool, device=device)
    nl = torch.nonzero(~loop, as_tuple=True)[0]
    if cfg.r_minus > 0 and len(nl):
        deg = torch.bincount(dst[nl], minlength=n)
        n_drop = torch.floor(deg.float() * cfg.r_minus).long()
        rank = _rank_within(dst[nl], score_a0[nl])
        drop = rank < n_drop[dst[nl]]
        keep[nl[drop]] = False

    # Undirected A0 pairs: never re-added as "new".
    a0_ids = torch.unique(pair_ids(src[~loop], dst[~loop], n))

    # 2. Candidates: top-k by E, not A0 neighbours, above the score floor.
    cidx, cval = scorer.topk(min(cfg.cand_topk, n - 1))
    rows = torch.arange(n, device=device)[:, None].expand_as(cidx)
    valid = (cval >= cfg.score_floor) & ~in_sorted(pair_ids(rows, cidx, n).reshape(-1), a0_ids).reshape(cidx.shape)
    order_in_row = torch.cumsum(valid.long(), 1)
    add_mask = valid & (order_in_row <= cfg.m_plus)

    def _dedup(u, v, sc, kind):
        _, first = np.unique(pair_ids(u, v, n).cpu().numpy(), return_index=True)
        first = torch.from_numpy(np.sort(first)).to(device)
        return u[first], v[first], sc[first], kind[first]

    au, av, asc = rows[add_mask], cidx[add_mask], cval[add_mask]
    au, av, asc, ak = _dedup(au, av, asc, torch.zeros_like(au))

    # 3. Min-degree top-up. Degree = in-degree without loops; a new pair adds 1 to both ends.
    deg = torch.bincount(dst[keep & ~loop], minlength=n) + torch.bincount(au, minlength=n) \
        + torch.bincount(av, minlength=n)
    need = (cfg.d_min - deg).clamp(min=0)
    if bool((need > 0).any()):
        chosen = torch.sort(pair_ids(au, av, n)).values
        free = valid & ~in_sorted(pair_ids(rows, cidx, n).reshape(-1), chosen).reshape(cidx.shape)
        top = free & (torch.cumsum(free.long(), 1) <= need[:, None])
        tu, tv, ts = rows[top], cidx[top], cval[top]
        tu, tv, ts, tk = _dedup(tu, tv, ts, torch.ones_like(tu))
        au, av, asc, ak = torch.cat([au, tu]), torch.cat([av, tv]), torch.cat([asc, ts]), torch.cat([ak, tk])
        short = int(((need - torch.bincount(tu, minlength=n) - torch.bincount(tv, minlength=n)) > 0).sum())
        if short:
            logger.warning("%d genes stay below d_min=%d (not enough valid candidates)", short, cfg.d_min)

    # 4. Degree cap on new edges: target in-degree (no loops) <= k + m_plus, keeping best-scored new edges.
    new_src = torch.cat([au, av])
    new_dst = torch.cat([av, au])
    new_sc = torch.cat([asc, asc])
    new_kind = torch.cat([ak, ak])
    cap = k_gears + cfg.m_plus
    base_deg = torch.bincount(dst[keep & ~loop], minlength=n)
    rank = _rank_within(new_dst, -new_sc)
    room = (cap - base_deg).clamp(min=max(cfg.d_min, 0))
    ok = rank < room[new_dst]
    new_src, new_dst, new_sc, new_kind = new_src[ok], new_dst[ok], new_sc[ok], new_kind[ok]

    # 5. Weights: kept A0 edges keep theirs; new edges get gamma x quantile-mapped score.
    ref = ~loop
    new_w = cfg.gamma * _quantile_map(new_sc, score_a0[ref], w0[ref]).float()

    ei = torch.cat([torch.stack([src[keep], dst[keep]]), torch.stack([new_src, new_dst])], 1)
    ew = torch.cat([w0[keep], new_w])

    removed = torch.nonzero(~keep, as_tuple=True)[0]
    action = np.array(["add", "topup"])
    edits = pd.concat([
        pd.DataFrame({"source": src[removed].cpu().numpy(), "target": dst[removed].cpu().numpy(),
                      "action": "remove", "score": score_a0[removed].cpu().numpy()}),
        pd.DataFrame({"source": new_src.cpu().numpy(), "target": new_dst.cpu().numpy(),
                      "action": action[new_kind.cpu().numpy()], "score": new_sc.cpu().numpy()}),
    ], ignore_index=True)

    stats = {
        "a0_edges": int(len(src)), "removed": int(len(removed)),
        "added_directed": int((new_kind == 0).sum()), "topup_directed": int((new_kind == 1).sum()),
        "refined_edges": int(ei.shape[1]),
        "new_weight_mean": float(new_w.mean()) if len(new_w) else float("nan"),
        "a0_weight_mean": float(w0[ref].mean()) if ref.any() else float("nan"),
    }
    logger.info("refinement: %s", stats)
    return RefinedGraph(ei.cpu(), ew.cpu().float(), edits, stats)
