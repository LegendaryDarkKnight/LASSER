"""Self-supervised link-prediction pretraining (plan §4), in PyTorch + PyG.

- Positives: Pearson top-k pairs on half H1, kept only if they stay strong in
  >= ``bootstrap_keep`` of bootstrap resamples of H1.
- Negatives: K per query, ~25% hard (high cosine in a view, low |r| on H1), never
  an A0 or positive neighbour of the query, never the query itself.
- Each epoch the training positives are split 70/30: the encoders pass messages
  over the 70% only and predict the 30%, so a target edge is never in the GNN input.
- Per-view full-batch encoders (MLP -> 2x SGConv -> linear + LayerNorm, no output
  ReLU), momentum key encoders, intra- and inter-view InfoNCE, VICReg variance guard,
  early stopping on held-out link AUC.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GSRConfig
from .data import TrainingCells, rows_csr
from .views import ViewFeatures

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ pair helpers
def pair_ids(u: torch.Tensor, v: torch.Tensor, n: int) -> torch.Tensor:
    """Order-free id of a gene pair."""
    return torch.minimum(u, v) * n + torch.maximum(u, v)


def in_sorted(ids: torch.Tensor, sorted_ids: torch.Tensor) -> torch.Tensor:
    if sorted_ids.numel() == 0:
        return torch.zeros_like(ids, dtype=torch.bool)
    pos = torch.searchsorted(sorted_ids, ids).clamp(max=sorted_ids.numel() - 1)
    return sorted_ids[pos] == ids


def both_directions(pairs: torch.Tensor) -> torch.Tensor:
    """(P, 2) undirected pairs -> (2, 2P) edge_index."""
    return torch.cat([pairs.T, pairs.T.flip(0)], 1)


# ------------------------------------------------------------------- positives
@dataclass
class Positives:
    pairs: torch.Tensor           # (P, 2) long, undirected, u < v, on CPU
    abs_corr_h1: torch.Tensor     # (G, G) float16 |r| on H1, on device
    stats: Dict[str, Any] = field(default_factory=dict)


def _standardise(X: torch.Tensor) -> torch.Tensor:
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, unbiased=False, keepdim=True)
    sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
    Z = (X - mu) / sd
    return Z


def _corr(X: torch.Tensor) -> torch.Tensor:
    Z = _standardise(X)
    return (Z.T @ Z) / X.shape[0]


def _top_pairs(A: torch.Tensor, k: int, threshold: float) -> torch.Tensor:
    """Top-k |r| neighbours per gene above ``threshold`` -> undirected unique pairs."""
    n = A.shape[0]
    A = A.clone()
    A.fill_diagonal_(0)
    vals, idx = A.topk(k, dim=1)
    rows = torch.arange(n, device=A.device)[:, None].expand_as(idx)
    keep = vals > threshold
    ids = torch.unique(pair_ids(rows[keep], idx[keep], n))
    return torch.stack([ids // n, ids % n], 1)


@torch.no_grad()
def build_positives(adata, tc: TrainingCells, cfg: GSRConfig, seed: int, device: torch.device,
                    response_profiles: Optional[np.ndarray] = None) -> Positives:
    """Bootstrapped H1 co-expression pairs (same cell population GEARS builds A0 from)."""
    rng = np.random.default_rng(seed)
    idx = np.intersect1d(tc.h1_idx, tc.a0_cell_idx)
    if len(idx) > cfg.pos_max_cells:
        idx = np.sort(rng.choice(idx, cfg.pos_max_cells, replace=False))
    X = torch.from_numpy(rows_csr(adata, idx).toarray()).to(device)
    n_cells, n = X.shape

    C = _corr(X)
    pairs = _top_pairs(C.abs(), cfg.pos_k, cfg.pos_threshold)
    cand = pairs.to(device)
    hits = torch.zeros(len(cand), device=device)
    for _ in range(cfg.n_bootstrap):
        rows = torch.from_numpy(rng.integers(0, n_cells, n_cells)).to(device)
        Cb = _corr(X[rows])
        hits += (Cb[cand[:, 0], cand[:, 1]].abs() > cfg.pos_threshold).float()
        del Cb
    freq = hits / max(cfg.n_bootstrap, 1)
    keep = freq >= cfg.bootstrap_keep if cfg.n_bootstrap > 0 else torch.ones_like(freq, dtype=torch.bool)
    pos = cand[keep].cpu()
    stats = {"h1_cells_used": int(n_cells), "candidate_pairs": int(len(cand)),
             "bootstrap_kept_pairs": int(len(pos))}

    if cfg.response_pos_threshold is not None and response_profiles is not None:
        R = torch.from_numpy(np.asarray(response_profiles, dtype=np.float32)).to(device)
        extra = _top_pairs(_corr(R.T).abs(), cfg.pos_k, cfg.response_pos_threshold).cpu()
        ids = torch.unique(torch.cat([pair_ids(pos[:, 0], pos[:, 1], n), pair_ids(extra[:, 0], extra[:, 1], n)]))
        pos = torch.stack([ids // n, ids % n], 1)
        stats["response_pairs"] = int(len(extra))

    stats["positive_pairs"] = int(len(pos))
    logger.info("positives: %s", stats)
    if len(pos) < 10:
        raise RuntimeError(f"only {len(pos)} positive pairs; check pos_threshold / bootstrap_keep")
    abs_corr = C.abs().half()
    del X, C
    return Positives(pos, abs_corr, stats)


# --------------------------------------------------------------------- models
class ViewEncoder(nn.Module):
    """MLP -> n x SGConv over message edges -> linear head + LayerNorm (no output ReLU)."""

    def __init__(self, in_dim: int, hidden: int, out: int, n_layers: int):
        super().__init__()
        from torch_geometric.nn import SGConv

        self.missing_token = nn.Parameter(torch.zeros(in_dim))
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList([SGConv(hidden, hidden, K=1) for _ in range(n_layers)])
        self.head = nn.Linear(hidden, out)
        self.norm = nn.LayerNorm(out)

    def forward(self, x: torch.Tensor, missing: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.where(missing[:, None], self.missing_token.expand_as(x), x)
        h = self.mlp(x)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
        return self.norm(self.head(h))


class GSRModel(nn.Module):
    def __init__(self, in_dims: Dict[str, int], cfg: GSRConfig):
        super().__init__()
        self.views = list(in_dims)
        self.encoders = nn.ModuleDict({v: ViewEncoder(d, cfg.hidden_dim, cfg.emb_dim, cfg.n_conv_layers)
                                       for v, d in in_dims.items()})
        self.decoders = nn.ModuleDict({
            f"{s}->{t}": nn.Sequential(nn.Linear(cfg.emb_dim, cfg.decoder_hidden), nn.ELU(),
                                       nn.Linear(cfg.decoder_hidden, cfg.emb_dim))
            for s in self.views for t in self.views if s != t})

    def encode(self, feats: Dict[str, torch.Tensor], missing: Dict[str, torch.Tensor],
               edge_index: torch.Tensor, feat_mask: float = 0.0, edge_drop: float = 0.0) -> Dict[str, torch.Tensor]:
        out = {}
        for v in self.views:
            x = feats[v]
            ei = edge_index
            if feat_mask > 0:
                x = x * (torch.rand(x.shape[1], device=x.device) >= feat_mask).float()
            if edge_drop > 0:
                ei = ei[:, torch.rand(ei.shape[1], device=ei.device) >= edge_drop]
            out[v] = self.encoders[v](x, missing[v], ei)
        return out


@torch.no_grad()
def momentum_update(model: nn.Module, ema: nn.Module, m: float) -> None:
    for p, q in zip(model.parameters(), ema.parameters()):
        q.mul_(m).add_(p.detach(), alpha=1 - m)


# ---------------------------------------------------------------- monitoring
def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    s = torch.cat([pos, neg]).double()
    ranks = torch.empty_like(s)
    ranks[s.argsort()] = torch.arange(1, len(s) + 1, dtype=s.dtype, device=s.device)
    n_p, n_n = len(pos), len(neg)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


@torch.no_grad()
def collapse_metrics(z: torch.Tensor, sample: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Std of pairwise cosine and effective rank (exp entropy of normalised singular values)."""
    if sample is not None:
        z = z[sample]
    zn = F.normalize(z.float(), dim=-1)
    cos = zn @ zn.T
    iu = torch.triu_indices(len(zn), len(zn), 1, device=zn.device)
    c = cos[iu[0], iu[1]]
    s = torch.linalg.svdvals(z.float() - z.float().mean(0))
    p = s / s.sum().clamp_min(1e-12)
    erank = float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))
    return {"cos_mean": float(c.mean()), "cos_std": float(c.std()), "effective_rank": erank}


# ------------------------------------------------------------------- pretrain
@dataclass
class PretrainResult:
    embeddings: Dict[str, torch.Tensor]       # view -> (G, emb_dim), CPU, from the best step
    state_dict: Dict[str, torch.Tensor]       # best query encoders + decoders, CPU
    history: List[Dict[str, Any]]
    best_step: int
    best_auc: float
    baseline_auc: Dict[str, float]
    stats: Dict[str, Any]


class _NegSampler:
    def __init__(self, n: int, excluded: torch.Tensor, cfg: GSRConfig, device: torch.device):
        self.n, self.excluded, self.cfg, self.device = n, excluded, cfg, device
        self.pool: Optional[torch.Tensor] = None

    @torch.no_grad()
    def refresh_pool(self, embs: Dict[str, torch.Tensor], abs_corr: torch.Tensor) -> None:
        """Hard candidates: top cosine in any view, but weakly correlated on H1 and not excluded."""
        cfg, n = self.cfg, self.n
        h = min(cfg.hard_pool_size, n - 1)
        if cfg.hard_neg_frac <= 0 or h <= 0:
            self.pool = None
            return
        parts = []
        for z in embs.values():
            zn = F.normalize(z.float(), dim=-1)
            top = []
            for s in range(0, n, cfg.chunk_size):
                S = zn[s:s + cfg.chunk_size] @ zn.T
                r = torch.arange(s, min(s + cfg.chunk_size, n), device=S.device)
                S[r - s, r] = -float("inf")
                top.append(S.topk(h, dim=1).indices)
            parts.append(torch.cat(top))
        pool = torch.cat(parts, 1)
        rows = torch.arange(n, device=pool.device)[:, None].expand_as(pool)
        bad = (abs_corr[rows, pool].float() >= cfg.hard_corr_max) | (pool == rows)
        bad |= in_sorted(pair_ids(rows, pool, n).reshape(-1), self.excluded).reshape(pool.shape)
        self.pool = torch.where(bad, torch.full_like(pool, -1), pool)

    def sample(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg, n = self.cfg, self.n
        K = cfg.n_neg
        neg = torch.randint(0, n, (len(q), K), device=self.device)
        n_hard = int(round(K * cfg.hard_neg_frac)) if self.pool is not None else 0
        if n_hard > 0:
            cols = torch.randint(0, self.pool.shape[1], (len(q), n_hard), device=self.device)
            hard = self.pool[q[:, None], cols]
            neg[:, :n_hard] = torch.where(hard >= 0, hard, neg[:, :n_hard])
        valid = neg != q[:, None]
        valid &= ~in_sorted(pair_ids(q[:, None].expand_as(neg), neg, n).reshape(-1), self.excluded).reshape(neg.shape)
        return neg, valid


def _info_nce(h: torch.Tensor, kpos: torch.Tensor, kneg: torch.Tensor, valid: torch.Tensor, tau: float) -> torch.Tensor:
    h = F.normalize(h, dim=-1)
    lpos = (h * kpos).sum(-1, keepdim=True)
    lneg = torch.einsum("qd,qkd->qk", h, kneg).masked_fill(~valid, float("-inf"))
    logits = torch.cat([lpos, lneg], 1) / tau
    return F.cross_entropy(logits, torch.zeros(len(h), dtype=torch.long, device=h.device))


def _vicreg_std(z: torch.Tensor, target: float) -> torch.Tensor:
    return F.relu(target - torch.sqrt(z.var(0) + 1e-4)).mean()


def pretrain(views: Dict[str, ViewFeatures], positives: Positives, a0_undirected: torch.Tensor,
             cfg: GSRConfig, seed: int, device: torch.device) -> PretrainResult:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    names = list(views)
    n = next(iter(views.values())).x.shape[0]
    feats = {v: views[v].x.to(device) for v in names}
    missing = {v: views[v].missing.to(device) for v in names}
    beta = {v: cfg.beta[v] / sum(cfg.beta[w] for w in names) for v in names}

    P = positives.pairs
    excluded = torch.unique(torch.cat([pair_ids(P[:, 0], P[:, 1], n),
                                       pair_ids(a0_undirected[0], a0_undirected[1], n)])).to(device)

    perm = torch.randperm(len(P), generator=g)
    n_val = max(1, int(cfg.link_val_frac * len(P)))
    val_pos, train_pairs = P[perm[:n_val]], P[perm[n_val:]]
    cand = torch.randint(0, n, (4 * n_val, 2), generator=g)
    ok = (cand[:, 0] != cand[:, 1]) & ~in_sorted(pair_ids(cand[:, 0], cand[:, 1], n).to(device), excluded).cpu()
    val_neg = cand[ok][:n_val]
    train_msg_full = both_directions(train_pairs).to(device)
    val_pos_d, val_neg_d = val_pos.to(device), val_neg.to(device)
    mon_sample = torch.randperm(n, generator=g)[:min(cfg.collapse_sample, n)].to(device)

    def link_scores(z: Dict[str, torch.Tensor], pairs: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {v: F.cosine_similarity(z[v][pairs[:, 0]], z[v][pairs[:, 1]], dim=-1) for v in names}
        out["combined"] = sum(beta[v] * out[v] for v in names)
        return out

    def link_auc(z: Dict[str, torch.Tensor]) -> Dict[str, float]:
        sp, sn = link_scores(z, val_pos_d), link_scores(z, val_neg_d)
        return {k: auc(sp[k], sn[k]) for k in sp}

    baseline_auc = link_auc(feats)
    logger.info("link AUC of raw view features (baseline): %s",
                {k: round(v, 4) for k, v in baseline_auc.items()})

    model = GSRModel({v: feats[v].shape[1] for v in names}, cfg).to(device)
    key_model = copy.deepcopy(model)
    for p in key_model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sampler = _NegSampler(n, excluded, cfg, device)
    sampler.refresh_pool(feats, positives.abs_corr_h1)

    history: List[Dict[str, Any]] = []
    best_auc, best_step, best_state = -1.0, 0, copy.deepcopy(model.state_dict())
    targets = torch.empty(0, 2, dtype=torch.long)
    msg_ei = train_msg_full
    cursor, epoch = 0, 0

    for step in range(1, cfg.max_steps + 1):
        if cursor >= len(targets):  # new epoch: fresh 70/30 message/target split
            sp = train_pairs[torch.randperm(len(train_pairs), generator=g)]
            n_msg = int(cfg.msg_edge_frac * len(sp))
            msg_ei = both_directions(sp[:n_msg]).to(device)
            tgt = both_directions(sp[n_msg:]).T
            targets = tgt[torch.randperm(len(tgt), generator=g)]
            cursor, epoch = 0, epoch + 1
        batch = targets[cursor:cursor + cfg.batch_edges].to(device)
        cursor += cfg.batch_edges
        qi, pj = batch[:, 0], batch[:, 1]

        model.train()
        zq = model.encode(feats, missing, msg_ei, cfg.feat_mask, cfg.edge_drop)
        with torch.no_grad():
            zk = key_model.encode(feats, missing, msg_ei, cfg.feat_mask, cfg.edge_drop)
            zk = {v: F.normalize(z, dim=-1) for v, z in zk.items()}
        neg, valid = sampler.sample(qi)

        intra, inter = [], []
        for t in names:
            kpos, kneg = zk[t][pj], zk[t][neg]
            for s in names:
                h = zq[s][qi]
                if s == t:
                    intra.append(_info_nce(h, kpos, kneg, valid, cfg.tau))
                else:
                    inter.append(_info_nce(model.decoders[f"{s}->{t}"](h), kpos, kneg, valid, cfg.tau))
        l_intra = torch.stack(intra).mean()
        l_inter = torch.stack(inter).mean() if inter else torch.zeros((), device=device)
        alpha = cfg.alpha if inter else 1.0
        l_var = torch.stack([_vicreg_std(zq[v], cfg.vicreg_std_target) for v in names]).mean()
        loss = alpha * l_intra + (1 - alpha) * l_inter + cfg.vicreg_weight * l_var

        opt.zero_grad()
        loss.backward()
        opt.step()
        momentum_update(model, key_model, cfg.momentum)

        if cfg.hard_refresh_every and step % cfg.hard_refresh_every == 0:
            with torch.no_grad():
                key_model.eval()
                sampler.refresh_pool(key_model.encode(feats, missing, train_msg_full), positives.abs_corr_h1)

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            model.eval()
            with torch.no_grad():
                z = model.encode(feats, missing, train_msg_full)
                aucs = link_auc(z)
                mon = {v: collapse_metrics(z[v], mon_sample) for v in names}
            rec = {"step": step, "epoch": epoch, "loss": float(loss), "intra": float(l_intra),
                   "inter": float(l_inter), "var": float(l_var),
                   **{f"auc_{k}": v for k, v in aucs.items()},
                   **{f"{m}_{v}": mon[v][m] for v in names for m in mon[v]}}
            history.append(rec)
            logger.info("pretrain step %d (epoch %d): loss %.4f intra %.4f inter %.4f var %.4f | AUC %s | cos_std %s",
                        step, epoch, rec["loss"], rec["intra"], rec["inter"], rec["var"],
                        {k: round(v, 4) for k, v in aucs.items()},
                        {v: round(mon[v]["cos_std"], 3) for v in names})
            if aucs["combined"] > best_auc + 1e-4:
                best_auc, best_step = aucs["combined"], step
                best_state = copy.deepcopy(model.state_dict())
            elif step - best_step >= cfg.patience:
                logger.info("early stop at step %d (best step %d, AUC %.4f)", step, best_step, best_auc)
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z = model.encode(feats, missing, both_directions(P).to(device))
    stats = {"positives": positives.stats, "val_pos": int(len(val_pos)), "val_neg": int(len(val_neg)),
             "final_collapse": {v: collapse_metrics(z[v], mon_sample) for v in names}}
    logger.info("pretraining done: best step %d, held-out link AUC %.4f (raw features %.4f)",
                best_step, best_auc, baseline_auc["combined"])
    return PretrainResult(
        embeddings={v: z[v].detach().cpu() for v in names},
        state_dict={k: v.detach().cpu() for k, v in best_state.items()},
        history=history, best_step=best_step, best_auc=best_auc,
        baseline_auc=baseline_auc, stats=stats)
