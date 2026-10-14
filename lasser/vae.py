"""VAE-based co-expression graph learning (``LasserConfig.graph_mode = "vae"``).

1. **Model.** A cell-level VAE with a *linear* decoder, ``x ≈ W z + b`` (in the spirit of
   scVI's LDVAE), trained on half H1 of the training cells (control + training-perturbation
   cells of the split). Half H2 stays held out for the graph checks. Gaussian likelihood
   with a learned per-gene variance on GEARS's log-normalised expression.
2. **Gene embeddings and similarities.** Each gene g has a decoder loading ``W_g`` (its
   participation in the latent expression programs). Three similarities, each the cosine
   of a linear transform of ``W_g``, so no gene x gene matrix is ever formed:

   - ``load``: cosine of ``W_g``. Genes used by the same programs.
   - ``denoise``: correlation of the VAE's *reconstructed* expression across cells. With
     ``C = cov(z) = L L^T``, ``corr(g, h) = cos(L^T W_g, L^T W_h)``. Pearson correlation
     with dropout noise removed.
   - ``resp``: correlation, across training perturbations, of the decoded change from
     control, ``D_c W_g`` with ``D`` = per-perturbation mean latent minus the control mean
     (centred across perturbations).
3. **Graph.** Combined score ``sum_s beta_s * minmax(cos_s)``. The same refinement as
   LASSER-GSR is applied to GEARS's A0 (``lasser.refine.refine_graph``: remove the weakest
   A0 edges, add top candidates, top up to ``d_min``, quantile-mapped weights).
4. **Edge confidence.** Latents are resampled from the posterior ``n_posterior_samples``
   times. An edge's confidence is the share of samples in which it stays in the top
   ``confidence_k`` of one of its endpoints. New-edge weights are scaled by it.

Training-only data, as for GSR: ``data.build_training_cells`` asserts that no validation or
test condition is used.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from .config import GSRConfig, LasserConfig, VAEConfig
from .data import build_training_cells, rows_csr
from .graph import CoexpressGraph, GSRCoexpressProvider
from .refine import EdgeScorer, RefinedGraph, randomize_new_edges, refine_graph
from .utils import phase

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- model
class LinearVAE(nn.Module):
    """Encoder MLP -> (mu, logvar); linear decoder x_hat = z W^T + b; per-gene Gaussian variance."""

    def __init__(self, n_genes: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_genes, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.W = nn.Parameter(torch.randn(n_genes, latent_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_genes))
        self.log_s2 = nn.Parameter(torch.zeros(n_genes))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.mu(h), self.logvar(h).clamp(-10.0, 10.0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W.T + self.b


def _nll(x: torch.Tensor, x_hat: torch.Tensor, log_s2: torch.Tensor) -> torch.Tensor:
    return 0.5 * (((x - x_hat) ** 2) * torch.exp(-log_s2) + log_s2).sum(1).mean()


def _kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()


def _dense(X: sp.csr_matrix, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(X[rows].toarray()).to(device)


def _sparse_mean_var(X: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(X.mean(0)).ravel()
    sq = np.asarray(X.multiply(X).mean(0)).ravel()
    return mean, np.maximum(sq - mean ** 2, 0.0)


def train_vae(X: sp.csr_matrix, cfg: VAEConfig, seed: int, device: torch.device
              ) -> Tuple[LinearVAE, List[Dict[str, float]], int]:
    """Train on rows of X (cells x genes); a ``val_frac`` share of rows is held out for early stopping."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    n, n_genes = X.shape
    perm = torch.randperm(n, generator=g).numpy()
    n_val = max(1, int(cfg.val_frac * n))
    val_rows, tr_rows = np.sort(perm[:n_val]), perm[n_val:]

    mean, var = _sparse_mean_var(X[tr_rows])
    model = LinearVAE(n_genes, cfg.latent_dim, cfg.hidden_dim).to(device)
    with torch.no_grad():
        model.b.copy_(torch.from_numpy(mean).float())
        model.log_s2.copy_(torch.log(torch.from_numpy(var).float().clamp_min(1e-4)))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: List[Dict[str, float]] = []
    best, best_epoch, best_state, bad = float("inf"), 0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        kl_w = cfg.kl_weight * min(1.0, epoch / max(cfg.kl_warmup_epochs, 1))
        order = tr_rows[torch.randperm(len(tr_rows), generator=g).numpy()]
        tot = rec_t = kl_t = 0.0
        steps = 0
        for i in range(0, len(order), cfg.batch_size):
            xb = _dense(X, order[i:i + cfg.batch_size], device)
            mu, lv = model.encode(xb)
            z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
            rec, kl = _nll(xb, model.decode(z), model.log_s2), _kl(mu, lv)
            loss = rec + kl_w * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, rec_t, kl_t, steps = tot + loss.item(), rec_t + rec.item(), kl_t + kl.item(), steps + 1
        model.eval()
        with torch.no_grad():
            v_rec = v_kl = 0.0
            nb = 0
            for i in range(0, len(val_rows), cfg.batch_size):
                xb = _dense(X, val_rows[i:i + cfg.batch_size], device)
                mu, lv = model.encode(xb)
                v_rec += _nll(xb, model.decode(mu), model.log_s2).item()
                v_kl += _kl(mu, lv).item()
                nb += 1
            v_rec, v_kl = v_rec / nb, v_kl / nb
        val = v_rec + cfg.kl_weight * v_kl
        history.append({"epoch": epoch, "loss": tot / steps, "recon": rec_t / steps, "kl": kl_t / steps,
                        "kl_weight": kl_w, "val_recon": v_rec, "val_kl": v_kl, "val_elbo": val})
        if epoch % 5 == 0 or epoch == 1:
            logger.info("vae epoch %d: loss %.2f (recon %.2f, kl %.2f) | val recon %.2f kl %.2f",
                        epoch, tot / steps, rec_t / steps, kl_t / steps, v_rec, v_kl)
        if epoch >= cfg.kl_warmup_epochs and val < best - 1e-3:
            best, best_epoch, best_state, bad = val, epoch, copy.deepcopy(model.state_dict()), 0
        elif epoch >= cfg.kl_warmup_epochs:
            bad += 1
            if bad >= cfg.patience:
                logger.info("vae early stop at epoch %d (best %d)", epoch, best_epoch)
                break
    if best_epoch == 0:  # never past warm-up (very short runs): keep the last state
        best_epoch, best_state = history[-1]["epoch"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


@torch.no_grad()
def posterior(model: LinearVAE, X: sp.csr_matrix, rows: np.ndarray, batch_size: int,
              device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Posterior means and standard deviations (on CPU) for rows of X."""
    mus, sds = [], []
    for i in range(0, len(rows), batch_size):
        mu, lv = model.encode(_dense(X, rows[i:i + batch_size], device))
        mus.append(mu.cpu())
        sds.append(torch.exp(0.5 * lv).cpu())
    return torch.cat(mus), torch.cat(sds)


# ------------------------------------------------------------- embeddings
def gene_embeddings(W: torch.Tensor, Z: torch.Tensor, conditions: np.ndarray, missing: torch.Tensor,
                    similarities) -> Dict[str, torch.Tensor]:
    """Gene embeddings whose pairwise cosines are the VAE similarities (see module docstring).

    ``Z``: latents of the training cells (posterior means or a posterior sample);
    ``conditions``: their condition labels (``"ctrl"`` and training perturbations).
    Genes in ``missing`` (no variance in the training cells) get zero rows.
    """
    W = W.detach().float().cpu()
    Z = Z.float().cpu()
    out: Dict[str, torch.Tensor] = {}
    if "load" in similarities:
        out["load"] = W.clone()
    if "denoise" in similarities:
        Zc = Z - Z.mean(0)
        C = Zc.T @ Zc / max(len(Z) - 1, 1) + 1e-6 * torch.eye(Z.shape[1])
        L = torch.linalg.cholesky(C)
        out["denoise"] = W @ L
    if "resp" in similarities:
        labels, inv = np.unique(conditions, return_inverse=True)
        inv_t = torch.from_numpy(inv)
        sums = torch.zeros(len(labels), Z.shape[1]).index_add_(0, inv_t, Z)
        counts = torch.bincount(inv_t, minlength=len(labels)).clamp_min(1).float()[:, None]
        means = sums / counts
        is_ctrl = labels == "ctrl"
        if not is_ctrl.any() or (~is_ctrl).sum() < 2:
            raise ValueError("response similarity needs control cells and >= 2 training perturbations")
        D = means[torch.from_numpy(~is_ctrl)] - means[torch.from_numpy(is_ctrl)][0]
        out["resp"] = W @ (D - D.mean(0)).T
    for k in out:
        out[k][missing] = 0.0
    return out


@torch.no_grad()
def edge_confidence(W: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor, conditions: np.ndarray,
                    missing: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, cfg: VAEConfig,
                    seed: int, device: torch.device) -> torch.Tensor:
    """Share of posterior samples in which each edge (src, dst) is in the top-k of either endpoint."""
    if len(src) == 0:
        return torch.zeros(0)
    g = torch.Generator().manual_seed(seed + 7919)
    beta = {s: cfg.beta[s] for s in cfg.similarities}
    hits = torch.zeros(len(src))
    s_d, d_d = src.to(device), dst.to(device)
    for _ in range(cfg.n_posterior_samples):
        Zs = mu + sd * torch.randn(mu.shape, generator=g)
        embs = gene_embeddings(W, Zs, conditions, missing, cfg.similarities)
        top, _ = EdgeScorer(embs, beta, cfg.chunk_size, device).topk(cfg.confidence_k)
        member = (top[d_d] == s_d[:, None]).any(1) | (top[s_d] == d_d[:, None]).any(1)
        hits += member.float().cpu()
    return hits / cfg.n_posterior_samples


# --------------------------------------------------------------- pipeline
def refine_settings(v: VAEConfig) -> GSRConfig:
    """GSRConfig carrying the VAE's refinement and check settings (for refine_graph / graph_checks)."""
    return GSRConfig(beta={s: v.beta[s] for s in v.similarities}, r_minus=v.r_minus, m_plus=v.m_plus,
                     d_min=v.d_min, gamma=v.gamma, score_floor=v.score_floor, cand_topk=v.cand_topk,
                     chunk_size=v.chunk_size, h2_hit_threshold=v.h2_hit_threshold,
                     pos_max_cells=v.pos_max_cells, collapse_sample=v.collapse_sample)


@torch.no_grad()
def heldout_r2(model: LinearVAE, X: sp.csr_matrix, rows: np.ndarray, batch_size: int,
               device: torch.device) -> float:
    """Share of variance of held-out cells (H2) explained by the VAE reconstruction (posterior mean)."""
    sse = 0.0
    s1 = torch.zeros(X.shape[1], dtype=torch.float64)
    s2 = torch.zeros(X.shape[1], dtype=torch.float64)
    for i in range(0, len(rows), batch_size):
        xb = _dense(X, rows[i:i + batch_size], device)
        mu, _ = model.encode(xb)
        sse += ((xb - model.decode(mu)) ** 2).sum().item()
        s1 += xb.double().sum(0).cpu()
        s2 += (xb.double() ** 2).sum(0).cpu()
    n = len(rows)
    sst = float((s2 - s1 ** 2 / n).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def run_vae_pipeline(adata, set2conditions, a0: CoexpressGraph, gene_list, cfg: LasserConfig,
                     device: torch.device, timings: Optional[Dict[str, Any]] = None) -> CoexpressGraph:
    """Pure function of (training cells of adata, split, A0, config): VAE -> similarities -> refine -> checks."""
    from .checks import graph_checks
    from .utils import describe_device

    v = cfg.vae
    n = len(gene_list)
    rng = np.random.default_rng(cfg.seed)
    logger.info("VAE graph learning on %s", describe_device(device))
    with phase("vae/data", timings):
        tc = build_training_cells(adata, set2conditions, cfg.seed, 0.5)
        idx = tc.h1_idx
        if len(idx) > v.max_cells:
            idx = np.sort(rng.choice(idx, v.max_cells, replace=False))
        X = rows_csr(adata, idx)
        _, var = _sparse_mean_var(X)
        missing = torch.from_numpy(var < 1e-12)
        conds = tc.conditions[idx]
        logger.info("VAE training cells: %d (H1), genes without variance: %d", len(idx), int(missing.sum()))
    with phase("vae/train", timings):
        model, history, best_epoch = train_vae(X, v, cfg.seed, device)
    with phase("vae/embeddings", timings):
        mu, sd = posterior(model, X, np.arange(X.shape[0]), v.batch_size, device)
        W = model.W.detach().cpu()
        embs = gene_embeddings(W, mu, conds, missing, v.similarities)
    rcfg = refine_settings(v)
    with phase("vae/refine", timings):
        refined = refine_graph(embs, a0.edge_index, a0.edge_weight, rcfg,
                               cfg.num_similar_genes_co_express_graph, device)
        n_kept = refined.stats["a0_edges"] - refined.stats["removed"]
        new = refined.edge_index[:, n_kept:]
        conf = edge_confidence(W, mu, sd, conds, missing, new[0], new[1], v, cfg.seed, device)
        if v.confidence_weighting and len(conf):
            w = refined.edge_weight.clone()
            w[n_kept:] = w[n_kept:] * conf.clamp_min(v.confidence_floor)
            refined = RefinedGraph(refined.edge_index, w, refined.edits, refined.stats)
        edits = refined.edits.copy()
        edits["confidence"] = np.nan
        is_new = edits.action.isin(["add", "topup"]).to_numpy()
        edits.loc[is_new, "confidence"] = conf.numpy()
        stats = dict(refined.stats, confidence_mean=float(conf.mean()) if len(conf) else float("nan"),
                     confidence_ge_half=float((conf >= 0.5).float().mean()) if len(conf) else float("nan"))
        refined = RefinedGraph(refined.edge_index, refined.edge_weight, edits, stats)
        if v.edge_control == "random":
            refined = randomize_new_edges(refined, a0.edge_index, n, cfg.seed)
    with phase("vae/checks", timings):
        checks = graph_checks(adata, tc, a0.edge_index, refined.edge_index, refined.edits, embs, rcfg,
                              cfg.seed, device)
        h2 = tc.h2_idx
        if len(h2) > v.pos_max_cells:
            h2 = np.sort(rng.choice(h2, v.pos_max_cells, replace=False))
        r2 = heldout_r2(model, rows_csr(adata, h2), np.arange(len(h2)), v.batch_size, device)
    active = int((mu.var(0) > 0.01).sum())
    checks["vae"] = {"best_epoch": best_epoch, "epochs_run": len(history),
                     "val_recon": history[best_epoch - 1]["val_recon"], "val_kl": history[best_epoch - 1]["val_kl"],
                     "heldout_h2_r2": r2, "active_latent_dims": active, "latent_dim": v.latent_dim,
                     "genes_without_variance": int(missing.sum()), "training_cells": int(len(idx))}
    checks["refine"] = refined.stats
    logger.info("VAE: best epoch %d, held-out (H2) R^2 %.3f, %d/%d active latent dims, edge confidence %.2f",
                best_epoch, r2, active, v.latent_dim, refined.stats.get("confidence_mean", float("nan")))
    source = "vae_random_control" if v.edge_control == "random" else "vae_refined"
    return CoexpressGraph(
        refined.edge_index.long(), refined.edge_weight.float(), source, a0=a0, edits=refined.edits,
        checks=checks, embeddings=embs, encoder_state={k: t.detach().cpu() for k, t in model.state_dict().items()},
        info={"history": history, "edge_confidence": conf, "missing": missing})


class VAECoexpressProvider(GSRCoexpressProvider):
    """VAE-refined co-expression graph, cached like the GSR graph (folder ``cache_dir/vae``)."""

    LABEL = "VAE"

    def __init__(self, cfg: LasserConfig):
        self.cfg = cfg
        self.key = cfg.vae_cache_key()
        self.cache_path = os.path.join(cfg.cache_dir, "vae",
                                       f"{cfg.dataset}_{cfg.split}_seed{cfg.seed}_{cfg.train_gene_set_size}_{self.key}")
        self.from_cache = False

    def _source(self) -> str:
        return "vae_random_control" if self.cfg.vae.edge_control == "random" else "vae_refined"

    def _pipeline(self, pert_data, a0: CoexpressGraph, device: torch.device,
                  timings: Optional[Dict[str, Any]]) -> CoexpressGraph:
        gene_list = pert_data.gene_names.values.tolist()
        return run_vae_pipeline(pert_data.adata, pert_data.set2conditions, a0, gene_list, self.cfg, device, timings)
