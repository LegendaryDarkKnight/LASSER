"""
Memory-light evaluation for GEARS-style models.

Two metric families are computed in a single streaming pass over the test set:

1. Paper metrics (Roohani et al., Nat. Biotechnol. 2024, Fig. 2), computed per
   perturbation on the MEAN predicted and MEAN true expression over its cells:
     - mse_top20_de        m.s.e. on the 20 most DE genes (scanpy ranking
                           stored in adata.uns['rank_genes_groups_cov_all'])
     - nmse_top20_de       the same, normalised to the no-perturbation baseline
                           (predicting mean control expression)
     - pearson_delta       Pearson of (pred - ctrl) vs (true - ctrl), all genes
     - pearson_delta_top20_de   same on the top-20 DE genes
     - frac_opposite_top20_de   fraction of top-20 DE genes whose predicted
                                change has the opposite sign to the true change
     - mse / pearson / pearson_top20_de on absolute expression
   Aggregated as the mean over test perturbations, overall and per subgroup.

2. LASSER metrics (Thesis-main/Thesis/LASSER/evaluation.py, GEARSEvaluator),
   which are per cell. They reproduce the numbers in best_model_evaluation.json
   so the new model can be compared with GL-GEARS on the same footing.
"""

import numpy as np
import torch
from scipy.stats import rankdata


def _rowwise_pearson(a, b, min_std=None):
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    num = (a * b).sum(1)
    sa = a.pow(2).sum(1).sqrt()
    sb = b.pow(2).sum(1).sqrt()
    r = num / (sa * sb)
    valid = torch.isfinite(r)
    if min_std is not None:
        n = a.shape[1]
        valid &= (sa / np.sqrt(n) > min_std) & (sb / np.sqrt(n) > min_std)
    return r, valid


def _pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return 0.0 if np.isnan(r) else float(r)


class LasserStream:
    """Streaming re-implementation of LASSER's GEARSEvaluator.compute_all_metrics."""

    def __init__(self, ctrl, device):
        self.c = ctrl.to(device)
        g = ctrl.numel()
        z = lambda: torch.zeros(g, dtype=torch.float64, device=device)
        self.sx, self.sy, self.sxx, self.syy, self.sxy = z(), z(), z(), z(), z()
        self.n = 0
        self.sse = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.mse_de = []
        self.pearson = []
        self.spearman = []
        self.prec10 = []
        self.prec20 = []
        self.dir_correct = 0
        self.dir_total = 0

    @torch.no_grad()
    def update(self, p, t):
        p = p.double()
        t = t.double()
        c = self.c.double()
        self.n += p.shape[0]
        self.sse += (p - t).pow(2).sum().item()
        self.sum_y += t.sum().item()
        self.sum_y2 += t.pow(2).sum().item()
        self.sx += p.sum(0); self.sy += t.sum(0)
        self.sxx += p.pow(2).sum(0); self.syy += t.pow(2).sum(0); self.sxy += (p * t).sum(0)

        tfc = (t - c).abs()
        pfc = (p - c).abs()
        top20 = tfc.topk(20, dim=1).indices
        self.mse_de += (p.gather(1, top20) - t.gather(1, top20)).pow(2).mean(1).tolist()

        r, valid = _rowwise_pearson(p, t, min_std=1e-6)
        self.pearson += r[valid].tolist()

        rp = torch.tensor(rankdata(p.cpu().numpy(), axis=1), device=p.device)
        rt = torch.tensor(rankdata(t.cpu().numpy(), axis=1), device=p.device)
        rs, valid = _rowwise_pearson(rp, rt)
        self.spearman += rs[valid].tolist()

        for k, store in ((10, self.prec10), (20, self.prec20)):
            tk = tfc.topk(k, dim=1).indices
            pk = pfc.topk(k, dim=1).indices
            hit = (pk.unsqueeze(2) == tk.unsqueeze(1)).any(2).sum(1).double() / k
            store += hit.tolist()

        sig = tfc > 0.1
        same = torch.sign(p - c) == torch.sign(t - c)
        self.dir_correct += (same & sig).sum().item()
        self.dir_total += sig.sum().item()

    def result(self):
        n = self.n
        mx, my = self.sx / n, self.sy / n
        vx = (self.sxx / n - mx ** 2).clamp_min(0)
        vy = (self.syy / n - my ** 2).clamp_min(0)
        cov = self.sxy / n - mx * my
        ok = (vx.sqrt() > 1e-6) & (vy.sqrt() > 1e-6)
        r_gene = cov[ok] / (vx[ok].sqrt() * vy[ok].sqrt())
        r_gene = r_gene[torch.isfinite(r_gene)]
        n_ent = n * self.c.numel()
        sst = self.sum_y2 - self.sum_y ** 2 / n_ent
        return {
            'mse': self.sse / n_ent,
            'mse_de': float(np.mean(self.mse_de)),
            'pearson': float(np.mean(self.pearson)) if self.pearson else 0.0,
            'pearson_per_gene': float(r_gene.mean()) if r_gene.numel() else 0.0,
            'spearman_per_pert': float(np.mean(self.spearman)) if self.spearman else 0.0,
            'r_squared': 1 - self.sse / sst,
            'precision_at_10': float(np.mean(self.prec10)),
            'precision_at_20': float(np.mean(self.prec20)),
            'direction_accuracy': self.dir_correct / self.dir_total if self.dir_total else 1.0,
        }


def de_indices(adata, k=20):
    """condition -> indices of its top-k DE genes (GEARS / paper definition)."""
    cond2name = dict(adata.obs[['condition', 'condition_name']].values)
    gid2idx = {g: i for i, g in enumerate(adata.var.index.values)}
    ranks = adata.uns['rank_genes_groups_cov_all']
    return {c: np.array([gid2idx[g] for g in ranks[n][:k]])
            for c, n in cond2name.items() if n in ranks}


def paper_pert_metrics(pred_mean, true_mean, ctrl, de):
    d_pred, d_true = pred_mean - ctrl, true_mean - ctrl
    mse_de = float(np.mean((pred_mean[de] - true_mean[de]) ** 2))
    mse_de_noperturb = float(np.mean((ctrl[de] - true_mean[de]) ** 2))
    sp, st = np.sign(d_pred[de]), np.sign(d_true[de])
    return {
        'mse': float(np.mean((pred_mean - true_mean) ** 2)),
        'pearson': _pearson(pred_mean, true_mean),
        'mse_top20_de': mse_de,
        'mse_top20_de_noperturb': mse_de_noperturb,
        'nmse_top20_de': mse_de / mse_de_noperturb if mse_de_noperturb > 0 else np.nan,
        'pearson_top20_de': _pearson(pred_mean[de], true_mean[de]),
        'pearson_delta': _pearson(d_pred, d_true),
        'pearson_delta_top20_de': _pearson(d_pred[de], d_true[de]),
        'frac_opposite_top20_de': float(np.mean(sp * st < 0)),
        'frac_correct_direction_top20_de': float(np.mean(sp == st)),
    }


def aggregate(per_pert, perts):
    perts = [p for p in perts if p in per_pert]
    out = {'n_perts': len(perts)}
    if not perts:
        return out
    for k in per_pert[perts[0]]:
        vals = np.array([per_pert[p][k] for p in perts], dtype=float)
        out[k] = float(np.nanmean(vals))
    # normalised m.s.e. as ratio of mean errors (robust to tiny denominators)
    out['nmse_top20_de_ratio_of_means'] = out['mse_top20_de'] / out['mse_top20_de_noperturb']
    return out


def paper_summary(means, ctrl, de_idx, subgroups):
    """means: pert -> (pred_mean, true_mean). Returns (per_pert, overall, per_subgroup)."""
    per_pert = {p: paper_pert_metrics(pm, tm, ctrl, de_idx[p])
                for p, (pm, tm) in means.items() if p != 'ctrl' and p in de_idx}
    overall = aggregate(per_pert, list(per_pert))
    groups = {name: aggregate(per_pert, perts) for name, perts in subgroups.items()}
    return per_pert, overall, groups


@torch.no_grad()
def train_mean_expression(train_loader, num_genes):
    s = torch.zeros(num_genes, dtype=torch.float64)
    n = 0
    for batch in train_loader:
        y = batch.y.reshape(-1, num_genes).double()
        s += y.sum(0)
        n += y.shape[0]
    return (s / n).float()


@torch.no_grad()
def evaluate_streaming(model, loader, ctrl, mean_train, device):
    """
    One pass over the loader. Returns per-perturbation mean predictions/truths and
    LASSER metrics for the model and the no-perturb / mean-perturb baselines.
    """
    model.eval()
    ctrl_d = ctrl.to(device)
    mean_d = mean_train.to(device)
    streams = {'model': LasserStream(ctrl, device),
               'no_perturb': LasserStream(ctrl, device),
               'mean_perturb': LasserStream(ctrl, device)}
    sums = {}
    for batch in loader:
        batch = batch.to(device)
        p = model(batch)
        t = batch.y.reshape(p.shape)
        streams['model'].update(p, t)
        streams['no_perturb'].update(ctrl_d.expand_as(t), t)
        streams['mean_perturb'].update(mean_d.expand_as(t), t)
        for i, name in enumerate(batch.pert):
            if name not in sums:
                sums[name] = [torch.zeros_like(p[0], dtype=torch.float64),
                              torch.zeros_like(p[0], dtype=torch.float64), 0]
            sums[name][0] += p[i].double()
            sums[name][1] += t[i].double()
            sums[name][2] += 1
    means = {k: ((v[0] / v[2]).float().cpu().numpy(), (v[1] / v[2]).float().cpu().numpy())
             for k, v in sums.items()}
    counts = {k: v[2] for k, v in sums.items()}
    return means, counts, {k: s.result() for k, s in streams.items()}