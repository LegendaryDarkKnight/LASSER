"""Study analysis: tables, paired statistics, baselines, gene groups, graphs. CPU only.

Everything reads the saved run folders (``out_dir/runs.csv`` and each run's files), so it
can run in a separate CPU session on a Kaggle output dataset. Metrics come from eval.py
(saved in ``metrics/test_metrics.json``); the reference baselines below are scored with
the same eval.py functions.

Arms are identified by ``run_tag`` = ``<arm>_seed<seed>`` (set by ``lasser.study``).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import _eval
from .tracking import read_runs_csv

logger = logging.getLogger(__name__)

HIGHER_BETTER = {"pearson", "pearson_top20_de", "pearson_delta", "pearson_delta_top20_de",
                 "frac_correct_direction_top20_de", "pearson_per_gene", "spearman_per_pert", "r_squared",
                 "precision_at_10", "precision_at_20", "direction_accuracy"}
LOWER_BETTER = {"mse", "mse_top20_de", "nmse_top20_de", "nmse_top20_de_ratio_of_means",
                "frac_opposite_top20_de", "mse_de", "mse_top20_de_noperturb"}
# The paper's headline metrics (Roohani et al. 2024, Fig. 2) + the robust nMSE.
HEADLINE = ["pearson_delta", "pearson_delta_top20_de", "mse_top20_de", "nmse_top20_de_ratio_of_means",
            "frac_opposite_top20_de"]
SUBGROUPS = ["unseen_single", "combo_seen0", "combo_seen1", "combo_seen2"]


def better_sign(metric: str) -> int:
    """+1 if higher is better, -1 if lower is better."""
    if metric in HIGHER_BETTER:
        return 1
    if metric in LOWER_BETTER:
        return -1
    raise KeyError(f"unknown direction for metric {metric!r}")


# ------------------------------------------------------------------ loading
def load_runs(out_dir: str, include_subsampled: bool = False) -> pd.DataFrame:
    """runs.csv with an ``arm`` column; one run per (arm, seed) - the latest."""
    t = read_runs_csv(os.path.join(out_dir, "runs.csv"))
    if not include_subsampled and "subsample_cells_per_condition" in t:
        t = t[t["subsample_cells_per_condition"].isna()]
    t = t[t["run_tag"].notna() & t["run_tag"].str.contains("_seed")].copy()
    t["arm"] = t["run_tag"].str.rsplit("_seed", n=1).str[0]
    t = t.sort_values("timestamp").groupby(["arm", "seed"], as_index=False).tail(1)
    return t.reset_index(drop=True)


def _json(run_dir: str, rel: str) -> Optional[dict]:
    p = os.path.join(run_dir, rel)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _pkl(run_dir: str, rel: str):
    p = os.path.join(run_dir, rel)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def metrics_long(runs: pd.DataFrame) -> pd.DataFrame:
    """Long table: arm, seed, family (paper|lasser), scope (overall|subgroup), metric, value.

    Also adds the eval.py reference streams No-perturb and Mean-perturb (LASSER metrics),
    taken once per seed.
    """
    rows: List[Dict[str, Any]] = []
    refs_done = set()
    for r in runs.itertuples():
        tm = _json(r.run_dir, "metrics/test_metrics.json")
        if tm is None:
            logger.warning("no test_metrics.json in %s", r.run_dir)
            continue
        scopes = {"overall": tm["paper"]["overall"], **tm["paper"]["subgroups"]}
        for scope, d in scopes.items():
            for m, v in d.items():
                if m != "n_perts":
                    rows.append(dict(arm=r.arm, seed=r.seed, family="paper", scope=scope, metric=m, value=v))
        lsc = {"overall": tm["lasser"]["overall"], **tm["lasser"]["subgroups"]}
        for scope, d in lsc.items():
            for stream, arm in (("model", r.arm), ("no_perturb", "No-perturb"), ("mean_perturb", "Mean-perturb")):
                if stream not in d or (stream != "model" and (r.seed, scope, stream) in refs_done):
                    continue
                refs_done.add((r.seed, scope, stream))
                for m, v in d[stream].items():
                    rows.append(dict(arm=arm, seed=r.seed, family="lasser", scope=scope, metric=m, value=v))
    return pd.DataFrame(rows)


# ------------------------------------------------------- reference baselines
def paper_reference_baselines(runs: pd.DataFrame, h5ad_path: str) -> pd.DataFrame:
    """No-perturbation and additive baselines, scored with eval.py's paper metrics.

    Additive (GEARS paper): prediction = ctrl + sum of the single-gene deltas seen in
    training (a gene without a training single contributes 0). For unseen singles it
    equals no-perturbation. Uses each seed's saved split (``splits/split.json``).
    """
    import scanpy as sc

    from .data import group_means, rows_csr

    adata = sc.read_h5ad(h5ad_path)
    cond = np.asarray(adata.obs["condition"].astype(str).values)
    labels, means = group_means(rows_csr(adata, np.arange(adata.n_obs)), cond)
    mean_of = dict(zip(labels, means))
    ctrl = mean_of["ctrl"]
    de_idx = _eval.de_indices(adata)
    rows = []
    for seed, grp in runs.groupby("seed"):
        split = _json(grp.iloc[0].run_dir, "splits/split.json")
        s2c = split["set2conditions"]
        subgroups = (split.get("subgroup") or {}).get("test_subgroup", {})
        delta = {}
        for c in s2c["train"]:
            parts = c.split("+")
            if c != "ctrl" and "ctrl" in parts:
                gene = [p for p in parts if p != "ctrl"][0]
                delta[gene] = mean_of[c] - ctrl
        for name in ("No-perturb", "Additive"):
            means_d = {}
            for p in s2c["test"]:
                if p == "ctrl" or p not in mean_of:
                    continue
                genes = [g for g in p.split("+") if g != "ctrl"]
                pred = ctrl.copy()
                if name == "Additive":
                    for g in genes:
                        pred = pred + delta.get(g, 0.0)
                means_d[p] = (pred.astype(np.float32), mean_of[p].astype(np.float32))
            _, overall, groups = _eval.paper_summary(means_d, ctrl.astype(np.float32), de_idx, subgroups)
            for scope, d in {"overall": overall, **groups}.items():
                for m, v in d.items():
                    if m != "n_perts":
                        rows.append(dict(arm=name, seed=seed, family="paper", scope=scope, metric=m, value=v))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- tables
def summary_table(long: pd.DataFrame, metrics: Sequence[str], family: str = "paper",
                  scopes: Sequence[str] = ("overall",), arms: Optional[Sequence[str]] = None,
                  digits: int = 3) -> pd.DataFrame:
    """Rows = arms, columns = (scope, metric), cells = 'mean ± sd (n)'."""
    d = long[(long.family == family) & long.metric.isin(metrics) & long.scope.isin(scopes)]
    g = d.groupby(["arm", "scope", "metric"])["value"].agg(["mean", "std", "count"]).reset_index()
    g["cell"] = g.apply(lambda r: f"{r['mean']:.{digits}f} ± {0 if pd.isna(r['std']) else r['std']:.{digits}f}"
                                  f" (n={int(r['count'])})", axis=1)
    out = g.pivot_table(index="arm", columns=["scope", "metric"], values="cell", aggfunc="first")
    out = out.reindex(columns=pd.MultiIndex.from_product([list(scopes), list(metrics)]))
    if arms is not None:
        out = out.reindex([a for a in arms if a in out.index])
    return out


def _holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    adj = np.empty_like(pv)
    running = 0.0
    m = len(pv)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pv[i]))
        adj[i] = running
    out[ok] = adj
    return out


def paired_tests(long: pd.DataFrame, ref: str, arms: Sequence[str], metrics: Sequence[str],
                 family: str = "paper", scopes: Sequence[str] = ("overall",)) -> pd.DataFrame:
    """Per-split paired comparison of each arm against ``ref`` (same seeds only).

    ``improvement`` is oriented so that > 0 means the arm is better. p-values: paired
    t-test and Wilcoxon signed-rank (two-sided; with 5 splits Wilcoxon cannot go below
    0.0625). ``p_t_holm`` is Holm-corrected across all rows of the returned table.
    """
    from scipy import stats

    d = long[(long.family == family) & long.metric.isin(metrics) & long.scope.isin(scopes)]
    w = d.pivot_table(index=["scope", "metric", "seed"], columns="arm", values="value")
    rows = []
    for arm in arms:
        if arm == ref or arm not in w or ref not in w:
            continue
        for (scope, metric), sub in w[[ref, arm]].dropna().groupby(level=[0, 1]):
            a, b = sub[ref].to_numpy(), sub[arm].to_numpy()
            n = len(a)
            sgn = better_sign(metric)
            imp = sgn * (b - a)
            mean, sd = imp.mean(), imp.std(ddof=1) if n > 1 else np.nan
            half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
            p_t = stats.ttest_rel(b, a).pvalue if n > 1 and sd > 0 else np.nan
            try:
                p_w = stats.wilcoxon(b, a).pvalue if n > 1 and np.any(b != a) else np.nan
            except ValueError:
                p_w = np.nan
            rows.append(dict(arm=arm, ref=ref, scope=scope, metric=metric, n_splits=n,
                             ref_mean=a.mean(), arm_mean=b.mean(), improvement=mean,
                             ci95_low=mean - half, ci95_high=mean + half,
                             rel_improvement_pct=100 * mean / abs(a.mean()) if a.mean() != 0 else np.nan,
                             splits_better=int((imp > 0).sum()), p_t=p_t, p_wilcoxon=p_w))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_t_holm"] = _holm(out["p_t"].to_numpy())
    return out


# --------------------------------------------------------- per perturbation
def per_perturbation(runs: pd.DataFrame) -> pd.DataFrame:
    """Long: arm, seed, pert, subgroup, metric, value (eval.py paper metrics per perturbation)."""
    rows = []
    for r in runs.itertuples():
        tm = _json(r.run_dir, "metrics/test_metrics.json")
        split = _json(r.run_dir, "splits/split.json") or {}
        sub = (split.get("subgroup") or {}).get("test_subgroup", {})
        pert2sub = {p: name for name, ps in sub.items() for p in ps}
        if tm is None:
            continue
        for p, d in tm["per_perturbation"].items():
            for m, v in d.items():
                rows.append(dict(arm=r.arm, seed=r.seed, pert=p, subgroup=pert2sub.get(p, "other"),
                                 metric=m, value=v))
    return pd.DataFrame(rows)


def _cluster_bootstrap(diff: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Mean improvement, resampling splits and then perturbations within each split."""
    by_seed = [g["imp"].to_numpy() for _, g in diff.groupby("seed")]
    k = len(by_seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        picks = rng.integers(0, k, k)
        vals = [by_seed[i][rng.integers(0, len(by_seed[i]), len(by_seed[i]))] for i in picks]
        out[b] = np.concatenate(vals).mean()
    return out


def paired_perturbation(pp: pd.DataFrame, ref: str, arm: str, metrics: Sequence[str],
                        n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per-perturbation paired comparison, pooled over splits, overall and per subgroup.

    Pairs are (split, perturbation). Reports the share of perturbations where the arm is
    better, a Wilcoxon signed-rank p over all pairs, and a 95% CI of the mean improvement
    from a bootstrap that resamples splits and then perturbations (so splits aren't
    treated as independent of their perturbations).
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    rows = []
    for metric in metrics:
        a = pp[(pp.arm == ref) & (pp.metric == metric)].set_index(["seed", "pert"])
        b = pp[(pp.arm == arm) & (pp.metric == metric)].set_index(["seed", "pert"])
        j = a[["value", "subgroup"]].join(b[["value"]], lsuffix="_ref", rsuffix="_arm", how="inner").dropna()
        j["imp"] = better_sign(metric) * (j["value_arm"] - j["value_ref"])
        j = j.reset_index()
        for scope, sub in [("overall", j)] + list(j.groupby("subgroup")):
            if len(sub) < 2:
                continue
            boot = _cluster_bootstrap(sub, n_boot, rng)
            try:
                p_w = stats.wilcoxon(sub["value_arm"], sub["value_ref"]).pvalue
            except ValueError:
                p_w = np.nan
            rows.append(dict(arm=arm, ref=ref, scope=scope, metric=metric, n_pairs=len(sub),
                             n_splits=sub["seed"].nunique(), mean_improvement=sub["imp"].mean(),
                             ci95_low=np.quantile(boot, 0.025), ci95_high=np.quantile(boot, 0.975),
                             frac_better=float((sub["imp"] > 0).mean()),
                             frac_worse=float((sub["imp"] < 0).mean()), p_wilcoxon=p_w))
    return pd.DataFrame(rows)


# ------------------------------------------------------------- gene groups
def a0_isolated_by_seed(runs: pd.DataFrame) -> Dict[int, np.ndarray]:
    """Genes with no co-expression neighbour in GEARS's A0 graph, per split."""
    out = {}
    for seed, grp in runs.groupby("seed"):
        g = None
        for r in grp.itertuples():
            rec = _pkl(r.run_dir, "graphs/a0_graph.pkl")
            if rec is None and str(r.graph_source) == "gears_static" and r.arm == "GEARS":
                rec = _pkl(r.run_dir, "graphs/coexpress_graph.pkl")
            if rec is not None:
                g = rec
                break
        if g is None:
            continue
        s, d = g["edge_index"]
        deg = np.bincount(d[s != d], minlength=len(g["gene_list"]))
        out[int(seed)] = deg == 0
    return out


def gene_group_metrics(runs: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """Per perturbation, metrics on genes isolated in A0 vs genes with A0 neighbours.

    ``pearson_delta``: Pearson of predicted vs true change over the group's genes.
    ``mse_topk``: MSE on the group's ``top_k`` genes with the largest true change.
    ``frac_opposite_topk``: share of those genes whose predicted change has the wrong sign.
    """
    iso = a0_isolated_by_seed(runs)
    rows = []
    for r in runs.itertuples():
        if int(r.seed) not in iso:
            continue
        pr = _pkl(r.run_dir, "metrics/predictions.pkl")
        if pr is None:
            continue
        ctrl = np.asarray(pr["ctrl"], dtype=np.float64)
        groups = {"isolated_in_A0": iso[int(r.seed)], "connected_in_A0": ~iso[int(r.seed)]}
        for p, pred in pr["pred_mean"].items():
            dp = np.asarray(pred, dtype=np.float64) - ctrl
            dt = np.asarray(pr["true_mean"][p], dtype=np.float64) - ctrl
            for gname, mask in groups.items():
                idx = np.flatnonzero(mask)
                if len(idx) < 3:
                    continue
                top = idx[np.argsort(-np.abs(dt[idx]))[:min(top_k, len(idx))]]
                vals = {"pearson_delta": _eval._pearson(dp[idx], dt[idx]),
                        "mse_topk": float(np.mean((dp[top] - dt[top]) ** 2)),
                        "frac_opposite_topk": float(np.mean(np.sign(dp[top]) * np.sign(dt[top]) < 0))}
                for m, v in vals.items():
                    rows.append(dict(arm=r.arm, seed=r.seed, pert=p, gene_group=gname, metric=m, value=v))
    return pd.DataFrame(rows)


GENE_GROUP_DIRECTION = {"pearson_delta": 1, "mse_topk": -1, "frac_opposite_topk": -1}


def gene_group_paired(gg: pd.DataFrame, ref: str, arms: Sequence[str]) -> pd.DataFrame:
    """Per gene group: mean over splits of the per-split mean improvement (arm vs ref)."""
    rows = []
    for arm in arms:
        for (group, metric), sub in gg[gg.arm.isin([ref, arm])].groupby(["gene_group", "metric"]):
            w = sub.pivot_table(index=["seed", "pert"], columns="arm", values="value").dropna()
            if ref not in w or arm not in w:
                continue
            imp = GENE_GROUP_DIRECTION[metric] * (w[arm] - w[ref])
            per_split = imp.groupby(level=0).mean()
            rows.append(dict(arm=arm, ref=ref, gene_group=group, metric=metric,
                             ref_mean=w[ref].mean(), arm_mean=w[arm].mean(),
                             mean_improvement=per_split.mean(),
                             sd_over_splits=per_split.std(ddof=1), n_splits=len(per_split),
                             frac_perts_better=float((imp > 0).mean())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ curves/graphs
def training_curves(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in runs.itertuples():
        p = os.path.join(r.run_dir, "metrics", "epoch_log.csv")
        if os.path.exists(p):
            e = pd.read_csv(p)
            e["arm"], e["seed"] = r.arm, r.seed
            rows.append(e)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def graph_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Graph statistics and label-free checks per run (one row per arm and split)."""
    rows = []
    for r in runs.itertuples():
        g = _pkl(r.run_dir, "graphs/coexpress_graph.pkl")
        if g is None:
            continue
        row = dict(arm=r.arm, seed=r.seed, source=g["source"], **g["stats"])
        ch = _json(r.run_dir, "graphs/checks.json")
        if ch:
            h2 = ch.get("h2", {})
            row.update(h2_hit_added=h2.get("hit_rate_added"), h2_hit_random=h2.get("hit_rate_degree_matched_random"),
                       h2_hit_a0=h2.get("hit_rate_a0"), link_auc=ch.get("pretrain", {}).get("best_link_auc"),
                       link_auc_raw=(ch.get("pretrain", {}).get("baseline_link_auc") or {}).get("combined"),
                       jaccard_to_a0=ch.get("jaccard_to_a0", {}).get("mean"),
                       **{f"cos_std_{v}": m.get("cos_std") for v, m in ch.get("collapse", {}).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def added_pairs(run_dir: str, actions: Iterable[str] = ("add", "topup", "random")) -> pd.DataFrame:
    """Undirected new pairs of a GSR run (from graphs/edits.csv) with their score."""
    p = os.path.join(run_dir, "graphs", "edits.csv")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["u", "v", "score"])
    e = pd.read_csv(p)
    e = e[e.action.isin(list(actions))]
    u = np.minimum(e.source, e.target)
    v = np.maximum(e.source, e.target)
    return pd.DataFrame({"u": u, "v": v, "score": e.score}).groupby(["u", "v"], as_index=False)["score"].mean()


def edge_stability(runs: pd.DataFrame, arm: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Jaccard of the new-edge sets between splits, and how many splits each pair appears in."""
    sets, scores = {}, []
    for r in runs[runs.arm == arm].itertuples():
        a = added_pairs(r.run_dir)
        sets[int(r.seed)] = set(zip(a.u, a.v))
        a["seed"] = r.seed
        scores.append(a)
    seeds = sorted(sets)
    jac = pd.DataFrame(index=seeds, columns=seeds, dtype=float)
    for s in seeds:
        for t in seeds:
            inter = len(sets[s] & sets[t])
            union = len(sets[s] | sets[t])
            jac.loc[s, t] = inter / union if union else np.nan
    if not scores:
        return jac, pd.DataFrame()
    all_pairs = pd.concat(scores)
    counts = all_pairs.groupby(["u", "v"]).agg(n_splits=("seed", "nunique"), mean_score=("score", "mean")).reset_index()
    return jac, counts.sort_values(["n_splits", "mean_score"], ascending=False)


def consensus_edges(runs: pd.DataFrame, arm: str = "LASSER", min_splits: int = 3) -> pd.DataFrame:
    """New edges found in at least ``min_splits`` splits, with gene names (for interpretation).

    ``in_knn8`` says whether the pair is also in the density-matched Pearson graph
    (GEARS-kNN8) of any split, i.e. whether plain correlation would have found it.
    """
    _, counts = edge_stability(runs, arm)
    if counts.empty:
        return counts
    r0 = runs[runs.arm == arm].iloc[0]
    genes = _pkl(r0.run_dir, "graphs/coexpress_graph.pkl")["gene_list"]
    knn = set()
    for r in runs[runs.arm == "GEARS-kNN8"].itertuples():
        g = _pkl(r.run_dir, "graphs/coexpress_graph.pkl")
        s, d = g["edge_index"]
        knn |= set(zip(np.minimum(s, d).tolist(), np.maximum(s, d).tolist()))
    c = counts[counts.n_splits >= min_splits].copy()
    c["gene_u"] = [genes[i] for i in c.u]
    c["gene_v"] = [genes[i] for i in c.v]
    c["in_knn8"] = [(u, v) in knn for u, v in zip(c.u, c.v)] if knn else np.nan
    return c[["gene_u", "gene_v", "n_splits", "mean_score", "in_knn8", "u", "v"]]


def overlap_with_knn8(runs: pd.DataFrame, arm: str = "LASSER") -> pd.DataFrame:
    """Per split: share of the arm's new edges that the density-matched Pearson graph also has."""
    rows = []
    for r in runs[runs.arm == arm].itertuples():
        k = runs[(runs.arm == "GEARS-kNN8") & (runs.seed == r.seed)]
        if k.empty:
            continue
        g = _pkl(k.iloc[0].run_dir, "graphs/coexpress_graph.pkl")
        s, d = g["edge_index"]
        knn = set(zip(np.minimum(s, d).tolist(), np.maximum(s, d).tolist()))
        a = added_pairs(r.run_dir)
        pairs = set(zip(a.u, a.v))
        rows.append(dict(arm=arm, seed=r.seed, new_pairs=len(pairs),
                         share_in_knn8=len(pairs & knn) / len(pairs) if pairs else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ export
def export_tables(tables: Dict[str, pd.DataFrame], out_dir: str) -> List[str]:
    """Write each table as CSV and LaTeX (booktabs) into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, df in tables.items():
        if df is None or len(df) == 0:
            continue
        csv = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(csv)
        tex = os.path.join(out_dir, f"{name}.tex")
        try:
            with open(tex, "w") as f:
                f.write(df.to_latex(float_format="%.3f", escape=True))
        except Exception as e:  # LaTeX export needs jinja2 in recent pandas; CSV is the source of truth
            logger.warning("LaTeX export of %s failed: %s", name, e)
        paths.append(csv)
    return paths
