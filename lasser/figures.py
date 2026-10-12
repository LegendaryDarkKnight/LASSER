"""Paper figures (matplotlib, saved as PDF + PNG). Needs matplotlib (preinstalled on Kaggle).

Design rules (validated palette, fixed colour per arm, one axis per panel, thin marks,
recessive grid, arms always labelled on the axis so colour never carries identity alone;
every figure has a matching table from ``lasser.analysis``).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .analysis import GENE_GROUP_DIRECTION, HEADLINE, SUBGROUPS, better_sign

# Validated categorical slots 1-4 (light surface); colour follows the arm, never its rank.
ARM_COLORS: Dict[str, str] = {"GEARS": "#2a78d6", "LASSER": "#eb6834",
                              "GEARS-kNN8": "#1baf7a", "LASSER-random": "#eda100"}
REF_COLOR = "#898781"        # reference baselines (No-perturb, Additive): muted ink, not a series
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
METRIC_LABELS = {
    "pearson_delta": "Pearson Δ (all genes)",
    "pearson_delta_top20_de": "Pearson Δ (top-20 DE)",
    "mse_top20_de": "MSE (top-20 DE)",
    "nmse_top20_de_ratio_of_means": "Normalised MSE (top-20 DE)",
    "frac_opposite_top20_de": "Opposite direction (top-20 DE)",
    "mse_topk": "MSE (top-20 genes of group)",
    "frac_opposite_topk": "Opposite direction (top-20 of group)",
}
SUBGROUP_LABELS = {"unseen_single": "Unseen single", "combo_seen0": "Combo, 0/2 seen",
                   "combo_seen1": "Combo, 1/2 seen", "combo_seen2": "Combo, 2/2 seen", "overall": "All test"}


def _style():
    import matplotlib as mpl

    if os.environ.get("MPLBACKEND") is None and not os.environ.get("DISPLAY"):
        mpl.use("Agg")

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def _save(fig, out_dir: str, name: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p)
        paths.append(p)
    return paths


def _color(arm: str) -> str:
    return ARM_COLORS.get(arm, REF_COLOR)


def _arm_panel(ax, d: pd.DataFrame, arms: Sequence[str], metric: str, show_pairs: bool = True):
    """Per-split points + mean with 95% CI per arm; thin grey lines join the same split."""
    from scipy import stats

    xs = {a: i for i, a in enumerate(arms)}
    if show_pairs:
        w = d.pivot_table(index="seed", columns="arm", values="value")
        main = [a for a in arms if a in w and a in ARM_COLORS]
        for _, row in w[main].iterrows():
            ax.plot([xs[a] for a in main], [row[a] for a in main], color=GRID, lw=0.8, zorder=1)
    for a in arms:
        v = d.loc[d.arm == a, "value"].dropna().to_numpy()
        if len(v) == 0:
            continue
        c = _color(a)
        jitter = (np.arange(len(v)) - (len(v) - 1) / 2) * 0.05
        ax.scatter(xs[a] + jitter, v, s=14, color=c, alpha=0.55, lw=0, zorder=2)
        m = v.mean()
        half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
        ax.errorbar(xs[a] + 0.22, m, yerr=half, fmt="o", ms=5, color=c, ecolor=c, elinewidth=1.5,
                    capsize=0, zorder=3, mec="white", mew=1.0)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=30, ha="right")
    ax.set_xlim(-0.6, len(arms) - 0.4)
    try:
        arrow = "↑" if better_sign(metric) > 0 else "↓"
    except KeyError:
        arrow = ""
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)}  ({arrow} better)", loc="left")


def fig_headline(long: pd.DataFrame, arms: Sequence[str], out_dir: str, scope: str = "overall",
                 metrics: Sequence[str] = ("pearson_delta", "pearson_delta_top20_de", "mse_top20_de",
                                           "frac_opposite_top20_de")) -> List[str]:
    """Figure 1: headline metrics per arm (points = splits, marker = mean ± 95% CI)."""
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(1.9 * len(metrics) + 0.6, 2.8))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        d = long[(long.family == "paper") & (long.scope == scope) & (long.metric == m) & long.arm.isin(arms)]
        _arm_panel(ax, d, [a for a in arms if a in set(d.arm)], m)
    fig.suptitle(f"Test performance, {SUBGROUP_LABELS.get(scope, scope).lower()} perturbations "
                 f"(dots: splits; marker: mean ± 95% CI)", x=0.01, ha="left", fontsize=8.5, color=INK)
    fig.tight_layout()
    paths = _save(fig, out_dir, f"fig1_headline_{scope}")
    plt.close(fig)
    return paths


def fig_subgroups(long: pd.DataFrame, arms: Sequence[str], out_dir: str, metric: str = "pearson_delta") -> List[str]:
    """Figure 2: one metric per test subgroup (small multiples)."""
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(1, len(SUBGROUPS), figsize=(8.2, 2.8), sharey=True)
    for ax, sg in zip(axes, SUBGROUPS):
        d = long[(long.family == "paper") & (long.scope == sg) & (long.metric == metric) & long.arm.isin(arms)]
        _arm_panel(ax, d, [a for a in arms if a in set(d.arm)], metric)
        ax.set_title(SUBGROUP_LABELS[sg], loc="left")
    axes[0].set_ylabel(METRIC_LABELS.get(metric, metric))
    fig.tight_layout()
    paths = _save(fig, out_dir, f"fig2_subgroups_{metric}")
    plt.close(fig)
    return paths


def fig_forest(paired: pd.DataFrame, out_dir: str, name: str, title: str,
               metrics: Sequence[str] = ("pearson_delta", "mse_top20_de", "frac_opposite_top20_de"),
               scope: str = "overall") -> List[str]:
    """Improvement (arm vs ref, oriented so > 0 = better) with 95% CI over splits; one row per comparison."""
    import matplotlib.pyplot as plt

    _style()
    d = paired[(paired.scope == scope) & paired.metric.isin(metrics)]
    comps = list(dict.fromkeys(zip(d.arm, d.ref)))
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.7 * len(metrics), 0.45 * len(comps) + 1.0), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), metrics):
        sub = d[d.metric == m]
        for i, (arm, ref) in enumerate(comps):
            r = sub[(sub.arm == arm) & (sub.ref == ref)]
            if r.empty:
                continue
            r = r.iloc[0]
            c = _color(arm)
            ax.plot([r.ci95_low, r.ci95_high], [i, i], color=c, lw=1.5, solid_capstyle="round")
            ax.plot(r.improvement, i, "o", ms=5, color=c, mec="white", mew=1.0)
            ax.annotate(f"{r.splits_better}/{r.n_splits}", (r.ci95_high, i), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=6.5, color=INK2)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{METRIC_LABELS.get(m, m)}", loc="left")
        ax.set_xlabel("improvement (> 0 = better)")
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(range(len(comps)))
    ax0.set_yticklabels([f"{a} vs {r}" for a, r in comps])
    ax0.invert_yaxis()
    fig.suptitle(title + "  (dot: mean over splits; bar: 95% CI; text: splits better)", x=0.01, ha="left",
                 fontsize=8.5)
    fig.tight_layout()
    paths = _save(fig, out_dir, name)
    plt.close(fig)
    return paths


def fig_gene_groups(gg_paired: pd.DataFrame, out_dir: str, arm: str = "LASSER") -> List[str]:
    """Improvement over GEARS on genes isolated vs connected in A0."""
    import matplotlib.pyplot as plt

    _style()
    d = gg_paired[gg_paired.arm == arm]
    metrics = [m for m in GENE_GROUP_DIRECTION if m in set(d.metric)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.4 * len(metrics), 2.4))
    groups = ["isolated_in_A0", "connected_in_A0"]
    for ax, m in zip(np.atleast_1d(axes), metrics):
        for i, g in enumerate(groups):
            r = d[(d.metric == m) & (d.gene_group == g)]
            if r.empty:
                continue
            r = r.iloc[0]
            sd = 0 if pd.isna(r.sd_over_splits) else r.sd_over_splits
            ax.bar(i, r.mean_improvement, width=0.55, color=_color(arm), edgecolor="white", linewidth=1)
            ax.errorbar(i, r.mean_improvement, yerr=sd, color=INK2, elinewidth=1, capsize=0)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(["isolated\nin A0", "connected\nin A0"])
        ax.set_title(METRIC_LABELS.get(m, m), loc="left")
    np.atleast_1d(axes)[0].set_ylabel(f"{arm} − GEARS (> 0 = better)")
    fig.suptitle("Where the gain comes from: output genes by A0 connectivity (mean ± sd over splits)",
                 x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout()
    paths = _save(fig, out_dir, "fig4_gene_groups")
    plt.close(fig)
    return paths


def fig_training(curves: pd.DataFrame, arms: Sequence[str], out_dir: str, col: str = "val_mse_de") -> List[str]:
    """Validation DE-MSE per epoch: mean over splits with a ±1 sd band; arms labelled at the line end."""
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for a in arms:
        d = curves[curves.arm == a]
        if d.empty or col not in d:
            continue
        g = d.groupby("epoch")[col].agg(["mean", "std"]).reset_index()
        c = _color(a)
        ax.fill_between(g.epoch, g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), color=c, alpha=0.12, lw=0)
        ax.plot(g.epoch, g["mean"], color=c, lw=2)
        ax.annotate(a, (g.epoch.iloc[-1], g["mean"].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MSE (top-20 DE)")
    ax.set_title("Training curves (mean ± sd over splits)", loc="left")
    fig.tight_layout()
    paths = _save(fig, out_dir, "fig5_training_curves")
    plt.close(fig)
    return paths


def fig_graphs(graphs: pd.DataFrame, runs: pd.DataFrame, out_dir: str, seed: Optional[int] = None) -> List[str]:
    """Graph structure: in-degree distribution (A0 vs LASSER) and H2 hit rates."""
    import pickle

    import matplotlib.pyplot as plt

    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    seed = seed if seed is not None else int(runs[runs.arm == "LASSER"].seed.min())
    for arm, label in (("GEARS", "A0 (GEARS)"), ("GEARS-kNN8", "Pearson kNN-8"), ("LASSER", "LASSER")):
        r = runs[(runs.arm == arm) & (runs.seed == seed)]
        if r.empty:
            continue
        with open(os.path.join(r.iloc[0].run_dir, "graphs", "coexpress_graph.pkl"), "rb") as f:
            g = pickle.load(f)
        s, d = g["edge_index"]
        deg = np.bincount(d[s != d], minlength=len(g["gene_list"]))
        vals, counts = np.unique(np.minimum(deg, 30), return_counts=True)
        ax1.plot(vals, counts, "-o", ms=2.5, lw=1.5, color=_color(arm), label=label)
    ax1.set_yscale("log")
    ax1.set_xlabel("in-degree (self-loops excluded; ≥30 pooled)")
    ax1.set_ylabel("genes")
    ax1.set_title(f"Degree distribution (split {seed})", loc="left")
    ax1.legend(fontsize=7)
    cols = [("h2_hit_a0", "A0 edges", REF_COLOR), ("h2_hit_added", "LASSER new", _color("LASSER")),
            ("h2_hit_random", "degree-matched random", _color("LASSER-random"))]
    gl = graphs[graphs.arm == "LASSER"]
    for i, (c, lab, color) in enumerate(cols):
        v = gl[c].dropna().to_numpy() if c in gl else np.array([])
        if len(v) == 0:
            continue
        ax2.bar(i, v.mean(), width=0.55, color=color, edgecolor="white", linewidth=1)
        ax2.errorbar(i, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, color=INK2, elinewidth=1)
        ax2.annotate(f"{v.mean():.3f}", (i, v.mean()), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=7, color=INK2)
    ax2.set_xticks(range(len(cols)))
    ax2.set_xticklabels([c[1] for c in cols], rotation=15, ha="right")
    ax2.set_ylabel("share with |r| > 0.2 on held-out half H2")
    ax2.set_title("Are new edges real co-expression?", loc="left")
    fig.tight_layout()
    paths = _save(fig, out_dir, "fig6_graphs")
    plt.close(fig)
    return paths


def fig_per_perturbation(pp: pd.DataFrame, out_dir: str, ref: str = "GEARS", arm: str = "LASSER",
                         metric: str = "pearson_delta") -> List[str]:
    """Per-perturbation scatter, arm vs ref, one panel per test subgroup (all splits pooled)."""
    import matplotlib.pyplot as plt

    _style()
    w = pp[pp.metric == metric].pivot_table(index=["seed", "pert", "subgroup"], columns="arm", values="value").reset_index()
    if ref not in w or arm not in w:
        return []
    fig, axes = plt.subplots(1, len(SUBGROUPS), figsize=(8.2, 2.4), sharex=True, sharey=True)
    lo = np.nanmin(w[[ref, arm]].to_numpy())
    hi = np.nanmax(w[[ref, arm]].to_numpy())
    for ax, sg in zip(axes, SUBGROUPS):
        d = w[w.subgroup == sg].dropna(subset=[ref, arm])
        ax.plot([lo, hi], [lo, hi], color=AXIS, lw=0.8)
        ax.scatter(d[ref], d[arm], s=9, color=_color(arm), alpha=0.6, lw=0)
        better = (better_sign(metric) * (d[arm] - d[ref]) > 0).mean() if len(d) else np.nan
        ax.set_title(f"{SUBGROUP_LABELS[sg]}\n{arm} better: {better:.0%} of {len(d)}", loc="left")
        ax.set_xlabel(ref)
        ax.grid(axis="both", color=GRID, lw=0.6)
    axes[0].set_ylabel(arm)
    fig.suptitle(f"{METRIC_LABELS.get(metric, metric)} per test perturbation (all splits)", x=0.01, ha="left",
                 fontsize=8.5)
    fig.tight_layout()
    paths = _save(fig, out_dir, f"fig3_per_perturbation_{metric}")
    plt.close(fig)
    return paths
