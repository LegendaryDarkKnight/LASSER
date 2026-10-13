"""Study tooling on synthetic data: random-edge control, job table, analysis, figures.

Needs no dataset or GPU, so it runs in seconds and should pass before any long study run.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import torch

from lasser.analysis import (HEADLINE, edge_stability, export_tables, gene_group_metrics, gene_group_paired,
                             graph_table, load_runs, metrics_long, paired_perturbation, paired_tests,
                             per_perturbation, summary_table, training_curves)
from lasser.refine import RefinedGraph, randomize_new_edges
from lasser.study import ARMS, DEFAULT_PLAN, make_jobs


# ----------------------------------------------------------- random control
def test_random_control_keeps_in_degree_and_weights():
    n = 200
    rng = np.random.default_rng(0)
    a0 = torch.tensor([[1, 2, 3, 0, 5], [0, 0, 0, 0, 6]])           # 4 real edges + self-loop (0,0)
    new_src = torch.from_numpy(rng.integers(0, n, 300))
    new_dst = torch.from_numpy(rng.integers(0, n, 300))
    ok = new_src != new_dst
    new_src, new_dst = new_src[ok], new_dst[ok]
    ei = torch.cat([a0, torch.stack([new_src, new_dst])], 1)
    ew = torch.cat([torch.full((5,), 0.5), torch.rand(len(new_src))])
    edits = pd.DataFrame({"source": new_src.numpy(), "target": new_dst.numpy(), "action": "add", "score": 0.9})
    rg = RefinedGraph(ei, ew, edits, {"a0_edges": 5, "removed": 0})
    out = randomize_new_edges(rg, a0, n, seed=1)

    assert torch.equal(out.edge_index[:, :5], a0)                      # A0 block untouched
    s, d = out.edge_index[:, 5:]
    assert torch.equal(torch.bincount(d, minlength=n), torch.bincount(new_dst, minlength=n))
    assert bool((s != d).all())
    a0_pairs = {(min(u, v), max(u, v)) for u, v in a0.T.tolist() if u != v}
    assert not any((min(u, v), max(u, v)) in a0_pairs for u, v in zip(s.tolist(), d.tolist()))
    assert len(set(zip(s.tolist(), d.tolist()))) == len(s)             # no duplicate edges
    assert torch.equal(torch.sort(out.edge_weight[5:]).values, torch.sort(ew[5:]).values)
    assert set(out.edits.action) == {"random"}


# ---------------------------------------------------------------- job table
def test_job_table():
    base = dict(data_dir="/d", out_dir="/o", cache_dir="/c", epochs=20)
    jobs = make_jobs(DEFAULT_PLAN, base)
    assert len(jobs) == 4 * 5 + 6 * 3
    assert len({j["config_hash"] for j in jobs}) == len(jobs)          # every job is distinct
    assert {j["arm"] for j in jobs} == set(ARMS)
    j = {x["job_id"]: x for x in jobs}
    assert j["GEARS_seed1"]["config"]["graph_learning"] is False
    assert j["GEARS-kNN8_seed2"]["config"]["coexpress_threshold"] == 0.0
    assert j["LASSER-random_seed3"]["config"]["gsr"]["edge_control"] == "random"
    assert list(j["LASSER-noS_seed1"]["config"]["gsr"]["views"]) == ["E", "R"]
    assert j["LASSER_seed1"]["config"]["run_tag"] == "LASSER_seed1"


# ------------------------------------------------------- synthetic run dirs
GENES = [f"G{i}" for i in range(40)]


def _fake_run(out_dir, arm, seed, shift, rng):
    run_dir = os.path.join(out_dir, f"norman_seed{seed}_{arm}")
    for sub in ("metrics", "graphs", "splits"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    perts = {"unseen_single": [f"A{k}+ctrl" for k in range(4)], "combo_seen0": [f"A{k}+B{k}" for k in range(3)],
             "combo_seen1": [f"C{k}+B{k}" for k in range(3)], "combo_seen2": [f"C{k}+D{k}" for k in range(3)]}
    allp = [p for ps in perts.values() for p in ps]
    per = {p: {"pearson_delta": 0.5 + shift + rng.normal(0, 0.01), "pearson_delta_top20_de": 0.6 + shift,
               "mse_top20_de": 0.2 - shift / 10, "nmse_top20_de": 0.5, "frac_opposite_top20_de": 0.3 - shift / 10,
               "mse_top20_de_noperturb": 0.4, "mse": 0.01, "pearson": 0.98,
               "frac_correct_direction_top20_de": 0.7, "nmse_top20_de_ratio_of_means": 0.5} for p in allp}

    def agg(ps):
        d = {m: float(np.mean([per[p][m] for p in ps])) for m in per[ps[0]]}
        d["n_perts"] = len(ps)
        return d

    lstream = {"mse": 0.01, "mse_de": 0.2, "pearson": 0.98, "pearson_per_gene": 0.3, "spearman_per_pert": 0.9,
               "r_squared": 0.9, "precision_at_10": 0.4, "precision_at_20": 0.4, "direction_accuracy": 0.7}
    tm = {"paper": {"overall": agg(allp), "subgroups": {k: agg(v) for k, v in perts.items()}},
          "lasser": {"overall": {"model": lstream, "no_perturb": lstream, "mean_perturb": lstream},
                     "subgroups": {k: {"model": lstream, "no_perturb": lstream, "mean_perturb": lstream, "n_cells": 9}
                                   for k in perts}},
          "per_perturbation": per}
    json.dump(tm, open(os.path.join(run_dir, "metrics", "test_metrics.json"), "w"))
    json.dump({"set2conditions": {"train": [], "val": [], "test": allp},
               "subgroup": {"test_subgroup": perts}, "split_hash": f"h{seed}"},
              open(os.path.join(run_dir, "splits", "split.json"), "w"))
    ctrl = rng.random(len(GENES))
    pred = {p: ctrl + rng.normal(0, 0.1, len(GENES)) for p in allp}
    true = {p: ctrl + rng.normal(0, 0.1, len(GENES)) for p in allp}
    pickle.dump({"ctrl": ctrl, "pred_mean": pred, "true_mean": true, "gene_list": GENES},
                open(os.path.join(run_dir, "metrics", "predictions.pkl"), "wb"))
    a0 = np.array([[1, 2, 3, 0, 4, 5, 6], [0, 0, 1, 0, 2, 3, 4]])   # genes 0-4 connected (>= 3 per group)
    ei = a0 if arm == "GEARS" else np.concatenate([a0, rng.integers(0, len(GENES), (2, 30))], 1)
    rec = {"edge_index": ei, "edge_weight": np.ones(ei.shape[1], np.float32), "gene_list": GENES,
           "source": "gears_static" if arm == "GEARS" else "gsr_refined",
           "stats": {"num_edges": int(ei.shape[1]), "isolated_genes": 1}}
    pickle.dump(rec, open(os.path.join(run_dir, "graphs", "coexpress_graph.pkl"), "wb"))
    if arm != "GEARS":
        pickle.dump(dict(rec, edge_index=a0, source="gears_static"),
                    open(os.path.join(run_dir, "graphs", "a0_graph.pkl"), "wb"))
        k = a0.shape[1]
        pd.DataFrame({"source": ei[0, k:], "target": ei[1, k:], "action": "add", "score": 0.8}) \
            .to_csv(os.path.join(run_dir, "graphs", "edits.csv"), index=False)
        json.dump({"h2": {"hit_rate_added": 0.04, "hit_rate_degree_matched_random": 0.002, "hit_rate_a0": 0.5},
                   "pretrain": {"best_link_auc": 0.98, "baseline_link_auc": {"combined": 0.97}}},
                  open(os.path.join(run_dir, "graphs", "checks.json"), "w"))
    pd.DataFrame({"epoch": [1, 2], "val_mse_de": [0.3, 0.25 - shift / 10]}) \
        .to_csv(os.path.join(run_dir, "metrics", "epoch_log.csv"), index=False)
    return {"run_id": f"{arm}{seed}", "timestamp": f"2026-10-0{seed}T00:00:00", "seed": seed,
            "run_tag": f"{arm}_seed{seed}", "config_hash": f"{arm}{seed}", "subsample_cells_per_condition": None,
            "graph_source": rec["source"], "run_dir": run_dir}


@pytest.fixture(scope="module")
def study_dir(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("study"))
    rng = np.random.default_rng(0)
    rows = [_fake_run(out, arm, seed, shift, rng)
            for seed in (1, 2, 3) for arm, shift in (("GEARS", 0.0), ("LASSER", 0.05), ("GEARS-kNN8", 0.01))]
    pd.DataFrame(rows).to_csv(os.path.join(out, "runs.csv"), index=False)
    return out


def test_analysis_tables_and_stats(study_dir):
    runs = load_runs(study_dir)
    assert set(runs.arm) == {"GEARS", "LASSER", "GEARS-kNN8"} and len(runs) == 9
    long = metrics_long(runs)
    assert {"No-perturb", "Mean-perturb"} <= set(long.arm)
    tab = summary_table(long, HEADLINE, scopes=("overall", "combo_seen2"), arms=["GEARS", "LASSER"])
    assert tab.shape == (2, 2 * len(HEADLINE))
    pt = paired_tests(long, "GEARS", ["LASSER", "GEARS-kNN8"], HEADLINE)
    r = pt[(pt.arm == "LASSER") & (pt.metric == "pearson_delta")].iloc[0]
    assert r.improvement == pytest.approx(0.05, abs=0.01) and r.splits_better == 3
    r = pt[(pt.arm == "LASSER") & (pt.metric == "mse_top20_de")].iloc[0]
    assert r.improvement > 0                                          # lower MSE counts as improvement
    pp = per_perturbation(runs)
    ppt = paired_perturbation(pp, "GEARS", "LASSER", ["pearson_delta"], n_boot=200)
    o = ppt[ppt.scope == "overall"].iloc[0]
    assert o.frac_better > 0.9 and o.ci95_low > 0
    gg = gene_group_metrics(runs)
    assert set(gg.gene_group) == {"isolated_in_A0", "connected_in_A0"}
    assert len(gene_group_paired(gg, "GEARS", ["LASSER"])) == 6
    assert len(training_curves(runs)) == 18
    gt = graph_table(runs)
    assert gt.loc[gt.arm == "LASSER", "h2_hit_added"].notna().all()
    jac, counts = edge_stability(runs, "LASSER")
    assert jac.shape == (3, 3) and np.allclose(np.diag(jac.to_numpy(dtype=float)), 1)
    paths = export_tables({"summary": tab, "paired": pt}, os.path.join(study_dir, "tables"))
    assert all(os.path.exists(p) for p in paths)


def test_figures_render(study_dir):
    pytest.importorskip("matplotlib")
    from lasser import figures

    runs = load_runs(study_dir)
    long = metrics_long(runs)
    arms = ["GEARS", "LASSER", "GEARS-kNN8"]
    fig_dir = os.path.join(study_dir, "figures")
    paths = []
    paths += figures.fig_headline(long, ["No-perturb"] + arms, fig_dir)
    paths += figures.fig_subgroups(long, arms, fig_dir)
    paths += figures.fig_forest(paired_tests(long, "GEARS", arms, HEADLINE), fig_dir, "fig_test_forest", "test")
    gg = gene_group_paired(gene_group_metrics(runs), "GEARS", ["LASSER"])
    paths += figures.fig_gene_groups(gg, fig_dir)
    paths += figures.fig_training(training_curves(runs), arms, fig_dir)
    paths += figures.fig_graphs(graph_table(runs), runs, fig_dir)
    paths += figures.fig_per_perturbation(per_perturbation(runs), fig_dir)
    assert paths and all(os.path.getsize(p) > 0 for p in paths)
