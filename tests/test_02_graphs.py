"""Test 4 (saved graph = used graph) and test 5 (flag-on shape contract)."""

import json
import os
import pickle

import numpy as np
import torch

from lasser.run import used_graph


def _load(run, name="coexpress_graph.pkl"):
    with open(os.path.join(run["run_dir"], "graphs", name), "rb") as f:
        return pickle.load(f)


def test_saved_graph_is_used_graph(runs):
    for kind, run in runs.items():
        saved = _load(run)
        for model in (run["gears"].model, run["gears"].best_model):
            ei = model.G_coexpress.detach().cpu().numpy()
            ew = model.G_coexpress_weight.detach().cpu().numpy()
            assert np.array_equal(saved["edge_index"], ei), kind
            assert np.array_equal(saved["edge_weight"], ew), kind
            assert saved["edge_index"].dtype == ei.dtype and saved["edge_weight"].dtype == ew.dtype
        expected = "gsr_refined" if kind == "gl" else "gears_static"
        assert saved["source"] == expected
        print(kind, saved["source"], saved["stats"])


def test_flag_on_shape_contract(runs, pert_data):
    run = runs["gl"]
    ei, ew = used_graph(run["gears"])
    n = len(pert_data.gene_names)
    with open(os.path.join(run["run_dir"], "config.json")) as f:
        target = json.load(f)["gsr"]["d_min"]
    saved = _load(run)
    assert saved["gene_list"] == pert_data.gene_names.values.tolist()
    assert saved["node_map"] == pert_data.node_map
    assert ei.dtype == torch.long and ei.shape[0] == 2
    assert ew.dtype == torch.float32 and ew.shape == (ei.shape[1],)
    assert int(ei.min()) >= 0 and int(ei.max()) < n
    assert torch.isfinite(ew).all() and (ew >= 0).all()

    loops = ei[0] == ei[1]
    deg = torch.bincount(ei[1][~loops], minlength=n)
    low = torch.nonzero(deg < target, as_tuple=True)[0]
    print(f"in-degree (no self-loops): min {int(deg.min())}, median {float(deg.float().median())}, "
          f"max {int(deg.max())}; genes below d_min={target}: {len(low)}")
    assert len(low) == 0, f"genes below d_min: {low[:20].tolist()}"

    # Kept A0 edges come first, in GEARS's direction and order.
    a0 = _load(run, "a0_graph.pkl")
    assert a0["source"] == "gears_static" and a0["gene_list"] == saved["gene_list"]
    a0_ids = a0["edge_index"][0].astype(np.int64) * n + a0["edge_index"][1]
    ids = saved["edge_index"][0].astype(np.int64) * n + saved["edge_index"][1]
    kept = np.isin(a0_ids, ids)
    assert np.array_equal(ids[:kept.sum()], a0_ids[kept])
