"""Test 7 (cache), test 8 (tracking) and the eval.py copy check."""

import hashlib
import os

import pandas as pd
import torch

import lasser
from conftest import GPU, make_cfg, tiny_gsr
from lasser import load_run

BASE_FILES = ["run.log", "config.json", "env.json", "splits/split.json", "splits/split_hash.txt",
              "graphs/coexpress_graph.pkl", "metrics/epoch_log.csv", "metrics/test_metrics.json",
              "metrics/predictions.pkl", "checkpoints/gears_model/model.pt"]
GSR_FILES = ["graphs/a0_graph.pkl", "graphs/edits.csv", "graphs/checks.json", "checkpoints/gsr_encoders.pt"]


def test_cache_returns_identical_graph(runs, pert_data):
    from lasser.graph import GSRCoexpressProvider

    cfg = make_cfg(graph_learning=True, gsr=tiny_gsr(), run_tag="test")
    provider = GSRCoexpressProvider(cfg)
    g = provider.build(pert_data, torch.device(GPU))
    assert provider.from_cache, f"expected a cache hit at {provider.cache_path}"
    used = runs["gl"]["gears"].model
    assert torch.equal(g.edge_index, used.G_coexpress.cpu())
    assert torch.equal(g.edge_weight, used.G_coexpress_weight.cpu())
    # A different GSR setting must not hit the same cache entry.
    other = GSRCoexpressProvider(make_cfg(graph_learning=True, gsr=tiny_gsr(m_plus=2)))
    assert other.cache_path != provider.cache_path


def test_run_folders_and_runs_csv(runs):
    for kind, run in runs.items():
        files = BASE_FILES + (GSR_FILES if kind == "gl" else [])
        missing = [f for f in files if not os.path.exists(os.path.join(run["run_dir"], f))]
        assert not missing, f"{kind}: missing {missing}"
        split = os.listdir(os.path.join(run["run_dir"], "splits"))
        assert any(f.endswith(".pkl") for f in split), f"{kind}: GEARS split pkl not copied ({split})"

    out_dir = os.path.dirname(runs["base"]["run_dir"])
    table = pd.read_csv(os.path.join(out_dir, "runs.csv"))
    assert {runs["base"]["run_id"], runs["gl"]["run_id"]} <= set(table.run_id.astype(str))
    # Same seed -> same split.
    assert runs["base"]["split_hash"] == runs["gl"]["split_hash"]
    print(table.tail(2).T.to_string())


def test_load_run_reads_everything(runs):
    for kind, run in runs.items():
        r = load_run(run["run_dir"])
        for key in ("config", "env", "split", "graph", "epoch_log", "test_metrics", "predictions"):
            assert r[key] is not None, f"{kind}: load_run missing {key}"
        assert r["lasser_config"].graph_learning == (kind == "gl")
        assert r["split"]["split_hash"] == run["split_hash"]
        assert len(r["epoch_log"]) == 1, r["epoch_log"]
        assert "subgroups" in r["test_metrics"]["paper"]
        if kind == "gl":
            assert r["a0_graph"] is not None and r["edits"] is not None and r["checks"] is not None


def test_eval_copy_is_verbatim():
    pkg = os.path.dirname(os.path.abspath(lasser.__file__))
    candidates = [os.path.join(pkg, os.pardir, "eval.py"),
                  os.path.join(os.path.dirname(__file__), os.pardir, "eval.py")]
    original = next((p for p in candidates if os.path.exists(p)), None)
    if original is None:
        import pytest
        pytest.skip("repo eval.py not found next to lasser/ or tests/")
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert h(original) == h(os.path.join(pkg, "_eval.py"))
