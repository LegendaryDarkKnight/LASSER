"""Option-1 additions on synthetic data (no dataset): batching fix, train_seed, fix plan, replicates."""

import os

import pandas as pd
import pytest
import torch
from torch_geometric.nn import SGConv

from lasser.batching import BatchedSGConv, batch_edges, enable_batched_coexpress
from lasser.config import LasserConfig, stable_hash


class _Holder(torch.nn.Module):
    """Stands in for GEARS_Model: only the attribute the fix touches."""

    def __init__(self, conv):
        super().__init__()
        self.layers_emb_pos = torch.nn.ModuleList([conv])


def _graph(n=6):
    ei = torch.tensor([[1, 2, 3, 4, 0, 5], [0, 0, 1, 2, 3, 3]])
    ew = torch.tensor([0.5, 0.9, 0.7, 0.4, 0.6, 0.8])
    return n, ei, ew


def test_batched_sgconv_equals_per_cell_sgconv():
    torch.manual_seed(0)
    n, ei, ew = _graph()
    plain = SGConv(4, 3, K=1)
    fixed = SGConv(4, 3, K=1)
    fixed.load_state_dict(plain.state_dict())
    keys_before = set(fixed.state_dict())
    assert enable_batched_coexpress(_Holder(fixed), n) == 1
    assert isinstance(fixed, BatchedSGConv) and set(fixed.state_dict()) == keys_before

    x = torch.randn(3 * n, 4)                                  # 3 cells stacked, like GEARS
    out = fixed(x, ei, ew)
    per_cell = torch.cat([plain(x[b * n:(b + 1) * n], ei, ew) for b in range(3)])
    assert torch.allclose(out, per_cell, atol=1e-6)            # every cell gets its own messages

    upstream = plain(x, ei, ew)                                # GEARS's behaviour: only cell 0 does
    assert torch.allclose(upstream[:n], per_cell[:n], atol=1e-6)
    assert not torch.allclose(upstream[n:], per_cell[n:], atol=1e-4)
    assert torch.allclose(fixed(x[:n], ei, ew), plain(x[:n], ei, ew), atol=1e-6)   # one cell: identical


def test_batch_edges_offsets():
    n, ei, ew = _graph()
    bei, bew = batch_edges(ei, ew, n, 3)
    assert bei.shape == (2, 3 * ei.shape[1]) and bew.shape == (3 * ew.shape[0],)
    assert torch.equal(bei[:, ei.shape[1]:2 * ei.shape[1]], ei + n)
    assert int(bei.max()) < 3 * n


def test_new_fields_keep_old_hashes_and_seed():
    cfg = LasserConfig()
    d = {k: v for k, v in cfg.to_dict().items()
         if k not in {"data_dir", "out_dir", "cache_dir", "log_level", "repo_dir", "device", "run_tag",
                      "gsr", "fix_coexpress_batching", "train_seed"}}
    assert cfg.config_hash() == stable_hash(d)                 # runs from before these fields still match
    assert cfg.rng_seed == cfg.seed
    assert LasserConfig(seed=2, train_seed=101).rng_seed == 101
    assert LasserConfig(fix_coexpress_batching=True).config_hash() != cfg.config_hash()
    assert LasserConfig(train_seed=101).config_hash() != cfg.config_hash()


def test_fix_plan_jobs():
    from lasser.study import FIX_PLAN, make_jobs

    jobs = make_jobs(FIX_PLAN, dict(data_dir="/d", out_dir="/o", cache_dir="/c", epochs=10))
    ids = [j["job_id"] for j in jobs]
    assert ids[:2] == ["GEARS-fix_seed1", "LASSER-fix_seed1"]  # paired, interleaved by split
    assert len(ids) == len(set(ids)) == len({j["config_hash"] for j in jobs})
    assert "LASSER-fix_seed2_ts101" in ids and "GEARS_seed3" in ids
    j = {x["job_id"]: x for x in jobs}
    assert j["LASSER-fix_seed2_ts101"]["config"]["train_seed"] == 101
    assert j["GEARS-fix_seed1"]["config"]["fix_coexpress_batching"] is True
    assert j["GEARS_seed1"]["config"]["fix_coexpress_batching"] is False
    assert j["LASSER-random-fix_seed1"]["config"]["gsr"]["edge_control"] == "random"
    # replicates share the split's learned graph (cache key ignores the training seed)
    c1 = LasserConfig.from_dict(j["LASSER-fix_seed1"]["config"])
    c2 = LasserConfig.from_dict(j["LASSER-fix_seed1_ts101"]["config"])
    assert c1.gsr_cache_key() == c2.gsr_cache_key()


def test_load_runs_parses_replicates(tmp_path):
    from lasser.analysis import load_runs

    rows = [dict(run_tag=t, seed=s, timestamp=f"2026-10-13T0{i}:00:00", run_dir="x",
                 subsample_cells_per_condition=None, config_hash=f"h{i}")
            for i, (t, s) in enumerate([("GEARS-fix_seed1", 1), ("LASSER-fix_seed1_ts101", 1),
                                        ("LASSER-fix_seed1", 1), ("LASSER-random-fix_seed2", 2)])]
    pd.DataFrame(rows).to_csv(os.path.join(tmp_path, "runs.csv"), index=False)
    r = load_runs(str(tmp_path)).set_index("run_tag")
    assert r.loc["LASSER-fix_seed1_ts101", "arm"] == "LASSER-fix" and r.loc["LASSER-fix_seed1_ts101", "rep"] == 101
    assert r.loc["LASSER-fix_seed1", "rep"] == 0 and r.loc["LASSER-random-fix_seed2", "arm"] == "LASSER-random-fix"
