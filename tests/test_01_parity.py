"""Test 1: with graph_learning=False, lasser == plain GEARS from pip, exactly.

Runs on CPU so every op is deterministic and equality can be exact. The reference
below uses only GEARS's public API plus the same seeds (see the call order in
lasser/run.py); it does not call any lasser code.
"""

import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from conftest import make_cfg
from lasser.run import prepare, run_experiment, used_graph

N_STEPS = 5


def _seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def reference_gears(pert_data, cfg):
    """Plain GEARS, as a user would run it from pip."""
    import gears  # noqa: F401
    from gears import GEARS

    _seed(cfg.seed)
    pert_data.prepare_split(split=cfg.split, seed=cfg.seed, train_gene_set_size=cfg.train_gene_set_size)
    full = pert_data.dataset_processed
    pert_data.dataset_processed = {k: v[:cfg.subsample_cells_per_condition] for k, v in full.items()}
    try:
        pert_data.get_dataloader(batch_size=cfg.batch_size, test_batch_size=cfg.test_batch_size)
    finally:
        pert_data.dataset_processed = full
    _seed(cfg.seed)
    g = GEARS(pert_data, device=cfg.device)
    g.model_initialize()  # GEARS defaults
    return g


def _assert_same_state(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    diff = [k for k in sa if not torch.equal(sa[k], sb[k])]
    assert not diff, f"state_dict differs in {diff[:10]}"


def _train_steps(g, n, seed):
    """GEARS's own loop body (gears.py, train()), for the first n steps."""
    from gears.utils import loss_fct

    _seed(seed)
    model = g.model
    model.train()
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    losses = []
    for step, batch in enumerate(g.dataloader["train_loader"]):
        if step == n:
            break
        batch.to(g.device)
        opt.zero_grad()
        pred = model(batch)
        loss = loss_fct(pred, batch.y, batch.pert, ctrl=g.ctrl_expression,
                        dict_filter=g.dict_filter, direction_lambda=g.config["direction_lambda"])
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        opt.step()
        losses.append(loss.item())
    return losses


@pytest.fixture
def cpu_cfg():
    return make_cfg(device="cpu", run_tag="parity")


def test_graph_and_init_identical(pert_data, cpu_cfg):
    ours = prepare(cpu_cfg, pert_data).gears
    ref = reference_gears(pert_data, cpu_cfg)
    ei, ew = used_graph(ours)
    print(f"co-expression graph: {ei.shape[1]} edges")
    assert torch.equal(ei, ref.config["G_coexpress"].cpu())
    assert torch.equal(ew, ref.config["G_coexpress_weight"].cpu())
    assert ei.dtype == ref.config["G_coexpress"].dtype and ew.dtype == ref.config["G_coexpress_weight"].dtype
    assert torch.equal(ours.config["G_go"].cpu(), ref.config["G_go"].cpu())
    _assert_same_state(ours.model, ref.model)

    # The provider used for A0 in GSR runs builds the same graph.
    from lasser.graph import StaticCoexpressProvider
    a0 = StaticCoexpressProvider(cpu_cfg).build(pert_data)
    assert torch.equal(a0.edge_index, ei) and torch.equal(a0.edge_weight, ew)


def test_first_training_steps_identical(pert_data, cpu_cfg):
    ours = prepare(cpu_cfg, pert_data).gears
    l_ours = _train_steps(ours, N_STEPS, cpu_cfg.seed)
    ref = reference_gears(pert_data, cpu_cfg)
    l_ref = _train_steps(ref, N_STEPS, cpu_cfg.seed)
    print("losses lasser:", l_ours)
    print("losses GEARS: ", l_ref)
    assert len(l_ours) == N_STEPS
    assert l_ours == l_ref
    _assert_same_state(ours.model, ref.model)


def test_full_run_identical(pert_data, cpu_cfg):
    res = run_experiment(cpu_cfg, pert_data)
    ref = reference_gears(pert_data, cpu_cfg)
    ref.train(epochs=cpu_cfg.epochs, lr=cpu_cfg.lr, weight_decay=cpu_cfg.weight_decay)
    _assert_same_state(res["gears"].best_model, ref.best_model)
