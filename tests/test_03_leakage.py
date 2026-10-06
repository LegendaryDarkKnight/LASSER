"""Test 6: shuffling / rescaling only validation and test cells changes nothing the GSR
pipeline produces (A0, features, positives, refined graph). Runs on CPU, deterministic.
"""

import os

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from conftest import TEST_DIR, make_cfg, tiny_gsr
from lasser.utils import seed_everything


def _perturb_heldout(adata, set2conditions, seed=0):
    """Copy of adata where val/test cells are permuted among themselves and scaled by 3."""
    held = set(set2conditions.get("val", [])) | set(set2conditions.get("test", []))
    held.discard("ctrl")
    cond = adata.obs["condition"].astype(str).values
    rows = np.flatnonzero(np.isin(cond, list(held)))
    assert len(rows) > 0
    perm = np.arange(adata.n_obs)
    perm[rows] = np.random.default_rng(seed).permutation(rows)
    scale = np.ones(adata.n_obs, dtype=np.float32)
    scale[rows] = 3.0
    X = sp.csr_matrix(adata.X)
    new = adata.copy()
    new.X = sp.diags(scale) @ X[perm]
    changed = abs(new.X[rows] - X[rows]).sum()
    assert changed > 0
    return new, len(rows)


def _a0(adata, pert_data, cfg, tag):
    from gears.utils import GeneSimNetwork, get_coexpression_network_from_train

    root = os.path.join(TEST_DIR, "leakage_a0")
    os.makedirs(os.path.join(root, tag), exist_ok=True)
    fname = os.path.join(root, tag, f"{pert_data.split}_{pert_data.seed}_{pert_data.train_gene_set_size}_"
                                    f"{cfg.coexpress_threshold}_{cfg.num_similar_genes_co_express_graph}"
                                    "_co_expression_network.csv")
    if os.path.exists(fname):
        os.remove(fname)  # always recompute from this adata
    df = get_coexpression_network_from_train(adata, cfg.coexpress_threshold, cfg.num_similar_genes_co_express_graph,
                                             root, tag, pert_data.split, pert_data.seed,
                                             pert_data.train_gene_set_size, pert_data.set2conditions)
    return df, GeneSimNetwork(df, pert_data.gene_names.values.tolist(), node_map=pert_data.node_map)


def _pipeline(adata, pert_data, cfg, a0_net):
    from lasser.graph import CoexpressGraph, run_gsr_pipeline

    seed_everything(cfg.seed, deterministic=True)
    a0 = CoexpressGraph(a0_net.edge_index, a0_net.edge_weight, "gears_static")
    return run_gsr_pipeline(adata, pert_data.set2conditions, a0, pert_data.gene_names.values.tolist(),
                            cfg, torch.device("cpu"))


def test_heldout_cells_do_not_change_graph(pert_data):
    cfg = make_cfg(device="cpu", graph_learning=True, gsr=tiny_gsr(max_steps=10))
    pert_data.prepare_split(split=cfg.split, seed=cfg.seed, train_gene_set_size=cfg.train_gene_set_size)
    pert_data.get_dataloader(batch_size=cfg.batch_size, test_batch_size=cfg.test_batch_size)
    shuffled, n_rows = _perturb_heldout(pert_data.adata, pert_data.set2conditions)
    print(f"perturbed {n_rows} validation/test cells")

    df1, a0_1 = _a0(pert_data.adata, pert_data, cfg, "orig")
    df2, a0_2 = _a0(shuffled, pert_data, cfg, "shuffled")
    assert df1.equals(df2), "GEARS's A0 changed"
    assert torch.equal(a0_1.edge_index, a0_2.edge_index) and torch.equal(a0_1.edge_weight, a0_2.edge_weight)

    g1 = _pipeline(pert_data.adata, pert_data, cfg, a0_1)
    g2 = _pipeline(shuffled, pert_data, cfg, a0_2)

    for v, f in g1.info["view_features"].items():
        f2 = g2.info["view_features"][v]
        assert torch.equal(f["x"], f2["x"]) and torch.equal(f["missing"], f2["missing"]), f"view {v} changed"
    assert torch.equal(g1.info["positive_pairs"], g2.info["positive_pairs"]), "positives changed"
    for v in g1.embeddings:
        assert torch.equal(g1.embeddings[v], g2.embeddings[v]), f"embedding {v} changed"
    assert torch.equal(g1.edge_index, g2.edge_index), "refined graph edges changed"
    assert torch.equal(g1.edge_weight, g2.edge_weight), "refined graph weights changed"
    print("views, positives, embeddings and refined graph identical;", g1.checks["refine"])
