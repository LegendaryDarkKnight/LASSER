"""Adapter from a trained GEARS object to the metrics in eval.py.

``_eval.py`` is a byte-for-byte copy of the repo's ``eval.py`` (it has to live inside
the package so ``lasser/`` works on its own). This module only builds the inputs
eval.py needs and calls its functions; no metric is re-implemented here.

Inputs built here, because GEARS doesn't hand them over directly:
- ``ctrl``: GEARS's mean control expression (``GEARS.ctrl_expression``).
- ``mean_train``: mean training expression, from the train loader (eval.py helper).
- DE gene indices: ``eval.de_indices(adata)``.
- test subgroups: ``pert_data.subgroup['test_subgroup']`` (simulation split).
- per-subgroup loaders: the cells of the test loader, filtered by condition.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch

from . import _eval

logger = logging.getLogger(__name__)


def _loader(data_list, batch_size):
    from torch_geometric.loader import DataLoader
    return DataLoader(data_list, batch_size=batch_size, shuffle=False)


def test_subgroups(pert_data) -> Dict[str, list]:
    sub = getattr(pert_data, "subgroup", None)
    if not sub or "test_subgroup" not in sub:
        return {}
    return {name: list(perts) for name, perts in sub["test_subgroup"].items()}


@torch.no_grad()
def evaluate_gears(gears_obj, pert_data, batch_size: int, device: torch.device
                   ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run eval.py on ``gears_obj.best_model`` over the test set, overall and per subgroup.

    Returns (test_metrics, predictions).
    """
    model = gears_obj.best_model.to(device)
    num_genes = gears_obj.num_genes
    ctrl = gears_obj.ctrl_expression.detach().float().cpu().reshape(-1)
    train_loader = pert_data.dataloader["train_loader"]
    test_loader = pert_data.dataloader["test_loader"]
    mean_train = _eval.train_mean_expression(train_loader, num_genes)

    means, counts, lasser_overall = _eval.evaluate_streaming(model, test_loader, ctrl, mean_train, device)

    subgroups = test_subgroups(pert_data)
    de_idx = _eval.de_indices(pert_data.adata)
    per_pert, paper_overall, paper_groups = _eval.paper_summary(means, ctrl.numpy(), de_idx, subgroups)

    lasser_groups: Dict[str, Any] = {}
    test_cells = list(test_loader.dataset)
    for name, perts in subgroups.items():
        keep = set(perts)
        cells = [d for d in test_cells if d.pert in keep]
        if not cells:
            lasser_groups[name] = {"n_cells": 0}
            continue
        _, _, res = _eval.evaluate_streaming(model, _loader(cells, batch_size), ctrl, mean_train, device)
        res["n_cells"] = len(cells)
        lasser_groups[name] = res

    metrics = {
        "paper": {"overall": paper_overall, "subgroups": paper_groups},
        "lasser": {"overall": lasser_overall, "subgroups": lasser_groups},
        "per_perturbation": per_pert,
    }
    gene_ids = list(map(str, pert_data.adata.var.index.values))
    preds = {
        "gene_list": list(gears_obj.gene_list),
        "gene_ids": gene_ids,
        "ctrl": ctrl.numpy(),
        "mean_train": mean_train.numpy(),
        "pred_mean": {p: v[0] for p, v in means.items()},
        "true_mean": {p: v[1] for p, v in means.items()},
        "n_cells": counts,
        "subgroups": subgroups,
    }
    o = paper_overall
    logger.info("test (eval.py): pearson_delta=%.4f mse_top20_de=%.4f nmse_top20_de=%.4f over %d perts",
                o.get("pearson_delta", np.nan), o.get("mse_top20_de", np.nan),
                o.get("nmse_top20_de", np.nan), o.get("n_perts", 0))
    for name, g in paper_groups.items():
        logger.info("  %-14s pearson_delta=%.4f mse_top20_de=%.4f (%d perts)", name,
                    g.get("pearson_delta", np.nan), g.get("mse_top20_de", np.nan), g.get("n_perts", 0))
    return metrics, preds


def headline(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """The columns that go into runs.csv."""
    row: Dict[str, Any] = {}
    po = metrics["paper"]["overall"]
    for k in ("pearson_delta", "pearson_delta_top20_de", "mse_top20_de", "nmse_top20_de",
              "nmse_top20_de_ratio_of_means", "frac_opposite_top20_de", "mse", "pearson"):
        row[f"test_{k}"] = po.get(k)
    for name, g in metrics["paper"]["subgroups"].items():
        for k in ("pearson_delta", "mse_top20_de", "nmse_top20_de"):
            row[f"test_{name}_{k}"] = g.get(k)
    lo = metrics["lasser"]["overall"]["model"]
    for k in ("mse", "mse_de", "pearson", "r_squared", "direction_accuracy"):
        row[f"lasser_{k}"] = lo.get(k)
    return row
