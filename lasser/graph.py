"""Co-expression graph providers.

- ``StaticCoexpressProvider``: GEARS's own co-expression graph (A0), built by GEARS's
  functions with exactly the arguments ``GEARS.model_initialize`` uses.
- ``GSRCoexpressProvider``: A0 -> views -> pretraining -> refinement -> checks, cached
  on disk by (dataset, split, seed, train_gene_set_size, GSR config hash).

Both return ``edge_index`` (2, M) long and ``edge_weight`` (M,) float32 in GEARS's node
order (``pert_data.node_map``). GSR modules are imported lazily, so a baseline run
never imports them.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from .config import LasserConfig
from .utils import phase

logger = logging.getLogger(__name__)


@dataclass
class CoexpressGraph:
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    source: str                                   # "gears_static" or "gsr_refined"
    a0: Optional["CoexpressGraph"] = None
    edits: Any = None                             # pandas DataFrame (GSR only)
    checks: Optional[Dict[str, Any]] = None
    embeddings: Optional[Dict[str, torch.Tensor]] = None
    encoder_state: Optional[Dict[str, torch.Tensor]] = None
    info: Dict[str, Any] = field(default_factory=dict)


class StaticCoexpressProvider:
    """GEARS's co-expression graph, unchanged (same call as in ``GEARS.model_initialize``)."""

    def __init__(self, cfg: LasserConfig):
        self.cfg = cfg

    def build(self, pert_data) -> CoexpressGraph:
        from gears.utils import GeneSimNetwork, get_similarity_network

        edge_list = get_similarity_network(network_type="co-express",
                                           adata=pert_data.adata,
                                           threshold=self.cfg.coexpress_threshold,
                                           k=self.cfg.num_similar_genes_co_express_graph,
                                           data_path=pert_data.data_path,
                                           data_name=pert_data.dataset_name,
                                           split=pert_data.split, seed=pert_data.seed,
                                           train_gene_set_size=pert_data.train_gene_set_size,
                                           set2conditions=pert_data.set2conditions)
        gene_list = pert_data.gene_names.values.tolist()
        net = GeneSimNetwork(edge_list, gene_list, node_map=pert_data.node_map)
        return CoexpressGraph(net.edge_index, net.edge_weight, "gears_static")


def run_gsr_pipeline(adata, set2conditions, a0: CoexpressGraph, gene_list, cfg: LasserConfig,
                     device: torch.device, timings: Optional[Dict[str, Any]] = None) -> CoexpressGraph:
    """Pure function of (training cells of adata, split, A0, config): views -> pretrain -> refine -> checks."""
    from .checks import graph_checks
    from .data import build_training_cells
    from .pretrain import build_positives, pretrain
    from .refine import refine_graph
    from .views import build_views, response_profiles, undirected_simple

    g = cfg.gsr
    n = len(gene_list)
    with phase("gsr/data", timings):
        tc = build_training_cells(adata, set2conditions, cfg.seed, g.h1_frac)
    with phase("gsr/views", timings):
        views = build_views(adata, tc, a0.edge_index, gene_list, g, cfg.seed, device)
    with phase("gsr/positives", timings):
        resp = response_profiles(adata, tc) if g.response_pos_threshold is not None else None
        positives = build_positives(adata, tc, g, cfg.seed, device, resp)
    with phase("gsr/pretrain", timings):
        a0_und = undirected_simple(a0.edge_index, n)
        res = pretrain(views, positives, a0_und, g, cfg.seed, device)
    positive_pairs = positives.pairs
    del positives
    with phase("gsr/refine", timings):
        refined = refine_graph(res.embeddings, a0.edge_index, a0.edge_weight, g,
                               cfg.num_similar_genes_co_express_graph, device)
    with phase("gsr/checks", timings):
        checks = graph_checks(adata, tc, a0.edge_index, refined.edge_index, refined.edits,
                              res.embeddings, g, cfg.seed, device)
    checks["pretrain"] = {"best_step": res.best_step, "best_link_auc": res.best_auc,
                          "baseline_link_auc": res.baseline_auc, **res.stats}
    checks["refine"] = refined.stats
    return CoexpressGraph(
        refined.edge_index.long(), refined.edge_weight.float(), "gsr_refined", a0=a0,
        edits=refined.edits, checks=checks, embeddings=res.embeddings, encoder_state=res.state_dict,
        info={"history": res.history,
              "views_missing": {v: int(f.missing.sum()) for v, f in views.items()},
              "view_features": {v: {"x": f.x, "missing": f.missing} for v, f in views.items()},
              "positive_pairs": positive_pairs,
              "positives": res.stats.get("positives")})


class GSRCoexpressProvider:
    """GSR-refined co-expression graph, computed once per cache key."""

    FILES = ("graph.pt", "extras.pt", "edits.csv", "checks.json", "meta.json")

    def __init__(self, cfg: LasserConfig):
        self.cfg = cfg
        self.key = cfg.gsr_cache_key()
        self.cache_path = os.path.join(cfg.cache_dir, "gsr",
                                       f"{cfg.dataset}_{cfg.split}_seed{cfg.seed}_{cfg.train_gene_set_size}_{self.key}")
        self.from_cache = False

    def _cached(self) -> bool:
        return all(os.path.exists(os.path.join(self.cache_path, f)) for f in self.FILES)

    def _save(self, g: CoexpressGraph) -> None:
        os.makedirs(self.cache_path, exist_ok=True)
        torch.save({"edge_index": g.edge_index, "edge_weight": g.edge_weight,
                    "a0_edge_index": g.a0.edge_index, "a0_edge_weight": g.a0.edge_weight},
                   os.path.join(self.cache_path, "graph.pt"))
        torch.save({"embeddings": g.embeddings, "encoder_state": g.encoder_state, "info": g.info},
                   os.path.join(self.cache_path, "extras.pt"))
        g.edits.to_csv(os.path.join(self.cache_path, "edits.csv"), index=False)
        from .tracking import to_jsonable
        with open(os.path.join(self.cache_path, "checks.json"), "w") as f:
            json.dump(to_jsonable(g.checks), f, indent=2)
        with open(os.path.join(self.cache_path, "meta.json"), "w") as f:
            json.dump({"key": self.key, "config": to_jsonable(self.cfg.to_dict())}, f, indent=2)

    def _load(self) -> CoexpressGraph:
        import pandas as pd

        d = torch.load(os.path.join(self.cache_path, "graph.pt"), map_location="cpu", weights_only=False)
        x = torch.load(os.path.join(self.cache_path, "extras.pt"), map_location="cpu", weights_only=False)
        with open(os.path.join(self.cache_path, "checks.json")) as f:
            checks = json.load(f)
        a0 = CoexpressGraph(d["a0_edge_index"], d["a0_edge_weight"], "gears_static")
        return CoexpressGraph(d["edge_index"], d["edge_weight"], "gsr_refined", a0=a0,
                              edits=pd.read_csv(os.path.join(self.cache_path, "edits.csv")),
                              checks=checks, embeddings=x["embeddings"], encoder_state=x["encoder_state"],
                              info=x["info"])

    def build(self, pert_data, device: torch.device, timings: Optional[Dict[str, Any]] = None) -> CoexpressGraph:
        a0 = StaticCoexpressProvider(self.cfg).build(pert_data)
        if self._cached():
            g = self._load()
            if not (torch.equal(g.a0.edge_index, a0.edge_index) and torch.equal(g.a0.edge_weight, a0.edge_weight)):
                raise RuntimeError(f"cached A0 in {self.cache_path} differs from GEARS's A0; delete the cache")
            self.from_cache = True
            logger.info("GSR graph loaded from cache %s", self.cache_path)
            return g
        gene_list = pert_data.gene_names.values.tolist()
        g = run_gsr_pipeline(pert_data.adata, pert_data.set2conditions, a0, gene_list, self.cfg, device, timings)
        self._save(g)
        logger.info("GSR graph cached at %s", self.cache_path)
        return g
