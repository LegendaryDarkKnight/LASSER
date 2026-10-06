"""``run_experiment(cfg)``: seeds, data, graph, GEARS training, evaluation and saving.

Call order (the "reference protocol" the flag-off parity test reproduces with plain GEARS):

    import lasser.gears                  # vendored GEARS; seeds torch with 0 at import
    seed_everything(seed)
    pert_data = PertData(data_dir); pert_data.load(dataset)
    pert_data.prepare_split(split, seed, train_gene_set_size)
    pert_data.get_dataloader(batch_size, test_batch_size)
    [graph learning only: build the GSR graph]
    seed_everything(seed)                # so GEARS's init and batch order don't depend on the flag
    g = GEARS(pert_data, device); g.model_initialize(<GEARS defaults>[, G_coexpress=..., G_coexpress_weight=...])
    g.train(epochs, lr, weight_decay)

With ``graph_learning=False`` nothing else touches GEARS: the co-expression graph is
built by GEARS itself inside ``model_initialize``.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
import torch

from .config import LasserConfig
from .tracking import RunTracker, graph_record, parse_gears_output, write_json
from .utils import capture_stderr, describe_device, phase, resolve_device, seed_everything

logger = logging.getLogger(__name__)


@dataclass
class RunContext:
    cfg: LasserConfig
    pert_data: Any
    gears: Any
    device: torch.device
    graph: Any = None                       # CoexpressGraph when graph_learning
    timings: Dict[str, Any] = field(default_factory=dict)
    gears_lines: List[str] = field(default_factory=list)


def load_pert_data(cfg: LasserConfig):
    """GEARS's PertData for ``cfg.dataset`` (a GEARS dataset name, or a processed-data folder)."""
    from .gears import PertData

    os.makedirs(cfg.data_dir, exist_ok=True)
    pert_data = PertData(cfg.data_dir)
    if os.path.isdir(cfg.dataset):
        pert_data.load(data_path=cfg.dataset)
    else:
        pert_data.load(data_name=cfg.dataset)
    # GEARS derives the name with split('/'), which breaks on Windows paths; same value on Linux.
    pert_data.dataset_name = os.path.basename(os.path.normpath(pert_data.dataset_path))
    return pert_data


@contextlib.contextmanager
def subsampled(pert_data, n: Optional[int]) -> Iterator[None]:
    """Testing only: dataloaders built inside see the first ``n`` cells of each condition."""
    if not n:
        yield
        return
    full = pert_data.dataset_processed
    pert_data.dataset_processed = {k: v[:n] for k, v in full.items()}
    try:
        yield
    finally:
        pert_data.dataset_processed = full


@torch.no_grad()
def init_gene_embedding(gears_obj, emb: torch.Tensor) -> Dict[str, float]:
    """Copy a linear projection of the pretrained E-view embedding into GEARS's gene embedding.

    The projection is onto the top principal directions (identity if dims match, zero
    padding if the embedding is smaller); rows are rescaled to the mean row norm of
    GEARS's own initial table. GEARS's embedding has ``max_norm=True``, so rows are
    renormalised to norm <= 1 on lookup either way.
    """
    hidden = gears_obj.model.gene_emb.weight.shape[1]
    E = emb.float()
    E = E - E.mean(0)
    d = E.shape[1]
    if d > hidden:
        _, _, Vh = torch.linalg.svd(E, full_matrices=False)
        W = E @ Vh[:hidden].T
    elif d < hidden:
        W = torch.cat([E, torch.zeros(E.shape[0], hidden - d)], 1)
    else:
        W = E
    target = gears_obj.model.gene_emb.weight.norm(dim=1).mean().cpu()
    W = W * (target / W.norm(dim=1).mean().clamp_min(1e-12))
    for m in (gears_obj.model, gears_obj.best_model):
        m.gene_emb.weight.copy_(W.to(m.gene_emb.weight.device))
    logger.info("gene embedding initialised from the E view (%d -> %d dims)", d, hidden)
    return {"from_dim": d, "to_dim": hidden, "row_norm": float(target)}


def prepare(cfg: LasserConfig, pert_data=None, tracker: Optional[RunTracker] = None) -> RunContext:
    """Everything up to (and including) ``model_initialize``."""
    from . import gears as _gears  # noqa: F401  (GEARS seeds torch at import; import before seeding)

    cfg.validate()
    timings: Dict[str, Any] = {}
    lines: List[str] = []
    gears_log = tracker.gears_log if tracker else None
    seed_everything(cfg.seed, cfg.deterministic)
    device = resolve_device(cfg.device)
    logger.info("device: %s (GEARS training%s)", describe_device(device),
                " and GSR graph learning" if cfg.graph_learning else "")
    if device.type == "cpu" and torch.cuda.is_available():
        logger.warning("running on CPU although CUDA is available (cfg.device=%r)", cfg.device)

    with capture_stderr(lines, gears_log):
        if pert_data is None:
            with phase("load_data", timings):
                pert_data = load_pert_data(cfg)
        pert_data.prepare_split(split=cfg.split, seed=cfg.seed, train_gene_set_size=cfg.train_gene_set_size)
        with subsampled(pert_data, cfg.subsample_cells_per_condition):
            pert_data.get_dataloader(batch_size=cfg.batch_size, test_batch_size=cfg.test_batch_size)
    if tracker:
        tracker.save_split(pert_data)

    graph = None
    if cfg.graph_learning:
        from .graph import GSRCoexpressProvider

        with phase("gsr_graph", timings), capture_stderr(lines, gears_log):
            provider = GSRCoexpressProvider(cfg)
            graph = provider.build(pert_data, device, timings)
            timings["gsr_graph_from_cache"] = provider.from_cache

    seed_everything(cfg.seed, cfg.deterministic)
    from .gears import GEARS

    with phase("gears_init", timings), capture_stderr(lines, gears_log):
        gears_obj = GEARS(pert_data, device=cfg.device)
        kwargs = cfg.gears_model_kwargs()
        if graph is not None:
            kwargs.update(G_coexpress=graph.edge_index, G_coexpress_weight=graph.edge_weight)
        gears_obj.model_initialize(**kwargs)
    ctx = RunContext(cfg, pert_data, gears_obj, device, graph, timings, lines)
    if cfg.gsr_init_gene_emb:
        timings["gene_emb_init"] = init_gene_embedding(gears_obj, graph.embeddings["E"])
    return ctx


def used_graph(gears_obj):
    """The co-expression graph tensors the model actually holds."""
    m = gears_obj.model
    return m.G_coexpress.detach().cpu(), m.G_coexpress_weight.detach().cpu()


def run_experiment(cfg: LasserConfig, pert_data=None) -> Dict[str, Any]:
    """Run one experiment end to end. ``pert_data`` may be passed to reuse a loaded dataset."""
    cfg.validate()
    tracker = RunTracker(cfg)
    try:
        tracker.save_config()
        tracker.save_env()
        ctx = prepare(cfg, pert_data, tracker)
        return _train_evaluate_save(ctx, tracker)
    except Exception:
        logger.exception("run %s failed", tracker.run_id)
        raise
    finally:
        tracker.close()


def _train_evaluate_save(ctx: RunContext, tracker: RunTracker) -> Dict[str, Any]:
    from .evaluation import evaluate_gears, headline

    cfg, g, pd_ = ctx.cfg, ctx.gears, ctx.pert_data
    gene_list, node_map = g.gene_list, g.node_map

    # The graph handed to the model, read back from it.
    ei, ew = used_graph(g)
    source = ctx.graph.source if ctx.graph is not None else "gears_static"
    rec = graph_record(ei, ew, gene_list, node_map, source, cfg.seed, tracker.split_hash, tracker.config_hash)
    tracker.save_graph("coexpress_graph.pkl", rec)
    logger.info("co-expression graph used (%s): %s", source, rec["stats"])
    if ctx.graph is not None:
        a0 = ctx.graph.a0
        tracker.save_graph("a0_graph.pkl", graph_record(a0.edge_index, a0.edge_weight, gene_list, node_map,
                                                        "gears_static", cfg.seed, tracker.split_hash,
                                                        tracker.config_hash))
        ctx.graph.edits.to_csv(tracker.path("graphs", "edits.csv"), index=False)
        write_json(tracker.path("graphs", "checks.json"), ctx.graph.checks)
        tracker.save_checkpoint("gsr_encoders.pt", {"encoder_state": ctx.graph.encoder_state,
                                                    "embeddings": ctx.graph.embeddings,
                                                    "history": ctx.graph.info.get("history")})

    lines: List[str] = []
    logger.info("training GEARS on %s for %d epochs", describe_device(ctx.device), cfg.epochs)
    with phase("gears_train", ctx.timings), capture_stderr(lines, tracker.gears_log):
        g.train(epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ctx.gears_lines += lines
    # Save the trained model first, so nothing after this point can lose the training.
    g.save_model(tracker.path("checkpoints", "gears_model"))  # GEARS format: config.pkl + model.pt (best model)
    with open(tracker.path("metrics", "gears_output.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    try:
        parsed = parse_gears_output(lines)
    except Exception:  # bookkeeping only; never fail a trained run on it
        logger.exception("could not parse GEARS's training printout; raw text is in metrics/gears_output.txt")
        parsed = {"epochs": pd.DataFrame(), "gears_test": {}}
    tracker.save_epoch_log(parsed["epochs"])

    with phase("evaluate", ctx.timings):
        metrics, preds = evaluate_gears(g, pd_, cfg.test_batch_size, ctx.device)
    metrics["gears_inference"] = parsed["gears_test"]
    tracker.save_test_metrics(metrics)
    tracker.save_predictions(preds)

    finished = _dt.datetime.now().isoformat(timespec="seconds")
    tracker.save_env(finished=finished)
    write_json(tracker.path("metrics", "timings.json"), ctx.timings)

    row: Dict[str, Any] = {
        "run_id": tracker.run_id, "timestamp": tracker.started, "dataset": cfg.dataset, "split": cfg.split,
        "seed": cfg.seed, "graph_learning": cfg.graph_learning, "gsr_init_gene_emb": cfg.gsr_init_gene_emb,
        "epochs": cfg.epochs, "subsample_cells_per_condition": cfg.subsample_cells_per_condition,
        "run_tag": cfg.run_tag, "device": describe_device(ctx.device), "config_hash": tracker.config_hash, "split_hash": tracker.split_hash,
        "graph_source": source, **{f"graph_{k}": v for k, v in rec["stats"].items()},
    }
    if ctx.graph is not None:
        h2 = ctx.graph.checks.get("h2", {})
        row.update({"gsr_cache_key": cfg.gsr_cache_key(),
                    "gsr_from_cache": ctx.timings.get("gsr_graph_from_cache"),
                    "gsr_link_auc": ctx.graph.checks.get("pretrain", {}).get("best_link_auc"),
                    "gsr_h2_hit_added": h2.get("hit_rate_added"),
                    "gsr_h2_hit_random": h2.get("hit_rate_degree_matched_random"),
                    "gsr_checks_pass": bool(h2.get("passes")) and bool(ctx.graph.checks.get("collapse_passes"))})
    row.update(headline(metrics))
    row["train_seconds"] = ctx.timings.get("gears_train", {}).get("seconds")
    row["train_peak_cuda_gb"] = ctx.timings.get("gears_train", {}).get("peak_cuda_gb")
    row["gsr_peak_cuda_gb"] = max((v.get("peak_cuda_gb", 0) for k, v in ctx.timings.items()
                                   if k.startswith("gsr") and isinstance(v, dict)), default=None)
    row["run_dir"] = tracker.run_dir
    tracker.append_runs_csv(row)

    return {
        "run_id": tracker.run_id, "run_dir": tracker.run_dir, "config_hash": tracker.config_hash,
        "split_hash": tracker.split_hash, "graph_stats": rec["stats"], "test_metrics": metrics,
        "epoch_log": parsed["epochs"], "timings": ctx.timings, "row": row,
        "checks": ctx.graph.checks if ctx.graph is not None else None,
        "gears": g, "pert_data": pd_,
    }
