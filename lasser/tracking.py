"""Run folders, logging and the global ``runs.csv``; ``load_run`` reads a run back."""

from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import logging
import math
import os
import pickle
import platform
import re
import shutil
import subprocess
import sys
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .config import LasserConfig, stable_hash
from .utils import graph_stats

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def to_jsonable(obj: Any) -> Any:
    """Make numpy / torch values JSON-safe (NaN and inf become None)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def graph_record(edge_index: torch.Tensor, edge_weight: torch.Tensor, gene_list: List[str],
                 node_map: Dict[str, int], source: str, seed: int, split_hash: str,
                 config_hash: str) -> Dict[str, Any]:
    """The saved-graph format shared by coexpress_graph.pkl and a0_graph.pkl."""
    ei = edge_index.detach().cpu().numpy()
    return {
        "edge_index": ei,
        "edge_weight": edge_weight.detach().cpu().numpy(),
        "gene_list": list(gene_list),
        "node_map": dict(node_map),
        "source": source,
        "seed": seed,
        "split_hash": split_hash,
        "config_hash": config_hash,
        "stats": graph_stats(ei, len(gene_list)),
    }


def split_payload(pert_data) -> Dict[str, Any]:
    """The split exactly as GEARS uses it, in a readable form."""
    s2c = {k: sorted(map(str, v)) for k, v in pert_data.set2conditions.items()}
    subgroup = None
    if getattr(pert_data, "subgroup", None):
        subgroup = {grp: {name: sorted(map(str, perts)) for name, perts in d.items()}
                    for grp, d in pert_data.subgroup.items()}
    payload = {
        "dataset": pert_data.dataset_name,
        "split": pert_data.split,
        "seed": pert_data.seed,
        "train_gene_set_size": pert_data.train_gene_set_size,
        "set2conditions": s2c,
        "subgroup": subgroup,
    }
    payload["split_hash"] = stable_hash({"set2conditions": s2c, "subgroup": subgroup}, n=16)
    return payload


def _git_commit(repo_dir: Optional[str]) -> Optional[str]:
    if not repo_dir:
        return None
    try:
        out = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _version(pkg: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return None


class RunTracker:
    """Owns one run folder: logging, config/env/split/graph/metric files and runs.csv."""

    def __init__(self, cfg: LasserConfig):
        self.cfg = cfg
        self.started = _now()
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = uuid.uuid4().hex[:8]
        kind = "gl" if cfg.graph_learning else "base"
        name = f"{cfg.dataset}_seed{cfg.seed}_{kind}_{stamp}"
        if cfg.run_tag:
            name += f"_{cfg.run_tag}"
        self.run_dir = os.path.join(cfg.out_dir, name)
        if os.path.exists(self.run_dir):
            self.run_dir += f"_{self.run_id}"
        for sub in ("splits", "graphs", "metrics", "checkpoints"):
            os.makedirs(os.path.join(self.run_dir, sub), exist_ok=True)
        self.split_hash: Optional[str] = None
        self.config_hash = cfg.config_hash()
        self._handlers: List[logging.Handler] = []
        self.gears_log = logging.getLogger("lasser.gears_output")
        self._setup_logging()
        logger.info("run %s -> %s", self.run_id, self.run_dir)

    # ----------------------------------------------------------------- logging
    def _setup_logging(self) -> None:
        level = getattr(logging, self.cfg.log_level.upper(), logging.INFO)
        fmt = logging.Formatter(_LOG_FORMAT)
        fh = logging.FileHandler(os.path.join(self.run_dir, "run.log"))
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root = logging.getLogger("lasser")
        root.setLevel(level)
        for h in (fh, sh):
            root.addHandler(h)
        # GEARS's own stderr lines: already shown in the notebook, so file only.
        self.gears_log.propagate = False
        self.gears_log.setLevel(logging.INFO)
        self.gears_log.addHandler(fh)
        self._handlers = [fh, sh]

    def close(self) -> None:
        root = logging.getLogger("lasser")
        for h in self._handlers:
            root.removeHandler(h)
            self.gears_log.removeHandler(h)
            h.close()
        self._handlers = []

    # ------------------------------------------------------------------- files
    def path(self, *parts: str) -> str:
        return os.path.join(self.run_dir, *parts)

    def save_config(self) -> None:
        d = self.cfg.to_dict()
        d["config_hash"] = self.config_hash
        d["gsr_cache_key"] = self.cfg.gsr_cache_key() if self.cfg.graph_learning else None
        d["run_id"] = self.run_id
        write_json(self.path("config.json"), d)

    def save_env(self, finished: Optional[str] = None) -> None:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        write_json(self.path("env.json"), {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "torch_geometric": _version("torch_geometric"),
            "cell_gears": _version("cell-gears"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scanpy": _version("scanpy"),
            "scikit_learn": _version("scikit-learn"),
            "gpu": gpu,
            "git_commit": _git_commit(self.cfg.repo_dir),
            "started": self.started,
            "finished": finished,
        })

    def save_split(self, pert_data) -> str:
        """Copy GEARS's split pkl(s) as-is and write split.json; returns the split hash."""
        payload = split_payload(pert_data)
        self.split_hash = payload["split_hash"]
        split_dir = os.path.join(pert_data.dataset_path, "splits")
        stem = f"{pert_data.dataset_name}_{pert_data.split}_{pert_data.seed}_{pert_data.train_gene_set_size}"
        copied = []
        for f in glob.glob(os.path.join(split_dir, glob.escape(stem) + "*.pkl")):
            shutil.copy2(f, self.path("splits", os.path.basename(f)))
            copied.append(os.path.basename(f))
        payload["copied_files"] = sorted(copied)
        write_json(self.path("splits", "split.json"), payload)
        with open(self.path("splits", "split_hash.txt"), "w") as f:
            f.write(self.split_hash + "\n")
        logger.info("split hash %s (copied %s)", self.split_hash, copied)
        return self.split_hash

    def save_graph(self, name: str, record: Dict[str, Any]) -> None:
        with open(self.path("graphs", name), "wb") as f:
            pickle.dump(record, f)

    def save_epoch_log(self, df: pd.DataFrame) -> None:
        df.to_csv(self.path("metrics", "epoch_log.csv"), index=False)

    def save_test_metrics(self, metrics: Dict[str, Any]) -> None:
        write_json(self.path("metrics", "test_metrics.json"), metrics)

    def save_predictions(self, preds: Dict[str, Any]) -> None:
        with open(self.path("metrics", "predictions.pkl"), "wb") as f:
            pickle.dump(preds, f)

    def save_checkpoint(self, name: str, state: Dict[str, Any]) -> None:
        torch.save(state, self.path("checkpoints", name))

    def append_runs_csv(self, row: Dict[str, Any]) -> str:
        path = os.path.join(self.cfg.out_dir, "runs.csv")
        new = pd.DataFrame([to_jsonable(row)])
        if os.path.exists(path):
            new = pd.concat([pd.read_csv(path), new], ignore_index=True, sort=False)
        new.to_csv(path, index=False)
        logger.info("appended run %s to %s", self.run_id, path)
        return path


# --------------------------------------------------------------------- GEARS log
_RE_STEP = re.compile(r"Epoch (\d+) Step (\d+) Train Loss: ([-\d.eE+naif]+)")
_RE_EPOCH = re.compile(r"Epoch (\d+): Train Overall MSE: ([-\d.eE+naif]+) Validation Overall MSE: ([-\d.eE+naif]+)")
_RE_DE = re.compile(r"Train Top 20 DE MSE: ([-\d.eE+naif]+) Validation Top 20 DE MSE: ([-\d.eE+naif]+)")
_RE_TEST_DE = re.compile(r"Best performing model: Test Top 20 DE MSE: ([-\d.eE+naif]+)")
_RE_TEST_SUB = re.compile(r"^test_(.+?): ([-\d.eE+naif]+)$")


def parse_gears_output(lines: List[str]) -> Dict[str, Any]:
    """Turn GEARS's training printout into an epoch table and its own test metrics."""
    epochs: List[Dict[str, Any]] = []
    step_losses: Dict[int, List[float]] = {}
    gears_test: Dict[str, float] = {}
    for line in lines:
        line = line.strip()
        if m := _RE_STEP.search(line):
            step_losses.setdefault(int(m.group(1)), []).append(float(m.group(3)))
        elif m := _RE_EPOCH.search(line):
            epochs.append({"epoch": int(m.group(1)), "train_mse": float(m.group(2)),
                           "val_mse": float(m.group(3))})
        elif (m := _RE_DE.search(line)) and epochs:
            epochs[-1]["train_mse_de"] = float(m.group(1))
            epochs[-1]["val_mse_de"] = float(m.group(2))
        elif m := _RE_TEST_DE.search(line):
            gears_test["test_mse_de"] = float(m.group(1))
        elif m := _RE_TEST_SUB.match(line):
            gears_test["test_" + m.group(1)] = float(m.group(2))
    best = math.inf
    for e in epochs:
        losses = step_losses.get(e["epoch"], [])
        e["train_loss_logged_mean"] = float(np.mean(losses)) if losses else float("nan")
        e["n_logged_steps"] = len(losses)
        v = e.get("val_mse_de", math.inf)
        e["is_best"] = v < best   # GEARS keeps the model with the lowest val DE-MSE
        best = min(best, v)
    return {"epochs": pd.DataFrame(epochs), "gears_test": gears_test}


# ------------------------------------------------------------------- load_run
def load_run(path: str) -> Dict[str, Any]:
    """Load a run folder back into a dict (config, env, split, graphs, metrics)."""

    def _json(p):
        p = os.path.join(path, p)
        return json.load(open(p)) if os.path.exists(p) else None

    def _pkl(p):
        p = os.path.join(path, p)
        return pickle.load(open(p, "rb")) if os.path.exists(p) else None

    def _csv(p):
        p = os.path.join(path, p)
        return pd.read_csv(p) if os.path.exists(p) else None

    cfg = _json("config.json")
    return {
        "run_dir": path,
        "config": cfg,
        "lasser_config": LasserConfig.from_dict({k: v for k, v in cfg.items()
                                                 if k in LasserConfig.__dataclass_fields__})
        if cfg else None,
        "env": _json("env.json"),
        "split": _json("splits/split.json"),
        "graph": _pkl("graphs/coexpress_graph.pkl"),
        "a0_graph": _pkl("graphs/a0_graph.pkl"),
        "edits": _csv("graphs/edits.csv"),
        "checks": _json("graphs/checks.json"),
        "epoch_log": _csv("metrics/epoch_log.csv"),
        "test_metrics": _json("metrics/test_metrics.json"),
        "predictions": _pkl("metrics/predictions.pkl"),
        "checkpoints": sorted(os.listdir(os.path.join(path, "checkpoints")))
        if os.path.isdir(os.path.join(path, "checkpoints")) else [],
    }


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
