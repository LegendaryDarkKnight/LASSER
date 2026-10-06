"""Small shared helpers: seeding, phase timing, stderr capture and graph statistics."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import random
import sys
import time
from typing import Callable, Dict, Iterator, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """The one place where RNGs are seeded (python, numpy, torch CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requested but CUDA is not available")
    return torch.device(device)


@contextlib.contextmanager
def phase(name: str, timings: Optional[Dict[str, Dict[str, float]]] = None) -> Iterator[None]:
    """Log a phase's wall time and peak CUDA memory; optionally record them in ``timings``."""
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    logger.info("[%s] start", name)
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
        logger.info("[%s] done in %.1fs, peak CUDA memory %.2f GB", name, dt, peak)
        if timings is not None:
            timings[name] = {"seconds": round(dt, 2), "peak_cuda_gb": round(peak, 3)}


class _Tee(io.TextIOBase):
    """Write-through stream that also hands every complete line to a callback."""

    def __init__(self, stream, on_line: Callable[[str], None]):
        self._stream = stream
        self._on_line = on_line
        self._buf = ""

    def write(self, s: str) -> int:
        self._stream.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rsplit("\r", 1)[-1]  # keep only the final state of tqdm-style lines
            if line.strip():
                self._on_line(line)
        return len(s)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._stream, name)


@contextlib.contextmanager
def capture_stderr(lines: List[str], log: Optional[logging.Logger] = None) -> Iterator[None]:
    """Collect GEARS's ``print_sys`` output (stderr) while still showing it.

    GEARS reports training progress only through stderr, so this is how per-epoch
    numbers are recorded without touching its training loop.
    """

    def on_line(line: str) -> None:
        lines.append(line)
        if log is not None:
            log.info(line)

    old = sys.stderr
    sys.stderr = _Tee(old, on_line)
    try:
        yield
    finally:
        sys.stderr.flush()
        sys.stderr = old


def graph_stats(edge_index: np.ndarray, num_nodes: int) -> Dict[str, float]:
    """Degree statistics. Degree = in-degree without self-loops (what SGConv aggregates)."""
    src, dst = np.asarray(edge_index[0]), np.asarray(edge_index[1])
    loop = src == dst
    deg = np.bincount(dst[~loop], minlength=num_nodes)
    return {
        "num_nodes": int(num_nodes),
        "num_edges": int(len(src)),
        "num_self_loops": int(loop.sum()),
        "degree_min": int(deg.min()),
        "degree_median": float(np.median(deg)),
        "degree_mean": float(deg.mean()),
        "degree_max": int(deg.max()),
        "degree_gini": gini(deg),
        "isolated_genes": int((deg == 0).sum()),
    }


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
