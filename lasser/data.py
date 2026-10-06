"""Training-only cell selection, halves H1/H2 and metacells (plan §2).

Only control cells and cells of *training* perturbations are ever selected here.
``assert_training_only`` enforces that no validation or test condition gets in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


@dataclass
class TrainingCells:
    """Index sets into ``adata`` (rows), all drawn from control + training perturbations."""

    train_conditions: List[str]   # training conditions, excluding 'ctrl'
    cell_idx: np.ndarray          # control + all training-perturbation cells
    a0_cell_idx: np.ndarray       # the subset GEARS builds A0 from: control + training singles
    h1_idx: np.ndarray            # half H1 of cell_idx (stratified by condition)
    h2_idx: np.ndarray            # half H2 = cell_idx minus H1
    conditions: np.ndarray        # condition of every adata row (strings)


def assert_training_only(conditions: np.ndarray, idx: np.ndarray, set2conditions: Dict[str, list]) -> None:
    used = set(np.unique(conditions[idx]).tolist())
    held_out = set(set2conditions.get("val", [])) | set(set2conditions.get("test", []))
    held_out.discard("ctrl")
    leaked = used & held_out
    assert not leaked, f"validation/test conditions in the training mask: {sorted(leaked)[:10]}"
    allowed = set(set2conditions["train"]) | {"ctrl"}
    extra = used - allowed
    assert not extra, f"non-training conditions in the training mask: {sorted(extra)[:10]}"


def build_training_cells(adata, set2conditions: Dict[str, list], seed: int, h1_frac: float = 0.5) -> TrainingCells:
    conditions = np.asarray(adata.obs["condition"].astype(str).values)
    train_conditions = sorted(c for c in set2conditions["train"] if c != "ctrl")
    is_train = np.isin(conditions, train_conditions + ["ctrl"])
    cell_idx = np.flatnonzero(is_train)
    # GEARS builds its co-expression graph from conditions containing 'ctrl'
    # (control + single perturbations) among the training conditions.
    a0_conds = [c for c in set2conditions["train"] if "ctrl" in c]
    a0_cell_idx = np.flatnonzero(np.isin(conditions, a0_conds))

    # Stratified halves: within each condition, a random h1_frac of cells goes to H1.
    rng = np.random.default_rng(seed)
    cond = conditions[cell_idx]
    keys = rng.random(len(cell_idx))
    order = np.lexsort((keys, cond))
    sorted_cond = cond[order]
    starts = np.r_[0, np.flatnonzero(sorted_cond[1:] != sorted_cond[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(order)])
    rank = np.arange(len(order)) - np.repeat(starts, sizes)
    n_h1 = np.repeat(np.ceil(sizes * h1_frac).astype(int), sizes)
    in_h1 = np.zeros(len(cell_idx), dtype=bool)
    in_h1[order] = rank < n_h1
    h1_idx, h2_idx = np.sort(cell_idx[in_h1]), np.sort(cell_idx[~in_h1])

    for idx in (cell_idx, a0_cell_idx, h1_idx, h2_idx):
        assert_training_only(conditions, idx, set2conditions)
    logger.info("training cells: %d (A0 subset %d, H1 %d, H2 %d) over %d training perturbations",
                len(cell_idx), len(a0_cell_idx), len(h1_idx), len(h2_idx), len(train_conditions))
    return TrainingCells(train_conditions, cell_idx, a0_cell_idx, h1_idx, h2_idx, conditions)


def rows_csr(adata, idx: np.ndarray) -> sp.csr_matrix:
    X = adata.X
    X = X[idx] if sp.issparse(X) else sp.csr_matrix(np.asarray(X)[idx])
    return sp.csr_matrix(X, dtype=np.float32)


def group_means(X: sp.csr_matrix, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean row per label, via a sparse indicator matrix (no Python loop over cells)."""
    uniq, inv = np.unique(labels, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    ind = sp.csr_matrix((1.0 / counts[inv], (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))
    return uniq, np.asarray((ind @ X).todense(), dtype=np.float64)


def build_metacells(X: sp.csr_matrix, n_metacells: int, svd_dim: int, seed: int) -> np.ndarray:
    """Pseudobulk cells into k-means metacells (cells -> SVD space -> MiniBatchKMeans)."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD

    n = X.shape[0]
    k = min(n_metacells, n)
    dim = min(svd_dim, X.shape[1] - 1, n - 1)
    emb = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(X)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=3)
    labels = km.fit_predict(emb)
    _, M = group_means(X, labels)
    logger.info("metacells: %d cells -> %d metacells", n, M.shape[0])
    return M
