"""Run configuration: one ``LasserConfig`` (with a nested ``GSRConfig``) drives a whole run."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# Bump when a change to the GSR pipeline should invalidate cached graphs.
GSR_CACHE_VERSION = 2  # 2: canonical CSR input (run 02)

# Fields that say *where* or *how loudly* a run happens, not *what* it computes.
_NON_SEMANTIC = {"data_dir", "out_dir", "cache_dir", "log_level", "repo_dir", "device", "run_tag"}


@dataclass
class GSRConfig:
    """GSR-style co-expression graph learning. Defaults follow gsr_coexpression_plan.md §11."""

    # Views (plan §3). "X" also needs ``x_path``.
    views: Tuple[str, ...] = ("E", "R", "S")
    x_path: Optional[str] = None

    # Data (plan §2, §3)
    h1_frac: float = 0.5               # fraction of each training condition's cells in half H1
    n_metacells: int = 500             # k-means metacells for view E
    metacell_svd_dim: int = 50         # cell-space SVD dims used for k-means
    e_dim: int = 64                    # PCA dims of view E
    r_dim: int = 32                    # PCA dims of view R
    x_dim: int = 64                    # PCA dims of view X

    # Structure view S: DeepWalk on A0 (plan §3)
    s_dim: int = 64
    s_walk_length: int = 20
    s_walks_per_node: int = 10
    s_window: int = 5
    s_neg: int = 5
    s_epochs: int = 3
    s_batch_size: int = 16384
    s_lr: float = 0.01

    # Positives (plan §4.1)
    pos_k: int = 20                    # Pearson top-k per gene on H1 (GEARS k)
    pos_threshold: float = 0.4         # |r| threshold (GEARS coexpress_threshold)
    pos_max_cells: int = 20000         # cap on H1 cells used for correlations
    n_bootstrap: int = 20
    bootstrap_keep: float = 0.7        # keep pairs strong in >= 70% of bootstrap resamples
    response_pos_threshold: Optional[float] = None  # optional extra positives from view R

    # Negatives (plan §4.1)
    n_neg: int = 256
    hard_neg_frac: float = 0.25
    hard_pool_size: int = 32           # top cosine candidates per gene and view
    hard_corr_max: float = 0.1         # hard negatives must have |r_H1| below this
    hard_refresh_every: int = 50

    # Encoders (plan §4.2)
    emb_dim: int = 64
    hidden_dim: int = 256
    n_conv_layers: int = 2
    decoder_hidden: int = 64
    momentum: float = 0.99
    feat_mask: float = 0.2
    edge_drop: float = 0.2
    msg_edge_frac: float = 0.7         # per-epoch split of training positives: message vs target
    link_val_frac: float = 0.1         # positives held out for link AUC / early stopping

    # Losses (plan §4.3)
    tau: float = 0.2
    alpha: float = 0.75                # intra-view weight
    vicreg_weight: float = 0.1
    vicreg_std_target: float = 1.0

    # Schedule (plan §4.4)
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_steps: int = 300
    batch_edges: int = 2048            # target edges (queries) per step
    eval_every: int = 10
    patience: int = 50                 # steps without link-AUC improvement

    # Refinement (plan §5)
    beta: Dict[str, float] = field(default_factory=lambda: {"E": 0.4, "R": 0.3, "S": 0.3})
    r_minus: float = 0.1
    m_plus: int = 5
    d_min: int = 5
    gamma: float = 0.5
    score_floor: float = 0.0
    cand_topk: int = 100
    chunk_size: int = 512

    # Checks (plan §6.1)
    h2_hit_threshold: float = 0.2
    collapse_sample: int = 2000

    def validate(self) -> None:
        unknown = set(self.views) - {"E", "R", "S", "X"}
        if unknown:
            raise ValueError(f"unknown views {sorted(unknown)}")
        if "X" in self.views and not self.x_path:
            raise ValueError("view X needs gsr.x_path")
        missing_beta = [v for v in self.views if v not in self.beta]
        if missing_beta:
            raise ValueError(f"beta has no weight for views {missing_beta}")
        total = sum(self.beta[v] for v in self.views)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"beta over the active views must sum to 1, got {total}")
        if not 0 <= self.r_minus < 1:
            raise ValueError("r_minus must be in [0, 1)")
        if self.m_plus < 0 or self.d_min < 0:
            raise ValueError("m_plus and d_min must be >= 0")
        if not 0 <= self.hard_neg_frac < 1:
            raise ValueError("hard_neg_frac must be in [0, 1)")


@dataclass
class LasserConfig:
    """Everything a run needs. Defaults are GEARS's defaults and Kaggle paths."""

    # Data and split
    dataset: str = "norman"
    data_dir: str = "/kaggle/working/data"
    split: str = "simulation"
    seed: int = 1
    train_gene_set_size: float = 0.75

    # GEARS hyperparameters (GEARS defaults; batch sizes as in the GEARS demo)
    batch_size: int = 32
    test_batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 5e-4
    hidden_size: int = 64
    num_go_gnn_layers: int = 1
    num_gene_gnn_layers: int = 1
    decoder_hidden_size: int = 16
    num_similar_genes_go_graph: int = 20
    num_similar_genes_co_express_graph: int = 20
    coexpress_threshold: float = 0.4
    uncertainty: bool = False
    uncertainty_reg: float = 1.0
    direction_lambda: float = 1e-1

    # Flags
    graph_learning: bool = False
    gsr_init_gene_emb: bool = False

    # Runtime
    device: str = "cuda"
    deterministic: bool = False
    out_dir: str = "/kaggle/working/runs"
    cache_dir: str = "/kaggle/working/cache"
    log_level: str = "INFO"
    repo_dir: Optional[str] = None       # git checkout, for the commit in env.json
    run_tag: Optional[str] = None
    # Testing only: keep the first n cells of every condition in the dataloaders.
    subsample_cells_per_condition: Optional[int] = None

    gsr: GSRConfig = field(default_factory=GSRConfig)

    def validate(self) -> None:
        if self.gsr_init_gene_emb and not self.graph_learning:
            raise ValueError("gsr_init_gene_emb needs graph_learning=True")
        if self.uncertainty:
            raise ValueError("uncertainty mode is not supported: eval.py expects point predictions")
        if self.graph_learning:
            self.gsr.validate()
        if self.gsr_init_gene_emb and "E" not in self.gsr.views:
            raise ValueError("gsr_init_gene_emb uses the E view; add 'E' to gsr.views")

    def gears_model_kwargs(self) -> Dict[str, Any]:
        """Arguments for ``GEARS.model_initialize`` (graphs excluded)."""
        return dict(
            hidden_size=self.hidden_size,
            num_go_gnn_layers=self.num_go_gnn_layers,
            num_gene_gnn_layers=self.num_gene_gnn_layers,
            decoder_hidden_size=self.decoder_hidden_size,
            num_similar_genes_go_graph=self.num_similar_genes_go_graph,
            num_similar_genes_co_express_graph=self.num_similar_genes_co_express_graph,
            coexpress_threshold=self.coexpress_threshold,
            uncertainty=self.uncertainty,
            uncertainty_reg=self.uncertainty_reg,
            direction_lambda=self.direction_lambda,
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LasserConfig":
        d = dict(d)
        g = dict(d.pop("gsr", {}) or {})
        if "views" in g:
            g["views"] = tuple(g["views"])
        return cls(**d, gsr=GSRConfig(**g))

    def config_hash(self) -> str:
        d = {k: v for k, v in self.to_dict().items() if k not in _NON_SEMANTIC}
        if not self.graph_learning:
            d.pop("gsr")  # GSR settings don't affect a baseline run
        return stable_hash(d)

    def gsr_cache_key(self) -> str:
        """Everything the refined graph depends on."""
        return stable_hash({
            "version": GSR_CACHE_VERSION,
            "dataset": self.dataset,
            "split": self.split,
            "seed": self.seed,
            "train_gene_set_size": self.train_gene_set_size,
            "coexpress_threshold": self.coexpress_threshold,
            "num_similar_genes_co_express_graph": self.num_similar_genes_co_express_graph,
            "gsr": dataclasses.asdict(self.gsr),
        })


def stable_hash(obj: Any, n: int = 12) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:n]
