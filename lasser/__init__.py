"""LASSER: GEARS with a GSR-style learned co-expression graph.

Usage (notebook):
    from lasser import LasserConfig, run_experiment
    results = run_experiment(LasserConfig(dataset="norman", seed=1, graph_learning=True))

GSR modules (data, views, pretrain, refine, checks) are imported only when
``graph_learning=True``.
"""

from .config import GSRConfig, LasserConfig
from .run import load_pert_data, prepare, run_experiment
from .tracking import load_run

__all__ = ["GSRConfig", "LasserConfig", "load_pert_data", "load_run", "prepare", "run_experiment"]
__version__ = "0.1.0"
