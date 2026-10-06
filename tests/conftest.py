"""Shared fixtures. Tests run on Kaggle (see notebooks/run_lasser.ipynb).

Environment variables:
    LASSER_DATA_DIR    GEARS data dir (default /kaggle/working/data)
    LASSER_TEST_DIR    scratch dir for test runs (default /kaggle/working/lasser_tests)
    LASSER_SUBSAMPLE   cells kept per condition in test dataloaders (default 4)
    LASSER_SMOKE=1     also run the full-Norman smoke test (test 9)
"""

import os

import pytest
import torch

from lasser import GSRConfig, LasserConfig, load_pert_data

DATA_DIR = os.environ.get("LASSER_DATA_DIR", "/kaggle/working/data")
TEST_DIR = os.environ.get("LASSER_TEST_DIR", "/kaggle/working/lasser_tests")
SUBSAMPLE = int(os.environ.get("LASSER_SUBSAMPLE", "4"))
GPU = "cuda" if torch.cuda.is_available() else "cpu"


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: full-Norman smoke run (LASSER_SMOKE=1)")


def tiny_gsr(**kw) -> GSRConfig:
    """GSR settings small enough for tests (structure of the pipeline unchanged)."""
    base = dict(max_steps=20, eval_every=5, patience=100, n_bootstrap=3, pos_max_cells=3000,
                s_epochs=1, s_walks_per_node=4, n_metacells=200, hard_refresh_every=10)
    base.update(kw)
    return GSRConfig(**base)


def make_cfg(**kw) -> LasserConfig:
    base = dict(dataset="norman", data_dir=DATA_DIR, seed=1, epochs=1,
                out_dir=os.path.join(TEST_DIR, "runs"), cache_dir=os.path.join(TEST_DIR, "cache"),
                subsample_cells_per_condition=SUBSAMPLE, device=GPU, deterministic=True)
    base.update(kw)
    return LasserConfig(**base)


@pytest.fixture(scope="session")
def pert_data():
    return load_pert_data(make_cfg())


@pytest.fixture(scope="session")
def runs(pert_data):
    """One tiny flag-off and one tiny flag-on run, shared by the graph/tracking/cache tests."""
    from lasser import run_experiment

    base = run_experiment(make_cfg(run_tag="test"), pert_data)
    gl = run_experiment(make_cfg(graph_learning=True, gsr=tiny_gsr(), run_tag="test"), pert_data)
    return {"base": base, "gl": gl}
