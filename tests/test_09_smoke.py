"""Test 9: full Norman, seed 1, 1 epoch, flag off and on, T4 memory budget.

Skipped unless LASSER_SMOKE=1 (it trains on the full dataset).
"""

import os

import pytest
import torch

from conftest import make_cfg

pytestmark = pytest.mark.skipif(os.environ.get("LASSER_SMOKE") != "1", reason="set LASSER_SMOKE=1")

T4_GB = 15.0
GSR_BUDGET_GB = 4.0


@pytest.mark.parametrize("graph_learning", [False, True])
def test_smoke_full_norman(pert_data, graph_learning):
    from lasser import run_experiment

    assert torch.cuda.is_available(), "smoke test needs a GPU"
    cfg = make_cfg(graph_learning=graph_learning, subsample_cells_per_condition=None, epochs=1,
                   device="cuda", deterministic=False, run_tag="smoke")
    res = run_experiment(cfg, pert_data)
    t = res["timings"]
    print({k: v for k, v in t.items()})
    print({k: v for k, v in res["row"].items() if k.startswith("test_") or k.startswith("graph_")})
    peaks = [v["peak_cuda_gb"] for v in t.values() if isinstance(v, dict) and "peak_cuda_gb" in v]
    assert max(peaks) < T4_GB
    if graph_learning:
        gsr = [v["peak_cuda_gb"] for k, v in t.items() if k.startswith("gsr") and isinstance(v, dict)
               and "peak_cuda_gb" in v]
        assert max(gsr) < GSR_BUDGET_GB, f"GSR used {max(gsr):.2f} GB"
