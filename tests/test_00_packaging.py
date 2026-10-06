"""Tests 2 and 3: notebook packaging, flag-off import isolation, no external GEARS.

``lasser/`` alone is copied into a fresh temp dir; a subprocess started there imports
it and runs a tiny flag-off experiment. Runs first (file name order) so the parent
process hasn't loaded Norman yet.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

import pytest

import lasser
from conftest import DATA_DIR, GPU, TEST_DIR

GSR_MODULES = ["lasser.data", "lasser.views", "lasser.pretrain", "lasser.refine", "lasser.checks", "lasser.graph"]

SCRIPT = textwrap.dedent("""
    import json, sys
    import lasser
    from lasser import LasserConfig, run_experiment
    cfg = LasserConfig(dataset="norman", data_dir={data!r}, out_dir={out!r}, cache_dir={cache!r},
                       seed=1, epochs=1, subsample_cells_per_condition=2, device={dev!r},
                       graph_learning=False, run_tag="packaging")
    res = run_experiment(cfg)
    mods = sorted(m for m in sys.modules if m == "lasser" or m.startswith("lasser."))
    pip_gears = sorted(m for m in sys.modules if m == "gears" or m.startswith("gears."))
    print("RESULT_JSON:" + json.dumps({{"lasser_file": lasser.__file__, "modules": mods,
                                        "pip_gears": pip_gears, "run_dir": res["run_dir"]}}))
""")


@pytest.fixture(scope="module")
def packaged_run():
    src = os.path.dirname(os.path.abspath(lasser.__file__))
    work = tempfile.mkdtemp(prefix="lasser_pkg_")
    shutil.copytree(src, os.path.join(work, "lasser"), ignore=shutil.ignore_patterns("__pycache__"))
    script = SCRIPT.format(data=DATA_DIR, out=os.path.join(TEST_DIR, "pkg_runs"),
                           cache=os.path.join(TEST_DIR, "pkg_cache"), dev=GPU)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run([sys.executable, "-c", script], cwd=work, env=env,
                          capture_output=True, text=True, timeout=3600)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON:")]
    if proc.returncode != 0 or not lines:
        pytest.fail(f"subprocess failed (code {proc.returncode})\nSTDOUT tail:\n{proc.stdout[-3000:]}\n"
                    f"STDERR tail:\n{proc.stderr[-3000:]}")
    return work, json.loads(lines[-1][len("RESULT_JSON:"):])


def test_packaging_imports_copied_folder(packaged_run):
    work, res = packaged_run
    assert os.path.realpath(res["lasser_file"]).startswith(os.path.realpath(work)), res["lasser_file"]
    assert os.path.exists(os.path.join(res["run_dir"], "metrics", "test_metrics.json"))


def test_no_external_gears_package(packaged_run):
    _, res = packaged_run
    assert not res["pip_gears"], f"a top-level gears package was imported: {res['pip_gears']}"
    assert "lasser.gears" in res["modules"]


def test_flag_off_does_not_import_gsr_modules(packaged_run):
    _, res = packaged_run
    print("lasser modules loaded:", res["modules"])
    leaked = [m for m in GSR_MODULES if m in res["modules"]]
    assert not leaked, f"flag-off run imported {leaked}"
