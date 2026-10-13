"""Multi-split study runner: experiment table, data warm-up, per-GPU workers, launcher.

Typical use (Kaggle notebook):

    from lasser.study import ARMS, make_jobs, prepare_data, launch, study_status
    jobs = make_jobs(PLAN, base=dict(data_dir=..., out_dir=..., cache_dir=..., epochs=20))
    prepare_data(jobs)                       # one subprocess: downloads, splits, GEARS graphs
    launch(jobs, budget_hours=11.0)          # one worker process per GPU, resumable

A job is done when ``runs.csv`` has a row with the job's config hash (out_dir, device,
run_tag and log level are not part of the hash). Workers claim jobs through files under
``out_dir/claims/<session>/``, so two GPUs never run the same job, and a new session
retries anything that didn't finish. Workers stop taking jobs when the next one would
run past the session deadline.

Command line (used by the launcher):
    python -m lasser.study prep   --jobs jobs.json
    python -m lasser.study worker --jobs jobs.json --session S --deadline T --worker i
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import gc
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .config import LasserConfig
from .tracking import read_runs_csv

logger = logging.getLogger(__name__)

# Arm name -> overrides of LasserConfig (``gsr`` holds GSRConfig overrides).
ARMS: Dict[str, Dict[str, Any]] = {
    # core
    "GEARS": dict(graph_learning=False),
    "LASSER": dict(graph_learning=True),
    # controls
    "GEARS-kNN8": dict(graph_learning=False, coexpress_threshold=0.0, num_similar_genes_co_express_graph=8),
    "LASSER-random": dict(graph_learning=True, gsr=dict(edge_control="random")),
    # ablations
    "LASSER-noS": dict(graph_learning=True, gsr=dict(views=("E", "R"), beta={"E": 4 / 7, "R": 3 / 7})),
    "LASSER-Eonly": dict(graph_learning=True, gsr=dict(views=("E",), beta={"E": 1.0})),
    "LASSER-addonly": dict(graph_learning=True, gsr=dict(r_minus=0.0)),
    "LASSER-embinit": dict(graph_learning=True, gsr_init_gene_emb=True),
    "LASSER-m2": dict(graph_learning=True, gsr=dict(m_plus=2)),
    "LASSER-m10": dict(graph_learning=True, gsr=dict(m_plus=10)),
}

ARM_DESCRIPTIONS = {
    "GEARS": "GEARS, static Pearson co-expression graph (|r| > 0.4, top-20)",
    "LASSER": "GEARS on the GSR-refined co-expression graph (default settings)",
    "GEARS-kNN8": "GEARS, Pearson top-8 neighbours without threshold (density matched to LASSER)",
    "LASSER-random": "LASSER graph with every new edge rewired to a random gene (same in-degrees and weights)",
    "LASSER-noS": "LASSER without the structure view (views E + R)",
    "LASSER-Eonly": "LASSER with the expression view only",
    "LASSER-addonly": "LASSER without removing A0 edges (r- = 0)",
    "LASSER-embinit": "LASSER + gene embedding initialised from the pretrained E view",
    "LASSER-m2": "LASSER with m+ = 2 new edges per gene",
    "LASSER-m10": "LASSER with m+ = 10 new edges per gene",
}

# The plan the user chose: core + controls on 5 splits, ablations on 3 (in priority order).
DEFAULT_PLAN: List[Tuple[str, Sequence[int]]] = [
    ("GEARS", [1, 2, 3, 4, 5]),
    ("LASSER", [1, 2, 3, 4, 5]),
    ("GEARS-kNN8", [1, 2, 3, 4, 5]),
    ("LASSER-random", [1, 2, 3, 4, 5]),
    ("LASSER-noS", [1, 2, 3]),
    ("LASSER-Eonly", [1, 2, 3]),
    ("LASSER-addonly", [1, 2, 3]),
    ("LASSER-embinit", [1, 2, 3]),
    ("LASSER-m2", [1, 2, 3]),
    ("LASSER-m10", [1, 2, 3]),
]


def arm_config(arm: str, seed: int, base: Dict[str, Any]) -> LasserConfig:
    over = copy.deepcopy(ARMS[arm])
    gsr_over = over.pop("gsr", {})
    d = LasserConfig(**base).to_dict()
    d.update(over)
    d["gsr"].update(gsr_over)
    d["seed"] = seed
    d["run_tag"] = f"{arm}_seed{seed}"
    cfg = LasserConfig.from_dict(d)
    cfg.validate()
    return cfg


def make_jobs(plan: Iterable[Tuple[str, Sequence[int]]], base: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Jobs in priority order: all splits of the first arm first, then the next arm, ..."""
    jobs = []
    for arm, seeds in plan:
        for seed in seeds:
            cfg = arm_config(arm, seed, base)
            jobs.append({"job_id": cfg.run_tag, "arm": arm, "seed": seed,
                         "config_hash": cfg.config_hash(), "config": cfg.to_dict()})
    return jobs


def save_jobs(jobs: List[Dict[str, Any]], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(jobs, f, indent=1, default=list)
    return path


def load_jobs(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ status
def _runs_table(out_dir: str) -> pd.DataFrame:
    path = os.path.join(out_dir, "runs.csv")
    return read_runs_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["config_hash"])


def done_hashes(out_dir: str) -> set:
    t = _runs_table(out_dir)
    return set(t["config_hash"].astype(str)) if "config_hash" in t else set()


def study_status(jobs: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per job: arm, seed, done, run_dir (latest), test Pearson delta."""
    out_dir = jobs[0]["config"]["out_dir"]
    t = _runs_table(out_dir)
    rows = []
    for j in jobs:
        hit = t[t["config_hash"].astype(str) == j["config_hash"]] if len(t) else t
        last = hit.iloc[-1] if len(hit) else None
        rows.append({"job_id": j["job_id"], "arm": j["arm"], "seed": j["seed"], "done": last is not None,
                     "test_pearson_delta": None if last is None else last.get("test_pearson_delta"),
                     "train_minutes": None if last is None else round(float(last.get("train_seconds", 0)) / 60, 1),
                     "run_dir": None if last is None else last.get("run_dir")})
    return pd.DataFrame(rows)


def expected_minutes(out_dir: str, default: float = 95.0) -> float:
    """Typical wall time of one run, from finished runs (training + evaluation + overhead)."""
    t = _runs_table(out_dir)
    if "train_seconds" in t and t["train_seconds"].notna().any():
        full = t[t.get("subsample_cells_per_condition").isna()] if "subsample_cells_per_condition" in t else t
        if full["train_seconds"].notna().any():
            return float(full["train_seconds"].median()) / 60 + 8.0  # + data/eval overhead
    return default


# ----------------------------------------------------------------- workers
def _claim(out_dir: str, session: str, job_id: str) -> bool:
    d = os.path.join(out_dir, "claims", session)
    os.makedirs(d, exist_ok=True)
    try:
        fd = os.open(os.path.join(d, job_id), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(f"{os.getpid()} {_dt.datetime.now().isoformat(timespec='seconds')}\n")
    return True


def _release(out_dir: str, session: str, job_id: str) -> None:
    try:
        os.remove(os.path.join(out_dir, "claims", session, job_id))
    except FileNotFoundError:
        pass


def _record_failure(out_dir: str, job: Dict[str, Any], err: str) -> None:
    from .tracking import file_lock

    path = os.path.join(out_dir, "failures.csv")
    row = pd.DataFrame([{"time": _dt.datetime.now().isoformat(timespec="seconds"), "job_id": job["job_id"],
                         "config_hash": job["config_hash"], "error": err[-2000:]}])
    with file_lock(path + ".lock"):
        if os.path.exists(path):
            row = pd.concat([pd.read_csv(path), row], ignore_index=True)
        row.to_csv(path, index=False)


def run_worker(jobs: List[Dict[str, Any]], session: str, deadline: float, worker: int = 0) -> None:
    """Run jobs one after another on this process's GPU until none are left or time runs out."""
    import torch

    from .run import load_pert_data, run_experiment
    from .utils import describe_device

    out_dir = jobs[0]["config"]["out_dir"]
    logger.info("worker %d on %s (CUDA_VISIBLE_DEVICES=%s), %d jobs, deadline %s", worker,
                describe_device("cuda" if torch.cuda.is_available() else "cpu"),
                os.environ.get("CUDA_VISIBLE_DEVICES"), len(jobs),
                _dt.datetime.fromtimestamp(deadline).isoformat(timespec="minutes"))
    pert_data = None
    for job in jobs:
        if job["config_hash"] in done_hashes(out_dir):
            continue
        need = expected_minutes(out_dir) * 60
        if time.time() + need > deadline:
            logger.info("worker %d: next job needs ~%.0f min, past the deadline; stopping", worker, need / 60)
            break
        if not _claim(out_dir, session, job["job_id"]):
            continue
        cfg = LasserConfig.from_dict(job["config"])
        logger.info("worker %d: starting %s", worker, job["job_id"])
        res = None
        try:
            if pert_data is None:
                from .tracking import file_lock
                with file_lock(os.path.join(out_dir, "data_load.lock")):   # one worker loads at a time
                    pert_data = load_pert_data(cfg)
            res = run_experiment(cfg, pert_data)
            logger.info("worker %d: finished %s -> %s", worker, job["job_id"], res["run_dir"])
        except Exception as e:  # keep going; the next session retries unfinished jobs
            import traceback
            logger.exception("worker %d: %s failed", worker, job["job_id"])
            _record_failure(out_dir, job, f"{e!r}\n{traceback.format_exc()}")
            _release(out_dir, session, job["job_id"])
        finally:
            del res
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    logger.info("worker %d: done", worker)


def run_prep(jobs: List[Dict[str, Any]]) -> None:
    """Download data and build every split and GEARS graph the jobs need, once, before workers start
    (so parallel workers never download or write the same file at the same time)."""
    from .gears import GEARS
    from .run import load_pert_data

    cfgs = [LasserConfig.from_dict(j["config"]) for j in jobs]
    pert_data = load_pert_data(cfgs[0])
    variants = sorted({(c.coexpress_threshold, c.num_similar_genes_co_express_graph) for c in cfgs})
    for seed in sorted({c.seed for c in cfgs}):
        c = next(c for c in cfgs if c.seed == seed)
        pert_data.prepare_split(split=c.split, seed=seed, train_gene_set_size=c.train_gene_set_size)
        pert_data.get_dataloader(batch_size=c.batch_size, test_batch_size=c.test_batch_size)
        for thr, k in variants:
            g = GEARS(pert_data, device="cpu")
            g.model_initialize(coexpress_threshold=thr, num_similar_genes_co_express_graph=k)
            logger.info("prep: seed %d, co-expression graph (threshold %s, k %s): %d edges",
                        seed, thr, k, g.config["G_coexpress"].shape[1])
            del g
    logger.info("prep: data, splits and GEARS graphs ready for seeds %s", sorted({c.seed for c in cfgs}))


# ---------------------------------------------------------------- launcher
WORKER_RAM_GB = 15.0   # one worker's peak: Norman cell graphs (~6 GB) + GEARS's per-epoch prediction arrays


def total_ram_gb() -> Optional[float]:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def max_parallel_workers(n_gpus: int, worker_ram_gb: float = WORKER_RAM_GB) -> int:
    """How many workers fit in RAM. Kaggle's T4 x2 machine (~31 GB) fits one: run 1 of the
    study had its second worker killed by the OOM killer (exit -9)."""
    ram = total_ram_gb()
    if ram is None:
        return max(n_gpus, 1)
    # keep one worker's worth of RAM for the notebook kernel and the system
    return max(1, min(n_gpus, int(ram // worker_ram_gb) - 1))


def _python_cmd(*args: str) -> List[str]:
    return [sys.executable, "-m", "lasser.study", *args]


def _pkg_parent() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tail(path: str, n: int) -> str:
    try:
        with open(path, errors="replace") as f:
            return "\n".join("    " + l.rstrip() for l in f.readlines()[-n:])
    except OSError:
        return "    (no log)"


def _release_claims_of(out_dir: str, session: str, pid: int, jobs: List[Dict[str, Any]]) -> List[str]:
    """Remove the claims a dead worker held on unfinished jobs, so another worker can take them."""
    d = os.path.join(out_dir, "claims", session)
    if not os.path.isdir(d):
        return []
    done = done_hashes(out_dir)
    hashes = {j["job_id"]: j["config_hash"] for j in jobs}
    freed = []
    for job_id in os.listdir(d):
        try:
            with open(os.path.join(d, job_id)) as f:
                owner = int(f.read().split()[0])
        except (OSError, ValueError, IndexError):
            continue
        if owner == pid and hashes.get(job_id) not in done:
            _release(out_dir, session, job_id)
            freed.append(job_id)
    return freed


def prepare_data(jobs: List[Dict[str, Any]], jobs_path: Optional[str] = None) -> int:
    """Run ``run_prep`` in a subprocess (frees its memory when done); streams its log."""
    out_dir = jobs[0]["config"]["out_dir"]
    jobs_path = jobs_path or save_jobs(jobs, os.path.join(out_dir, "jobs.json"))
    log = os.path.join(out_dir, "logs", "prep.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as f:
        code = subprocess.call(_python_cmd("prep", "--jobs", jobs_path), cwd=_pkg_parent(),
                               stdout=f, stderr=subprocess.STDOUT)
    lines = [l for l in open(log).read().splitlines() if "lasser" in l or "Error" in l]
    print("\n".join(lines[-15:]))
    if code != 0:
        raise RuntimeError(f"data preparation failed (exit {code}); see {log}")
    return code


def launch(jobs: List[Dict[str, Any]], gpus: Optional[Sequence[int]] = None, budget_hours: float = 11.0,
           poll_minutes: float = 10.0, session: Optional[str] = None) -> pd.DataFrame:
    """Start one worker per GPU and wait, printing progress every ``poll_minutes``.

    ``budget_hours`` is measured from now; keep it under Kaggle's 12-hour session limit so
    outputs are saved before the session ends.
    """
    import torch

    out_dir = jobs[0]["config"]["out_dir"]
    if gpus is None:
        avail = list(range(torch.cuda.device_count())) or [None]
        n = max_parallel_workers(len(avail))
        gpus = avail[:n]
        if n < len(avail):
            print(f"{len(avail)} GPUs but RAM ({total_ram_gb():.0f} GB) fits {n} worker(s) of ~{WORKER_RAM_GB:.0f} GB; "
                  f"using GPU(s) {gpus}. Pass gpus=[0, 1] to override.")
    session = session or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    jobs_path = save_jobs(jobs, os.path.join(out_dir, "jobs.json"))
    deadline = time.time() + budget_hours * 3600
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    todo = [j for j in jobs if j["config_hash"] not in done_hashes(out_dir)]
    est = expected_minutes(out_dir)
    print(f"session {session}: {len(jobs) - len(todo)}/{len(jobs)} jobs already done, {len(todo)} to run "
          f"on {len(gpus)} GPU(s); ~{est:.0f} min each -> ~{len(todo) * est / max(len(gpus), 1) / 60:.1f} h "
          f"(budget {budget_hours:.1f} h)")
    procs = []
    for i, gpu in enumerate(gpus):
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        threads = str(max(1, (os.cpu_count() or 2) // len(gpus)))   # share the CPU cores fairly
        env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
        env.pop("MPLBACKEND", None)
        log = open(os.path.join(log_dir, f"worker{i}_{session}.log"), "w")
        p = subprocess.Popen(_python_cmd("worker", "--jobs", jobs_path, "--session", session,
                                         "--deadline", str(deadline), "--worker", str(i)),
                             cwd=_pkg_parent(), env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((p, log))
        print(f"worker {i}: GPU {gpu}, pid {p.pid}, log {log.name}")
        time.sleep(30)  # stagger data loading

    reported = set()
    while any(p.poll() is None for p, _ in procs):
        time.sleep(poll_minutes * 60)
        for i, (p, log) in enumerate(procs):
            if p.poll() is not None and i not in reported:
                reported.add(i)
                freed = _release_claims_of(out_dir, session, p.pid, jobs)
                hint = " (killed: most likely out of RAM)" if p.returncode == -9 else ""
                print(f"worker {i} exited with code {p.returncode}{hint}; freed {freed or 'no'} unfinished job(s); "
                      "last log lines:\n" + _tail(log.name, 8))
        st = study_status(jobs)
        stamp = _dt.datetime.now().strftime("%H:%M")
        running = sorted(os.listdir(os.path.join(out_dir, "claims", session))) \
            if os.path.isdir(os.path.join(out_dir, "claims", session)) else []
        running = [r for r in running if not st.loc[st.job_id == r, "done"].any()]
        print(f"[{stamp}] done {int(st.done.sum())}/{len(st)} | running: {', '.join(running) or '-'}")
    for i, (p, log) in enumerate(procs):
        log.close()
        hint = " (killed: most likely out of RAM)" if p.returncode == -9 else ""
        print(f"worker {i} exited with code {p.returncode}{hint}")
        _release_claims_of(out_dir, session, p.pid, jobs)
    fails = os.path.join(out_dir, "failures.csv")
    if os.path.exists(fails):
        print(f"failures so far: {len(pd.read_csv(fails))} (see {fails})")
    return study_status(jobs)


def _cli(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lasser.study")
    ap.add_argument("mode", choices=["prep", "worker"])
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--session", default="manual")
    ap.add_argument("--deadline", type=float, default=time.time() + 11 * 3600)
    ap.add_argument("--worker", type=int, default=0)
    a = ap.parse_args(argv)
    # Own handler for the study logger only: run logs get theirs from RunTracker.
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    lg = logging.getLogger("lasser.study")
    lg.setLevel(logging.INFO)
    lg.addHandler(h)
    lg.propagate = False
    jobs = load_jobs(a.jobs)
    if a.mode == "prep":
        run_prep(jobs)
    else:
        run_worker(jobs, a.session, a.deadline, a.worker)


if __name__ == "__main__":
    _cli()
