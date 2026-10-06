# LASSER

**Learning Adaptive Sparse Structures for gene Effect Reasoning.** This is a thesis project that extends GEARS (perturbation-response prediction) by **learning its gene co-expression graph** with a GSR-style self-supervised method, instead of building it once from Pearson correlation. A later goal is to reason about gene relationships from the learned graph.

Scope right now: **co-expression graph only.** The GO perturbation graph stays GEARS's static graph. Don't learn or modify it.

**Start every session by reading `summarizer.md`.** It records current status, decisions and next steps. Update it whenever the status changes.

## Workspace map

| Path | What it is | How to treat it |
|---|---|---|
| `summarizer.md` | **Handoff summary**: status, decisions, package layout, deviations, next steps | Read first; keep it current |
| `lasser/` | Our package (implemented), **including the vendored GEARS in `lasser/gears/`** | Everything new goes here. Self-contained: never import an external `gears` |
| `tests/` | pytest tests 1–9 from the spec | Run on Kaggle only |
| `notebooks/run_lasser.ipynb` | Kaggle notebook: install → clone → tests → baseline → graph learning → results | The way everything is run |
| `LASSER.md` | User docs: notebook steps, every config field, run-folder layout, `load_run` | Update it when behaviour or config changes |
| `GEARS/` | Upstream GEARS source (`snap-stanford/GEARS`, commit `f374e43`) | **Local read-only reference only** (git-ignored, never pushed, not used at runtime). The runtime copy is `lasser/gears/` |
| `WSDM23-GSR/` | GSR reference code (DGL-based) | **Local read-only reference only** (git-ignored, never pushed). Port ideas to PyTorch + PyG; never import it or add DGL |
| `s41587-023-01905-6.pdf` | GEARS paper (Nat Biotech 2024) | Model, data, splits and metric definitions |
| `3539597.3570455.pdf` | GSR paper (WSDM 2023) | Pretrain → refine → finetune method |
| `description.MD` | Project description | Project context |
| `claude_code_prompt_gsr_gears.md` | **The task spec**: packaging, flags, logging, tests, deliverables | Follow it |
| `gsr_coexpression_plan.md` | Method design (views, pretraining, refinement, §11 defaults) | The spec refers to it |
| `eval.py` | **The metrics to report** | Source of truth for evaluation (see below) |

## Non-negotiables

1. **`graph_learning=False` must reproduce upstream GEARS exactly**: same graph, initialisation, RNG stream and results. The flag-off parity test (`tests/test_01_parity.py`) must pass before any other work is considered done. The call and seeding order is documented at the top of `lasser/run.py`; don't reorder it.
2. **`lasser/` is self-contained and independent of any external GEARS.** GEARS lives in `lasser/gears/` (vendored); import it only relatively (`from .gears import GEARS, PertData`, `from .gears.utils import ...`). Never import a top-level `gears`, never use pip `cell-gears`, never reference the `GEARS/` folder from code or tests. Edit `lasser/gears/` only for import or compatibility fixes that keep behaviour identical, mark each with `# LASSER:` and list it in `lasser/gears/VENDORED.md`. It is used by `git clone` → `mv <repo>/lasser .` → `import lasser` inside a Kaggle notebook. Use relative imports only, with no `sys.path` hacks, no repo-root paths and no install step. GSR modules (`data`, `views`, `pretrain`, `refine`, `checks`, `graph`) must be imported lazily, only when `graph_learning=True`.
3. **Don't change GEARS behaviour** (in the vendored copy or around it). The GO graph, `GEARS_Model.forward`, loss, training loop and default hyperparameters are untouched. The graph enters only through `model_initialize(G_coexpress=..., G_coexpress_weight=...)`. GEARS's co-expression batching quirk (only the first cell per batch gets graph messages) is **kept by the user's decision**; don't "fix" it unless asked.
4. **No data leakage.** Features, positives, the A0 graph and refinement use only control cells plus training-perturbation cells of the current split seed. Validation perturbations are used only for model selection. Test perturbations are never read before evaluation. `lasser/data.py::assert_training_only` enforces this.
5. **Memory (Kaggle T4, 16 GB).** Never keep a dense N×N tensor in autograd, and build edge lists directly. No per-sample Python loops over a batch. Pretraining and refinement should use < 4 GB GPU memory.

## Running: Kaggle only

- **Don't run anything locally**: no package installs, venvs, data downloads, tests or training on this machine. Static checks such as `ast.parse` are fine.
- The user runs `notebooks/run_lasser.ipynb` on Kaggle (GPU, Internet on) and sends back the notebook or `test_output.txt` after each run. Fix issues from that output.
- Tests run on tiny subsets (`subsample_cells_per_condition`). The parity and leakage tests run on CPU for exact equality. The smoke test (test 9) needs `LASSER_SMOKE=1`.

## Evaluation: `eval.py`

- `eval.py` defines the metrics every run reports, baseline and graph learning alike. `lasser/_eval.py` is a **byte-for-byte copy** (a test checks the hash); if `eval.py` changes, recopy it.
- **Don't modify `eval.py`.** `lasser/evaluation.py` is the adapter that builds its inputs: `ctrl` from `GEARS.ctrl_expression`, the train mean, `de_indices`, the subgroups from `pert_data.subgroup`, and per-subgroup loaders.
- Every `eval.py` metric is reported overall and per test subgroup (unseen_single, combo_seen0/1/2) in `metrics/test_metrics.json`, and the headline ones in `runs.csv`. GEARS's own printed metrics are kept under `gears_inference`.

## Run tracking

Every run (flag on or off) writes `out_dir/<dataset>_seed<seed>_<gl|base>_<timestamp>/` with:

- `run.log`, `config.json`, `env.json`;
- `splits/`: the GEARS split pkl, `split.json` and the split hash;
- `graphs/coexpress_graph.pkl`: **the graph actually used in training**, read back from the model;
- for GSR runs, also `graphs/a0_graph.pkl`, `edits.csv` and `checks.json`;
- `metrics/`: `epoch_log.csv`, `test_metrics.json`, `predictions.pkl`, `timings.json`;
- `checkpoints/`.

A row is appended to `out_dir/runs.csv`. `lasser.load_run(path)` reads a run back. Full layout is in `LASSER.md`.

## Conventions

- Use Python `logging` (`logging.getLogger(__name__)`) for everything; no bare `print` in library code. GEARS's stderr output is captured with `utils.capture_stderr`.
- All settings live in the `LasserConfig` / `GSRConfig` dataclasses. No hard-coded paths; defaults are Kaggle-friendly (`/kaggle/working/...`). Adding a `GSRConfig` field changes the GSR cache key automatically; bump `GSR_CACHE_VERSION` when the pipeline's logic changes.
- Seed everything through `utils.seed_everything`. Tests use deterministic settings.
- Dependencies (`requirements.txt`): torch, torch_geometric, scanpy (+anndata), numpy, pandas, scipy, scikit-learn, networkx, tqdm, requests. On Kaggle only `torch_geometric scanpy pytest` are pip-installed. GEARS's optional extras (dcor, wandb, seaborn, matplotlib) must stay lazy imports. Ask before adding anything else (pyg-lib and torch-cluster are deliberately avoided).
- After changing imports, run the static import audit (parse every file under `lasser/` with `ast`, check that relative imports resolve and that there is no top-level `gears` import). That is allowed locally; executing the code is not.
- Keep functions small and typed. Shapes and node order follow GEARS's `node_map`.
- Git: local repo, commit in small steps with clear messages. Don't push unless asked. Keep `.gitignore` patterns for the reference folders anchored (`/GEARS/`, `/WSDM23-GSR/`); on case-insensitive Windows git an unanchored `GEARS/` would also ignore `lasser/gears/`.

## Working style

- Plan before large changes: report the integration points and wait for an OK.
- Work in small steps, and get Kaggle test output before building on unverified code. Don't claim something works until the user's Kaggle output shows it.
- When the real code disagrees with the spec or this file, follow the code and say so.
- Don't ask for long training runs unless needed; smoke tests use 1 epoch on a small subset.
- Keep `summarizer.md` (status and next steps) and `LASSER.md` (user-facing docs) up to date with every change.