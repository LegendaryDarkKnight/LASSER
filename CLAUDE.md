# LASSER

**Learning Adaptive Sparse Structures for gene Effect Reasoning.** This is a thesis project that extends GEARS (perturbation-response prediction) by **learning its gene co-expression graph** with a GSR-style self-supervised method, instead of building it once from Pearson correlation. A later goal is to reason about gene relationships from the learned graph.

Scope right now: **co-expression graph only.** The GO perturbation graph stays GEARS's static graph. Don't learn or modify it.

## Workspace map

| Path | What it is | How to treat it |
|---|---|---|
| `GEARS/` | Upstream GEARS source (`snap-stanford/GEARS`) | **Read-only reference.** At runtime GEARS comes from pip (`cell-gears`, pinned). Don't edit it. |
| `WSDM23-GSR/` | GSR reference code (DGL-based) | **Read-only reference** for the method. Port the ideas to PyTorch + PyG; never import it or add DGL. |
| `s41587-023-01905-6.pdf` | GEARS paper (Nat Biotech 2024) | Model, data, splits and metric definitions |
| `3539597.3570455.pdf` | GSR paper (WSDM 2023) | Pretrain → refine → finetune method |
| `description.MD` | Project description | Read it for project context |
| `claude_code_prompt_gsr_gears.md` | **The task spec**: packaging, flags, logging, tests, deliverables | Follow it |
| `gsr_coexpression_plan.md` | Method design (views, pretraining, refinement, defaults) | The spec refers to it. If it is missing, ask for it rather than guessing |
| `eval.py` | **The metrics to report** | Source of truth for evaluation (see below) |
| `lasser/` | Our code (to be created) | Everything new goes here |

## Non-negotiables

1. **`graph_learning=False` must reproduce upstream GEARS exactly**: same graph, init, RNG stream and results. The flag-off parity test must pass before any other work is considered done.
2. **`lasser/` is self-contained.** It is used by `git clone` → `mv <repo>/lasser .` → `import lasser` inside a Kaggle notebook. Use relative imports only, with no `sys.path` hacks, no repo-root paths and no install step.
3. **Don't change GEARS behaviour.** The GO graph, `GEARS_Model.forward`, loss, training loop and default hyperparameters are untouched. The graph enters through GEARS's public API or a thin subclass in `lasser/`. If that isn't possible, stop and explain before patching.
4. **No data leakage.** Features, positives, the A0 graph and refinement use only control cells plus training-perturbation cells of the current split seed. Validation perturbations are used only for model selection. Test perturbations are never read before evaluation.
5. **Memory (Kaggle T4, 16 GB).** Never keep a dense N×N tensor in autograd, and build edge lists directly. No per-sample Python loops over a batch. Pretraining and refinement should use < 4 GB GPU memory.

## Evaluation: `eval.py`

- `eval.py` defines the metrics to get from every run, baseline and graph learning alike. **Read it before writing evaluation code**, and call its functions rather than re-implementing them.
- **Don't modify `eval.py`.** If it needs inputs that GEARS doesn't produce directly (e.g. per-perturbation mean predictions, control means, DE gene indices, subgroup labels), write an adapter in `lasser/` that builds them, and say so.
- Report every metric `eval.py` returns, overall and per test subgroup (unseen_single, combo_seen0/1/2), in each run's `metrics/test_metrics.json` and the headline ones in `runs.csv`.
- GEARS's own `inference.py` metrics may be logged too, but `eval.py` is what results are compared on.

## Run tracking

Every run (flag on or off) writes `out_dir/<dataset>_seed<seed>_<gl|base>_<timestamp>/` with: `run.log`, `config.json`, `env.json`, `splits/` (the GEARS split pkl + `split.json` + split hash), `graphs/coexpress_graph.pkl` (**the graph actually used in training**, read back from the model), `metrics/` (`epoch_log.csv`, `test_metrics.json`, `predictions.pkl`) and `checkpoints/`. A row is appended to `out_dir/runs.csv`. Full details are in the task spec.

## Conventions

- Use Python `logging` (`logging.getLogger(__name__)`) for everything; no bare `print` in library code.
- All settings live in the `LasserConfig` / `GSRConfig` dataclasses. No hard-coded paths; defaults are Kaggle-friendly (`/kaggle/working/...`).
- Seed everything through one helper. Tests use deterministic settings.
- Dependencies are what GEARS already needs, plus `scikit-learn` if required. Ask before adding anything else.
- Keep functions small and typed. Shapes and node order follow GEARS's `node_map`.

## Working style

- **Plan before code.** Read the relevant GEARS files (`gears.py`, `model.py`, `utils.py`, `pertdata.py`, `inference.py`), the GSR code and `eval.py`, report the integration points, and wait for an OK.
- Work in small steps: config + tracking + flag-off parity test first, then data/views, pretraining, refinement and the embedding-init hook.
- Run the tests after each step and show their output. Don't claim something works without running it.
- When the real code disagrees with the spec or this file, follow the code and say so.
- Don't run long training jobs unless asked; smoke tests use 1 epoch on a small subset.
