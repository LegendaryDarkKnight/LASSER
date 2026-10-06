# Prompt for Claude Code: add GSR-style co-expression graph learning to GEARS behind a flag

> Paste everything below the line into Claude Code, in a workspace that contains the GEARS repo (`snap-stanford/GEARS`) and the GSR repo (`andyjzhao/WSDM23-GSR`). Put `gsr_coexpression_plan.md` in the workspace too, since the prompt refers to it.

---

## Context

I am extending **GEARS** (perturbation-response prediction, Norman dataset) so that its **gene co-expression graph** can be learned with a **GSR-style self-supervised method** (WSDM 2023: pretrain → refine graph once → finetune) instead of being built once from Pearson correlation. The design is in `gsr_coexpression_plan.md`. Read it first; it is the spec.

The workspace has:
- the GEARS source (the base we build on);
- the GSR reference code (for understanding the method only; **do not add it as a dependency**, since it is DGL-based and GEARS uses PyTorch Geometric).

Scope: **co-expression graph only.** The GO perturbation graph, the GEARS model architecture, the loss, the decoder and the metrics stay exactly as in GEARS.

Hardware target: a single Kaggle T4 (16 GB GPU, ~30 GB RAM).

## How I will run it (this decides the packaging)

I run everything from a **Kaggle/Colab notebook**, like this:

```python
# cell 1: dependencies
!pip install cell-gears==<pinned version> torch_geometric ...   # whatever LASSER.md lists
# cell 2: get the code
!git clone https://github.com/<me>/<lasser-repo>.git
!mv <lasser-repo>/lasser ./lasser
# cell 3: use it
from lasser import LasserConfig, run_experiment
cfg = LasserConfig(dataset="norman", seed=1, graph_learning=True, out_dir="/kaggle/working/runs")
results = run_experiment(cfg)
```

So:
- **`lasser/` must be a self-contained folder** that works when copied into the notebook's working directory and imported as `import lasser`. Use only relative imports inside it. No `setup.py` install, no `sys.path` hacks, no paths computed from the repo root.
- **GEARS comes from pip** (`cell-gears`, pinned to the version you read). Prefer **zero edits to GEARS itself**: integrate by passing the graph into GEARS's public API (e.g. `model_initialize(G_coexpress=..., G_coexpress_weight=...)` if that exists in the pinned version) or by a thin subclass of `GEARS` inside `lasser/`. If neither is possible, tell me why before patching anything, and keep any patch inside `lasser/` (no edits to site-packages, no monkey-patching of unrelated code).
- Everything needed for a run is reachable from **one config object and one function call** in a notebook cell. Also provide `notebooks/run_lasser.ipynb` (the cells above plus a baseline run, a graph-learning run and a results-loading cell). A CLI wrapper is optional.
- Every path (data dir, output dir, cache dir) is a config field with Kaggle-friendly defaults (`/kaggle/working/...`). Nothing assumes Colab or Google Drive.

## The core requirement: one flag, and GEARS stays GEARS when it is off

`graph_learning` is a config field, default **off**.

- **Off:** the code path is **exactly upstream GEARS**: same graph, same model, same initialisation, same random number stream, same results. Nothing GSR-related is imported, computed or cached. Logging and saving (below) still happen, so baseline runs are tracked the same way.
- **On:** the co-expression graph is replaced by the GSR-refined graph. Everything downstream is unchanged GEARS.
- A second flag, `gsr_init_gene_emb` (default off, only valid with `graph_learning` on), copies a linear projection of the pretrained expression-view gene embeddings into GEARS's gene embedding table, from outside the model.

Do not touch the GO graph, `GEARS_Model.forward`, the loss, the training loop, metrics or GEARS's default hyperparameters. If a change seems necessary, stop and ask.

## What `lasser/` must contain

Follow `gsr_coexpression_plan.md` §§2–5. Port the GSR ideas to PyTorch + PyG; don't copy DGL code.

- `config.py`: a `LasserConfig` dataclass with run settings (dataset, data dir, split, seed, train_gene_set_size, GEARS hyperparameters, out_dir, cache_dir, log level, flags) and a nested `GSRConfig` with every GSR hyperparameter at the plan's §11 defaults: views, embedding dim 64, hidden 256, τ 0.2, K 256 negatives with 25% hard, α 0.75, VICReg weight 0.1, momentum 0.99, β view weights, r⁻, m⁺, d_min 5, γ, steps, lr.
- `data.py`: from a `PertData` object and its split, build the **training-only** cell mask (control + training-perturbation cells), halves H1/H2, and metacells.
- `views.py`: the expression view E (metacell PCA), response view R (mean Δ over training perturbations, PCA) and structure view S (PyG `Node2Vec` on the GEARS A0 graph). Optional view X (pretrained gene embeddings from a file path) stays off by default. Handle zero-variance genes with a "missing" token, not NaN or zeros that look like real data.
- `pretrain.py`: positives from H1 (bootstrapped), negatives (random + hard, excluding A0 neighbours), a 70/30 edge split each epoch so target edges are never in the message-passing graph, per-view full-batch encoders (MLP → 2× SGConv → linear, no output ReLU), momentum key encoders, intra-view and inter-view InfoNCE, VICReg variance guard, early stopping on held-out link AUC.
- `refine.py`: chunked cosine scoring under `torch.no_grad()` (never keep an N×N tensor in autograd), per-gene remove r⁻ / add m⁺ edits, min-degree top-up, symmetrise, quantile-matched weights scaled by γ for new edges.
- `checks.py`: the plan §6.1 label-free checks (H2 hit rate vs degree-matched random, isolated genes, degree stats, Jaccard to A0, embedding collapse metrics). Return a dict.
- `graph.py`: `StaticCoexpressProvider` (wraps GEARS's own co-expression build, unchanged) and `GSRCoexpressProvider` (runs or loads from cache pretraining + refinement), both returning `edge_index`, `edge_weight` in GEARS's node order and dtype. Cache keyed on (dataset, split, seed, train_gene_set_size, config hash).
- `tracking.py`: the logging and saving below.
- `run.py`: `run_experiment(cfg)` that sets seeds, loads data, builds the graph, trains GEARS, evaluates, saves everything, and returns a results dict.

## Logging and saving (every run, flag on or off)

Each run writes to its own folder: `out_dir/<dataset>_seed<seed>_<gl|base>_<timestamp>/`, with a short `run_id`. Inside:

- **`run.log`**: Python `logging` to both the notebook output and this file. Use one logger per module (`logging.getLogger(__name__)`); no bare `print`. The level is a config field. Log phases with timings and `torch.cuda.max_memory_allocated()`.
- **`config.json`**: the full resolved config, plus the config hash.
- **`env.json`**: Python, torch, PyG, GEARS versions, GPU name, the git commit of the cloned repo (if available), and the start and end time.
- **`splits/`**: the split exactly as used: the GEARS split pkl copied as-is, plus a readable `split.json` with the train/val/test conditions and the test subgroups (unseen_single, combo_seen0/1/2), plus a `split_hash`. Runs on the same seed must be checkable as using the same split.
- **`graphs/coexpress_graph.pkl`**: the co-expression graph **actually used in training**. Take it from the tensors handed to the model (read back after `model_initialize`), not from an intermediate. Save it as a dict with: `edge_index` (numpy), `edge_weight` (numpy), `gene_list`, `node_map`, `source` (`"gears_static"` or `"gsr_refined"`), `seed`, `split_hash`, `config_hash` and basic stats (num edges, degree min/median/max, isolated genes). For GSR runs also save `graphs/a0_graph.pkl` (the GEARS static graph) in the same format, `graphs/edits.csv` (`source,target,action,score`) and `graphs/checks.json` (the §6.1 checks).
- **`metrics/`**: per-epoch train and val losses and metrics as `epoch_log.csv`, final test metrics per subgroup as `test_metrics.json`, and per-perturbation predictions and truth as `predictions.pkl` (mean predicted and true expression per test perturbation, with the gene order).
- **`checkpoints/`**: the best model state dict (and, for GSR runs, the pretrained encoders).
- **A global `out_dir/runs.csv`** with one row appended per run: run_id, timestamp, seed, flags, config hash, split hash, graph stats and the headline test metrics. This is the table I use to compare runs.

Provide `lasser.tracking.load_run(path)` that loads a run folder back into a dict (config, split, graph, metrics), so a notebook can inspect any past run.

## Leakage rules (hard requirements)

- Features, positives, A0 and refinement use **only training cells and training perturbations** of the given split seed. Validation perturbations are used only for model selection. Test perturbations are never read.
- Add an assertion in `data.py` that no validation or test condition is in the training mask.

## Memory and speed rules (from our past failures)

- Never materialise a dense N×N tensor inside autograd. Build edge lists directly; no dense-adjacency-to-edge-list conversion.
- No per-sample Python loops over the batch.
- Pretraining + refinement should use < 4 GB GPU and take minutes per seed on a T4.

## Tests (write them, run them, show me the output)

1. **Flag-off parity (most important).** With `graph_learning=False`, fixed seeds and deterministic settings, `run_experiment` and plain GEARS run directly from pip must produce identical `G_coexpress` / `G_coexpress_weight`, identical initial `state_dict`, and identical losses for the first N training steps on a small subset. Use exact equality, not tolerance, unless you can explain a specific non-deterministic op.
2. **Flag-off import isolation.** With the flag off, nothing under `lasser.pretrain`/`lasser.refine` is imported (check `sys.modules`).
3. **Notebook packaging.** Copy `lasser/` alone into a fresh temp directory, `cd` there and run `import lasser` and a tiny `run_experiment` in a subprocess. It must work with nothing else from the repo present.
4. **Saved graph = used graph.** The `coexpress_graph.pkl` loaded back equals the model's graph tensors exactly, for both flag settings.
5. **Flag-on shape contract.** Valid indices, GEARS node order, non-negative float weights, every gene with degree ≥ d_min.
6. **Leakage.** Perturb or shuffle only validation and test cells; the features, positives and refined graph must stay identical.
7. **Cache.** A second run with the same key loads from cache and returns identical tensors.
8. **Tracking.** After a run, every file listed above exists, `runs.csv` has a new row, and `load_run` reads it back.
9. **Smoke run.** One end-to-end run on Norman, seed 1, 1 epoch, flag on and flag off, within a T4-sized memory budget.

## Deliverables

- The `lasser/` package, `notebooks/run_lasser.ipynb` and the tests.
- A short `LASSER.md`: the notebook install-clone-move-run steps, every config field, the run-folder layout, and how to load a past run.
- At the end, a summary of: how GEARS is integrated (and whether any GEARS code had to change, with the reason), test results, measured memory and time, and anything in the plan you could not implement or changed (with the reason).

## Working style

- **Plan first.** Read `gears/gears.py` (`GEARS.model_initialize`, `train`, `predict`), `gears/model.py`, `gears/utils.py` (`get_similarity_network`, `GeneSimNetwork`), `gears/pertdata.py` and `gears/inference.py` in the pinned version, then the GSR code. Report the exact integration points and file list **before writing code**, and wait for my OK.
- Then implement in small commits: config + tracking + flag-off parity test first, then data/views, then pretraining, then refinement, then the init hook.
- If GEARS's actual code differs from what this prompt assumes (for example, if `model_initialize` doesn't accept `G_coexpress`), follow the code and tell me.
- Don't change GEARS default hyperparameters, and don't add dependencies beyond what GEARS already uses plus `scikit-learn` if needed for PCA/k-means.
