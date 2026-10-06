# LASSER handoff summary

Read this at the start of a new session. It records what has been built, what was decided, and what comes next. Details are in `LASSER.md` (usage, config fields, run-folder layout), `gsr_coexpression_plan.md` (method) and `claude_code_prompt_gsr_gears.md` (task spec).

## Status (2026-10-06)

- The `lasser/` package, `tests/`, `notebooks/run_lasser.ipynb` and `LASSER.md` are written and committed in a local git repo (commits on `master`, no remote yet; GEARS vendored into `lasser/gears/` on 2026-10-06).
- **Nothing has been executed yet.** The only check was syntax parsing. All tests run on Kaggle; the user sends back the notebook output or `test_output.txt` after each run.
- **Next step:** the user pushes to GitHub, sets `REPO_URL` in notebook cell 2 and runs cells 1–3 (tests). Then we fix whatever fails, starting with `test_01_parity.py`, the gate for everything else.

## Kaggle run log

**Run 01 (2026-10-06, `notebooks/run_lasser_01.ipynb`, Kaggle Python 3.13, commit `ab75303`).** All tests except the `eval.py` hash check failed or errored. There were two root causes, both now fixed:

1. `'Series' object has no attribute 'nonzero'`: the vendored `GEARS.__init__` indexed the sparse `adata.X` with a pandas boolean Series, which the newer scipy rejects. The mask is now wrapped in `np.asarray(...)` (listed in VENDORED.md). The other `.X[...]` uses in the vendored code index with numpy arrays, and `adata[...]` indexing is handled by AnnData, so they're fine.
2. `module 'torch' has no attribute 'flatnonzero'`: my bug in `refine.py` (and `tests/test_02_graphs.py`). Replaced with `torch.nonzero(..., as_tuple=True)[0]`. Every other `torch.*` name used was checked and exists.

Also: `float(loss)` became `loss.item()` in pretraining (it caused a requires_grad warning).

What run 01 showed before the failure. These come from the leakage test's GSR run on CPU with the tiny test config:

- 49,849 training cells (38,824 in the A0 subset), 138 training perturbations.
- Missing genes per view: E 234, R 271, **S 4419 of 5045**.
- The pipeline ran through data, views, positives and pretraining without errors. Held-out link AUC was 0.97 against 0.83 for the raw features.

The S count means about 4.4k genes have no non-self edge in GEARS's A0 (|r| > 0.4 is rare in single cells), so A0 is mostly self-loops. Confirm this on the next run from `graph_isolated_genes` in `runs.csv`. If it holds, the `d_min` top-up adds edges for most genes, which matters for interpreting results.

**Next:** re-run notebook cells 1–3 (and 3b) with the fixes and send the output.

## Decisions made with the user

1. **GEARS stays exactly as is**, including a known quirk. `GEARS_Model.forward` (`GEARS/gears/model.py:137-141`) runs the co-expression SGConv over B×N nodes (batch × genes), but `G_coexpress` only covers node ids 0..N-1. So only the first cell of each batch gets co-expression messages; the others get a self-loop only. The user chose to keep this rather than add a fix flag, and it is documented in LASSER.md. A learned graph therefore has limited effect by construction. Keep this in mind when reading results.
2. **Kaggle only.** Don't install packages, create a venv, download data or run tests or training locally. Static checks (`ast.parse`) are fine.
3. **`lasser` is fully independent of any external GEARS.** GEARS is vendored as `lasser/gears/` (copied from `GEARS/gears`, commit `f374e43`, v0.1.2), and `lasser` never imports a top-level `gears` (pip `cell-gears` isn't installed or used). The `GEARS/` and `WSDM23-GSR/` folders are local references only, are git-ignored (patterns anchored as `/GEARS/` and `/WSDM23-GSR/`, because Windows git is case-insensitive and an unanchored `GEARS/` also hid `lasser/gears/`) and are never pushed. The GEARS MIT license is copied to `lasser/gears/LICENSE`. The vendored copy is upstream except for the fixes listed in `lasser/gears/VENDORED.md`, each marked `# LASSER:`:
   - `dcor` is imported lazily;
   - `torch_geometric.loader.DataLoader` replaces `torch_geometric.data.DataLoader`;
   - two pandas-compatibility rewrites in `pertdata.py` that give the same results (the `prepare_split` groupby, and `.iloc[0]`).
4. **Dependencies:**
   - Runtime: torch, torch_geometric, scanpy (+anndata), numpy, pandas, scipy, scikit-learn, networkx, tqdm, requests (`requirements.txt`).
   - Kaggle needs only `pip install torch_geometric scanpy pytest`.
   - Optional GEARS extras (dcor, wandb, seaborn, matplotlib) are imported only inside functions lasser never calls.
   - An `ast`-based audit (2026-10-06) found every relative import resolving and no top-level `gears` import.

## Package layout (`lasser/`)

| File | Role |
|---|---|
| `gears/` | **Vendored GEARS** (`GEARS`, `PertData`, `model`, `utils`, `inference`, `data_utils`, `version`) + `VENDORED.md` listing every change. Don't edit except to fix import or compatibility problems (mark with `# LASSER:`) |
| `__init__.py` | Exports `LasserConfig`, `GSRConfig`, `run_experiment`, `prepare`, `load_pert_data`, `load_run`. Must not import GSR modules |
| `config.py` | `LasserConfig` (GEARS defaults, Kaggle paths, flags) and nested `GSRConfig` (plan §11 defaults); `config_hash()`, `gsr_cache_key()`, `from_dict` |
| `utils.py` | `seed_everything` (the only seeding helper), `phase` (time + peak CUDA memory), `capture_stderr` (tees GEARS's stderr `print_sys`), `graph_stats` |
| `run.py` | `run_experiment(cfg, pert_data=None)`, `prepare()` (everything up to `model_initialize`), `subsampled()` (tests only), `init_gene_embedding()` (the `gsr_init_gene_emb` hook), `used_graph()` |
| `tracking.py` | `RunTracker` (run folder, logging, config/env/split/graph/metrics/checkpoints, `runs.csv`), `parse_gears_output` (stderr → epoch table), `load_run` |
| `evaluation.py` | Adapter that builds `eval.py`'s inputs from GEARS and calls it, overall and per subgroup |
| `_eval.py` | Byte-for-byte copy of the root `eval.py` (a test checks the hash). Never edit; recopy if `eval.py` changes |
| `graph.py` | `StaticCoexpressProvider` (GEARS's own A0 build), `run_gsr_pipeline` (pure function), `GSRCoexpressProvider` (with disk cache) |
| `data.py` | Training-only cells, halves H1/H2, metacells, `assert_training_only` |
| `views.py` | Views E (metacell PCA), R (response Δ PCA), S (plain-torch DeepWalk on A0) and optional X; missing-gene flags |
| `pretrain.py` | Bootstrapped H1 positives, random + hard negatives, per-epoch 70/30 edge split, `GSRModel` (MLP → 2×SGConv → linear + LayerNorm), MoCo keys, intra/inter InfoNCE, VICReg, link-AUC early stopping |
| `refine.py` | Chunked `no_grad` scoring, per-gene remove r⁻ / add m⁺, `d_min` top-up, degree cap, quantile-mapped weights × γ |
| `checks.py` | Plan §6.1: H2 hit rate vs degree-matched random, coverage, Jaccard to A0, collapse metrics |

Only `run.py` imports `graph.py`, and only when `graph_learning=True`. That is what keeps the flag-off run free of GSR imports, which test 2 checks.

## Key mechanics

- **The order of calls is what parity depends on.**
  1. `import lasser.gears` (the vendored GEARS calls `torch.manual_seed(0)` when imported).
  2. `seed_everything(seed)`.
  3. `PertData.load`, then `prepare_split`, then `get_dataloader(32, 128)`.
  4. Graph-learning runs only: build the GSR graph.
  5. `seed_everything(seed)` again.
  6. `GEARS(pert_data, device)`, then `model_initialize(...)`, then `train(epochs, lr, wd)`.

  `tests/test_01_parity.py::reference_gears` repeats this order by calling `lasser.gears` directly, with no other lasser code.
- **Flag off:** lasser passes no graph, so GEARS builds A0 itself. **Flag on:** the graph goes in through `model_initialize(G_coexpress=..., G_coexpress_weight=...)`.
- **The saved graph is read back from the model** (`gears.model.G_coexpress` and `.G_coexpress_weight`).
- **A0 is built from control cells plus training *singles* only** (`lasser/gears/utils.py:294`). Positives are built from the same cells on half H1; views E and R use all training cells.
- **Graph format:** edges run source → target, where a target's sources are its neighbours. Kept A0 edges come first, in GEARS's order and direction; new edges are appended in both directions. Degree means in-degree without self-loops.
- **Windows fix:** `load_pert_data` resets `pert_data.dataset_name` from `dataset_path`, because GEARS derives it with `split('/')`.
- **GSR cache:** `cache_dir/gsr/<dataset>_<split>_seed<s>_<tgs>_<hash>/`. The hash covers the GSR config and the GEARS co-expression parameters. A cache hit is checked against the current A0.
- **`epoch_log.csv` is parsed from GEARS's stderr.** GEARS prints a loss only every 50 steps, so the train loss column is the mean of those printed losses.

## Deviations from the spec or plan (already told to the user)

- S view: plain-torch DeepWalk instead of PyG `Node2Vec`, which needs pyg-lib or torch-cluster.
- Not implemented: the STRING/TRRUST external-agreement check (no data files).
- A graph that fails the §6.1 checks is still used; the failure is logged and recorded as `gsr_checks_pass` in `runs.csv`.
- Defaults chosen where the plan gives a grid:
  - r⁻ 0.1, m⁺ 5, γ 0.5.
  - Pretraining: 300 steps, patience 50, `batch_edges` 2048.
  - 500 metacells; 20 bootstrap resamples, keep ≥ 70%.
- Genes with zero training variance have no signal in any view, so their `d_min` top-up edges are essentially arbitrary.
- `uncertainty=True` is rejected, because `eval.py` expects point predictions.

## Tests (`tests/`, run from `/kaggle/working` so `import lasser` finds the moved folder)

| File | Covers |
|---|---|
| `conftest.py` | Env vars `LASSER_DATA_DIR`, `LASSER_TEST_DIR`, `LASSER_SUBSAMPLE` (4), `LASSER_SMOKE`; session fixtures `pert_data` and `runs` (tiny flag-off and flag-on runs); `make_cfg`, `tiny_gsr` |
| `test_00_packaging.py` | Tests 2 and 3: `lasser/` copied alone into a temp dir and run in a subprocess; `sys.modules` checks that no GSR module and no top-level `gears` is imported |
| `test_01_parity.py` | Test 1: lasser flag-off vs the vendored GEARS called directly; exact equality on CPU (graph, GO graph, initial state, first 5 losses, best model after 1 epoch) |
| `test_02_graphs.py` | Tests 4 and 5: saved graph = used graph; shape contract and `d_min` |
| `test_03_leakage.py` | Test 6: permuting and rescaling val/test cells leaves A0, views, positives, embeddings and the refined graph identical (CPU) |
| `test_04_cache_tracking.py` | Tests 7 and 8 plus the `eval.py` hash check |
| `test_09_smoke.py` | Test 9: full Norman, 1 epoch, flag off and on, memory budget (`LASSER_SMOKE=1`) |

## Open items / next steps

1. Get the Kaggle test output, fix failures, and repeat until tests 1–8 pass, then test 9.
2. Record measured time and memory (from `metrics/timings.json` and `runs.csv`) here and in LASSER.md.
3. Then follow plan gates G1–G5: check link AUC and the H2 checks on seed 1, run the refinement grid (m⁺ × r⁻), select with 10-epoch runs on validation DE-MSE, confirm on seeds 1–5, then the ablations F3–F6.
