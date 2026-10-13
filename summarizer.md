# LASSER handoff summary

Read this at the start of a new session. It records what has been built, what was decided, and what comes next. Details are in `LASSER.md` (usage, config fields, run-folder layout), `gsr_coexpression_plan.md` (method) and `claude_code_prompt_gsr_gears.md` (task spec).

## Status (2026-10-13: experiments finished; paper drafted)

- **Decision:** the project is written up as an **analysis / negative-result paper**. The draft is in `paper/` (`main.tex`, `refs.bib`, `figures/`, `README.md` with venues and the to-do list). The user asked for no further code runs.
- **Core findings:**
  1. GEARS's co-expression prior is nearly empty (87–89% of genes have no neighbour) and, because of the batching defect, reaches only the first cell of each batch.
  2. With the fix, learned, original Pearson, density-matched Pearson and random-rewired graphs perform alike, within run-to-run noise.
  3. The learned graph is informative (held-out co-expression about 18× random; 346 edges stable across splits forming cell-cycle, proteasome and nucleolar modules).
  4. **Mechanism:** the co-expression GNN smooths a free learnable table (`emb_pos`) that is independent of the cell's expression, so any graph only re-parameterises per-gene biases. In GEARS, `x` enters only as the final residual. The user's earlier IDGL and Poincaré attempts also gave no gains, which is consistent with this.
- **Open:** see the to-do list in `paper/README.md` (authors, earlier-attempt details, citation checks, the Figure 1 title overlap, and optional strengthening runs).

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

**Run 02 (2026-10-06, `notebooks/run-lasser_02.ipynb`, commit `c7be65c`).**

Passed:
- `test_01_parity::test_graph_and_init_identical` and `::test_first_training_steps_identical`: flag-off lasser is identical to the vendored GEARS on graph, GO graph, initial state and the first 5 losses.
- Everything up to the end of GEARS training, in every run.

Failed, now fixed:

1. `ValueError: could not convert string to float: '0.0099.'`, in every run (tests, smoke, and the notebook baseline after 86 minutes of training).
   - Cause: GEARS prints `Validation Overall MSE: 0.0381. `, and the number pattern in `tracking.py` also captured the trailing full stop.
   - Fix: a strict number pattern. Also, (a) the GEARS checkpoint is now saved straight after `train()`, before anything else; (b) the raw printout is saved to `metrics/gears_output.txt`; (c) a parse failure is logged and no longer fails the run.
   - New `tests/test_05_parsing.py` checks the parser on real run-02 lines without needing data.
2. Leakage test: "view E changed".
   - Cause: the test's altered matrix (`sp.diags(...) @ X[perm]`) stores entries in a different order. Sparse SVD and k-means sum in storage order, so results differed in the last bits.
   - Fix: `data.rows_csr` now returns a canonical CSR (sorted indices, no duplicates, no stored zeros). `GSR_CACHE_VERSION` was bumped to 2.

Also added: the device is logged before anything runs (`device: cuda:0 (Tesla T4, ...)` or `cpu`, with a warning if CPU is used while CUDA is available). It is also logged at the start of GSR and of training, and recorded in `runs.csv` (`device`) and `env.json`.

What run 02 measured (smoke test: full Norman, T4):

- **A0 confirmed:** 6,364 edges, 4,699 of them self-loops; **4,419 of 5,045 genes have no co-expression neighbour**.
- GEARS training: about 6 min per epoch on the full data (20 epochs = 5,160 s); peak CUDA memory 0.87 GB.
- GSR with default config: about 25 s in total; peak CUDA memory 1.6 GB, within the 4 GB budget.
  - Positives: 564 pairs.
  - Link AUC: 0.988 (raw features 0.970).
  - Refinement: 20 A0 edges removed and 38,848 directed edges added; every gene now has a neighbour.
  - H2 hit rate: added edges 0.036, degree-matched random 0.002, A0 0.58. So the check passes, but the added edges are weakly co-expressed.
- **Issue to discuss (not fixed):** in the S view, every gene that is isolated in A0 (88% of genes) gets the same "missing" input. They all end up with identical S embeddings, so the S-view link AUC is about 0.23 (below chance). In refinement, the S term is constant for pairs of such genes. The same applies to E and R missing genes. Options: score only views where both genes are non-missing, renormalising β per pair; or drop S, since A0 is nearly empty.

**Next:** push, re-run notebook cells 1–3 (and 3b), and send the output. Decide what to do about the missing-gene / S-view issue.

**Study session 1 (2026-10-12/13, `notebooks/lasser-study_after_run.ipynb`, `lasser_study_results.zip`, commit `f72d2b6`).**

- **Only 7/38 runs finished:** GEARS splits 2–5 and LASSER splits 1–3.
  - Worker 0 (GPU 0) was killed at the start (exit -9, out of RAM: two workers each load Norman plus GEARS's per-epoch prediction arrays, about 15 GB each).
  - Its claim on `GEARS_seed1` was never released, so one GPU did all the work.
  - Fixed: `launch` now starts only as many workers as RAM allows (`max_parallel_workers`: RAM ÷ 15 GB − 1, which is **1 on Kaggle**); loads data one worker at a time (lock); detects dead workers, frees their claims and prints the exit code and the end of their log.
  - `test_06_study.py` failed only because of its own fake data (fewer than 3 connected genes); fixed.
- **Results so far: no detectable effect of the learned graph.**
  - Paired splits 2 and 3: Pearson Δ GEARS 0.524 / 0.547 vs LASSER 0.523 / 0.551.
  - Per-perturbation paired (226 pairs): mean improvement +0.002, 95% CI [−0.003, 0.006]; 54% of perturbations improved.
  - **Replicate check:** LASSER on split 1 scored 0.574 in run 04 and 0.491 now. Same split (hash `f93e7d59…`), nearly the same graph (45,192 vs 45,107 edges; the CSR canonicalisation changed it slightly). So run-to-run noise on one split is about 0.08, larger than run 04's apparent +0.04 gain. Likely sources: GPU nondeterminism (`deterministic=False`) and GEARS keeping the epoch with the best (noisy) validation DE-MSE.
  - This is expected from the batching quirk: the co-expression graph reaches only the first cell of each batch.
- **The learned edges carry real but weak signal:**
  - H2 hit rate 0.033 vs 0.001 for degree-matched random (A0: 0.605).
  - Between splits, new-edge sets have Jaccard 0.027; 333 pairs recur in at least 3 of 3 splits.
  - Top consensus edges form a coherent cell-cycle module (CDCA8, CENPA, AURKA, BIRC5, CKS2), which is useful for the "reasoning" part.
- **Decision (2026-10-13): option 1.** Add an opt-in batching fix plus training-seed replicates, and run a new study that fits a 10-hour session.
  - Implemented: `LasserConfig.fix_coexpress_batching` (`lasser/batching.py`, `BatchedSGConv`) and `LasserConfig.train_seed` (`cfg.rng_seed`). Both are left out of `config_hash` at their defaults, so old runs keep their hashes.
  - `study.FIX_PLAN` with arms GEARS-fix, LASSER-fix, GEARS-kNN8-fix and LASSER-random-fix, 10 epochs; the analysis handles replicates (`rep`, `unit="split"|"run"`).
  - `notebooks/lasser_fix_study.ipynb`; `tests/test_07_fix.py`.
  - **Next:** the user runs `lasser_fix_study.ipynb` (10 h) and sends back `lasser_fix_study_results.zip` plus the notebook.

**Study 2 / option-1 fix study (2026-10-13, `notebooks/lasser_fix_study_after_run.ipynb`, `lasser_fix_study_results.zip`).** 10 runs on one T4, 10 epochs each (42–50 min per run).
- Pearson Δ, mean ± sd over splits 1–3: GEARS-fix 0.540 ± 0.023, LASSER-fix 0.527 ± 0.008, LASSER-random-fix 0.526 ± 0.029; GEARS-kNN8-fix 0.509 (split 1 only). Additive 0.407, no-perturbation 0.
- Paired: LASSER-fix − GEARS-fix = −0.013, 95% CI [−0.052, 0.027], 0/3 splits better; per perturbation −0.012 [−0.029, 0.001], 43% improved. LASSER-fix − random = +0.002 [−0.081, 0.084]. MSE differences are ≤ 0.002.
- Gene groups: no gain on genes isolated in A0 (−0.015 ± 0.021).
- Graph: H2 hit rate of new edges 0.034–0.037 vs 0.001–0.002 random vs 0.58–0.66 A0; link AUC 0.973–0.997 (raw features 0.956–0.979); 346 consensus pairs, 44.5% of them also in the kNN8 graph.
- Not run: tiers 3 (partly), 4, 5 and 6, i.e. kNN8 on splits 2–3, unfixed GEARS at 10 epochs, replicates and splits 4–5.

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

1. Finish the paper (`paper/README.md` to-do list): authors, details of the earlier attempts, verifying citations, re-rendering Figure 1, choosing the venue.
2. Optional strengthening runs, only if the user wants them: splits 4–5, training-seed replicates, a second dataset, and the self-loops-only control that directly tests the re-parameterisation argument.
