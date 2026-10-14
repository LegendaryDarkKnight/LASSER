# LASSER: running GEARS with a learned co-expression graph

`lasser/` is a self-contained package. GEARS is **vendored inside it** as `lasser/gears/` (upstream `snap-stanford/GEARS` commit `f374e43`, v0.1.2, with import and pandas-compatibility fixes only; see `lasser/gears/VENDORED.md`), so neither the pip `cell-gears` package nor a GEARS checkout is needed. One flag, `graph_learning`, swaps GEARS's Pearson co-expression graph for a GSR-style refined graph (pretrain, then refine once, then train GEARS). The design is in `gsr_coexpression_plan.md`.

## Running on Kaggle

GPU (T4), **Internet on**. `notebooks/run_lasser.ipynb` contains these cells:

```python
!pip install -q torch_geometric scanpy pytest     # everything else is preinstalled on Kaggle
!git clone https://github.com/<me>/<lasser-repo>.git /kaggle/working/lasser-repo
!mv /kaggle/working/lasser-repo/lasser /kaggle/working/lasser

# tests 1-8 (tiny subsets), before any real training
!cd /kaggle/working && python -m pytest lasser-repo/tests -m "not smoke" -p no:cacheprovider -rA -s

from lasser import LasserConfig, GSRConfig, load_pert_data, run_experiment, load_run
cfg = LasserConfig(dataset="norman", seed=1, graph_learning=True, out_dir="/kaggle/working/runs")
pert_data = load_pert_data(cfg)              # optional: load once, reuse across runs
results = run_experiment(cfg, pert_data)     # or run_experiment(cfg)
```

`lasser/` only uses relative imports and has no install step, so copying the folder next to the notebook is enough. The first run downloads Norman and the GO graph into `data_dir`. GEARS's processed cell graphs for Norman need about 4 GB of RAM.

## Dependencies

| Package | Used by | Notes |
|---|---|---|
| torch | everything | preinstalled on Kaggle |
| torch_geometric | GEARS model (`SGConv`), cell graphs and loaders, GSR encoders | `pip install torch_geometric` (pure Python; no pyg-lib or torch-cluster needed) |
| scanpy (+ anndata) | GEARS data loading (`read_h5ad`) and DE ranking | `pip install scanpy` |
| numpy, pandas, scipy, scikit-learn, networkx, tqdm, requests | GEARS and lasser | preinstalled on Kaggle |
| dcor, wandb, seaborn, matplotlib | optional GEARS extras (GI analysis, wandb tracking, plots) | imported only inside those functions; not needed for lasser |

The full list is in `requirements.txt`. `lasser` never imports a top-level `gears` package; a test checks this.

## How it fits into GEARS (GEARS's behaviour is not changed)

| Flag | What happens |
|---|---|
| `graph_learning=False` | Plain GEARS: `model_initialize()` builds its own co-expression graph. lasser adds seeding, logging and saving around it. |
| `graph_learning=True` | The GSR graph is built, then passed through GEARS's public `model_initialize(G_coexpress=..., G_coexpress_weight=...)`. The GO graph, model, loss, training loop and hyperparameters are unchanged. |
| `gsr_init_gene_emb=True` | After `model_initialize`, a linear projection of the pretrained E-view embedding is copied into `model.gene_emb` (and `best_model.gene_emb`). |

Seeding: `seed_everything(seed)` runs at the start and again right before `GEARS(...)`, so GEARS's initialisation and batch order are the same with the flag on or off. The parity test reproduces this order by calling the vendored GEARS (`lasser.gears`) directly (see `lasser/run.py`).

### Behaviour inherited from GEARS

- **The co-expression graph only reaches the first cell of each batch.** `GEARS_Model.forward` runs the co-expression `SGConv` over `batch_size × num_genes` nodes, but `G_coexpress` only covers node ids `0..num_genes-1`. Only the first cell in a batch receives messages over the graph; the other cells get a self-loop only. This holds in training and evaluation. By decision, GEARS is kept as is, so a learned graph can only have a limited effect.
- **A0 uses only control cells and training singles.** GEARS builds its co-expression graph (A0) from control cells plus training *singles* (combos are excluded). Positives are built from the same population on half H1. Views E and R use all training cells, including combos.
- **Graph format.** Edges run `source → target`, where a target's sources are its top-k Pearson neighbours, plus a weight-1 self-loop for every gene with nonzero variance. Genes with zero variance in training have no edges at all. The refined graph keeps the kept A0 edges in GEARS's order and direction, and appends the new edges in both directions.
- **Windows paths.** `PertData.load` derives `dataset_name` with `split('/')`. `load_pert_data` resets it from `dataset_path`, which gives the same value on Linux.

## Config fields

### `LasserConfig`

| Field | Default | Meaning |
|---|---|---|
| `dataset` | `"norman"` | GEARS dataset name, or a folder with `perturb_processed.h5ad` |
| `data_dir` | `/kaggle/working/data` | GEARS data directory (downloads, splits, cached graphs) |
| `split` | `"simulation"` | GEARS split type |
| `seed` | `1` | Split seed; also the RNG seed for GEARS and GSR |
| `train_gene_set_size` | `0.75` | GEARS split parameter |
| `batch_size`, `test_batch_size` | `32`, `128` | GEARS dataloaders (GEARS demo values) |
| `epochs`, `lr`, `weight_decay` | `20`, `1e-3`, `5e-4` | `GEARS.train` |
| `hidden_size` … `direction_lambda` | GEARS defaults | `GEARS.model_initialize` arguments |
| `graph_learning` | `False` | Use the GSR-refined co-expression graph |
| `gsr_init_gene_emb` | `False` | Initialise GEARS's gene embedding from the E view (needs `graph_learning`) |
| `device` | `"cuda"` | Device for GEARS and GSR |
| `deterministic` | `False` | `torch.use_deterministic_algorithms(True, warn_only=True)` plus cuDNN flags (tests set this) |
| `out_dir` | `/kaggle/working/runs` | Run folders and `runs.csv` |
| `cache_dir` | `/kaggle/working/cache` | GSR graph cache |
| `log_level` | `"INFO"` | Level for the `lasser` loggers |
| `repo_dir` | `None` | Git checkout, so `env.json` records the commit |
| `run_tag` | `None` | Suffix for the run folder name |
| `subsample_cells_per_condition` | `None` | **Tests only**: keep the first n cells of each condition in the dataloaders |
| `uncertainty` | `False` | Must stay `False`: `eval.py` expects point predictions |
| `fix_coexpress_batching` | `False` | Opt-in fix: every cell in a batch gets co-expression messages (upstream GEARS: only the first). See below |
| `graph_mode` | `"gsr"` | Graph learner used when `graph_learning=True`: `"gsr"` (LASSER-GSR, unchanged default) or `"vae"` (see below) |
| `cell_signal` | `False` | Opt-in GEARS variant (needs `fix_coexpress_batching`): each cell's control expression feeds the co-expression GNN, so graph messages are cell-specific |
| `train_seed` | `None` | RNG seed for model initialisation and batch order; `None` = `seed`. For training-seed replicates on one split; the learned graph doesn't depend on it |

### `GSRConfig` (nested as `cfg.gsr`)

Defaults follow plan §11. Where the plan gives a grid, the default is one value from it.

| Group | Fields (default) |
|---|---|
| Views | `views` (`E`,`R`,`S`), `x_path` (None; view X needs a .csv/.tsv/.pkl/.pt file of gene → vector) |
| Data | `h1_frac` 0.5, `n_metacells` 500, `metacell_svd_dim` 50, `e_dim` 64, `r_dim` 32, `x_dim` 64 |
| Structure view | `s_dim` 64, `s_walk_length` 20, `s_walks_per_node` 10, `s_window` 5, `s_neg` 5, `s_epochs` 3, `s_batch_size` 16384, `s_lr` 0.01 |
| Positives | `pos_k` 20, `pos_threshold` 0.4, `pos_max_cells` 20000, `n_bootstrap` 20, `bootstrap_keep` 0.7, `response_pos_threshold` None (off) |
| Negatives | `n_neg` 256, `hard_neg_frac` 0.25, `hard_pool_size` 32, `hard_corr_max` 0.1, `hard_refresh_every` 50 |
| Encoders | `emb_dim` 64, `hidden_dim` 256, `n_conv_layers` 2, `decoder_hidden` 64, `momentum` 0.99, `feat_mask` 0.2, `edge_drop` 0.2, `msg_edge_frac` 0.7, `link_val_frac` 0.1 |
| Loss | `tau` 0.2, `alpha` 0.75, `vicreg_weight` 0.1, `vicreg_std_target` 1.0 |
| Schedule | `lr` 1e-3, `weight_decay` 0, `max_steps` 300, `batch_edges` 2048, `eval_every` 10, `patience` 50 |
| Refinement | `beta` {E 0.4, R 0.3, S 0.3}, `r_minus` 0.1, `m_plus` 5, `d_min` 5, `gamma` 0.5, `score_floor` 0, `cand_topk` 100, `chunk_size` 512 |
| Checks | `h2_hit_threshold` 0.2, `collapse_sample` 2000 |
| Control (not the method) | `edge_control` None; `"random"` rewires every new edge to a random source gene, keeping each gene's in-degree and the weight distribution (the `LASSER-random` arm) |

### The GSR pipeline (`graph_learning=True`)

1. **`data.py`** selects control cells plus training-perturbation cells, splits them into halves H1/H2 (stratified by condition), and builds metacells. An assertion stops any validation or test condition from entering.
2. **`views.py`** builds the per-gene views:
   - E: metacell profiles, then PCA.
   - R: mean Δ per training perturbation, then PCA.
   - S: DeepWalk on A0. This is plain torch, because PyG's `Node2Vec` needs `pyg-lib` or `torch-cluster`, which GEARS doesn't depend on.
   - X: optional external embeddings.

   Genes with no signal in a view get a zero row plus a `missing` flag; the encoder swaps in a learned token for them.
3. **`pretrain.py`** builds positives, negatives and the per-epoch 70/30 message/target split, then trains the per-view encoders (momentum keys, intra- and inter-view InfoNCE, VICReg guard) with early stopping on held-out link AUC.
   - Positives: H1 Pearson top-k pairs kept only if they reappear in ≥ 70% of bootstrap resamples.
   - Negatives: random plus 25% hard, never the query itself or an A0 or positive neighbour.
4. **`refine.py`** scores pairs in chunks under `no_grad` and keeps the top-100 candidates per gene. It removes the fraction r⁻ of each gene's A0 edges with the lowest scores, adds the top m⁺ new candidates per gene, tops every gene up to `d_min`, and caps new edges at in-degree k + m⁺. Kept edges keep their A0 weight; new edges get γ × a score quantile-mapped onto the A0 weights.
5. **`checks.py`** runs the plan §6.1 label-free checks:
   - H2 hit rate of added edges vs degree-matched random pairs and vs A0.
   - Isolated genes, degree stats and Gini, low-degree coverage.
   - Jaccard overlap with A0.
   - Collapse monitors (cosine std, effective rank).

   External agreement (STRING/TRRUST) isn't implemented, because it needs files that aren't available here. A failing check is logged as a warning and recorded in `checks.json` and `runs.csv` (`gsr_checks_pass`); the graph is still used.

Results are cached in `cache_dir/gsr/<dataset>_<split>_seed<seed>_<tgs>_<hash>/`. The hash covers dataset, split, seed, `train_gene_set_size`, the GEARS co-expression parameters and every `GSRConfig` field. A cache hit is checked against GEARS's A0.

## Run folder

```
out_dir/
  runs.csv                                   one row per run (flags, hashes, graph stats, headline metrics)
  norman_seed1_<gl|base>_<YYYYmmdd-HHMMSS>/
    run.log                                  lasser logs + GEARS's own printout
    config.json                              resolved config + config_hash (+ gsr_cache_key)
    env.json                                 python/torch/PyG/GEARS versions, GPU, git commit, start/end
    splits/  <GEARS split pkl(s) as-is>, split.json (train/val/test + subgroups + split_hash), split_hash.txt
    graphs/  coexpress_graph.pkl             graph read back from the model after model_initialize
             a0_graph.pkl, edits.csv, checks.json      (graph-learning runs)
    metrics/ epoch_log.csv                   per epoch: train/val MSE and DE-MSE, logged train loss, is_best
             test_metrics.json               eval.py: paper + LASSER metrics, overall and per subgroup; GEARS's own
             predictions.pkl                 mean pred/true per test perturbation, gene order, ctrl, counts
             timings.json                    seconds and peak CUDA memory per phase
             gears_output.txt                GEARS's raw training/test printout (source of epoch_log.csv)
    checkpoints/ gears_model/{config.pkl,model.pt}     GEARS format (GEARS.load_pretrained works)
                 gsr_encoders.pt             pretrained encoders, embeddings, pretraining history
```

The graph pickle is a dict with these keys: `edge_index`, `edge_weight` (numpy), `gene_list`, `node_map`, `source` (`gears_static` / `gsr_refined`), `seed`, `split_hash`, `config_hash`, and `stats` (edges, self-loops, in-degree min/median/mean/max/Gini, isolated genes).

The device (`cuda:0 (<GPU name>, <memory>)` or `cpu`) is logged before anything runs, again when GSR and training start, and stored in `runs.csv` (`device`) and `env.json`. The GEARS checkpoint is written straight after training, before metrics are parsed or computed.

`epoch_log.csv` is parsed from GEARS's printout, because its training loop reports nothing else. So the train loss column is the mean of the losses GEARS prints every 50 steps.

## VAE graph learning and the cell-signal variant

**Why.** In GEARS the co-expression GNN smooths a learnable table that is the same for every cell, so any graph only
re-parameterises per-gene biases (`paper/main.tex`, Section 5). This line of work changes both the graph learner and
where the graph acts.

**`graph_mode="vae"` (`lasser/vae.py`, settings in `VAEConfig`, nested as `cfg.vae`).**
- **Model:** a cell-level VAE with a linear decoder (`x ≈ W z + b`, Gaussian likelihood with a per-gene variance),
  trained on half H1 of the training cells. A small part of H1 is held out for early stopping.
- **Gene embeddings:** decoder loadings `W_g`.
- **Similarities** (all cosines of linear transforms of `W_g`, so no N×N matrix is formed):
  - `load`: cosine of loadings;
  - `denoise`: correlation of the VAE's reconstructed expression (`cos(L^T W_g, L^T W_h)`, with `cov(z) = L L^T`);
  - `resp`: correlation of decoded changes across training perturbations.
- **Score and graph:** the combined score is `sum beta_s * minmax(cos_s)`, and the graph is refined from A0 with the
  LASSER-GSR rules (`refine_graph`).
- **Edge confidence:** the share of `n_posterior_samples` posterior samples in which an edge stays in an endpoint's
  top `confidence_k`. New-edge weights are multiplied by `max(confidence, confidence_floor)`.
- **Control:** `vae.edge_control="random"` rewires the new edges, as for GSR.
- **Checks:** held-out H2 reconstruction R², active latent dimensions and edge confidence, in `checks.json["vae"]`
  and `runs.csv` (`vae_*`).
- **Caching:** under `cache_dir/vae/...`, keyed by `vae_cache_key()`. The checkpoint is `checkpoints/vae_model.pt`.

| `VAEConfig` group | Fields (default) |
|---|---|
| Model | `latent_dim` 24, `hidden_dim` 256 |
| Training | `epochs` 40, `batch_size` 512, `lr` 1e-3, `weight_decay` 0, `kl_weight` 1, `kl_warmup_epochs` 10, `max_cells` 60000, `val_frac` 0.05, `patience` 8 |
| Similarities | `similarities` (load, denoise, resp), `beta` {load 0.3, denoise 0.4, resp 0.3} |
| Confidence | `n_posterior_samples` 10, `confidence_k` 10, `confidence_weighting` True, `confidence_floor` 0.1 |
| Refinement | `r_minus` 0.1, `m_plus` 5, `d_min` 5, `gamma` 0.5, `score_floor` 0, `cand_topk` 100, `chunk_size` 512, `edge_control` None |
| Checks | `h2_hit_threshold` 0.2, `pos_max_cells` 20000, `collapse_sample` 2000 |

**`cell_signal=True` (`lasser/cellsignal.py`; needs `fix_coexpress_batching=True`).**
- The co-expression SGConv input becomes `emb_pos[g] + x[cell, g]·u + c`. Here `x` is the cell's control
  expression, and `u, c` are learned and start near zero.
- So a gene's embedding depends on its neighbours' expression in that cell.
- It's wired by class swaps (`CellSignalGEARSModel`, `CellSignalSGConv`): GEARS's vendored code is untouched.
- `u` is initialised from a private generator, so batch order matches the paired runs without the variant.

**Compatibility.** With `graph_mode="gsr"` and `cell_signal=False` (the defaults), nothing changes. Earlier config
hashes, GSR cache keys and studies stay valid; `tests/test_08_vae.py` checks this.

**`notebooks/lasser_vae_study.ipynb`** (`lasser.study.VAE_PLAN`, 10 epochs, about 10 runs per 10-hour session):
1. `GEARS-cs` vs `VAE-cs`, splits 1–3
2. `VAE-random-cs`
3. `GEARS-kNN8-cs`
4. `LASSER-cs` (GSR graph, same variant)
5. `VAE-fix` (VAE graph in GEARS's original slot)
6. training-seed replicates
7. splits 4–5

## The batching fix and the option-1 study

**The fix (`fix_coexpress_batching=True`).** `GEARS_Model.forward` stacks the gene embeddings of all cells in a batch into one `batch_size × num_genes` node set, but runs the co-expression SGConv with the single-cell `G_coexpress`. So only the first cell gets messages from its co-expressed genes.
- `lasser/batching.py` swaps each co-expression SGConv's class for `BatchedSGConv`, which repeats the graph once per cell (node ids offset by `num_genes`).
- Parameters, `state_dict` keys and GEARS's code are unchanged; with one cell per batch the output is identical.
- Off by default; it's recorded in `runs.csv` (`fix_coexpress_batching`, `train_seed`).
- `GEARS.load_pretrained` doesn't know about the flag: re-enable it after loading a fixed model.

**`notebooks/lasser_fix_study.ipynb`** (`lasser.study.FIX_PLAN`): 10 epochs for every arm, about 50 min per run on one GPU. Jobs run in priority tiers, interleaved by split:
1. `GEARS-fix` vs `LASSER-fix`, splits 1–3
2. `LASSER-random-fix`, splits 1–3
3. `GEARS-kNN8-fix`, splits 1–3
4. `GEARS` (no fix), splits 1–3
5. training-seed replicates `_ts101` of tier 1
6. splits 4–5

A 10-hour session finishes tiers 1–2 and part of tier 3; later sessions resume.

**How the analysis handles replicates.** Runs are keyed by (arm, split, `rep` = training seed; 0 = default). Tables and paired tests average replicates within a split (`unit="split"`); `unit="run"` pairs every (split, training seed). Per-perturbation tests pair on (split, rep, perturbation).

## The study (multi-split experiments and analysis)

`notebooks/lasser_study.ipynb` runs the paper's experiment set on Kaggle (GPU T4 x2) and analyses it. The modules:

| Module | Role |
|---|---|
| `lasser/study.py` | `ARMS` (arm name → config overrides), `DEFAULT_PLAN`, `make_jobs`, `prepare_data` (warms data, splits and GEARS graphs in a subprocess), `launch` (one worker process per GPU), `study_status`. Also runs as `python -m lasser.study prep|worker` |
| `lasser/analysis.py` | Reads saved runs: `metrics_long`, `summary_table`, `paired_tests` (across splits, Holm), `per_perturbation` / `paired_perturbation` (split-resampling bootstrap), `paper_reference_baselines` (no-perturbation, additive), `gene_group_metrics` (genes isolated vs connected in A0), `training_curves`, `graph_table`, `edge_stability`, `consensus_edges`, `overlap_with_knn8`, `export_tables` (CSV + LaTeX) |
| `lasser/figures.py` | Paper figures (PDF + PNG), one fixed validated colour per arm |

Arms:

| Arm | Config | Splits |
|---|---|---|
| `GEARS` | `graph_learning=False` | 1–5 |
| `LASSER` | `graph_learning=True` (defaults) | 1–5 |
| `GEARS-kNN8` | GEARS with `coexpress_threshold=0.0`, `num_similar_genes_co_express_graph=8` (Pearson graph about as dense as LASSER's) | 1–5 |
| `LASSER-random` | `gsr.edge_control="random"`: every new edge gets a random source gene (same in-degrees, shuffled weights) | 1–5 |
| `LASSER-noS` / `-Eonly` | `gsr.views=("E","R")` / `("E",)` | 1–3 |
| `LASSER-addonly` | `gsr.r_minus=0` | 1–3 |
| `LASSER-embinit` | `gsr_init_gene_emb=True` | 1–3 |
| `LASSER-m2` / `-m10` | `gsr.m_plus=2` / `10` | 1–3 |

**How jobs run:**

- **Done check.** A job is done when `runs.csv` has a row with its config hash, so a new session skips finished work. Copy the previous output into `study/` (`RESUME_FROM`).
- **Claims.** Workers claim jobs with files in `runs/claims/<session>/`, so two GPUs never run the same job.
- **Deadline.** Workers stop before the session deadline.
- **Failures** are written to `runs/failures.csv` and retried in the next session.
- **Sharing.** Each worker gets `cpu_count / n_gpus` CPU threads. `runs.csv` appends are file-locked, and GSR cache entries are written to a temporary folder and then renamed.

Data lives in `/tmp/lasser_data`, so it isn't saved as output (it's several GB) and is re-downloaded each session.

## Loading a past run

```python
from lasser import load_run
r = load_run("/kaggle/working/runs/norman_seed1_gl_20261006-120000")
r["config"], r["split"]["split_hash"], r["graph"]["stats"], r["epoch_log"], r["test_metrics"]["paper"]["subgroups"]
```

## Evaluation

`lasser/_eval.py` is a byte-for-byte copy of the repo's `eval.py`; a test checks the hash. `lasser/evaluation.py` only builds its inputs:

- `ctrl` from `GEARS.ctrl_expression`;
- the mean training expression from the train loader;
- the DE indices from `eval.de_indices`;
- the subgroups from `pert_data.subgroup['test_subgroup']`;
- per-subgroup loaders (the test cells filtered by condition).

It then calls `evaluate_streaming` and `paper_summary` on GEARS's `best_model`.

## Tests (`tests/`, run on Kaggle)

| File | Test |
|---|---|
| `test_00_packaging.py` | 3: `lasser/` copied alone into a temp dir, imported and run in a subprocess. 2: that flag-off run imports no GSR module. Also checks that no top-level `gears` package is imported |
| `test_01_parity.py` | 1: flag off vs the vendored GEARS called directly, on CPU: identical graph, GO graph, initial `state_dict`, first 5 losses, and best model after 1 epoch (exact equality) |
| `test_02_graphs.py` | 4: saved graph = model's graph (both flags). 5: valid indices, node order, float32 weights ≥ 0, every gene in-degree ≥ `d_min`, kept A0 edges first |
| `test_03_leakage.py` | 6: validation/test cells permuted and rescaled: identical A0, views, positives, embeddings and refined graph |
| `test_04_cache_tracking.py` | 7: cache hit returns identical tensors. 8: every file exists, `runs.csv` has the rows, `load_run` reads them. Also checks the `eval.py` copy |
| `test_05_parsing.py` | GEARS printout parsing on real lines (no data needed) |
| `test_06_study.py` | Synthetic data, no dataset: random-edge control keeps in-degrees and weights; job table; analysis functions; figures render |
| `test_07_fix.py` | Synthetic: batched SGConv equals per-cell SGConv (and upstream differs); new config fields keep old hashes; fix plan; replicate parsing |
| `test_08_vae.py` | Synthetic: the VAE recovers planted modules; the similarity formulas equal explicit correlations; edge confidence; upstream GEARS's predicted change is cell-independent while the cell-signal variant's is not; config/hash compatibility; the VAE plan. `test_03_leakage.py` also checks the VAE graph against held-out-cell changes |
| `test_09_smoke.py` | 9: full Norman, 1 epoch, flag off and on, peak memory < 15 GB, GSR < 4 GB (`LASSER_SMOKE=1`) |

Environment variables: `LASSER_DATA_DIR`, `LASSER_TEST_DIR`, `LASSER_SUBSAMPLE` (default 4 cells per condition), `LASSER_SMOKE`.
