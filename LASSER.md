# LASSER: running GEARS with a learned co-expression graph

`lasser/` wraps GEARS (pip `cell-gears==0.1.2`). One flag, `graph_learning`, swaps GEARS's Pearson co-expression graph for a GSR-style refined graph (pretrain, then refine once, then train GEARS). The design is in `gsr_coexpression_plan.md`.

## Running on Kaggle

GPU (T4), **Internet on**. `notebooks/run_lasser.ipynb` contains these cells:

```python
!pip install -q cell-gears==0.1.2 torch_geometric pytest
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

## How it fits into GEARS (no GEARS code is changed)

| Flag | What happens |
|---|---|
| `graph_learning=False` | Plain GEARS: `model_initialize()` builds its own co-expression graph. lasser adds seeding, logging and saving around it. |
| `graph_learning=True` | The GSR graph is built, then passed through GEARS's public `model_initialize(G_coexpress=..., G_coexpress_weight=...)`. The GO graph, model, loss, training loop and hyperparameters are unchanged. |
| `gsr_init_gene_emb=True` | After `model_initialize`, a linear projection of the pretrained E-view embedding is copied into `model.gene_emb` (and `best_model.gene_emb`). |

Seeding: `seed_everything(seed)` runs at the start and again right before `GEARS(...)`, so GEARS's initialisation and batch order are the same with the flag on or off. The parity test reproduces this order with plain GEARS (see `lasser/run.py`).

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
    checkpoints/ gears_model/{config.pkl,model.pt}     GEARS format (GEARS.load_pretrained works)
                 gsr_encoders.pt             pretrained encoders, embeddings, pretraining history
```

The graph pickle is a dict with these keys: `edge_index`, `edge_weight` (numpy), `gene_list`, `node_map`, `source` (`gears_static` / `gsr_refined`), `seed`, `split_hash`, `config_hash`, and `stats` (edges, self-loops, in-degree min/median/mean/max/Gini, isolated genes).

`epoch_log.csv` is parsed from GEARS's printout, because its training loop reports nothing else. So the train loss column is the mean of the losses GEARS prints every 50 steps.

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
| `test_00_packaging.py` | 3: `lasser/` copied alone into a temp dir, imported and run in a subprocess. 2: that flag-off run imports no GSR module |
| `test_01_parity.py` | 1: flag off vs plain GEARS on CPU: identical graph, GO graph, initial `state_dict`, first 5 losses, and best model after 1 epoch (exact equality) |
| `test_02_graphs.py` | 4: saved graph = model's graph (both flags). 5: valid indices, node order, float32 weights ≥ 0, every gene in-degree ≥ `d_min`, kept A0 edges first |
| `test_03_leakage.py` | 6: validation/test cells permuted and rescaled: identical A0, views, positives, embeddings and refined graph |
| `test_04_cache_tracking.py` | 7: cache hit returns identical tensors. 8: every file exists, `runs.csv` has the rows, `load_run` reads them. Also checks the `eval.py` copy |
| `test_09_smoke.py` | 9: full Norman, 1 epoch, flag off and on, peak memory < 15 GB, GSR < 4 GB (`LASSER_SMOKE=1`) |

Environment variables: `LASSER_DATA_DIR`, `LASSER_TEST_DIR`, `LASSER_SUBSAMPLE` (default 4 cells per condition), `LASSER_SMOKE`.
