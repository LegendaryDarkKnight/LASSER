# LASSER-GSR: self-supervised learning of the co-expression graph

Written 2026-10-04 for Somik. Scope: the GEARS **gene co-expression graph only**. The GO perturbation graph stays GEARS's static Jaccard graph. Target hardware: Kaggle T4 (16 GB), using the splits and data from the official GEARS baseline run (S0 in `lasser_plan.md`). Background: GSR in `notes.md` §3, the v9 diagnosis in `lasser_plan.md` §1.

This plan expands `lasser_plan.md` §3.6 (stage S3) into a full route of its own.

---

## 0. TL;DR

1. **Pretrain** gene encoders by contrastive link prediction over several views of each gene: expression profile, perturbation-response profile, and graph structure on A0. Use an intra-view loss plus GSR's inter-view loss.
2. **Refine once.** Score candidate gene pairs from the pretrained embeddings, add the top m⁺ new edges and remove the weakest m⁻ A0 edges **per gene**, then freeze the graph.
3. **Fine-tune GEARS unchanged** on the refined graph, optionally initialising GEARS's gene embedding from the pretrained encoder. If the graph is written in GEARS's own co-expression format, GEARS code needs no changes.
4. Choose m⁺, m⁻ and view weights on **validation** DE-MSE (the only task feedback), then confirm on 5 seeds against the S0 baseline.
5. Memory is never a problem: N ≈ 5k genes, so full-batch encoders and chunked N×N scoring under `no_grad` fit easily in under 2 GB. The v9 problems (dense graphs in autograd, nonzero conversion, moving graph, collapse) don't arise because the graph is built once, offline, as an edge list.

---

## 1. Why GSR fits this problem

- **Goal mismatch (GSR's critique of IDGL)** is worse in GEARS than in the GSR paper. Norman has only ~100 training perturbations, so a graph trained only through the GEARS loss gets a weak, noisy signal (this is v9 problem 3). Self-supervision uses **every training cell** to learn which genes belong together.
- **Static graph = stable training.** GEARS trains on a fixed graph exactly as in the paper, so any change in metrics is attributable to the graph.
- **Cheap.** Pretraining and refinement take minutes. GEARS fine-tuning costs the same as the baseline.
- **Where it should help (inferred).** In GEARS, the co-expression graph propagates information between *output genes* (the gene embeddings), while unseen perturbations generalise mainly through the GO graph. So gains should show most on response genes that are poorly connected in A0, including the genes whose zero variance gave NaN correlations in v9 (about 1.8k, set to 0, so no edges). Evaluate a **low-degree output-gene subgroup** as well as the paper's perturbation subgroups.
- **Risk.** GSR's refinement is task-agnostic, and on noisy graphs it lowered homophily. Selecting edit sizes on validation and the ablations in §7 are how we find out whether it helps here.

---

## 2. Data and leakage rules

All of this is recomputed **per split seed** (1–5), because training perturbations differ by seed.

- **Source:** `perturb_processed.h5ad` plus the split pkl from the GEARS run. Only **control cells + training-perturbation cells** are used for anything that builds features, positives or graphs. Validation perturbations are used only for model selection, and test perturbations never.
- **A0:** the GEARS co-expression CSV for that seed (GEARS builds it from training data: Pearson, top-k per gene, threshold). Keep a copy as the baseline graph.
- **Cell halves for honest positives:** split training cells randomly into halves H1/H2 (stratified by condition). Positives are built from H1, and H2 is used to check whether added edges are real co-expression (§6.1).

---

## 3. Views (node properties)

GSR uses node features and DeepWalk structure as two views. For genes I propose three views, with an optional fourth:

| View | Per-gene vector | How | Notes |
|---|---|---|---|
| **E: expression** | PCA-64 of the gene's (log-normalised) expression over training cells | Pseudobulk cells into ~500 metacells (k-means or per-condition mini-bulks) first, to reduce dropout noise and memory | The core co-expression signal |
| **R: response** | The gene's mean Δ (pert − ctrl) across training perturbations, PCA-32 | One row per training perturbation, so about 100 columns before PCA | Captures "responds together", which is closer to the GEARS task |
| **S: structure** | node2vec/DeepWalk-64 on A0 | `torch_geometric.nn.Node2Vec`, a few epochs, under 1 min | GSR's structure view. Isolated genes get uninformative vectors |
| X: external (optional) | Pretrained gene embeddings, e.g. Gene2vec, GenePT, scGPT gene tokens | Lookup, then PCA-64 | The only view that has information for zero-variance genes. Off by default, turned on if the low-degree subgroup matters |

GO-derived features are deliberately not used, to keep GO out of scope.

Preprocessing for every view: standardise each dimension, mean-centre, and **no ReLU** on encoder outputs used for cosine (the v9 collapse cause). Genes with zero variance get a zero vector in E and R, plus a learned "missing" token, so the encoder doesn't treat them as identical real genes.

---

## 4. Pretraining (self-supervised link prediction)

### 4.1 Positives and negatives

- **Positives P:** A0 edges recomputed on half H1 (Pearson top-20 with the GEARS threshold). Keep only pairs that are also strongly correlated in at least 70% of bootstrap resamples of H1, which removes edges that are noise. Optionally add pairs whose response vectors (view R) correlate above a high threshold, so positives aren't only A0.
- **Negatives:** per query, K = 256 random genes, excluding the query's A0 and P neighbours to reduce false negatives (co-expression modules are large). Mix in about 25% **hard negatives**: candidates with high cosine in one view and low H1 correlation.
- **Edge split for message passing:** each epoch, split P 70/30. The encoder's GNN only passes messages over the 70% part and predicts the 30% part. Otherwise the target edge is already inside the GNN input, and link prediction becomes trivial (standard link-prediction practice; GSR avoids this through ego-subgraph sampling).

### 4.2 Encoders

- **Per-view encoder f_φ:** `MLP(view features) → 2-layer SGConv/GCN on the 70% message edges → linear head (dim 64)`, LayerNorm and no ReLU at the output. N ≈ 5k, so it runs **full batch**: every step encodes all genes at once (~5k × 256 activations, a few MB).
- **Key encoder:** a momentum copy (MoCo, m = 0.99), as in GSR. With full batch, a plain shared encoder with stop-gradient on keys is a reasonable simplification. Start with the momentum version to stay close to GSR.
- **Augmentations:** feature masking (20%) and message-edge dropout (20%) applied independently to query and key, which also discourages collapse.

### 4.3 Losses

- **Intra-view (per view φ):** InfoNCE over (query i, positive j, K negatives), with cosine/τ, τ = 0.2.
- **Inter-view (each ordered pair s→t):** `ĝ = g_{s→t}(z_i^s)` with a 2-layer MLP, contrasted against target-view keys `z_j^t` with the same InfoNCE. This makes each view predict links using another view's information, which is what lets a gene weak in one view (say, isolated in S) borrow from another (E or R).
- **Total:** `L_P = α·mean_φ L_intra + (1−α)·mean_{s≠t} L_inter`, with α = 0.75 (GSR's best).
- **Anti-collapse guard:** plus 0.1 × a VICReg variance hinge on each view's z (std per dimension ≥ 1). Cheap insurance against the v9 failure.

### 4.4 Schedule and cost

- Adam, lr 1e-3, 200–500 full-batch steps, early stop on held-out link AUC (the 30% split) and the H2 check (§6.1).
- Cost on T4: under 5 minutes per seed and under 2 GB. Save each view's embeddings per seed (`emb_seed{s}.pt`).

---

## 5. Refinement (build the graph once)

### 5.1 Edge probabilities

`E_ij = Σ_φ β_φ · Norm_φ(cos(z_i^φ, z_j^φ))`, where Norm_φ min-max scales each view's cosines to [0, 1] (GSR) and β sums to 1.

- Computed **chunked under `no_grad`**: 512 query genes × 5k at a time, keeping only the top-100 per gene. The full 5k × 5k matrix (100 MB fp32) would actually fit, but chunking keeps it safe for larger datasets.
- Default β: E 0.4, R 0.3, S 0.3. Tuned in {E-heavy, R-heavy, uniform}.

### 5.2 Edits per gene (not global)

GSR edits globally. Per gene is better here, because global top-m⁺ would put most new edges on a few hub modules.

- **Remove:** for each gene, drop the fraction r⁻ of its A0 edges with the lowest E_ij. Grid r⁻ ∈ {0, 0.1, 0.25}.
- **Add:** for each gene, add the top m⁺ non-A0 candidates by E_ij, above a probability floor. Grid m⁺ ∈ {0, 2, 5, 10}.
- **Minimum degree:** genes with A0 degree below d_min = 5 (including isolated genes) are topped up to d_min from their best candidates. This is the main lever for poorly connected genes.
- **Symmetrise** and drop duplicates. Cap degree at the GEARS top-k plus m⁺.

### 5.3 Edge weights

GEARS passes the co-expression importance (Pearson) to SGConv as edge weight, so new edges need weights on a comparable scale.

- Kept A0 edges keep their Pearson weight.
- New edges get γ × (their E_ij mapped to the A0 weight distribution by quantile), with γ ∈ {0.5, 1}. GSR found that too many equal-weight new edges dilute reliable ones; γ < 1 is the soft version of that lesson.

### 5.4 Output

Write the refined graph in GEARS's co-expression CSV format (`source, target, importance`), one per seed, with a sibling file listing which edges were added or removed. GEARS can then load it as if it were its own co-expression network. Either overwrite the cached file GEARS reads (check the exact name in `gears/utils.py`, `get_similarity_network`), or pass it through `model_initialize(G_coexpress=…, G_coexpress_weight=…)`, which I believe GEARS exposes (to verify on the installed version).

---

## 6. Checking the graph before spending GPU time on GEARS

### 6.1 Cheap, label-free checks (seconds per graph)

- **H2 hit rate:** the fraction of added edges whose correlation on the held-out half H2 exceeds 0.2, versus degree-matched random pairs and versus A0 edges. Added edges should be well above random.
- **Coverage:** the number of isolated genes, the degree distribution (min, median, Gini), and how much the low-degree genes gained.
- **Overlap with A0:** Jaccard of each gene's neighbourhood.
- **External agreement:** the fraction of added edges in STRING (combined score ≥ 700) or TRRUST/DoRothEA, versus random. This is also used later for "reasoning", so it is never used for selection.
- **Collapse monitors:** the std of pairwise cosine and the effective rank of each view's embeddings.

Graphs that fail the H2 check or show collapse are discarded before any GEARS run.

### 6.2 Task selection (GPU)

- Run GEARS on seed 1 with **10 epochs** for each surviving configuration, and select on validation normalised top-20 DE-MSE.
- Confirm the best 2–3 configurations with the full 20 epochs on **seeds 1–5**, and compare with S0 using the paper's metrics per subgroup plus the low-degree output-gene subgroup.

---

## 7. Fine-tuning variants and ablations

| ID | Variant | Question it answers |
|---|---|---|
| F0 | GEARS on A0 (the S0 baseline) | Reference |
| F1 | GEARS on the refined graph, standard init | Does the refined graph alone help? (pure GSR) |
| F2 | F1 + GEARS gene embedding initialised from the pretrained E-view embedding (linear projection to hidden 64) | GSR's pretrained init. GSR found random init hurts |
| F3 | Add-only (r⁻ = 0) and remove-only (m⁺ = 0) | Which edit matters? GSR found it differs between clean and noisy graphs |
| F4 | Single view (E only, S only) and no inter-view loss | GSR's multi-view ablations |
| F5 | Random edits with the same counts | Rules out "any extra edges help" |
| F6 | Positives from A0 only vs bootstrapped positives | Does denoising positives matter? |
| F7 (hybrid) | The refined graph as A0 for the IDGL-style learner (`lasser_plan.md` S2), with the pretrained scorer as its initialisation | Bridge to the task-driven route, only if F1/F2 show a gain |

---

## 8. Compute budget on Kaggle T4

- Pretraining plus refinement: about 5 min and under 2 GB per seed, so all 5 seeds fit in one session.
- GEARS runs dominate. Measure the time per epoch in S0 and size the grid from it. The full grid (3 r⁻ × 4 m⁺ × 3 β × 2 γ = 72) is too large, so do it in two passes: first m⁺ × r⁻ with default β and γ (12 graphs, minus those failing §6.1), then β and γ around the best one.
- Kaggle gives 12 h per session and a weekly GPU quota, so save checkpoints, embeddings, graphs and per-perturbation predictions to the output dir after every run and keep a results table (config, seed, metrics) that is appended to, so sessions can resume.

---

## 9. Code layout (one Kaggle notebook, or small modules)

1. `data.py`: load the h5ad and split pkl for a seed; build the training-cell mask, halves H1/H2 and metacells.
2. `views.py`: build E, R, S (and optionally X) features per seed.
3. `pretrain.py`: positives/negatives, edge split, encoders, intra/inter InfoNCE, VICReg guard, early stopping.
4. `refine.py`: chunked scoring, per-gene add/remove, min degree, weights, GEARS-format CSV plus the edit list.
5. `graph_checks.py`: §6.1 checks and monitors.
6. `run_gears.py`: GEARS training on a given graph and seed, saving per-perturbation predictions.
7. `evaluate.py`: the S0 harness metrics (shared with the baseline).

---

## 10. Stages and gates

| Stage | What | Gate to continue |
|---|---|---|
| G0 | S0 baseline done (GEARS, no-perturb, additive; 5 seeds) | Paper numbers reproduced |
| G1 | Views + pretraining for seed 1; link AUC and H2 checks | Held-out link AUC clearly above a correlation-only scorer; no collapse |
| G2 | Refinement grid for seed 1, §6.1 filtering | At least some graphs pass the H2 and external checks |
| G3 | GEARS 10-epoch selection on seed 1 (F1, then F2 on the best) | Val DE-MSE better than F0 |
| G4 | 5-seed confirmation of the best 2–3 configs | Gain beyond seed CI on ≥1 subgroup, no loss elsewhere |
| G5 | Ablations F3–F6 | Explains where the gain comes from |
| G6 | Reasoning analysis on consensus edges (`lasser_plan.md` §5), F7 if G4 passed | — |

If G3/G4 show no gain, that is still a result: the task-agnostic graph doesn't help, which motivates the task-driven route (`lasser_plan.md` S1/S2) with this pretrained scorer as its initialisation.

---

## 11. Defaults I picked

- Views E + R + S; X off. Embedding dim 64; encoder hidden 256; 2 SGConv layers.
- InfoNCE τ = 0.2, K = 256 negatives with 25% hard; α = 0.75; VICReg weight 0.1; momentum 0.99.
- β = (0.4, 0.3, 0.3); r⁻ ∈ {0, 0.1, 0.25}; m⁺ ∈ {0, 2, 5, 10}; d_min = 5; γ ∈ {0.5, 1}.
- GEARS hyperparameters exactly as in S0 (hidden 64, batch 32, 20 epochs; 10 epochs for selection).
