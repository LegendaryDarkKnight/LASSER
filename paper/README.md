# Paper draft: "Does the gene co-expression graph matter in GEARS?"

`main.tex` + `refs.bib` + `figures/` (copied from `lasser_fix_study_results.zip`). Build with
`pdflatex main && bibtex main && pdflatex main && pdflatex main`, or upload the folder to Overleaf.

## What kind of paper this is

An **analysis / negative-result paper**: the experiments show no gain from learning the co-expression graph, and
the paper explains why. Its claims, and the evidence for each:

| Claim | Evidence |
|---|---|
| The GEARS co-expression prior is nearly empty | 87–89% of genes have no neighbour; most edges are self-loops (`graph_stats`) |
| A batching defect limits it to the first cell of each batch | Code (`model.py`), `batching.py`, `tests/test_07_fix.py` |
| The learned graph is informative | Held-out H2 hit rate 3.4–3.7% vs 0.1–0.2% for random; 346 edges stable across splits with coherent modules |
| Graph structure doesn't change predictions | Studies 1 and 2: learned ≈ original ≈ random ≈ density-matched, all within noise (Tables 1–2) |
| Why: re-parameterisation | The co-expression GNN smooths a free, cell-independent table (Section 5); argument plus controls |

## Suggested venues

- **As it stands:** a workshop or short-paper venue that values careful negative results, e.g. NeurIPS "I Can't
  Believe It's Not Better" (ICBINB), ICML/NeurIPS computational-biology workshops, MLCB, or ICLR LMRL.
- **Journal (e.g. *Bioinformatics*, *Briefings in Bioinformatics*, *NAR Genomics and Bioinformatics*,
  *PLOS Computational Biology*):** needs a little more evidence first (see below).

## To do before submission

1. Fill in the `\todo{}` items: authors, the repository URL, and one or two sentences with settings and numbers
   from your earlier attempts (IDGL-style task-driven graph learning, Poincaré GO embeddings).
2. Verify the bibliography entries marked `VERIFY` in `refs.bib`.
3. Figure 1: two panel titles touch. Re-render with shorter titles or a wider figure
   (`lasser/figures.py::fig_headline`), or crop in a vector editor.
4. Optional, to make the claims harder to dispute (each is one config away in `lasser.study.FIX_PLAN`, but
   needs GPU time):
   - splits 4–5 for GEARS-fix / LASSER-fix / LASSER-random-fix, and GEARS-kNN8-fix on splits 2–3;
   - training-seed replicates (`_ts101` jobs), to replace the cross-session noise estimate with a planned one;
   - a second dataset (e.g. Adamson or Replogle K562, which GEARS supports);
   - a direct test of the re-parameterisation argument: GEARS-fix with the co-expression graph replaced by
     self-loops only (prediction unchanged if the argument holds).

All numbers in the draft come from `lasser_fix_study_results.zip` (study 2) and `lasser_study_results.zip`
(study 1), as recorded in `summarizer.md`.
