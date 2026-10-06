# Vendored GEARS

This folder is a copy of GEARS (`snap-stanford/GEARS`, commit `f374e43`, version 0.1.2), so `lasser` doesn't need the pip `cell-gears` package or a GEARS checkout. It is imported only through relative imports (`from .gears import GEARS, PertData`).

Upstream license: MIT (Yusuf Roohani, Kexin Huang, Jure Leskovec).

## Changes from upstream (every change is marked `# LASSER:` in the code)

| File | Change | Why |
|---|---|---|
| `utils.py` | `from dcor import distance_correlation` moved from module level into `get_coeffs` | `dcor` is only needed for gene-interaction (GI) analysis, so it isn't an install requirement |
| `pertdata.py`, `gears.py` | `torch_geometric.data.DataLoader` → `torch_geometric.loader.DataLoader` | Same class; the old location is a deprecated alias in current PyG |
| `pertdata.py` (`prepare_split`) | `groupby('split').agg({'condition': lambda x: x})` → iterate `groupby('split')['condition']` | Same per-split condition lists in the same order; newer pandas rejects a non-scalar `agg` result. Only runs when a split is first created |
| `pertdata.py` (`create_cell_graph_dataset`) | `obs['condition_name'][0]` → `.iloc[0]` | Same value; positional `[0]` on a string index warns in pandas 2 and fails in pandas 3. Only runs when `cell_graphs.pkl` is first built |

Nothing else is changed. In particular, the model, the loss, the training loop, the co-expression graph construction and the `torch.manual_seed(0)` at import are all upstream. The upstream quirks below are kept on purpose:

- `GEARS_Model.forward` (`model.py`) applies the co-expression graph only to the first cell of each batch.
- `PertData.load` derives `dataset_name` with `split('/')`; `lasser.run.load_pert_data` corrects it.
- `print('here1')` in `prepare_split`.

Don't edit these files except to fix an import or compatibility problem. Mark any such edit with `# LASSER:` and add it to the table above.
