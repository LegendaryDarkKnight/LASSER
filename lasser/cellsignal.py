"""Opt-in GEARS variant: cell-specific co-expression messages (``LasserConfig.cell_signal``).

In upstream GEARS the co-expression GNN only smooths a learnable table (``emb_pos``) that is
the same for every cell, so graph structure can only re-parameterise per-gene biases (see
``paper/main.tex``, Section 5). This variant adds each cell's own control expression as a
node feature *before* the co-expression SGConv:

    pos_in[cell, g] = emb_pos[g] + x[cell, g] * u + c          (u, c in R^hidden, learned)
    pos_emb = SGConv_A(pos_in)                                  (batched: one graph per cell)

so a gene's embedding now depends on the control expression of its graph neighbours in
that cell, and the graph's structure can matter. Everything else in GEARS is unchanged.

How it is wired, without editing the vendored ``GEARS_Model.forward``:

- the model's class is switched to ``CellSignalGEARSModel``, whose ``forward`` stores the
  batch's control expression (``data.x``) on the co-expression convs, then calls the
  upstream forward;
- each co-expression SGConv's class is switched to ``CellSignalSGConv`` (a ``BatchedSGConv``,
  so the batching fix is included), which adds the projected expression to its input.

``u`` and ``c`` start small (std 0.01) from a dedicated generator, so the global RNG, and
therefore batch order, is the same as in the matching run without the variant.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .batching import BatchedSGConv, enable_batched_coexpress
from .gears.model import GEARS_Model

logger = logging.getLogger(__name__)


class CellSignalSGConv(BatchedSGConv):
    """BatchedSGConv whose input gets ``x * u + c`` added per node (x = the cell's control expression)."""

    def forward(self, x, edge_index, edge_weight=None):
        cell_x = getattr(self, "_lasser_cell_x", None)
        if cell_x is not None:
            if cell_x.numel() != x.size(0):
                raise ValueError(f"cell expression has {cell_x.numel()} entries for {x.size(0)} nodes")
            x = x + cell_x.reshape(-1, 1).to(x.dtype) * self.cell_u + self.cell_c
        return super().forward(x, edge_index, edge_weight)


class CellSignalGEARSModel(GEARS_Model):
    """GEARS_Model whose co-expression convs see the batch's control expression."""

    def forward(self, data):
        convs = [c for c in self.layers_emb_pos if isinstance(c, CellSignalSGConv)]
        for c in convs[:1]:            # only the first layer gets the raw expression
            c._lasser_cell_x = data.x
        try:
            return super().forward(data)
        finally:
            for c in convs:
                c._lasser_cell_x = None


def enable_cell_signal(model: GEARS_Model, num_genes: int, seed: int) -> int:
    """Switch a GEARS_Model to the cell-signal variant (implies the batching fix); returns #convs."""
    n = enable_batched_coexpress(model, num_genes)
    g = torch.Generator().manual_seed(int(seed) + 104729)   # private RNG: global stream untouched
    for conv in model.layers_emb_pos:
        if isinstance(conv, BatchedSGConv) and not isinstance(conv, CellSignalSGConv):
            conv.__class__ = CellSignalSGConv
            dim = conv.lin.in_features
            dev = conv.lin.weight.device
            conv.cell_u = nn.Parameter((torch.randn(dim, generator=g) * 0.01).to(dev))
            conv.cell_c = nn.Parameter(torch.zeros(dim, device=dev))
            conv._lasser_cell_x = None
    model.__class__ = CellSignalGEARSModel
    return n
