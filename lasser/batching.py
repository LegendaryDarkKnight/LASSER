"""Opt-in fix for GEARS's co-expression batching (``LasserConfig.fix_coexpress_batching``).

``GEARS_Model.forward`` stacks the gene embeddings of all cells in a batch into one
(batch_size * num_genes) node set and runs the co-expression SGConv with the single-cell
``G_coexpress`` (node ids 0..num_genes-1). So only the first cell's copy gets messages
from its co-expressed genes; every other cell gets a self-loop only.

``enable_batched_coexpress`` changes the class of each co-expression SGConv to
``BatchedSGConv``, which repeats the graph once per cell (ids offset by num_genes) before
calling SGConv. Parameters, ``state_dict`` keys and GEARS's code are unchanged; with one
cell per batch the output is identical. Flag off (default): nothing here runs.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch_geometric.nn import SGConv

logger = logging.getLogger(__name__)


def batch_edges(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor], num_nodes: int,
                copies: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Repeat a graph ``copies`` times, copy b using node ids b*num_nodes ... (b+1)*num_nodes-1."""
    offsets = torch.arange(copies, device=edge_index.device) * num_nodes
    ei = (edge_index.unsqueeze(1) + offsets.view(1, -1, 1)).reshape(2, -1)
    ew = edge_weight.repeat(copies) if edge_weight is not None else None
    return ei, ew


class BatchedSGConv(SGConv):
    """SGConv that applies a per-cell graph to every cell of a batch-stacked node set."""

    lasser_num_genes: int = 0

    def forward(self, x, edge_index, edge_weight=None):  # noqa: D401 - same signature as SGConv
        n = self.lasser_num_genes
        copies = x.size(0) // n if n else 1
        if copies > 1:
            if x.size(0) != copies * n:
                raise ValueError(f"{x.size(0)} nodes is not a multiple of {n} genes")
            edge_index, edge_weight = batch_edges(edge_index, edge_weight, n, copies)
        return super().forward(x, edge_index, edge_weight)


def enable_batched_coexpress(model: torch.nn.Module, num_genes: int) -> int:
    """Switch every co-expression SGConv of a GEARS_Model to ``BatchedSGConv``; returns how many."""
    n = 0
    for conv in model.layers_emb_pos:
        if isinstance(conv, SGConv) and not isinstance(conv, BatchedSGConv):
            conv.__class__ = BatchedSGConv
        if isinstance(conv, BatchedSGConv):
            conv.lasser_num_genes = num_genes
            n += 1
    return n
