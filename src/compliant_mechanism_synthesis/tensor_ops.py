from __future__ import annotations

import torch


def symmetrize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    diagonal = torch.diagonal(symmetric, dim1=-2, dim2=-1)
    return symmetric - torch.diag_embed(diagonal)


def upper_triangle_edge_index(num_nodes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
    return edge_index[0], edge_index[1]
