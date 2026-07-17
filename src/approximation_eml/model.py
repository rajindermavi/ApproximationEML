from typing import cast

import torch
import torch.nn as nn

from .components import SoftLeaf, EMLNode
from .config import DEFAULT_CONFIG


class EMLTree(nn.Module):
    """Fixed-depth scalar-output EML tree for R^p -> R."""

    tau: torch.Tensor  # registered buffer — declared here so type checkers know the attribute type

    def __init__(
        self,
        input_dim: int,
        depth: int,
        use_gates: bool = DEFAULT_CONFIG.use_gates,
        eps: float = DEFAULT_CONFIG.eps,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.use_gates = use_gates

        # tau is a non-trainable buffer: checkpointed, moves with .to(device), not in optimizer
        self.register_buffer("tau", torch.tensor(1.0))

        n_leaves = 2 ** depth
        n_internal = n_leaves - 1

        # BFS order: leaves[i] corresponds to BFS index n_internal + i
        self.leaves = nn.ModuleList([SoftLeaf(input_dim) for _ in range(n_leaves)])
        # BFS order: nodes[i] corresponds to BFS index i (root = 0)
        self.nodes = nn.ModuleList([EMLNode(use_gates=use_gates, eps=eps) for _ in range(n_internal)])

    def update_tau(self, tau: float) -> None:
        """Set the current temperature. Called by the training loop each epoch."""
        self.tau.fill_(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor| None:
        """Return predictions of shape (batch_size,). Uses self.tau for all leaves and gates."""
        tau = self.tau.item()
        n_internal = 2 ** self.depth - 1
        n_leaves = 2 ** self.depth

        values: list[torch.Tensor | None] = [None] * (n_internal + n_leaves)

        # Evaluate leaves (bottom level)
        for i, leaf in enumerate(self.leaves):
            values[n_internal + i] = leaf(x, tau)

        # Bottom-up: process internal nodes from deepest level up to root
        # Iterating n_internal-1 down to 0 guarantees children are ready before parents
        for bfs_idx in range(n_internal - 1, -1, -1):
            left_child = 2 * bfs_idx + 1
            right_child = 2 * bfs_idx + 2
            values[bfs_idx] = self.nodes[bfs_idx](
                values[left_child], values[right_child], tau
            )

        return values[0]

    def leaf_modules(self) -> list[SoftLeaf]:
        """Return leaves in BFS order for inspection/export."""
        return cast(list[SoftLeaf], list(self.leaves))

    def node_modules(self) -> list[EMLNode]:
        """Return internal nodes in BFS order for inspection/export."""
        return cast(list[EMLNode], list(self.nodes))
