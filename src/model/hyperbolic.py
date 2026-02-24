"""
Hyperbolic projection layer — maps Euclidean ViT features to Poincaré ball.

Responsibilities:
- Linear projection from backbone embed_dim (768 or 512) to hyperbolic_dim (512 default)
- Map to Poincaré ball via exponential map at origin (geoopt)
- ALWAYS clip output norm to < norm_clip (0.99) — hard constraint
- Optionally learn curvature c (DECISION-3: fixed -1.0 default)
- Verify forward pass produces points with norm < 1 (Phase 3 validation gate)

DECISION-2: hyperbolic_dim 512 (default). Config key: model.hyperbolic_dim.
DECISION-3: Fixed curvature -1.0 (default). Config key: model.learn_curvature.

From SPEC §4.3:
    class HyperbolicProjection(nn.Module):
        def __init__(self, in_dim=768, out_dim=512, curvature=-1.0):
            self.linear = nn.Linear(in_dim, out_dim, bias=False)
            self.c = curvature
        def forward(self, x):
            x = self.linear(x)
            x = F.normalize(x, dim=-1) * 0.9
            return geoopt.manifolds.PoincareBall(c=abs(self.c)).expmap0(x)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO(phase3): implement HyperbolicProjection with geoopt
# TODO(phase3): implement learned curvature variant (nn.Parameter)
# TODO(phase3): add assertion that output norms < norm_clip in forward()


class HyperbolicProjection(nn.Module):
    """Projects Euclidean backbone features onto the Poincaré ball.

    Args:
        in_dim: Input dimension (backbone embed_dim: 768 for ViT-L/14).
        out_dim: Output dimension on the Poincaré ball (DECISION-2, default 512).
        curvature: Curvature of Poincaré ball (DECISION-3, default -1.0).
        learn_curvature: If True, curvature is a learnable parameter.
        norm_clip: Hard max norm of output points (must be < 1.0).
    """

    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 512,
        curvature: float = -1.0,
        learn_curvature: bool = False,
        norm_clip: float = 0.99,
    ):
        super().__init__()
        # TODO(phase3): self.linear = nn.Linear(in_dim, out_dim, bias=False)
        # TODO(phase3): handle learn_curvature with nn.Parameter
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map Euclidean features to Poincaré ball.

        Args:
            x: Tensor of shape (..., in_dim).

        Returns:
            Tensor of shape (..., out_dim) with all norms < norm_clip.
        """
        raise NotImplementedError
