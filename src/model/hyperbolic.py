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

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperbolicProjection(nn.Module):
    """Projects Euclidean backbone features onto the Poincaré ball.

    Pipeline: linear → unit-normalize → scale by (norm_clip * 0.95) → expmap0.
    The pre-expmap scale is chosen so that after expmap0 the output norm is
    safely below norm_clip for any input.

    Args:
        in_dim: Input dimension (backbone embed_dim: 768 for ViT-L/14).
        out_dim: Output dimension on the Poincaré ball (DECISION-2, default 512).
        curvature: Curvature of Poincaré ball (DECISION-3, default -1.0).
        learn_curvature: If True, curvature is a learnable scalar parameter.
        norm_clip: Hard max norm of output points (must be < 1.0, default 0.99).
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
        assert 0.0 < norm_clip < 1.0, f"norm_clip must be in (0, 1), got {norm_clip}"
        self.norm_clip = norm_clip
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

        if learn_curvature:
            # Store log(|c|) so we can keep |c| > 0 via exp
            self.log_curvature = nn.Parameter(
                torch.tensor(abs(curvature)).log()
            )
        else:
            self.log_curvature = None
            self.register_buffer("_curvature", torch.tensor(abs(curvature)))

    @property
    def curvature(self) -> torch.Tensor:
        if self.log_curvature is not None:
            return self.log_curvature.exp()
        return self._curvature  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map Euclidean features to Poincaré ball.

        Args:
            x: Tensor of shape (..., in_dim).

        Returns:
            Tensor of shape (..., out_dim) with all norms < norm_clip.
        """
        x = self.linear(x)
        # Scale to slightly inside the ball before expmap0 so norms stay < norm_clip
        # after the exponential map.  0.9 * norm_clip gives headroom.
        x = F.normalize(x, dim=-1) * (self.norm_clip * 0.9)
        # Use float() to avoid geoopt mutating the curvature buffer in-place
        ball = geoopt.PoincareBall(c=float(self.curvature.item()))
        out = ball.expmap0(x)
        # Hard clip as final safety net (handles numerical edge cases)
        norms = out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        too_large = norms >= self.norm_clip
        out = torch.where(
            too_large.expand_as(out),
            out / norms * (self.norm_clip - 1e-4),
            out,
        )
        return out
