"""Per-frame structural descriptors of the motion field (Phase 5).

All functions take the block MV field `mvs` [B, 2, H_b, W_b] holding (dy, dx) in
full-resolution pixels under the matching convention `curr(y, x) ~= ref(y + dy, x + dx)`,
and return one value per frame.

Sign convention for the differential features: they are computed on the **forward
flow** `-mvs`, i.e. how content moves from the reference into the current frame. That
is the physically intuitive direction, and it makes a zoom-in give a positive
divergence (content moves outward from the centre) rather than a negative one.
"""
from typing import Dict

import torch
import torch.nn.functional as F


def weighted_median(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Per-row weighted median of [B, N] values with [B, N] non-negative weights."""
    order = values.argsort(dim=1)
    v = torch.gather(values, 1, order)
    w = torch.gather(weights, 1, order)
    cum = w.cumsum(dim=1)
    half = 0.5 * w.sum(dim=1, keepdim=True)
    idx = (cum < half).sum(dim=1, keepdim=True).clamp_(max=values.shape[1] - 1)
    return torch.gather(v, 1, idx).squeeze(1)


def global_mv(mvs: torch.Tensor, sad_map: torch.Tensor) -> torch.Tensor:
    """SAD-weighted median motion vector per frame, [B, 2] as (dy, dx).

    Blocks are weighted by `1 / (SAD + 1)`: a low matching cost means the vector is
    well determined, so reliable blocks dominate the estimate. The median (rather than
    the mean) keeps a moving foreground object from dragging the global estimate away
    from the true background motion.
    """
    B = mvs.shape[0]
    flat = mvs.reshape(B, 2, -1)
    w = (1.0 / (sad_map.reshape(B, -1) + 1.0))
    return torch.stack([weighted_median(flat[:, 0], w),
                        weighted_median(flat[:, 1], w)], dim=1)


def mv_coherence(mvs: torch.Tensor, gmv: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Fraction of blocks whose vector lies within `eps` pixels of the global vector.

    Near 1 for a rigid pan, low where the field is dominated by independent object
    motion or by matching noise.
    """
    B = mvs.shape[0]
    diff = mvs.reshape(B, 2, -1) - gmv.view(B, 2, 1)
    return (diff.pow(2).sum(dim=1).sqrt() <= eps).float().mean(dim=1)


def _central_differences(field: torch.Tensor):
    """d/dx and d/dy of a [B, 2, H_b, W_b] field, replicate-padded at the border."""
    padded = F.pad(field, (1, 1, 1, 1), mode='replicate')
    d_dx = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) * 0.5
    d_dy = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) * 0.5
    return d_dx, d_dy


def divergence_curl(mvs: torch.Tensor):
    """Mean divergence and curl of the forward flow, one value per frame.

    Zoom-in gives divergence > 0, in-plane rotation gives curl != 0, and a pure
    translation gives both ~ 0 because the field is spatially constant.
    """
    flow = -mvs.float()
    d_dx, d_dy = _central_differences(flow)
    # channel 0 is dy (vertical), channel 1 is dx (horizontal)
    div = d_dx[:, 1] + d_dy[:, 0]
    curl = d_dx[:, 0] - d_dy[:, 1]
    return div.mean(dim=[1, 2]), curl.mean(dim=[1, 2])


def affine_fit(mvs: torch.Tensor, sad_map: torch.Tensor):
    """Weighted least-squares affine model of the forward flow.

    Solves `flow(p) ~= A p + t` over all blocks with inverse-SAD weights, where `p` is
    the block centre in units of half the frame size (so the coefficients are
    resolution independent). Returns (A [B, 2, 2], t [B, 2]) ordered (y, x).
    """
    B, _, Hb, Wb = mvs.shape
    device = mvs.device
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, Hb, device=device),
        torch.linspace(-1, 1, Wb, device=device), indexing='ij')
    ones = torch.ones_like(yy)
    design = torch.stack([yy, xx, ones], dim=-1).reshape(1, -1, 3).expand(B, -1, -1)

    flow = (-mvs.float()).reshape(B, 2, -1).transpose(1, 2)      # [B, N, 2] as (dy, dx)
    w = (1.0 / (sad_map.reshape(B, -1) + 1.0)).sqrt().unsqueeze(-1)
    sol = torch.linalg.lstsq(design * w, flow * w).solution      # [B, 3, 2]

    A = sol[:, :2, :].transpose(1, 2)                            # [B, 2(out), 2(in)]
    t = sol[:, 2, :]
    return A, t


def mvd_cost(mvs: torch.Tensor) -> torch.Tensor:
    """Mean L1 distance between each block vector and its 3x3 median predictor.

    A proxy for what an encoder would spend signalling the motion field.
    """
    from libs.motion_estimation import median_predictor
    return (mvs.float() - median_predictor(mvs.float())).abs().sum(dim=1).mean(dim=[1, 2])


def compute_motion_features(mvs: torch.Tensor, sad_map: torch.Tensor,
                            coherence_eps: float = 1.0) -> Dict[str, torch.Tensor]:
    """Every per-frame motion descriptor, as {column_name: [B] tensor}."""
    gmv = global_mv(mvs, sad_map)
    div, curl = divergence_curl(mvs)
    A, t = affine_fit(mvs, sad_map)
    return {
        'GMV_y': gmv[:, 0],
        'GMV_x': gmv[:, 1],
        'GMV_mag': gmv.pow(2).sum(dim=1).sqrt(),
        'MV_coherence': mv_coherence(mvs, gmv, coherence_eps),
        'MV_div': div,
        'MV_curl': curl,
        'MVD_cost': mvd_cost(mvs),
        'aff_a11': A[:, 0, 0], 'aff_a12': A[:, 0, 1],
        'aff_a21': A[:, 1, 0], 'aff_a22': A[:, 1, 1],
        'aff_ty': t[:, 0], 'aff_tx': t[:, 1],
    }
