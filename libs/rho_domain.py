"""rho-domain rate estimation from the motion-compensated residual (Phase 6).

The rho-domain model takes the coded rate of a block to be roughly linear in the
fraction of transform coefficients that survive quantisation. For a quantiser step
`Qstep`, a coefficient survives when `|c| > Qstep / 2`, so counting those gives a
QP-specific complexity feature that can be compared directly against the ground truth
at the matching QP.

`Qstep(QP) = 2 ** ((QP - 4) / 6)` is the standard HEVC/H.264 step-size ladder, defined
for **orthonormal** transform coefficients. The pipeline's DCT is the unnormalised
DCT-II (`X[k] = 2 * sum_m x[m] cos(...)`, matching `torch_dct` with `norm=None`), whose
coefficients are larger by `sqrt(2N)` per dimension. They are rescaled here before
thresholding; without that every coefficient clears the threshold and rho saturates
at 1.
"""
from typing import Dict, Sequence

import torch

DEFAULT_QPS = (22, 27, 32, 37)


def qstep(qp: float) -> float:
    """HEVC quantiser step size for a given QP."""
    return 2.0 ** ((qp - 4) / 6.0)


def orthonormal_scale(block_size: int) -> float:
    """Factor converting the pipeline's unnormalised 2D DCT-II to orthonormal scale.

    Per dimension the unnormalised AC coefficient is `sqrt(2N)` times the orthonormal
    one, so the separable 2D transform is off by `2N`.
    """
    return 2.0 * block_size


def rho_features(coeffs: torch.Tensor, block_size: int, n_frames: int,
                 qps: Sequence[int] = DEFAULT_QPS) -> Dict[str, torch.Tensor]:
    """Per-frame surviving-coefficient fraction at each QP.

    `coeffs` is the residual transform as produced by `apply_luma_transform`, shaped
    [total_blocks, block_size, block_size]. Returns {'rho_qp22': [n_frames], ...}.
    """
    magnitude = coeffs.reshape(n_frames, -1).abs() / orthonormal_scale(block_size)
    return {f'rho_qp{qp}': (magnitude > (qstep(qp) / 2.0)).float().mean(dim=1)
            for qp in qps}
