import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from libs.motion_compensation import DenseMC, MotionCompensator
# Re-exported so existing imports of the estimator from this module keep working.
from libs.motion_estimation import (HierarchicalBlockMatcher,  # noqa: F401
                                    SparsePatternBlockMatcher)


@dataclass
class TemporalState:
    """Unified memory footprint (and lazy-evaluation graph) for the metric pipeline."""
    # Inputs
    current_frame: torch.Tensor     #[B, C, H, W]
    ref_frame: torch.Tensor         #[B, C, H, W]
    bs: int = 32
    # Motion Estimation Outputs
    mvs: torch.Tensor = None        #[B, 2, H_blocks, W_blocks]
    sad_map: torch.Tensor = None    #[B, 1, H_blocks, W_blocks]
    # Motion Compensation strategy (defaults to the Gaussian-smoothed dense warp)
    compensator: Optional[MotionCompensator] = None
    # Motion Compensation Outputs (Lazy/Optional)
    _mc_blocks: torch.Tensor = field(default=None, repr=False)  # Unfolded, motion-compensated blocks
    _residual: torch.Tensor = field(default=None, repr=False)   # current_blocks - mc_blocks
    _mc_frame: torch.Tensor = field(default=None, repr=False)   # Full-resolution warped reference

    @property
    def mc_frame(self) -> torch.Tensor:
        """Lazy evaluation of the motion-compensated reference frame."""
        if self._mc_frame is None:
            if self.mvs is None:
                raise ValueError("Motion vectors must be evaluated before extracting mc_blocks")
            compensator = self.compensator if self.compensator is not None else DenseMC('gauss')
            self._mc_frame = compensator(self.ref_frame, self.mvs, self.bs)
        return self._mc_frame

    @property
    def mc_blocks(self) -> torch.Tensor:
        """Lazy evaluation of Motion-Compensated blocks, unfolded to the block grid."""
        if self._mc_blocks is None:
            self._mc_blocks = self.mc_frame.unfold(2, self.bs, self.bs).unfold(3, self.bs, self.bs).contiguous()
        return self._mc_blocks
    
    @property
    def residual(self) -> torch.Tensor:
        """Lazy evaluation of the motion-compensated spatial residual."""
        if self._residual is None:
            mc = self.mc_blocks
            curr_blocks = self.current_frame.unfold(2, self.bs, self.bs).unfold(3, self.bs, self.bs).contiguous()
            self._residual = curr_blocks - mc
        return self._residual

    @property
    def residual_frame(self) -> torch.Tensor:
        """Full-resolution motion-compensated residual [B, C, H, W]."""
        return self.current_frame - self.mc_frame
        

class EVCATemporalMetric(nn.Module):
    """Abstract Base Class for Temporal Plugins."""
    def forward(self, state: TemporalState) -> torch.Tensor:
        raise NotImplementedError

class MetricMVC(EVCATemporalMetric):
    """Calculates Motion Vector Field Complexity (Entropy)"""
    def __init__(self):
        super().__init__()
        # fixed Laplacian edge-detector kernel to find spatial variance in the MV field
        laplacian = torch.tensor([[[[0., 1., 0.],
                                    [1., -4., 1.],
                                    [0., 1., 0.]]]])
        self.register_buffer('laplacian_kernel', laplacian.repeat(2, 1, 1, 1))
    
    def forward(self, state: TemporalState) -> torch.Tensor:
        # F.conv2d requires [B, C, H, W] -> mvs are [B, 2, H_b, W_b]
        # Replicate-pad instead of zero-padding: border blocks would otherwise see
        # phantom zero-motion neighbors, producing spurious gradients on global motion.
        mvs_padded = F.pad(state.mvs, (1, 1, 1, 1), mode='replicate')
        mv_gradients = F.conv2d(mvs_padded, self.laplacian_kernel, groups=2)
        # average absolute spatial variance (chaos of the motion field)
        return torch.mean(torch.abs(mv_gradients), dim=[1, 2, 3])

class MetricsTCSAD(EVCATemporalMetric):
    """Outputs the minimum SAD cost of the residual."""
    def forward(self, state: TemporalState) -> torch.Tensor:
        return torch.mean(state.sad_map, dim=[1, 2, 3])

class EVCATemporalEngine(nn.Module):
    """Orchestrator for Motion Estimation, Motion Compensation and Metric Plugins."""
    def __init__(self, motion_estimator: nn.Module, metrics: Dict[str, EVCATemporalMetric],
                 compensator: MotionCompensator = None):
        super().__init__()
        self.me_module = motion_estimator
        self.metrics = nn.ModuleDict(metrics)
        self.compensator = compensator if compensator is not None else DenseMC('gauss')

    def forward(self, current_frame: torch.Tensor, ref_frame: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], TemporalState]:
        state = TemporalState(current_frame, ref_frame, bs=self.me_module.bs,
                              compensator=self.compensator)
        # 1. hardware-accelerated batched ME
        state.mvs, state.sad_map = self.me_module(current_frame, ref_frame)
        # 2. evaluate registered plugins dynamically
        results = {}
        for name, metric_module in self.metrics.items():
            results[name] = metric_module(state)
        
        return results, state


