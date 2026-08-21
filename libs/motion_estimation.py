"""Motion-estimation strategies.

Both estimators share the interface consumed by `EVCATemporalEngine`:

    mvs, sad_map = estimator(curr_frame, ref_frame)

where `mvs` is [B, 2, H_b, W_b] holding (dy, dx) in full-resolution pixels and
`sad_map` is [B, 1, H_b, W_b] holding the winning block cost, with the convention
`curr(y, x) ~= ref(y + dy, x + dx)`.

`HierarchicalBlockMatcher` walks a resolution pyramid: an exhaustive search at the
coarsest level fixes the large-displacement part, and each finer level upsamples the
field and corrects it by a small radius. All candidates of a level are evaluated by
stacking shifted references along the channel dimension and issuing a single
`avg_pool2d`, so the number of kernel launches is independent of the candidate count.
"""
import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Peak bytes allowed for the [b, K, H, W] candidate-difference tensor of one level.
# Larger frame batches are split into chunks that each stay under this budget; the
# candidate axis is never split, so every chunk is still one batched pooling op.
_SAD_BUDGET_BYTES = 512 * 1024 * 1024

# Transform size for the SATD criterion.
_HADAMARD_N = 8


def phase_correlation_gmv(curr: torch.Tensor, ref: torch.Tensor,
                          scale: int = 8) -> torch.Tensor:
    """Per-frame global translation (dy, dx) in full-resolution pixels.

    Normalised cross-power spectrum on a `1/scale` luma pyramid level: a pure
    translation is a linear phase ramp, whose inverse transform is a delta at the
    shift. Returns [B, 2] in the same convention as the MV field.
    """
    small_c = F.avg_pool2d(curr, kernel_size=scale, stride=scale)
    small_r = F.avg_pool2d(ref, kernel_size=scale, stride=scale)
    B, _, h, w = small_c.shape

    # Hann window suppresses the wrap-around edge discontinuity that would otherwise
    # put a strong spurious peak at zero shift.
    win = (torch.hann_window(h, device=curr.device).unsqueeze(1)
           * torch.hann_window(w, device=curr.device).unsqueeze(0))
    Fc = torch.fft.rfft2(small_c.squeeze(1) * win)
    Fr = torch.fft.rfft2(small_r.squeeze(1) * win)
    cross = Fc * Fr.conj()
    cross = cross / (cross.abs() + 1e-8)
    corr = torch.fft.irfft2(cross, s=(h, w))

    flat = corr.reshape(B, -1).argmax(dim=1)
    peak_y = torch.div(flat, w, rounding_mode='floor')
    peak_x = flat % w
    # Peaks past the midpoint are negative shifts (circular correlation).
    peak_y = torch.where(peak_y > h // 2, peak_y - h, peak_y)
    peak_x = torch.where(peak_x > w // 2, peak_x - w, peak_x)
    # irfft2 of Fc * conj(Fr) peaks at (curr - ref); the MV convention is ref - curr.
    return torch.stack((-peak_y, -peak_x), dim=1).float() * scale


def hadamard_matrix(n: int, device: torch.device) -> torch.Tensor:
    """Sylvester-construction Hadamard matrix (n a power of two), entries +/-1."""
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1),
                       torch.cat([h, -h], dim=1)], dim=0)
    return h


def satd_blocks(diff: torch.Tensor, block: int, hadamard: int = 8) -> torch.Tensor:
    """Block-mean SATD: sum of absolute 8x8 Hadamard-transformed differences.

    SATD approximates transform-domain rate better than SAD, because it charges for
    how the residual is distributed across frequencies rather than only its size.
    `diff` is [B, K, H, W]; the result is [B, K, H_b, W_b].
    """
    B, K, H, W = diff.shape
    n = hadamard
    hm = hadamard_matrix(n, diff.device)
    # Tile into n x n sub-blocks: [B, K, H/n, n, W/n, n] -> [..., n, n]
    tiles = diff.view(B, K, H // n, n, W // n, n).permute(0, 1, 2, 4, 3, 5)
    transformed = hm @ tiles @ hm
    # Mean |coefficient| per n x n tile, then average over the tiles of a block.
    per_tile = transformed.abs().mean(dim=(-2, -1)) / n
    return F.avg_pool2d(per_tile, kernel_size=block // n, stride=block // n)


def _offsets_square(radius: int) -> List[Tuple[int, int]]:
    """Every integer offset in the (2r+1)^2 square, centre first."""
    offs = [(0, 0)]
    offs += [(dy, dx)
             for dy in range(-radius, radius + 1)
             for dx in range(-radius, radius + 1)
             if (dy, dx) != (0, 0)]
    return offs


def batched_block_cost(curr: torch.Tensor, ref: torch.Tensor,
                       offsets: Sequence[Tuple[int, int]], block: int,
                       criterion: str = 'sad') -> torch.Tensor:
    """Block matching cost for every candidate offset.

    `curr`/`ref` are [B, 1, H, W]; the result is [B, K, H_b, W_b]. The shifted
    references are stacked on the channel axis so a single pooling op reduces all
    candidates at once. The frame batch is split into chunks that keep the
    [b, K, H, W] intermediates inside `_SAD_BUDGET_BYTES`; the candidate axis is never
    split, so each chunk remains one batched op.

    `criterion='satd'` uses an 8x8 Hadamard transform of the residual. It needs blocks
    of at least 8 samples, so coarse pyramid levels whose block is smaller fall back
    to SAD.
    """
    B, _, H, W = curr.shape
    K = len(offsets)
    reach = max(max(abs(dy), abs(dx)) for dy, dx in offsets)
    ref_padded = F.pad(ref, (reach, reach, reach, reach), mode='replicate')
    use_satd = (criterion == 'satd' and block % _HADAMARD_N == 0
                and block >= _HADAMARD_N)

    # Two K-sized intermediates are alive at once (the stack and the difference).
    chunk = max(1, int(_SAD_BUDGET_BYTES // max(1, 2 * K * H * W * 4)))
    out = []
    for start in range(0, B, chunk):
        stop = min(B, start + chunk)
        shifted = torch.cat(
            [ref_padded[start:stop, :, reach + dy:reach + dy + H,
                        reach + dx:reach + dx + W] for dy, dx in offsets], dim=1)
        diff = curr[start:stop] - shifted
        if use_satd:
            out.append(satd_blocks(diff, block, _HADAMARD_N))
        else:
            out.append(F.avg_pool2d(diff.abs_(), kernel_size=block, stride=block))
    return torch.cat(out, dim=0)


# Backwards-compatible alias for the SAD-only entry point.
def batched_block_sad(curr, ref, offsets, block):
    return batched_block_cost(curr, ref, offsets, block, 'sad')


def median_predictor(mvs: torch.Tensor) -> torch.Tensor:
    """Component-wise median of each block's 3x3 MV neighbourhood.

    Used as the MV-cost reference. The neighbourhood includes the block itself, so a
    locally uniform field costs nothing and the penalty only bites where a block
    disagrees with its surroundings.
    """
    B, C, Hb, Wb = mvs.shape
    padded = F.pad(mvs, (1, 1, 1, 1), mode='replicate')
    patches = F.unfold(padded, kernel_size=3).view(B, C, 9, Hb, Wb)
    return patches.median(dim=2).values


def _mv_cost(candidate_mvs: torch.Tensor, predictor: torch.Tensor,
             lam: float) -> torch.Tensor:
    """lambda * L1 distance between each candidate vector and the median predictor."""
    return lam * (candidate_mvs - predictor.unsqueeze(1)).abs().sum(dim=2)


class SparsePatternBlockMatcher(nn.Module):
    """
    Sparse Pattern Block Matcher (The 'Fixed Diamond').
    Evaluates a static, deterministic diamond of motion vectors to achieve
    fast O(1) search complexity while preserving highly accurate heuristics.
    """
    def __init__(self, block_size: int = 32, heuristic: str = 'diamond', dilation: int = 1):
        super().__init__()
        self.bs = block_size
        self.bs_c = block_size // 2     # coarse block size

        if heuristic == 'diamond':
        # 1.a 13-point Large Diamond Pattern (Half-Resolution Offsets)
        # The search runs on a 2x2-pooled image, so offsets are half-res and the
        # effective full-res search radius is +/- 6 pixels (even-valued MVs only).
            base_pattern = [
                (0, 0),
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-2, 0), (2, 0), (0, -2), (0, 2),
                (-3, 0), (3, 0), (0, -3), (0, 3)
            ]
        elif heuristic == 'square':
        # 1.b 9-point Sparse Square (Radius = 2 at half-res -> Effective +/- 4 pixels)
        # Captures center, cross, and extreme diagonals.
            base_pattern = [
                (0, 0),
                (-2, 0), (2, 0), (0, -2), (0, 2),    # Cardinal directions
                (-2, -2), (-2, 2), (2, -2), (2, 2)   # The Diagonals (1,1), (-1,-1), etc.
            ]
        else:
            raise ValueError(f"unknown heuristic pattern: {heuristic}")

        # 1.c apply resolution-aware dilation: multiply offsets by dilation factor to
        # stretch the search horizon for large resolutions
        self.pattern = [(dy * dilation, dx * dilation) for dy, dx in base_pattern]
        self.num_cands = len(self.pattern)

        # Calculate padding dynamically based on the pattern's maximum reach
        self.R_c = max(max(abs(dy), abs(dx)) for dy, dx in self.pattern)

        # Maximum decodable full-resolution MV magnitude (L-inf). A block whose MV
        # reaches this bound sits on the search-pattern boundary (used for MV_sat_frac).
        self.max_reach_fullres = float(2 * self.R_c)

        # 2. Vectorized O(1) Lookup Table for Coordinate Decoding
        # The search runs at half resolution (2x2 avg_pool), so we pre-multiply by 2.0
        # to decode candidate offsets into full-resolution (even-valued) vectors.
        # register_buffer ensures this tensor automatically moves to MPS/CUDA alongside the model.
        lookup_tensor = torch.tensor(self.pattern, dtype=torch.float32) * 2.0
        self.register_buffer('pattern_lookup', lookup_tensor)

    def forward(self, curr_frame: torch.Tensor, ref_frame: torch.Tensor):
        B, C, H, W = curr_frame.shape
        H_b, W_b = H // self.bs, W // self.bs

        # =====================================================================
        # COARSE SEARCH: Shift-and-Pool over the Fixed Diamond
        # =====================================================================
        curr_c = F.avg_pool2d(curr_frame, kernel_size=2, stride=2)
        ref_c = F.avg_pool2d(ref_frame, kernel_size=2, stride=2)

        ref_c_padded = F.pad(ref_c, (self.R_c, self.R_c, self.R_c, self.R_c), mode='replicate')

        # Pre-allocate SAD tensor for exactly 13 candidates instead of 49
        sads = torch.empty((B, self.num_cands, H_b, W_b), device=curr_frame.device, dtype=curr_frame.dtype)

        for idx, (dy, dx) in enumerate(self.pattern):
            # Zero-copy tensor slice
            ref_slice = ref_c_padded[:, :, self.R_c+dy : (H//2)+self.R_c+dy, self.R_c+dx : (W//2)+self.R_c+dx]

            # Hardware-accelerated block SAD
            abs_diff = torch.abs(curr_c - ref_slice)
            sads[:, idx] = F.avg_pool2d(abs_diff, kernel_size=self.bs_c, stride=self.bs_c).squeeze(1)

        # Global Hardware Reduction
        # best_idx is a 3D tensor of shape [B, H_b, W_b] containing values 0-12
        best_sad, best_idx = torch.min(sads, dim=1)

        # =====================================================================
        # COORDINATE DECODING: Vectorized Advanced Indexing
        # =====================================================================
        # We pass the entire batch's index tensor into the 2D lookup table.
        # PyTorch advanced indexing automatically expands this into shape [B, H_b, W_b, 2]
        decoded_mvs = self.pattern_lookup[best_idx]

        # Split the vectors and reshape to [B, 1, H_b, W_b] to match EVCA plugin formats
        best_dy = decoded_mvs[..., 0].unsqueeze(1)
        best_dx = decoded_mvs[..., 1].unsqueeze(1)

        best_mv = torch.cat([best_dy, best_dx], dim=1)

        return best_mv, best_sad.unsqueeze(1)


class HierarchicalBlockMatcher(nn.Module):
    """Pyramid search: exhaustive at the coarsest level, +/- refine_radius per level.

    Levels are powers-of-two downscales of the luma plane. The coarse level is searched
    exhaustively within `coarse_radius` (in that level's pixels), which sets the total
    reach at `coarse_radius * coarsest_scale` full-resolution pixels. Each finer level
    doubles the field and corrects it within `refine_radius`, so the reachable
    correction around the coarse estimate is `sum(refine_radius * scale)` and the final
    vectors are integer-pel rather than the even-only vectors of the sparse pattern.
    """

    def __init__(self, block_size: int = 32, width: int = 1920,
                 coarse_radius: int = 8, refine_radius: int = 1,
                 max_range: int = None, subpel: int = 0,
                 predictor: str = 'none', mv_lambda: float = 0.0,
                 merge: bool = False, criterion: str = 'sad'):
        super().__init__()
        self.bs = block_size
        self.refine_radius = refine_radius
        self.subpel = subpel
        self.predictor = predictor
        self.mv_lambda = mv_lambda
        self.merge = merge
        self.criterion = criterion
        self.last_gmv = None        # [B, 2], populated when predictor == 'global'

        # Pyramid at 1/4, 1/2 and full resolution; 4K and wider get a 1/8 level so the
        # coarse search still reaches far enough without an enormous candidate count.
        self.scales = [8, 4, 2, 1] if width >= 3840 else [4, 2, 1]
        self.coarse_scale = self.scales[0]

        if max_range is not None:
            coarse_radius = max(1, int(math.ceil(max_range / self.coarse_scale)))
        self.coarse_radius = coarse_radius

        self.coarse_offsets = _offsets_square(coarse_radius)
        self.refine_offsets = _offsets_square(refine_radius)

        # Full-resolution reach of the exhaustive coarse stage, plus what the
        # refinement levels can add on top.
        refine_reach = sum(refine_radius * s for s in self.scales[1:])
        self.max_reach_fullres = float(coarse_radius * self.coarse_scale + refine_reach)

        self.register_buffer('coarse_lookup',
                             torch.tensor(self.coarse_offsets, dtype=torch.float32))
        self.register_buffer('refine_lookup',
                             torch.tensor(self.refine_offsets, dtype=torch.float32))

    def _level_inputs(self, frame: torch.Tensor, scale: int, Hb: int, Wb: int) -> torch.Tensor:
        """Downscales to `scale` and crops to a whole number of blocks.

        Every level must produce the same H_b x W_b block grid, otherwise the upsampled
        field from the coarser level lands on the wrong blocks. Cropping to
        `Hb * (bs // scale)` guarantees that even when the frame height is not a
        multiple of the block size.
        """
        if scale != 1:
            frame = F.avg_pool2d(frame, kernel_size=scale, stride=scale)
        block_l = max(1, self.bs // scale)
        return frame[:, :, :Hb * block_l, :Wb * block_l]

    def forward(self, curr_frame: torch.Tensor, ref_frame: torch.Tensor):
        from libs.motion_compensation import warp

        H, W = curr_frame.shape[-2:]
        Hb, Wb = H // self.bs, W // self.bs

        gmv = None
        if self.predictor == 'global':
            # One translation per frame, in full-resolution pixels.
            gmv = phase_correlation_gmv(curr_frame, ref_frame, scale=8)
            self.last_gmv = gmv

        mvs = None          # [B, 2, Hb, Wb] in *current level* pixel units
        cost = None
        for scale in self.scales:
            curr_l = self._level_inputs(curr_frame, scale, Hb, Wb)
            ref_l = self._level_inputs(ref_frame, scale, Hb, Wb)
            block_l = max(1, self.bs // scale)

            if mvs is None:
                # Coarsest level: exhaustive search about the origin.
                costs = batched_block_cost(curr_l, ref_l, self.coarse_offsets,
                                           block_l, self.criterion)
                cost, best = torch.min(costs, dim=1)
                mvs = self.coarse_lookup[best].permute(0, 3, 1, 2).contiguous()

                if gmv is not None:
                    # Same pattern re-centred on the global vector: warp by the GMV,
                    # search around it, then keep whichever centre won per block.
                    gmv_l = (gmv / scale).view(-1, 2, 1, 1).expand(-1, -1, Hb, Wb)
                    H_l, W_l = curr_l.shape[-2:]
                    ref_g = warp(ref_l, F.interpolate(gmv_l, size=(H_l, W_l),
                                                      mode='nearest'))
                    costs_g = batched_block_cost(curr_l, ref_g, self.coarse_offsets,
                                                 block_l, self.criterion)
                    cost_g, best_g = torch.min(costs_g, dim=1)
                    mvs_g = (gmv_l
                             + self.coarse_lookup[best_g].permute(0, 3, 1, 2))
                    take_g = (cost_g < cost).unsqueeze(1)
                    mvs = torch.where(take_g, mvs_g, mvs)
                    cost = torch.minimum(cost, cost_g)
            else:
                # Finer level: the field doubles along with the resolution. Rather than
                # searching each block around its own predictor, pre-warp the reference
                # by the upsampled field once; the remaining search is then a plain
                # origin-centred one and reuses the cheap stack-and-pool path. Exact for
                # integer vectors, since bilinear sampling at integer coordinates is a copy.
                mvs = mvs * 2.0
                H_l, W_l = curr_l.shape[-2:]
                pixel_mvs = F.interpolate(mvs, size=(H_l, W_l), mode='nearest')
                ref_pred = warp(ref_l, pixel_mvs)
                costs = batched_block_cost(curr_l, ref_pred, self.refine_offsets,
                                           block_l, self.criterion)

                if self.mv_lambda > 0:
                    # Bias ambiguous blocks toward the local consensus vector. Rate cost
                    # is charged in this level's units so lambda means the same thing
                    # at every scale.
                    pred = median_predictor(mvs)
                    cand = (mvs.unsqueeze(1)
                            + self.refine_lookup.view(1, -1, 2, 1, 1))
                    costs = costs + _mv_cost(cand, pred, self.mv_lambda)

                cost, best = torch.min(costs, dim=1)
                delta = self.refine_lookup[best].permute(0, 3, 1, 2).contiguous()
                mvs = mvs + delta

                if self.merge and scale == 1:
                    mvs, cost = self._merge_pass(curr_l, ref_l, mvs, block_l)

        for step in range(self.subpel):
            # Half-pel, then quarter-pel. Each stage searches the 8 neighbours at the
            # current fraction around the integer winner, on a reference pre-upsampled
            # by that factor, where the fractional offsets become integer ones.
            mvs, cost = self._refine_subpel(curr_frame, ref_frame, mvs, 2 ** (step + 1),
                                            Hb, Wb)

        return mvs, cost.unsqueeze(1)

    def _merge_pass(self, curr_l: torch.Tensor, ref_l: torch.Tensor,
                    mvs: torch.Tensor, block_l: int):
        """Re-evaluates each block against its four neighbours' vectors.

        A block whose own winner barely beat a neighbour's vector is better off
        adopting the neighbour's: it costs no extra rate to signal in a real encoder
        and yields a smoother field. Ties keep the block's own vector.
        """
        from libs.motion_compensation import warp

        H_l, W_l = curr_l.shape[-2:]
        padded = F.pad(mvs, (1, 1, 1, 1), mode='replicate')
        Hb, Wb = mvs.shape[-2:]
        candidates = [
            mvs,
            padded[:, :, 0:Hb, 1:1 + Wb],       # up
            padded[:, :, 2:2 + Hb, 1:1 + Wb],   # down
            padded[:, :, 1:1 + Hb, 0:Wb],       # left
            padded[:, :, 1:1 + Hb, 2:2 + Wb],   # right
        ]
        best_mv, best_cost = None, None
        for cand in candidates:
            pred = warp(ref_l, F.interpolate(cand, size=(H_l, W_l), mode='nearest'))
            diff = curr_l - pred
            if self.criterion == 'satd' and block_l % _HADAMARD_N == 0:
                c = satd_blocks(diff, block_l, _HADAMARD_N).squeeze(1)
            else:
                c = F.avg_pool2d(diff.abs(), block_l, block_l).squeeze(1)
            if best_cost is None:
                best_mv, best_cost = cand, c
            else:
                take = (c < best_cost).unsqueeze(1)
                best_mv = torch.where(take, cand, best_mv)
                best_cost = torch.minimum(best_cost, c)
        return best_mv.contiguous(), best_cost

    def _refine_subpel(self, curr_frame: torch.Tensor, ref_frame: torch.Tensor,
                       mvs: torch.Tensor, factor: int, Hb: int, Wb: int):
        """Refines `mvs` to 1/`factor`-pel precision.

        Each of the 9 neighbours at the current fraction is sampled directly from the
        full-resolution reference through `grid_sample`, whose bilinear interpolation
        is exactly the sub-pel filter. Pre-interpolating the reference by `factor`
        would be equivalent but allocates factor^2 times the frame: at 1080p with
        quarter-pel that is ~17 GB for a 32-frame GOP, whereas this path costs the same
        as an integer refinement level.
        """
        from libs.motion_compensation import warp

        curr = curr_frame[:, :, :Hb * self.bs, :Wb * self.bs]
        ref = ref_frame[:, :, :Hb * self.bs, :Wb * self.bs]
        B, _, H, W = curr.shape
        step = 1.0 / factor
        K = len(self.refine_offsets)

        chunk = max(1, int(_SAD_BUDGET_BYTES // max(1, 2 * K * H * W * 4)))
        costs = []
        for start in range(0, B, chunk):
            stop = min(B, start + chunk)
            mv_chunk = mvs[start:stop]
            preds = []
            for dy, dx in self.refine_offsets:
                cand = mv_chunk.clone()
                cand[:, 0] += dy * step
                cand[:, 1] += dx * step
                preds.append(warp(ref[start:stop],
                                  F.interpolate(cand, size=(H, W), mode='nearest')))
            diff = curr[start:stop] - torch.cat(preds, dim=1)
            if self.criterion == 'satd' and self.bs % _HADAMARD_N == 0:
                costs.append(satd_blocks(diff, self.bs, _HADAMARD_N))
            else:
                costs.append(F.avg_pool2d(diff.abs_(), self.bs, self.bs))
        costs = torch.cat(costs, dim=0)

        cost, best = torch.min(costs, dim=1)
        delta = self.refine_lookup[best].permute(0, 3, 1, 2).contiguous() * step
        return mvs + delta, cost


def build_motion_estimator(args, width: int) -> nn.Module:
    """Constructs the motion estimator selected by the `--me*` flags.

    Options accepted by the parser but not yet implemented raise here rather than
    silently degrading to the default search, so an ablation can never report a
    variant it did not actually run.
    """
    # The refinement flags are properties of the pyramid search; the sparse pattern
    # has no refinement stage to attach them to.
    hierarchical_only = []
    if args.me_subpel != 0:
        hierarchical_only.append(f'--me-subpel {args.me_subpel}')
    if args.me_predictor != 'none':
        hierarchical_only.append(f'--me-predictor {args.me_predictor}')
    if args.me_lambda != 0.0:
        hierarchical_only.append(f'--me-lambda {args.me_lambda}')
    if args.me_merge:
        hierarchical_only.append('--me-merge')
    if args.me_criterion != 'sad':
        hierarchical_only.append(f'--me-criterion {args.me_criterion}')
    if hierarchical_only and args.me != 'hierarchical':
        raise NotImplementedError(
            'these flags require --me hierarchical: ' + ', '.join(hierarchical_only))

    if args.me == 'hierarchical':
        return HierarchicalBlockMatcher(
            block_size=args.block_size,
            width=width,
            coarse_radius=getattr(args, 'me_coarse_radius', 8),
            refine_radius=getattr(args, 'me_refine_radius', 1),
            subpel=args.me_subpel,
            predictor=args.me_predictor,
            mv_lambda=args.me_lambda,
            merge=args.me_merge,
            criterion=args.me_criterion,
        )

    dilation_factor = max(1, width // 1920)  # 1080p has multiplier of 1
    return SparsePatternBlockMatcher(
        block_size=args.block_size,
        heuristic=args.heuristic,
        dilation=dilation_factor,
    )
