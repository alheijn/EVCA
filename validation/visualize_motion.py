"""Visualise the motion-vector field the `-me` search finds for one frame pair.

    python validation/visualize_motion.py -i seq.yuv -r 1920x1080 --frame 30

The `-me` path exports four scalars per frame (`MVC`, `TC_SAD`, `mean_mv_mag`,
`intra_frac`) and, with `-bi`, a grayscale map of MV *magnitude*. Direction, which
candidate the search picked, and how close the decision was are all invisible. That
matters here because `PatternBlockMatcher` is deliberately coarse: with the default
`diamond` / `--me-offset 2` every block picks one of five vectors, and every non-zero
one has magnitude exactly 2. Such a field can look plausible in aggregate while being
close to meaningless per block.

So this renders one pair F(t) -> F(t+1) as four figures, ordered as the pipeline is,
each answering one question, plus a per-block drill-down:

    1 evidence      what actually moved          frames, raw difference, anaglyph
    2 field         what the search decided      quiver, flow colour, winning candidate
    3 confidence    how much to trust it         SAD, decision margin, MVC integrand
    4 compensation  what compensation bought     warped reference, residual, gain
    5 blocks        why *this* block decided     patch quartet + the SAD cost surface

Everything is computed by the real pipeline, not reimplemented: the CLI is `main.py`'s
own parser with a `visualization` group bolted on, frames come from `libs.video_loader`,
and the search and warp are `libs.EVCA.build_motion_estimator` /
`libs.motion_compensation.build_compensator` driven by `EVCATemporalEngine`. So every
analysis flag (`--heuristic`, `--me-offset`, `--temporal-pool`, `--mc`, ...) means here
exactly what it means to `main.py`, and what you look at is what gets measured.

MV sign convention, stated on every figure that draws direction: the estimate (dy, dx)
of a block satisfies `curr(y, x) ~= ref(y + dy, x + dx)`, i.e. it points *backwards*
into the reference. Apparent forward motion of the content is therefore `-(dy, dx)`,
and that is what the arrows and the flow colours show.

Run `--synthetic translation:0,4` first: it renders the same figures on a pan whose
answer is known and prints the endpoint error, which is how you confirm the tool's own
signs and alignment before trusting it on real footage.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
# Headless unless the user asked for a window. Selecting the backend after pyplot is
# imported does not reliably take effect, so the decision is made from argv here.
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle

import torch
import torch.nn.functional as F

from main import build_parser, select_device
from libs.EVCA import build_motion_estimator, resolve_pools
from libs.motion_compensation import BlockMC, DenseMC, OBMC, build_compensator, smooth_mv_field
from libs.temporal_engine import EVCATemporalEngine, MetricMVC, MetricsTCSAD
from libs.video_loader import load_gop

DEFAULT_OUT_DIR = PROJECT_ROOT / 'png' / 'motion'
FIGURE_NAMES = ('evidence', 'field', 'confidence', 'compensation', 'blocks')
# --synthetic exists to check the tool against known motion, not to synthesise footage.
# Without a cap, `--frame 900` quietly generates 900 frames -- and for a translation
# that also widens the canvas by 900 * |v| pixels before rendering a single figure.
SYNTHETIC_MAX_FRAMES = 64


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_pair(text: str, name: str, count: int = 2) -> Tuple[int, ...]:
    """`"a,b"` -> (a, b). Used for --roi, --block and the synthetic velocity."""
    parts = [p for p in text.replace(' ', '').split(',') if p]
    if len(parts) != count:
        raise argparse.ArgumentTypeError(f'{name} needs {count} comma-separated integers, '
                                         f'got {text!r}')
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{name} takes integers, got {text!r}')


def frame_selection(text: str) -> Tuple[int, int]:
    """`"30"` -> (30, 30); `"30-45"` -> (30, 45). Inclusive of both ends."""
    text = text.strip()
    if '-' in text.lstrip('-'):
        lo, _, hi = text.lstrip().partition('-')
        try:
            lo_i, hi_i = int(lo), int(hi)
        except ValueError:
            raise argparse.ArgumentTypeError(f'--frame takes T or T1-T2, got {text!r}')
        if hi_i < lo_i:
            raise argparse.ArgumentTypeError(f'--frame range is empty: {text!r}')
        return lo_i, hi_i
    try:
        t = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f'--frame takes T or T1-T2, got {text!r}')
    return t, t


def roi_arg(text: str) -> Tuple[int, int, int, int]:
    return parse_pair(text, '--roi', count=4)


def block_arg(text: str) -> Tuple[int, int]:
    return parse_pair(text, '--block', count=2)


def build_viz_parser() -> argparse.ArgumentParser:
    """`main.py`'s parser plus a visualization group.

    Extending the real parser rather than declaring a subset is the point: every
    analysis flag keeps its name, default and validation, so a figure can be reproduced
    by pasting the same flags into `main.py` and the reverse.
    """
    parser = build_parser()
    parser.prog = 'validation/visualize_motion.py'
    parser.description = ('Visualise the motion-vector field for one frame pair '
                          'F(t) -> F(t+1). Accepts every analysis flag main.py does.')
    parser.epilog = (
        'examples:\n'
        '  # sanity-check the tool itself against a known 4 px pan\n'
        '  python validation/visualize_motion.py --synthetic translation:0,4 --me-offset 4\n'
        '\n'
        '  # one pair of a real sequence, with the four worst blocks drilled into\n'
        '  python validation/visualize_motion.py -i seq.yuv -r 1920x1080 --frame 30 \\\n'
        '      --worst sad 4\n'
        '\n'
        '  # a wide grid search under pooling, cropped to one region\n'
        '  python validation/visualize_motion.py -i seq.yuv -r 1920x1080 --frame 30 \\\n'
        '      --heuristic grid --me-offset 8 --temporal-pool 2 --roi 300,600,256,384\n'
        '\n'
        'MV convention: curr(y, x) ~= ref(y + dy, x + dx), so vectors point backwards\n'
        'into the reference. Arrows and flow colours are drawn as -(dy, dx), the\n'
        'apparent forward motion of the content.\n')

    viz = parser.add_argument_group(
        'visualization',
        'The pair is (reference = F(t), current = F(t+1)); with -s/--sample_rate N the\n'
        'partner is F(t+N), matching what the analyzer compares. EVCA reports that\n'
        "pair on CSV row t+1, which the tool prints so numbers can be cross-checked.")
    viz.add_argument('--frame', type=frame_selection, default=(0, 0), metavar='T[-T2]',
                     help='base frame F(t), or an inclusive range to render in turn')
    viz.add_argument('--out', default=str(DEFAULT_OUT_DIR), metavar='DIR',
                     help='directory for the rendered figures')
    viz.add_argument('--show', action='store_true',
                     help='open the figures in a window instead of only writing them')
    viz.add_argument('--figures', default='all', metavar='LIST',
                     help='comma-separated subset of '
                          f'{",".join(FIGURE_NAMES)}, or "all"')
    viz.add_argument('--roi', type=roi_arg, default=None, metavar='Y,X,H,W',
                     help='crop every spatial panel to this region, in full-resolution '
                          'pixels; snapped outwards to block boundaries')
    viz.add_argument('--quiver-stride', dest='quiver_stride', type=int, default=0,
                     metavar='N',
                     help='draw every Nth block arrow; 0 picks a stride that keeps the '
                          'field readable at the current size')
    viz.add_argument('--arrow-scale', dest='arrow_scale', type=float, default=0.0,
                     metavar='X',
                     help='arrow length multiplier; 0 auto-scales so the pattern reach '
                          'spans half a block, since a 2 px vector on a 32 px block is '
                          'otherwise invisible')
    viz.add_argument('--grid', dest='block_grid', action='store_true',
                     help='overlay the block grid on spatial panels (automatic when the '
                          'view is small enough for it to stay legible)')
    viz.add_argument('--block', type=block_arg, action='append', default=None,
                     metavar='BY,BX',
                     help='drill into the block at this grid position; repeatable')
    viz.add_argument('--worst', nargs=2, default=None, metavar=('METRIC', 'K'),
                     help='drill into the K worst blocks by METRIC: sad (worst '
                          'prediction), margin (least decisive choice) or gain (where '
                          'compensation helped least)')
    viz.add_argument('--check-csv', dest='check_csv', default=None, metavar='FILE',
                     help='cross-check the recomputed MVC / TC_SAD / mean_mv_mag against '
                          'row t+1 of a CSV written by a matching main.py run')
    viz.add_argument('--synthetic', default=None, metavar='SPEC',
                     help='ignore --input and use a generated sequence with known '
                          'ground truth: translation:VY,VX (camera pan over texture), '
                          'square:VY,VX (a red square moving over black, where the '
                          'expected vector is the negated velocity), static_noise, '
                          'or cut')
    viz.add_argument('--synthetic-size', dest='synthetic_size', default='384x256',
                     metavar='WxH', help='resolution of the --synthetic sequence')
    return parser


# --------------------------------------------------------------------------------------
# Scene assembly
# --------------------------------------------------------------------------------------

@dataclass
class Geometry:
    width: int
    height: int
    pix_size: float
    luma_size: int
    chroma_size: int
    uv_w: int
    uv_h: int
    cb_size: int


def frame_geometry(args: argparse.Namespace) -> Geometry:
    """Byte layout of the input, mirroring the arithmetic at the top of `libs.EVCA.EVCA`.

    Kept local rather than imported because `EVCA()` computes it inline; if it ever
    moves into a helper, this should call that helper instead.
    """
    width, height = map(int, args.resolution.split('x'))
    bytes_per_sample = 1 if args.bit_depth == 8 else 2
    if args.pix_fmt == 'yuv420':
        pix_size = 1.5 * bytes_per_sample
        uv_w, uv_h = width // 2, height // 2
        cb_size = args.block_size // 2
    else:
        pix_size = 3.0 * bytes_per_sample
        uv_w, uv_h = width, height
        cb_size = args.block_size
    return Geometry(width, height, pix_size, width * height, uv_w * uv_h,
                    uv_w, uv_h, cb_size)


def frame_count(args: argparse.Namespace, geo: Geometry) -> int:
    return int(Path(args.input).stat().st_size // (geo.width * geo.height * geo.pix_size))


def load_frame_pair(args: argparse.Namespace, t: int, device) -> torch.Tensor:
    """(reference, current) = (F(t), F(t + sample_rate)) as one [2, 1, H, W] batch.

    Goes through the analyzer's own `load_gop`, so the pixels shown are byte-identical
    to the pixels measured -- including the crop to whole blocks, which at 1080p and
    --block_size 32 drops the frame to 1056 rows.
    """
    geo = frame_geometry(args)
    stream = open(args.input, 'rb')
    try:
        # arange(t, t + s + 1, s) is exactly [t, t + s] for any s >= 1.
        *_, frames = load_gop(args, stream, t, t + args.sample_rate + 1, device,
                              geo.width, geo.height, geo.pix_size, geo.luma_size,
                              geo.chroma_size, geo.uv_w, geo.uv_h, geo.cb_size)
    finally:
        stream.close()
    if frames.shape[0] != 2:
        raise SystemExit(f'frame {t} (+{args.sample_rate}) is out of range for this file')
    return frames


def effective_pixel_field(compensator, mvs: torch.Tensor,
                          size: Tuple[int, int]) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """The pixel-grid MV field the compensator really warps with, in full-resolution px.

    Not the same object as the block field the search emitted, which is the point of
    showing it: `DenseMC` filters and bilinearly upsamples, `BlockMC` replicates, and
    `OBMC` has no single field at all -- it blends five warps -- so its own-block field
    is returned with a note saying so. The library's own `smooth_mv_field` does the
    filtering, so only the one interpolate call is restated here.
    """
    if isinstance(compensator, DenseMC):
        field = smooth_mv_field(mvs, compensator.smooth)
        return F.interpolate(field, size=size, mode='bilinear', align_corners=False), None
    if isinstance(compensator, BlockMC):
        return F.interpolate(mvs.float(), size=size, mode='nearest'), None
    if isinstance(compensator, OBMC):
        return (F.interpolate(mvs.float(), size=size, mode='nearest'),
                'own-block field; OBMC blends five warps, so no single field is applied')
    return None, f'no pixel field defined for {type(compensator).__name__}'


@dataclass
class MotionScene:
    """Everything one frame pair produced, on the CPU, in the units each panel needs."""
    args: argparse.Namespace
    label: str
    t_ref: int
    t_curr: int
    csv_row: int
    # geometry (full-resolution, after the block-multiple crop)
    H: int
    W: int
    bs: int
    Hb: int
    Wb: int
    pool: int              # residual-domain pooling factor (--temporal-pool)
    me_pool: int           # search pooling factor (--me-pool)
    reach_px: int          # largest vector this search can report, in full-res px
    pattern: np.ndarray    # [N, 2] candidate (dy, dx), full-resolution px
    # full-resolution planes
    ref: np.ndarray        # [H, W]
    curr: np.ndarray       # [H, W]
    pixel_mv: Optional[np.ndarray]   # [2, H, W] the field actually warped, full-res px
    pixel_mv_note: Optional[str]
    # block-resolution fields (always H/bs x W/bs, whatever the pooling)
    mv: np.ndarray         # [2, Hb, Wb], full-resolution px
    sad: np.ndarray        # [Hb, Wb], the winning cost
    costs: np.ndarray      # [N, Hb, Wb], the whole cost volume the search reduced over
    best_idx: np.ndarray   # [Hb, Wb], winning pattern index
    mvc_map: np.ndarray    # [Hb, Wb], |Laplacian of the MV field|, what MVC averages
    # residual domain (pooled by `pool`; identical to full resolution when pool == 1)
    ref_r: np.ndarray
    curr_r: np.ndarray
    mc_r: np.ndarray
    bs_r: int
    metrics: dict
    gt: Optional[Tuple[float, float]] = None   # ground-truth (dy, dx), --synthetic only

    # -- derived quantities ------------------------------------------------------------

    @property
    def mv_mag(self) -> np.ndarray:
        return np.hypot(self.mv[0], self.mv[1])

    @property
    def apparent(self) -> np.ndarray:
        """[2, Hb, Wb] apparent forward motion of the content, i.e. -(dy, dx)."""
        return -self.mv

    @property
    def margin(self) -> np.ndarray:
        """Second-best cost minus best cost: how decisive each block's choice was.

        A block whose margin is ~0 chose between candidates it could not tell apart, so
        its vector carries no information however small its SAD is.
        """
        if self.costs.shape[0] < 2:
            return np.zeros_like(self.sad)
        two = np.partition(self.costs, 1, axis=0)[:2]
        return two[1] - two[0]

    @property
    def clipped(self) -> np.ndarray:
        """Blocks whose winner sits on the pattern boundary, so true motion may exceed it."""
        return (np.abs(self.mv[0]) >= self.reach_px) | (np.abs(self.mv[1]) >= self.reach_px)

    @property
    def diff_r(self) -> np.ndarray:
        """Uncompensated frame difference, in the residual domain."""
        return self.curr_r - self.ref_r

    @property
    def resid_r(self) -> np.ndarray:
        """Motion-compensated residual, in the residual domain."""
        return self.curr_r - self.mc_r

    def block_mean_abs(self, plane_r: np.ndarray) -> np.ndarray:
        """Mean |value| per block of a residual-domain plane, on the MV grid."""
        b = self.bs_r
        return np.abs(plane_r).reshape(self.Hb, b, self.Wb, b).mean(axis=(1, 3))

    @property
    def gain(self) -> np.ndarray:
        """Per-block mean|difference| - mean|residual|: what compensation bought.

        Positive where the warp predicted better than standing still, negative where it
        predicted worse -- which happens at occlusions and wherever the smoothed field
        drags a block off its own motion.
        """
        return self.block_mean_abs(self.diff_r) - self.block_mean_abs(self.resid_r)


def build_scene(args: argparse.Namespace, t: int, device) -> MotionScene:
    """Runs the real analysis pipeline on one pair and captures every intermediate."""
    frames = load_frame_pair(args, t, device)
    curr_frames, ref_frames = frames[1:], frames[:-1]

    matcher = build_motion_estimator(args).to(device)
    compensator = build_compensator(args.mc, args.mc_smooth).to(device)
    me_pool, residual_pool = resolve_pools(args)
    mvc_metric = MetricMVC().to(device)
    engine = EVCATemporalEngine(matcher,
                                {'mvc': mvc_metric, 'tc_sad': MetricsTCSAD().to(device)},
                                compensator, residual_pool=residual_pool).to(device)

    with torch.no_grad():
        results, state = engine(curr_frames, ref_frames, frame_stack=frames)
        # Second call for the cost volume. It is pure search with no orchestration, and
        # re-deriving the same vectors is a free consistency check on this whole path.
        mvs2, _, costs, best_idx = matcher(curr_frames, ref_frames, frame_stack=frames,
                                           return_costs=True)
        if not torch.equal(mvs2, state.mvs):
            raise AssertionError('the cost-volume pass disagreed with the engine pass; '
                                 'the search is not deterministic')

        lap = F.conv2d(F.pad(state.mvs, (1, 1, 1, 1), mode='replicate'),
                       mvc_metric.laplacian_kernel, groups=2)
        mvc_map = lap.abs().mean(dim=1)[0]

        pixel_mv, pixel_note = effective_pixel_field(
            compensator, state.mvs, (frames.shape[-2], frames.shape[-1]))

        mv_mag = torch.sqrt(state.mvs[:, 0] ** 2 + state.mvs[:, 1] ** 2)
        metrics = {
            'MVC': float(results['mvc'].ravel()[0]),
            'TC_SAD': float(results['tc_sad'].ravel()[0]),
            'mean_mv_mag': float(mv_mag.mean()),
        }

        def np_of(x):
            return x.detach().float().cpu().numpy()

        H, W = frames.shape[-2], frames.shape[-1]
        scene = MotionScene(
            args=args,
            label=Path(args.input).stem,
            t_ref=t,
            t_curr=t + args.sample_rate,
            csv_row=(t + args.sample_rate) // args.sample_rate,
            H=H, W=W, bs=args.block_size,
            Hb=state.mvs.shape[2], Wb=state.mvs.shape[3],
            pool=residual_pool, me_pool=me_pool,
            reach_px=matcher.reach_px,
            pattern=np_of(matcher.pattern_lookup),
            ref=np_of(ref_frames[0, 0]),
            curr=np_of(curr_frames[0, 0]),
            pixel_mv=None if pixel_mv is None else np_of(pixel_mv[0]),
            pixel_mv_note=pixel_note,
            mv=np_of(state.mvs[0]),
            sad=np_of(state.sad_map[0, 0]),
            costs=np_of(costs[0]),
            best_idx=best_idx[0].detach().cpu().numpy(),
            mvc_map=np_of(mvc_map),
            ref_r=np_of(state.ref_frame[0, 0]),
            curr_r=np_of(state.current_frame[0, 0]),
            mc_r=np_of(state.mc_frame[0, 0]),
            bs_r=state.bs,
            metrics=metrics,
        )
    return scene


# --------------------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------------------

@dataclass
class View:
    """A block-aligned crop, in full-resolution pixels and in block indices.

    Every panel is drawn with `extent` in full-resolution pixel coordinates whatever
    domain its data lives in -- full resolution, pooled residual, or one value per block
    -- so a feature keeps the same position on the page across all of them and the eye
    can move between panels without re-registering.
    """
    y0: int
    x0: int
    h: int
    w: int

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.x0 + self.w, self.y0 + self.h, self.y0)

    def full(self, arr: np.ndarray) -> np.ndarray:
        return arr[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]

    def pooled(self, arr: np.ndarray, pool: int) -> np.ndarray:
        return arr[self.y0 // pool:(self.y0 + self.h) // pool,
                   self.x0 // pool:(self.x0 + self.w) // pool]

    def blocks(self, arr: np.ndarray, bs: int) -> np.ndarray:
        sl = (Ellipsis,
              slice(self.y0 // bs, (self.y0 + self.h) // bs),
              slice(self.x0 // bs, (self.x0 + self.w) // bs))
        return arr[sl]


def make_view(scene: MotionScene, roi: Optional[Sequence[int]]) -> View:
    """The ROI snapped outwards to whole blocks, so every domain crops consistently."""
    if roi is None:
        return View(0, 0, scene.H, scene.W)
    y, x, h, w = roi
    bs = scene.bs
    y0 = max(0, (y // bs) * bs)
    x0 = max(0, (x // bs) * bs)
    y1 = min(scene.H, -(-(y + h) // bs) * bs)
    x1 = min(scene.W, -(-(x + w) // bs) * bs)
    if y1 <= y0 or x1 <= x0:
        raise SystemExit(f'--roi {roi} does not overlap the {scene.W}x{scene.H} frame')
    return View(y0, x0, y1 - y0, x1 - x0)


def flow_to_rgb(dy: np.ndarray, dx: np.ndarray, max_mag: float) -> np.ndarray:
    """Direction as hue, magnitude as brightness. Zero motion is black.

    Takes displacement in *display* orientation (+y downwards), so callers pass the
    apparent motion -(dy, dx) rather than the raw estimate.
    """
    hue = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
    val = np.clip(np.hypot(dy, dx) / max(max_mag, 1e-9), 0.0, 1.0)
    return hsv_to_rgb(np.stack([hue, np.ones_like(hue), val], axis=-1))


def draw_colour_wheel(ax, max_mag: float, size: str = '28%') -> None:
    """Inset legend for `flow_to_rgb`: hue is the direction content appears to move."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    inner = inset_axes(ax, width=size, height=size, loc='lower right', borderpad=0.35)
    n = 96
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    rgb = flow_to_rgb(yy * max_mag, xx * max_mag, max_mag)
    rgba = np.dstack([rgb, (np.hypot(yy, xx) <= 1.0).astype(float)])
    inner.imshow(rgba, origin='upper')
    inner.set_xticks([])
    inner.set_yticks([])
    for spine in inner.spines.values():
        spine.set_visible(False)


def candidate_colours(scene: MotionScene) -> np.ndarray:
    """[N, 3] colour per search candidate, sharing the flow map's hue convention.

    The zero candidate is grey rather than black so "stayed put" stays legible against
    the moving candidates, and brightness is floored so a `grid` pattern's short inner
    vectors do not collapse into one another.
    """
    colours = np.zeros((len(scene.pattern), 3))
    for i, (dy, dx) in enumerate(scene.pattern):
        mag = float(np.hypot(dy, dx))
        if mag == 0.0:
            colours[i] = (0.62, 0.62, 0.62)
            continue
        hue = (np.arctan2(-dy, -dx) + np.pi) / (2 * np.pi)
        val = 0.45 + 0.55 * min(mag / max(scene.reach_px, 1e-9), 1.0)
        colours[i] = hsv_to_rgb(np.array([hue, 1.0, val]))
    return colours


def bare(ax, title: str = None, pad: float = None) -> None:
    if title:
        ax.set_title(title, fontsize=9, pad=pad)
    ax.set_xticks([])
    ax.set_yticks([])


def stamp(ax, text: str) -> None:
    """Small domain/units note in the corner, so no panel is read in the wrong units."""
    ax.text(0.015, 0.015, text, transform=ax.transAxes, fontsize=6.5, color='w',
            va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', ec='none', alpha=0.55))


def colourbar(fig, im, ax, label: str = None) -> None:
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    if label:
        cb.set_label(label, fontsize=7)


def show(ax, data: np.ndarray, view: View, **kw):
    """imshow in the shared full-resolution coordinate frame."""
    kw.setdefault('interpolation', 'nearest')
    return ax.imshow(data, extent=view.extent, origin='upper', aspect='equal', **kw)


def draw_block_grid(ax, scene: MotionScene, view: View, force: bool = False) -> None:
    """Block boundaries, drawn only when they would stay legible."""
    n_blocks = (view.h // scene.bs) * (view.w // scene.bs)
    if not force and n_blocks > 900:
        return
    for y in range(view.y0, view.y0 + view.h + 1, scene.bs):
        ax.axhline(y, color='w', lw=0.25, alpha=0.28)
    for x in range(view.x0, view.x0 + view.w + 1, scene.bs):
        ax.axvline(x, color='w', lw=0.25, alpha=0.28)


def auto_stride(scene: MotionScene, view: View, target: int = 26) -> int:
    """Arrow stride that keeps roughly `target` arrows across the widest axis."""
    blocks = max(view.w // scene.bs, view.h // scene.bs)
    return max(1, int(np.ceil(blocks / target)))


def auto_arrow_scale(scene: MotionScene, stride: int) -> float:
    """Amplification that maps the pattern's reach onto half the drawn arrow spacing.

    Without amplification the default search is invisible: `--me-offset 2` on 32 px
    blocks draws arrows two pixels long inside squares sixteen times that size. Scaling
    against the *spacing* rather than the block matters once the stride exceeds 1, which
    it does at 1080p -- otherwise thinning the field also shrinks every arrow in it.
    """
    return (scene.bs * stride * 0.45) / max(scene.reach_px, 1e-9)


def robust_max(arr: np.ndarray, pct: float = 99.5, floor: float = 1e-6) -> float:
    return max(float(np.percentile(np.abs(arr), pct)), floor)


@dataclass
class Style:
    stride: int
    amp: float
    force_grid: bool = False


# --------------------------------------------------------------------------------------
# Layer 1 -- Evidence: what actually moved
# --------------------------------------------------------------------------------------

def panel_frame(fig, ax, scene, view, which: str) -> None:
    plane = scene.ref if which == 'ref' else scene.curr
    t = scene.t_ref if which == 'ref' else scene.t_curr
    role = 'reference' if which == 'ref' else 'current'
    show(ax, view.full(plane), view, cmap='gray', vmin=0, vmax=255)
    draw_block_grid(ax, scene, view)
    bare(ax, f'F({t}) — {role}')
    stamp(ax, f'{view.w}x{view.h} px')


def panel_raw_difference(fig, ax, scene, view) -> None:
    diff = np.abs(view.full(scene.curr) - view.full(scene.ref))
    im = show(ax, diff, view, cmap='magma', vmin=0, vmax=robust_max(diff))
    bare(ax, f'|F({scene.t_curr}) − F({scene.t_ref})| — motion as raw evidence')
    colourbar(fig, im, ax)
    stamp(ax, f'mean {diff.mean():.2f}')


def panel_anaglyph(fig, ax, scene, view) -> None:
    """Reference in red, current in cyan: displacement reads directly as colour fringing.

    Worth looking at before any arrow, because it owes nothing to the estimator -- it
    shows what moved, against which the estimate can then be judged.
    """
    a, b = view.full(scene.ref), view.full(scene.curr)
    lo, hi = np.percentile(np.concatenate([a.ravel(), b.ravel()]), [1, 99])
    norm = lambda x: np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    show(ax, np.dstack([norm(a), norm(b), norm(b)]), view)
    bare(ax, 'Anaglyph: F(t) red / F(t+1) cyan')
    stamp(ax, 'fringe width = displacement')


# --------------------------------------------------------------------------------------
# Layer 2 -- Field: what the search decided
# --------------------------------------------------------------------------------------

def panel_quiver(fig, ax, scene, view, style: Style) -> None:
    mv = view.blocks(scene.mv, scene.bs)
    show(ax, view.full(scene.curr), view, cmap='gray', vmin=0, vmax=255, alpha=0.85)
    draw_block_grid(ax, scene, view, force=style.force_grid)

    by, bx = np.mgrid[0:mv.shape[1], 0:mv.shape[2]]
    cy = (by + 0.5) * scene.bs + view.y0
    cx = (bx + 0.5) * scene.bs + view.x0
    s = style.stride
    cy, cx, mv = cy[::s, ::s], cx[::s, ::s], mv[:, ::s, ::s]
    mag = np.hypot(mv[0], mv[1])

    still = mag == 0
    if still.any():
        # Drawn explicitly: a block that chose the collocated candidate is a decision,
        # and a quiver simply omits it.
        ax.plot(cx[still], cy[still], '.', ms=1.6, color='0.75', alpha=0.8)
    moving = ~still
    if moving.any():
        q = ax.quiver(cx[moving], cy[moving], -mv[1][moving], -mv[0][moving], mag[moving],
                      angles='xy', scale_units='xy', scale=1.0 / style.amp,
                      cmap='turbo', clim=(0, scene.reach_px),
                      width=0.0032, headwidth=3.4, headlength=3.8)
        ax.quiverkey(q, 0.02, 1.055, scene.reach_px, f'= reach, {scene.reach_px}px',
                     labelpos='E', coordinates='axes', fontproperties={'size': 7})
        colourbar(fig, q, ax, '|MV| px')
    ax.set_xlim(view.x0, view.x0 + view.w)
    ax.set_ylim(view.y0 + view.h, view.y0)
    bare(ax, f'Motion vectors over F({scene.t_curr})', pad=12)
    stamp(ax, f'drawn as −(dy,dx) = apparent motion, ×{style.amp:.1f} · every {s} block'
              f'{"s" if s > 1 else ""}')


def panel_flow(fig, ax, scene, view, source: str = 'block') -> None:
    """The MV field as hue/brightness -- the readable form once arrows get too dense."""
    if source == 'block':
        field = view.blocks(scene.mv, scene.bs)
        title = 'MV field the search emitted'
        note = f'block grid {scene.Hb}x{scene.Wb}'
    else:
        if scene.pixel_mv is None:
            bare(ax, 'Pixel MV field — unavailable')
            ax.text(0.5, 0.5, scene.pixel_mv_note or '', ha='center', va='center',
                    fontsize=8, transform=ax.transAxes, wrap=True)
            return
        field = view.full(scene.pixel_mv.transpose(1, 2, 0)).transpose(2, 0, 1)
        title = f'MV field actually warped (--mc {scene.args.mc})'
        note = scene.pixel_mv_note or f'per pixel, after --mc-smooth {scene.args.mc_smooth}'
    show(ax, flow_to_rgb(-field[0], -field[1], scene.reach_px), view)
    draw_colour_wheel(ax, scene.reach_px)
    bare(ax, title)
    stamp(ax, note)


def panel_candidates(fig, ax, scene, view) -> None:
    """Which of the pattern's candidates won, as a categorical map.

    The panel that exposes how few choices there really are: with the default diamond
    the entire frame is painted in five colours, and a field that looked like flow in
    the quiver turns out to be a five-way vote.
    """
    colours = candidate_colours(scene)
    idx = view.blocks(scene.best_idx, scene.bs)
    show(ax, colours[idx], view)
    bare(ax, f'Winning candidate ({len(scene.pattern)} of them)')

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    inner = inset_axes(ax, width='30%', height='30%', loc='lower right', borderpad=0.35)
    r = scene.reach_px * 1.35
    inner.add_patch(plt.Circle((0, 0), scene.reach_px, fill=False, color='0.6',
                               lw=0.6, ls=':'))
    inner.scatter(scene.pattern[:, 1], scene.pattern[:, 0], c=colours,
                  s=max(12, 220 // max(len(scene.pattern), 1)), edgecolors='0.25', lw=0.3)
    inner.set_xlim(-r, r)
    inner.set_ylim(r, -r)
    inner.set_aspect('equal')
    inner.set_xticks([])
    inner.set_yticks([])
    inner.set_facecolor('white')
    inner.patch.set_alpha(0.8)
    counts = np.bincount(idx.ravel(), minlength=len(scene.pattern))
    zero_i = int(np.argmin(np.abs(scene.pattern).sum(axis=1)))
    # The legend's dots sit where each candidate *looks* in the reference, but are
    # coloured by the apparent motion that choice implies -- the two are opposite by
    # construction, so say which is which rather than leave it to be inferred.
    stamp(ax, f'{100 * counts[zero_i] / counts.sum():.0f}% chose (0,0)\n'
              f'legend = search geometry; hue = apparent motion')


def panel_clipped(fig, ax, scene, view) -> None:
    """Blocks whose winner sits on the pattern boundary: motion may exceed the reach."""
    clip = view.blocks(scene.clipped, scene.bs).astype(float)
    show(ax, view.full(scene.curr), view, cmap='gray', vmin=0, vmax=255)
    show(ax, np.ma.masked_where(clip == 0, clip), view, cmap='autumn', vmin=0, vmax=1,
         alpha=0.75)
    frac = 100.0 * clip.mean()
    bare(ax, f'Saturated at the pattern edge — {frac:.1f}% of blocks')
    stamp(ax, f'|MV| = {scene.reach_px}px · raise --me-offset if this is large')


def panel_field_delta(fig, ax, scene, view) -> None:
    """How far the applied pixel field drifted from the block field it came from.

    `--mc dense_smooth` filters the field and bilinearly upsamples it, so the motion a
    pixel is actually warped by is not the motion its own block voted for. Bright here
    means the compensator is predicting a block with somebody else's vector.
    """
    if scene.pixel_mv is None:
        bare(ax, 'Field displacement — unavailable')
        return
    blocky = np.repeat(np.repeat(scene.mv, scene.bs, axis=1), scene.bs, axis=2)
    delta = np.hypot(*(scene.pixel_mv - blocky))
    d = view.full(delta)
    im = show(ax, d, view, cmap='viridis', vmin=0, vmax=robust_max(d))
    bare(ax, 'Applied field − block field  (|Δ| px)')
    colourbar(fig, im, ax)
    stamp(ax, f'mean {d.mean():.2f}px · smoothing cost at motion boundaries')


# --------------------------------------------------------------------------------------
# Layer 3 -- Confidence: how much to trust each vector
# --------------------------------------------------------------------------------------

def pattern_grid_shape(scene: MotionScene) -> Optional[Tuple[int, int]]:
    """(ny, nx) when the pattern is a full rectangular grid, else None.

    `--heuristic grid` enumerates dy outer / dx inner, so its candidates can be shown in
    their own geometry instead of as an unreadable 25-bar chart.
    """
    uy = np.unique(scene.pattern[:, 0])
    ux = np.unique(scene.pattern[:, 1])
    if len(uy) * len(ux) == len(scene.pattern) and len(uy) > 1 and len(ux) > 1:
        return len(uy), len(ux)
    return None


def panel_sad(fig, ax, scene, view) -> None:
    sad = view.blocks(scene.sad, scene.bs)
    im = show(ax, sad, view, cmap='inferno', vmin=0, vmax=robust_max(sad))
    bare(ax, 'Winning cost (min SAD)')
    colourbar(fig, im, ax, 'mean |err| / px')
    stamp(ax, f'TC_SAD = {scene.metrics["TC_SAD"]:.3f} · searched at 1/{scene.me_pool}')


def panel_margin(fig, ax, scene, view) -> None:
    """How decisive each choice was: second-best cost minus best cost.

    Near zero means the search could not tell its candidates apart, so the vector is
    arbitrary no matter how small its SAD is. Nothing in the CSV exposes this.
    """
    margin = view.blocks(scene.margin, scene.bs)
    im = show(ax, margin, view, cmap='cividis', vmin=0, vmax=robust_max(margin))
    bare(ax, 'Decision margin (2nd best − best)')
    colourbar(fig, im, ax)
    stamp(ax, f'dark = arbitrary choice · median {np.median(margin):.3f}')


def panel_cost_scatter(fig, ax, scene, view) -> None:
    """Every block as (cost, margin). The lower-right corner is where the field lies."""
    sad = view.blocks(scene.sad, scene.bs).ravel()
    margin = view.blocks(scene.margin, scene.bs).ravel()
    clipped = view.blocks(scene.clipped, scene.bs).ravel()
    ax.scatter(sad[~clipped], margin[~clipped], s=2.5, alpha=0.35, c='#3b6ea5',
               linewidths=0, label='within reach')
    if clipped.any():
        ax.scatter(sad[clipped], margin[clipped], s=2.5, alpha=0.5, c='#d1495b',
                   linewidths=0, label='at pattern edge')
    sad_hi = float(np.percentile(sad, 75))
    margin_lo = float(np.percentile(margin, 25))
    ax.axvline(sad_hi, color='0.5', lw=0.6, ls='--')
    ax.axhline(margin_lo, color='0.5', lw=0.6, ls='--')
    bad = float(((sad > sad_hi) & (margin < margin_lo)).mean() * 100)
    ax.set_xlabel('min SAD', fontsize=8)
    ax.set_ylabel('margin', fontsize=8)
    ax.set_title('Cost vs. decisiveness', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, markerscale=3, loc='upper right', framealpha=0.85)
    ax.text(0.98, 0.02, f'{bad:.1f}% of blocks:\nbad fit AND arbitrary',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.25', fc='#ffe9ec', ec='#d1495b', lw=0.5))


def panel_mvc_map(fig, ax, scene, view) -> None:
    """The quantity `MetricMVC` averages, so the exported MVC scalar has a picture."""
    m = view.blocks(scene.mvc_map, scene.bs)
    im = show(ax, m, view, cmap='plasma', vmin=0, vmax=robust_max(m))
    bare(ax, 'MV-field roughness  |∇²MV|')
    colourbar(fig, im, ax)
    stamp(ax, f'mean = MVC = {scene.metrics["MVC"]:.3f}')


def panel_usage(fig, ax, scene, view) -> None:
    """How often each candidate won, drawn in the pattern's own geometry where it has one."""
    counts = np.bincount(view.blocks(scene.best_idx, scene.bs).ravel(),
                         minlength=len(scene.pattern))
    pct = 100.0 * counts / counts.sum()
    shape = pattern_grid_shape(scene)
    if shape is not None:
        im = ax.imshow(pct.reshape(shape), cmap='YlGnBu', origin='upper')
        ax.set_xticks(range(shape[1]))
        ax.set_yticks(range(shape[0]))
        ax.set_xticklabels(np.unique(scene.pattern[:, 1]).astype(int), fontsize=6)
        ax.set_yticklabels(np.unique(scene.pattern[:, 0]).astype(int), fontsize=6)
        ax.set_xlabel('dx (px)', fontsize=8)
        ax.set_ylabel('dy (px)', fontsize=8)
        colourbar(fig, im, ax, '% of blocks')
        ax.set_title('Candidate usage', fontsize=9)
        return
    colours = candidate_colours(scene)
    ax.bar(range(len(pct)), pct, color=colours, edgecolor='0.3', linewidth=0.4)
    ax.set_xticks(range(len(pct)))
    ax.set_xticklabels([f'({int(dy)},{int(dx)})' for dy, dx in scene.pattern],
                       fontsize=6.5, rotation=30)
    ax.set_ylabel('% of blocks', fontsize=8)
    ax.set_xlabel('candidate (dy, dx) px', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title('Candidate usage', fontsize=9)


def panel_unreliable(fig, ax, scene, view) -> None:
    """Where both conditions hold at once: poor prediction and an arbitrary choice."""
    sad = view.blocks(scene.sad, scene.bs)
    margin = view.blocks(scene.margin, scene.bs)
    sad_hi = float(np.percentile(sad, 75))
    margin_lo = float(np.percentile(margin, 25))
    mask = (sad > sad_hi) & (margin < margin_lo)
    show(ax, view.full(scene.curr), view, cmap='gray', vmin=0, vmax=255)
    m = mask.astype(float)
    show(ax, np.ma.masked_where(m == 0, m), view, cmap='cool', vmin=0, vmax=1, alpha=0.7)
    bare(ax, 'Untrustworthy vectors')
    stamp(ax, f'SAD > p75 and margin < p25 · {100 * mask.mean():.1f}% of blocks')


# --------------------------------------------------------------------------------------
# Layer 4 -- Compensation: what the warp bought
# --------------------------------------------------------------------------------------

def panel_mc_frame(fig, ax, scene, view) -> None:
    show(ax, view.pooled(scene.mc_r, scene.pool), view, cmap='gray', vmin=0, vmax=255)
    bare(ax, f'MC(F({scene.t_ref})) — reference warped onto F({scene.t_curr})')
    stamp(ax, f'--mc {scene.args.mc}'
              + (f' · domain ÷{scene.pool}' if scene.pool > 1 else ''))


def panel_error(fig, ax, scene, view, compensated: bool, vmax: float) -> None:
    """The A/B: uncompensated difference and compensated residual, on one colour scale.

    Same scale on both is the whole point -- an independently scaled residual can be
    made to look clean whatever it contains.
    """
    plane = scene.resid_r if compensated else scene.diff_r
    d = np.abs(view.pooled(plane, scene.pool))
    im = show(ax, d, view, cmap='magma', vmin=0, vmax=vmax)
    label = ('|F(t+1) − MC(F(t))|  compensated' if compensated
             else '|F(t+1) − F(t)|  uncompensated')
    bare(ax, label)
    colourbar(fig, im, ax)
    stamp(ax, f'mean {d.mean():.2f}'
              + (f' · domain ÷{scene.pool}' if scene.pool > 1 else ''))


def panel_gain(fig, ax, scene, view) -> None:
    gain = view.blocks(scene.gain, scene.bs)
    lim = robust_max(gain)
    # RdBu (not _r) puts positive gain on blue and loss on red, so the colour that
    # draws the eye is the case worth looking at.
    im = show(ax, gain, view, cmap='RdBu', vmin=-lim, vmax=lim)
    bare(ax, 'Per-block gain from compensation')
    colourbar(fig, im, ax, 'Δ mean |err|')
    hurt = 100.0 * (gain < 0).mean()
    stamp(ax, f'blue = helped, red = hurt · {hurt:.1f}% of blocks got worse')


def panel_block_residual(fig, ax, scene, view) -> None:
    """Per-block mean |residual|.

    Related to but not equal to the exported `TC_MC`, which additionally applies EVCA's
    frequency weighting to the residual DCT and caps the result at the block's own intra
    energy. For the real per-block TC_MC, run main.py with `-bi` and read
    `<csv>_TCMC_blocks.csv`.
    """
    r = view.blocks(scene.block_mean_abs(scene.resid_r), scene.bs)
    im = show(ax, r, view, cmap='inferno', vmin=0, vmax=robust_max(r))
    bare(ax, 'Residual energy per block')
    colourbar(fig, im, ax, 'mean |residual|')
    stamp(ax, 'unweighted, not intra-gated — see --block-csv for TC_MC')


def panel_error_histogram(fig, ax, scene, view) -> None:
    d = np.abs(view.pooled(scene.diff_r, scene.pool)).ravel()
    r = np.abs(view.pooled(scene.resid_r, scene.pool)).ravel()
    hi = float(np.percentile(d, 99.5))
    bins = np.linspace(0, max(hi, 1e-3), 80)
    ax.hist(d, bins=bins, histtype='step', color='#8c564b', lw=1.2,
            label=f'uncompensated (μ={d.mean():.2f})')
    ax.hist(r, bins=bins, histtype='step', color='#2ca02c', lw=1.2,
            label=f'compensated (μ={r.mean():.2f})')
    ax.set_yscale('log')
    ax.set_xlabel('|prediction error|', fontsize=8)
    ax.set_ylabel('pixels', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.set_title(f'Error distribution — {d.mean() / max(r.mean(), 1e-9):.2f}× better on average',
                 fontsize=9)


# --------------------------------------------------------------------------------------
# Layer 5 -- Drill-down: why this block chose what it chose
# --------------------------------------------------------------------------------------

def select_blocks(scene: MotionScene, args: argparse.Namespace,
                  view: View) -> List[Tuple[int, int]]:
    """Blocks to drill into, as absolute grid coordinates.

    `--worst` ranks within the view, so under `--roi` the drill-down concerns the same
    region every other panel does rather than jumping somewhere off-screen. `--block`
    is always absolute, since it names a specific block.
    """
    picks: List[Tuple[int, int]] = []
    for by, bx in (args.block or []):
        if not (0 <= by < scene.Hb and 0 <= bx < scene.Wb):
            raise SystemExit(f'--block {by},{bx} is outside the {scene.Hb}x{scene.Wb} grid')
        picks.append((by, bx))
    worst = args.worst if args.worst else (None if picks else ('sad', '3'))
    if worst:
        metric, k = worst[0], int(worst[1])
        ranking = {'sad': scene.sad, 'margin': -scene.margin, 'gain': -scene.gain}
        if metric not in ranking:
            raise SystemExit(f'--worst metric must be one of {sorted(ranking)}, got {metric!r}')
        ranked = view.blocks(ranking[metric], scene.bs)
        by0, bx0 = view.y0 // scene.bs, view.x0 // scene.bs
        order = np.argsort(ranked.ravel())[::-1][:k]
        picks += [(by0 + int(i // ranked.shape[1]), bx0 + int(i % ranked.shape[1]))
                  for i in order]
    seen, unique = set(), []
    for p in picks:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def draw_cost_surface(fig, ax, scene: MotionScene, by: int, bx: int) -> None:
    """The whole cost volume for one block: the evidence the decision was made on.

    A sharp well means the vector is pinned down; a flat valley means the winner beat
    its rivals by noise, and the reported motion is an artefact of the tie-break.
    """
    costs = scene.costs[:, by, bx]
    win = int(scene.best_idx[by, bx])
    shape = pattern_grid_shape(scene)
    if shape is not None:
        im = ax.imshow(costs.reshape(shape), cmap='inferno_r')
        wy, wx = divmod(win, shape[1])
        ax.add_patch(Rectangle((wx - 0.5, wy - 0.5), 1, 1, fill=False, ec='#00e5ff', lw=1.6))
        ax.set_xticks(range(shape[1]))
        ax.set_yticks(range(shape[0]))
        ax.set_xticklabels(np.unique(scene.pattern[:, 1]).astype(int), fontsize=5.5)
        ax.set_yticklabels(np.unique(scene.pattern[:, 0]).astype(int), fontsize=5.5)
        colourbar(fig, im, ax)
    else:
        colours = candidate_colours(scene)
        ax.bar(range(len(costs)), costs, color=colours, edgecolor='0.3', linewidth=0.4)
        ax.bar([win], [costs[win]], color='none', edgecolor='#00b0d0', linewidth=1.8)
        ax.set_xticks(range(len(costs)))
        ax.set_xticklabels([f'({int(dy)},{int(dx)})' for dy, dx in scene.pattern],
                           fontsize=5.5, rotation=45)
        # A block flat on both sides scores 0 for every candidate; without a floor
        # that collapses the axis and matplotlib warns about identical limits.
        ax.set_ylim(0, max(float(costs.max()) * 1.15, 1e-3))
        ax.tick_params(labelsize=6)
    second = float(np.partition(costs, 1)[1]) if len(costs) > 1 else float(costs[0])
    ax.set_title(f'cost surface · margin {second - costs.min():.3f}', fontsize=8)


def figure_blocks(scene: MotionScene, picks: Sequence[Tuple[int, int]]):
    """One row per block: where it looked, what it found, and what that cost."""
    bs, pad = scene.bs, max(scene.reach_px, 6)
    ref_pad = np.pad(scene.ref, pad, mode='edge')
    rows = len(picks)
    fig, axes = plt.subplots(rows, 5, figsize=(14.0, 2.7 * rows), squeeze=False)

    for row, (by, bx) in enumerate(picks):
        y0, x0 = by * bs, bx * bs
        dy, dx = scene.mv[0, by, bx], scene.mv[1, by, bx]
        ctx = ref_pad[y0:y0 + bs + 2 * pad, x0:x0 + bs + 2 * pad]
        pred = ref_pad[y0 + pad + int(dy):y0 + pad + int(dy) + bs,
                       x0 + pad + int(dx):x0 + pad + int(dx) + bs]
        cur = scene.curr[y0:y0 + bs, x0:x0 + bs]
        br = scene.bs_r
        res = scene.resid_r[by * br:(by + 1) * br, bx * br:(bx + 1) * br]

        ax = axes[row][0]
        ax.imshow(ctx, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax.add_patch(Rectangle((pad - 0.5, pad - 0.5), bs, bs, fill=False,
                               ec='#00e5ff', lw=1.2))
        ax.add_patch(Rectangle((pad + dx - 0.5, pad + dy - 0.5), bs, bs, fill=False,
                               ec='#ffd23f', lw=1.2, ls='--'))
        cy = pad + bs / 2 - 0.5
        ax.scatter(cy + scene.pattern[:, 1], cy + scene.pattern[:, 0],
                   c=candidate_colours(scene), s=14, edgecolors='k', linewidths=0.3)
        bare(ax, f'F({scene.t_ref}) around block ({by},{bx})')
        ax.set_ylabel(f'block ({by},{bx})\nSAD {scene.sad[by, bx]:.2f}',
                      fontsize=7.5, labelpad=6)

        ax = axes[row][1]
        ax.imshow(pred, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        bare(ax, f'predictor: ref shifted ({int(dy)},{int(dx)})')

        ax = axes[row][2]
        ax.imshow(cur, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        bare(ax, f'F({scene.t_curr}) block — the target')

        ax = axes[row][3]
        lim = max(float(np.abs(res).max()), 1e-3)
        im = ax.imshow(res, cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
        bare(ax, f'residual after --mc {scene.args.mc}')
        colourbar(fig, im, ax)

        draw_cost_surface(fig, axes[row][4], scene, by, bx)

    fig.suptitle('Per-block evidence — cyan: the block, dashed yellow: the predictor it '
                 'chose, dots: where each candidate looked', fontsize=9, y=0.995)
    finish(fig, 0.97, h_pad=1.2)
    return fig


# --------------------------------------------------------------------------------------
# Figure assembly
# --------------------------------------------------------------------------------------

def provenance(scene: MotionScene) -> str:
    a = scene.args
    still = 100.0 * (scene.mv_mag == 0).mean()
    return (
        f'{scene.label}   F({scene.t_ref}) → F({scene.t_curr})   {scene.W}×{scene.H}px   '
        f'block {scene.bs}   {a.heuristic} reach {a.me_offset}px '
        f'({len(scene.pattern)} candidates)   me-pool {scene.me_pool} / '
        f'temporal-pool {scene.pool}   --mc {a.mc}/{a.mc_smooth}\n'
        f'MVC {scene.metrics["MVC"]:.3f}   TC_SAD {scene.metrics["TC_SAD"]:.3f}   '
        f'mean |MV| {scene.metrics["mean_mv_mag"]:.2f}px   {still:.0f}% chose (0,0)   '
        f'{100 * scene.clipped.mean():.0f}% at the pattern edge   '
        f'(CSV row {scene.csv_row})')


def finish(fig, top: float, h_pad: float = 2.2, w_pad: float = 1.0) -> None:
    """tight_layout, minus the warning the colour-wheel and pattern insets provoke.

    Inset axes have no subplotspec, so tight_layout reports it cannot lay them out --
    but they are anchored to their parent axes, which it does lay out, and they follow.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*not compatible with tight_layout.*')
        fig.tight_layout(rect=(0, 0, 1, top), h_pad=h_pad, w_pad=w_pad)


def grid_figure(nrows: int, ncols: int, view: View, panel_w: float = 4.4):
    """Figure sized from the view's aspect so panels are neither squashed nor stretched."""
    aspect = np.clip(view.h / max(view.w, 1), 0.34, 1.35)
    # 0.55in per row for the panel title, plus 1.15in for the two-line suptitle.
    height = panel_w * aspect * nrows + 0.55 * nrows + 1.15
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, height),
                             squeeze=False)
    return fig, axes


def figure_evidence(scene, view, style):
    fig, ax = grid_figure(2, 2, view)
    panel_frame(fig, ax[0][0], scene, view, 'ref')
    panel_frame(fig, ax[0][1], scene, view, 'curr')
    panel_raw_difference(fig, ax[1][0], scene, view)
    panel_anaglyph(fig, ax[1][1], scene, view)
    fig.suptitle('1 · Evidence — what actually moved\n' + provenance(scene), fontsize=8.5)
    finish(fig, 0.93)
    return fig


def figure_field(scene, view, style):
    fig, ax = grid_figure(2, 3, view)
    panel_quiver(fig, ax[0][0], scene, view, style)
    panel_flow(fig, ax[0][1], scene, view, 'block')
    panel_flow(fig, ax[0][2], scene, view, 'pixel')
    panel_candidates(fig, ax[1][0], scene, view)
    panel_clipped(fig, ax[1][1], scene, view)
    panel_field_delta(fig, ax[1][2], scene, view)
    fig.suptitle('2 · Field — what the search decided   '
                 '(MV points back into the reference: curr(y,x) ≈ ref(y+dy, x+dx); '
                 'arrows and hues show −(dy,dx), the apparent motion)\n' + provenance(scene),
                 fontsize=8.5)
    finish(fig, 0.92)
    return fig


def figure_confidence(scene, view, style):
    fig, ax = grid_figure(2, 3, view)
    panel_sad(fig, ax[0][0], scene, view)
    panel_margin(fig, ax[0][1], scene, view)
    panel_cost_scatter(fig, ax[0][2], scene, view)
    panel_mvc_map(fig, ax[1][0], scene, view)
    panel_usage(fig, ax[1][1], scene, view)
    panel_unreliable(fig, ax[1][2], scene, view)
    fig.suptitle('3 · Confidence — how much to trust each vector\n' + provenance(scene),
                 fontsize=8.5)
    finish(fig, 0.92)
    return fig


def figure_compensation(scene, view, style):
    fig, ax = grid_figure(2, 3, view)
    vmax = robust_max(view.pooled(scene.diff_r, scene.pool))
    panel_mc_frame(fig, ax[0][0], scene, view)
    panel_error(fig, ax[0][1], scene, view, compensated=False, vmax=vmax)
    panel_error(fig, ax[0][2], scene, view, compensated=True, vmax=vmax)
    panel_gain(fig, ax[1][0], scene, view)
    panel_block_residual(fig, ax[1][1], scene, view)
    panel_error_histogram(fig, ax[1][2], scene, view)
    fig.suptitle('4 · Compensation — what the warp bought   '
                 '(both error panels share one colour scale)\n' + provenance(scene),
                 fontsize=8.5)
    finish(fig, 0.92)
    return fig


# --------------------------------------------------------------------------------------
# Self-verification
# --------------------------------------------------------------------------------------

def setup_synthetic(args: argparse.Namespace, tmpdir: str) -> Optional[Tuple[float, float]]:
    """Points `args` at a generated sequence with known motion. Returns the true (dy, dx).

    Renders the same figures on a case whose answer is known, which is how the tool's
    own signs, hues and grid alignment get checked before it is pointed at real footage.
    """
    from validation.synthetic import (gen_cut, gen_moving_square, gen_static_noise,
                                     gen_translation, write_yuv420)

    spec = args.synthetic
    width, height = map(int, args.synthetic_size.split('x'))
    # Frames t and t + sample_rate, so the default --frame 0 needs exactly two.
    needed = args.frame[1] + args.sample_rate + 1
    if needed > SYNTHETIC_MAX_FRAMES:
        raise SystemExit(f'--synthetic generates at most {SYNTHETIC_MAX_FRAMES} frames, '
                         f'but --frame {args.frame[0]}-{args.frame[1]} needs {needed}; '
                         'point --input at a real sequence for frames that far in')
    n_frames = max(needed, 2)
    kind, _, params = spec.partition(':')
    uv_frames = None
    if kind == 'translation':
        vy, vx = parse_pair(params or '0,4', '--synthetic translation')
        frames, gt = gen_translation(height, width, n_frames, vy=vy, vx=vx, seed=3)
        truth = (float(vy), float(vx))
    elif kind == 'square':
        # The object moves, so the vector to expect is the negation of its velocity.
        vy, vx = parse_pair(params or '0,-2', '--synthetic square')
        frames, uv_frames, gt = gen_moving_square(height, width, n_frames, vy=vy, vx=vx)
        truth = (float(-vy), float(-vx))
    elif kind == 'static_noise':
        frames, gt = gen_static_noise(height, width, n_frames, seed=8)
        truth = (0.0, 0.0)
    elif kind == 'cut':
        frames, gt = gen_cut(height, width, n_frames, seed=10)
        truth = None
    else:
        raise SystemExit(f'--synthetic takes translation:VY,VX, square:VY,VX, '
                         f'static_noise or cut; got {spec!r}')

    path = Path(tmpdir) / f'synthetic_{kind}.yuv'
    write_yuv420(path, frames, gt, uv_frames=uv_frames)
    args.input = str(path)
    args.resolution = f'{width}x{height}'
    args.pix_fmt = 'yuv420'
    args.bit_depth = 8
    return truth


def report_ground_truth(scene: MotionScene) -> None:
    """Endpoint error against known motion, split by whether the block could know it.

    A block whose candidates all score the same cost has no information about motion --
    its reported vector is whichever candidate the tie-break happened to reach. On flat
    synthetic content that is most of the frame, so a single averaged EPE would measure
    the tie-break rather than the search. Decisive blocks (margin > 0) are the ones the
    ground truth can fairly be held against, and they must match it exactly.
    """
    if scene.gt is None:
        return
    gy, gx = scene.gt
    inner = (slice(1, -1), slice(1, -1))
    dy, dx = scene.mv[0][inner], scene.mv[1][inner]
    epe = np.hypot(dy - gy, dx - gx)
    decisive = scene.margin[inner] > 1e-6
    print(f'  ground truth (dy,dx) = ({gy:g}, {gx:g})   '
          f'interior EPE mean {epe.mean():.4f} / max {epe.max():.4f}   '
          f'{100.0 * ((dy == gy) & (dx == gx)).mean():.1f}% exact')
    if decisive.any() and not decisive.all():
        d = epe[decisive]
        print(f'    of which decisive (margin > 0): {100.0 * decisive.mean():.1f}% of '
              f'blocks, EPE mean {d.mean():.4f} / max {d.max():.4f}, '
              f'{100.0 * (d == 0).mean():.1f}% exact')
        flat = (~decisive).mean()
        print(f'    the other {100.0 * flat:.1f}% are flat on both sides: every '
              f'candidate ties at SAD 0, so no vector is recoverable there')


def check_against_csv(scene: MotionScene, csv_path: str) -> bool:
    """Confirms the visualized state is the state that produced a main.py CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    if scene.csv_row >= len(df):
        print(f'  --check-csv: row {scene.csv_row} is past the end of {csv_path} '
              f'({len(df)} rows)')
        return False
    row = df.iloc[scene.csv_row]
    ok = True
    print(f'  --check-csv against row {scene.csv_row} of {csv_path}:')
    for name in ('MVC', 'TC_SAD', 'mean_mv_mag'):
        if name not in df.columns:
            print(f'    {name:<12} absent from the CSV')
            continue
        here, there = scene.metrics[name], float(row[name])
        match = bool(np.isclose(here, there, rtol=1e-4, atol=1e-6))
        ok &= match
        print(f'    {name:<12} here {here:<12.6f} csv {there:<12.6f} '
              f'{"PASS" if match else "MISMATCH"}')
    if not ok:
        print('    a mismatch means the CSV was written with different flags, a '
              'different frame numbering, or a different build')
    return ok


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def render(scene: MotionScene, args: argparse.Namespace, wanted) -> List[Path]:
    view = make_view(scene, args.roi)
    stride = args.quiver_stride or auto_stride(scene, view)
    style = Style(stride=stride,
                  amp=args.arrow_scale or auto_arrow_scale(scene, stride),
                  force_grid=args.block_grid)
    out_dir = Path(args.out) / f'{scene.label}_f{scene.t_ref:04d}'
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = [('evidence', figure_evidence), ('field', figure_field),
                ('confidence', figure_confidence), ('compensation', figure_compensation)]
    written = []
    for i, (name, build) in enumerate(builders, start=1):
        if name in wanted:
            fig = build(scene, view, style)
            path = out_dir / f'{i}_{name}.png'
            fig.savefig(path, dpi=args.dpi, bbox_inches='tight')
            written.append(path)
            if not args.show:
                plt.close(fig)
    if 'blocks' in wanted:
        fig = figure_blocks(scene, select_blocks(scene, args, view))
        path = out_dir / '5_blocks.png'
        fig.savefig(path, dpi=args.dpi, bbox_inches='tight')
        written.append(path)
        if not args.show:
            plt.close(fig)
    return written


def main(argv=None) -> int:
    args = build_viz_parser().parse_args(argv)

    if args.figures.strip() == 'all':
        wanted = set(FIGURE_NAMES)
    else:
        wanted = {f.strip() for f in args.figures.split(',') if f.strip()}
        unknown = wanted - set(FIGURE_NAMES)
        if unknown:
            raise SystemExit(f'--figures: unknown {sorted(unknown)}; '
                             f'choose from {", ".join(FIGURE_NAMES)}')

    tmp = None
    truth = None
    try:
        if args.synthetic:
            tmp = tempfile.TemporaryDirectory(prefix='evca_viz_')
            truth = setup_synthetic(args, tmp.name)
        if not Path(args.input).is_file():
            raise SystemExit(f'Input file not found: {args.input}')

        geo = frame_geometry(args)
        total = frame_count(args, geo)
        lo, hi = args.frame
        if lo < 0 or hi + args.sample_rate >= total:
            raise SystemExit(f'--frame {lo}-{hi} needs frames up to '
                             f'{hi + args.sample_rate}, but the file holds {total}')

        device = select_device(args.device)
        print(f'Visualizing {Path(args.input).name} on {device.type}: '
              f'{args.heuristic} pattern, reach {args.me_offset}px, '
              f'block {args.block_size}, --mc {args.mc}')

        all_ok = True
        for t in range(lo, hi + 1):
            scene = build_scene(args, t, device)
            scene.gt = truth
            print(f'F({scene.t_ref}) → F({scene.t_curr}): '
                  f'MVC {scene.metrics["MVC"]:.4f}  '
                  f'TC_SAD {scene.metrics["TC_SAD"]:.4f}  '
                  f'mean|MV| {scene.metrics["mean_mv_mag"]:.3f}px  '
                  f'{100 * scene.clipped.mean():.1f}% at reach')
            report_ground_truth(scene)
            if args.check_csv:
                all_ok &= check_against_csv(scene, args.check_csv)
            for path in render(scene, args, wanted):
                print(f'  wrote {path}')

        if args.show:
            plt.show()
        return 0 if all_ok else 1
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == '__main__':
    raise SystemExit(main())
