"""Synthetic 8-bit yuv420 sequences with known ground-truth motion, for the tests.

`write_yuv420` emits a `.yuv` (Y plane textured, U/V constant 128) plus a `.json`
sidecar holding the exact motion parameters; the generators return the frames and that
ground truth directly, which is how the tests consume them.

MV sign convention (matches SparsePatternBlockMatcher): the estimated MV (dy, dx)
of a block satisfies curr(y, x) ~= ref(y + dy, x + dx). For a sequence produced
by sliding a crop window over a fixed canvas with offset increment (vy, vx) per
frame, the expected MV is exactly (vy, vx).
"""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def make_base_texture(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Band-pass filtered noise plus a few hard edges.

    The band-pass component gives SAD a unique minimum under translation, and the
    injected rectangles/lines add strong edges so DCT-based metrics are non-trivial.
    Returns a float32 array in [16, 235].
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((height, width))
    bandpass = gaussian_filter(noise, sigma=1.5) - gaussian_filter(noise, sigma=6.0)
    bandpass /= np.abs(bandpass).max() + 1e-9
    img = 128.0 + 70.0 * bandpass

    # A few rectangles with hard edges
    for _ in range(8):
        h = int(rng.integers(height // 16, height // 4))
        w = int(rng.integers(width // 16, width // 4))
        y = int(rng.integers(0, height - h))
        x = int(rng.integers(0, width - w))
        img[y:y + h, x:x + w] += float(rng.choice([-45.0, 45.0]))

    # A couple of 3px-wide diagonal lines
    yy, xx = np.mgrid[0:height, 0:width]
    for _ in range(3):
        slope = float(rng.uniform(-1.5, 1.5))
        icpt = float(rng.uniform(0, height))
        mask = np.abs(yy - (slope * xx + icpt)) < 1.5
        img[mask] += float(rng.choice([-50.0, 50.0]))

    return np.clip(img, 16.0, 235.0).astype(np.float32)


def write_yuv420(path: Path, y_frames: list, gt: dict, uv_frames: list = None) -> None:
    """Writes 8-bit yuv420p plus a ground-truth JSON sidecar.

    Chroma is constant mid-grey unless `uv_frames` supplies one (U, V) pair per frame,
    which is what makes a generated clip genuinely coloured rather than grey. Motion
    estimation reads luma only, so chroma changes nothing the search sees -- it only
    makes the file honest when something other than EVCA opens it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = y_frames[0].shape
    flat = np.full((height // 2, width // 2), 128, dtype=np.uint8)
    as_bytes = lambda a: np.clip(np.rint(a), 0, 255).astype(np.uint8).tobytes()
    with open(path, 'wb') as f:
        for i, y in enumerate(y_frames):
            f.write(as_bytes(y))
            if uv_frames is None:
                f.write(flat.tobytes())
                f.write(flat.tobytes())
            else:
                u, v = uv_frames[i]
                f.write(as_bytes(u))
                f.write(as_bytes(v))
    gt = dict(gt, width=width, height=height, frames=len(y_frames),
              pix_fmt='yuv420', bit_depth=8)
    with open(path.with_suffix('.json'), 'w') as f:
        json.dump(gt, f, indent=2)


def gen_translation(height: int, width: int, n_frames: int, vy: int, vx: int,
                    seed: int = 0) -> tuple:
    """Integer translation: crop window slides (vy, vx) px/frame over a fixed canvas."""
    span_y, span_x = abs(vy) * (n_frames - 1), abs(vx) * (n_frames - 1)
    canvas = make_base_texture(height + span_y + 8, width + span_x + 8, seed)
    oy0 = 4 + (span_y if vy < 0 else 0)
    ox0 = 4 + (span_x if vx < 0 else 0)
    frames = []
    for f in range(n_frames):
        oy, ox = oy0 + f * vy, ox0 + f * vx
        frames.append(canvas[oy:oy + height, ox:ox + width])
    gt = {'type': 'translation', 'mv_dy': vy, 'mv_dx': vx}
    return frames, gt


def gen_static_noise(height: int, width: int, n_frames: int, sigma: float = 4.0,
                     seed: int = 0) -> tuple:
    """Static texture plus i.i.d. Gaussian noise per frame (zero true motion)."""
    base = make_base_texture(height, width, seed)
    rng = np.random.default_rng(seed + 1)
    frames = [np.clip(base + rng.standard_normal(base.shape).astype(np.float32) * sigma,
                      0, 255) for _ in range(n_frames)]
    gt = {'type': 'static_noise', 'sigma': sigma, 'mv_dy': 0, 'mv_dx': 0}
    return frames, gt


def gen_cut(height: int, width: int, n_frames: int, seed: int = 0) -> tuple:
    """Hard cut: static texture A, then an unrelated static texture B at n_frames//2."""
    a = make_base_texture(height, width, seed)
    b = make_base_texture(height, width, seed + 100)
    cut_at = n_frames // 2
    frames = [a] * cut_at + [b] * (n_frames - cut_at)
    gt = {'type': 'cut', 'cut_frame': cut_at}
    return frames, gt


# BT.601 limited-range YUV for saturated primaries on black. Only the luma gap matters
# to the search (red is Y=81 against a Y=16 background); the chroma is what makes the
# written file actually the colour it claims to be.
SOLID_COLOURS_YUV = {
    'red': (81, 90, 240),
    'green': (145, 54, 34),
    'blue': (41, 240, 110),
    'white': (235, 128, 128),
}
BLACK_YUV = (16, 128, 128)


def gen_moving_square(height: int, width: int, n_frames: int, vy: int, vx: int,
                      size: int = None, y0: int = None, x0: int = None,
                      colour: str = 'red') -> tuple:
    """A hard-edged solid square translating (vy, vx) px/frame over a flat background.

    `vy, vx` is the *object's* velocity, which is the negation of the motion vector the
    search should report. The object at p in the reference sits at p + v in the current
    frame, so curr(y, x) = ref(y - vy, x - vx) and the estimate is (-vy, -vx). That is
    the opposite sign to `gen_translation`, where the camera pans and the estimate
    equals the velocity -- checking a case of each is what actually pins the convention
    down, since a single sign error would satisfy one and fail the other.

    The square's interior and the background are both flat, so every candidate scores
    SAD 0 there and only blocks straddling an edge carry information. That is the
    aperture problem at its most extreme, and it is the point: on this clip the decision
    margin, not the vector, is the field to read.

    Returns (y_frames, uv_frames, gt) -- unlike the texture generators, which are luma
    only and return (frames, gt).
    """
    if colour not in SOLID_COLOURS_YUV:
        raise ValueError(f'unknown colour {colour!r}; '
                         f'choose from {sorted(SOLID_COLOURS_YUV)}')
    size = (min(height, width) // 3) & ~3 if size is None else size
    span_y, span_x = vy * (n_frames - 1), vx * (n_frames - 1)
    # Start so the square is centred over its whole trajectory, then verify it fits.
    y0 = (height - size - span_y) // 2 if y0 is None else y0
    x0 = (width - size - span_x) // 2 if x0 is None else x0
    last_y, last_x = y0 + span_y, x0 + span_x
    if min(y0, last_y) < 0 or max(y0, last_y) + size > height or \
       min(x0, last_x) < 0 or max(x0, last_x) + size > width:
        raise ValueError(f'a {size}px square moving ({vy}, {vx}) for {n_frames} frames '
                         f'leaves the {width}x{height} frame')

    yc, uc, vc = SOLID_COLOURS_YUV[colour]
    yb, ub, vb = BLACK_YUV
    y_frames, uv_frames = [], []
    for f in range(n_frames):
        oy, ox = y0 + f * vy, x0 + f * vx
        y = np.full((height, width), yb, dtype=np.float32)
        y[oy:oy + size, ox:ox + size] = yc
        u = np.full((height // 2, width // 2), ub, dtype=np.float32)
        v = np.full((height // 2, width // 2), vb, dtype=np.float32)
        u[oy // 2:(oy + size) // 2, ox // 2:(ox + size) // 2] = uc
        v[oy // 2:(oy + size) // 2, ox // 2:(ox + size) // 2] = vc
        y_frames.append(y)
        uv_frames.append((u, v))

    gt = {'type': 'moving_square', 'colour': colour, 'size': size,
          'obj_vy': vy, 'obj_vx': vx, 'mv_dy': -vy, 'mv_dx': -vx,
          'origin_y': y0, 'origin_x': x0}
    return y_frames, uv_frames, gt
