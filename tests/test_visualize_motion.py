"""The motion-vector visualizer: cost-volume hook, sign conventions, and rendering.

Renders at tiny sizes into tmp_path, so the suite stays CPU-only and fast. The point is
not that the figures look right -- that is what `--synthetic` is for, by eye -- but that
the quantities they are drawn from stay correct, above all the sign convention, which is
the one error that would make every figure confidently wrong.
"""
import numpy as np
import pytest
import torch

from libs.temporal_engine import PatternBlockMatcher
from validation import visualize_motion as vm
from validation.synthetic import gen_translation


def _batch(frame):
    return torch.from_numpy(np.ascontiguousarray(frame)).unsqueeze(0).unsqueeze(0).float()


def _pair(vy=0, vx=4, seed=21, h=128, w=160):
    frames, _ = gen_translation(h, w, 3, vy=vy, vx=vx, seed=seed)
    return _batch(frames[2]), _batch(frames[1])


# ------------------------------------------------------- the return_costs hook

def test_return_costs_is_consistent_with_the_reduction():
    """The exposed cost volume must be the volume the search actually reduced over."""
    curr, ref = _pair()
    matcher = PatternBlockMatcher(32, 'grid', offset=2)
    mvs, sad, costs, best_idx = matcher(curr, ref, return_costs=True)

    assert costs.shape == (1, matcher.num_cands, mvs.shape[2], mvs.shape[3])
    assert torch.equal(costs.argmin(dim=1), best_idx)
    assert torch.allclose(costs.min(dim=1).values.unsqueeze(1), sad)
    # The reported vector is the winning candidate's, in full-resolution pixels.
    assert torch.equal(matcher.pattern_lookup[best_idx].permute(0, 3, 1, 2), mvs)


def test_return_costs_leaves_the_default_path_alone():
    curr, ref = _pair()
    matcher = PatternBlockMatcher(32, 'diamond', offset=2)
    plain = matcher(curr, ref)
    assert len(plain) == 2
    withc = matcher(curr, ref, return_costs=True)
    assert len(withc) == 4
    assert torch.equal(plain[0], withc[0]) and torch.equal(plain[1], withc[1])


# ------------------------------------------------------------------- the scene

def _scene(tmp_path, vy=0, vx=4, size='160x128', extra=()):
    argv = ['--synthetic', f'translation:{vy},{vx}', '--synthetic-size', size,
            '--me-offset', str(max(abs(vy), abs(vx))), '--device', 'cpu', *extra]
    args = vm.build_viz_parser().parse_args(argv)
    import tempfile
    tmp = tempfile.TemporaryDirectory()
    truth = vm.setup_synthetic(args, tmp.name)
    scene = vm.build_scene(args, 0, torch.device('cpu'))
    scene.gt = truth
    tmp.cleanup()
    return scene


def test_scene_recovers_a_known_pan_exactly(tmp_path):
    scene = _scene(tmp_path, vy=0, vx=4)
    interior_dy = scene.mv[0][1:-1, 1:-1]
    interior_dx = scene.mv[1][1:-1, 1:-1]
    assert np.allclose(interior_dy, 0.0)
    assert np.allclose(interior_dx, 4.0)
    assert scene.metrics['mean_mv_mag'] == pytest.approx(4.0, abs=1e-6)


def test_arrows_are_drawn_as_apparent_motion(tmp_path):
    """The sign guard. `curr(y,x) ~= ref(y+dy, x+dx)`, so a crop window sliding right
    gives dx > 0 while the content appears to move *left*. Every direction the tool
    draws -- arrows and flow hues alike -- is the negated estimate, and a regression
    here would flip every figure without changing a single number."""
    scene = _scene(tmp_path, vy=0, vx=4)
    assert np.allclose(scene.apparent, -scene.mv)
    assert scene.mv[1].mean() > 0            # estimate points back into the reference
    assert scene.apparent[1].mean() < 0      # content is drawn moving left


def test_flow_colours_agree_with_the_wheel_they_are_read_against():
    """Zero motion is black; opposite directions must not collide onto one hue."""
    zero = vm.flow_to_rgb(np.zeros((2, 2)), np.zeros((2, 2)), 4.0)
    assert np.allclose(zero, 0.0)
    right = vm.flow_to_rgb(np.array([[0.0]]), np.array([[4.0]]), 4.0)
    left = vm.flow_to_rgb(np.array([[0.0]]), np.array([[-4.0]]), 4.0)
    assert not np.allclose(right, left)


def test_margin_is_the_gap_to_the_runner_up(tmp_path):
    scene = _scene(tmp_path, vy=0, vx=2, extra=['--heuristic', 'grid'])
    ordered = np.sort(scene.costs, axis=0)
    assert np.allclose(scene.margin, ordered[1] - ordered[0], atol=1e-5)
    assert np.allclose(ordered[0], scene.sad, atol=1e-5)
    assert (scene.margin >= 0).all()


def test_pattern_grid_shape_only_claims_real_grids(tmp_path):
    grid = _scene(tmp_path, extra=['--heuristic', 'grid'])
    assert vm.pattern_grid_shape(grid) == (9, 9)      # offset 4, pool 1 -> (2*4+1)^2
    assert len(grid.pattern) == 81
    diamond = _scene(tmp_path, extra=['--heuristic', 'diamond'])
    assert vm.pattern_grid_shape(diamond) is None


def test_pooled_run_keeps_the_two_domains_straight(tmp_path):
    """MVs stay in full-resolution pixels while the residual moves to the pooled grid."""
    scene = _scene(tmp_path, vy=0, vx=4, extra=['--temporal-pool', '2'])
    assert scene.pool == 2
    assert scene.bs_r == scene.bs // 2
    assert scene.curr_r.shape == (scene.H // 2, scene.W // 2)
    assert scene.mv.shape[1:] == (scene.Hb, scene.Wb) == (scene.H // scene.bs,
                                                          scene.W // scene.bs)
    # The per-block helpers must land on the MV grid despite the smaller blocks.
    assert scene.gain.shape == (scene.Hb, scene.Wb)


def test_compensation_beats_standing_still_on_a_clean_pan(tmp_path):
    scene = _scene(tmp_path, vy=0, vx=4)
    assert np.abs(scene.resid_r).mean() < 0.05 * np.abs(scene.diff_r).mean()
    interior = scene.gain[1:-1, 1:-1]
    assert (interior > 0).all(), 'compensation must help every interior block here'


# ------------------------------------------------------------------- the crop

def test_view_snaps_outwards_and_crops_every_domain_in_step(tmp_path):
    scene = _scene(tmp_path, extra=['--temporal-pool', '2'])
    view = vm.make_view(scene, (40, 40, 40, 40))
    assert (view.y0 % scene.bs, view.x0 % scene.bs) == (0, 0)
    assert view.h % scene.bs == 0 and view.w % scene.bs == 0
    assert view.y0 <= 40 and view.y0 + view.h >= 80
    # One region, three resolutions, same blocks.
    assert view.full(scene.curr).shape == (view.h, view.w)
    assert view.pooled(scene.curr_r, 2).shape == (view.h // 2, view.w // 2)
    assert view.blocks(scene.sad, scene.bs).shape == (view.h // scene.bs,
                                                      view.w // scene.bs)


def test_full_view_is_the_whole_frame(tmp_path):
    scene = _scene(tmp_path)
    view = vm.make_view(scene, None)
    assert np.array_equal(view.full(scene.curr), scene.curr)


# ------------------------------------------------------------------ end to end

def test_cli_renders_every_figure(tmp_path):
    """The whole tool, through the real CLI, on ground truth it must reproduce."""
    out = tmp_path / 'figs'
    rc = vm.main(['--synthetic', 'translation:0,4', '--synthetic-size', '160x128',
                  '--me-offset', '4', '--device', 'cpu', '--dpi', '50',
                  '--worst', 'sad', '2', '--out', str(out)])
    assert rc == 0
    written = sorted(p.name for p in (out / 'synthetic_translation_f0000').iterdir())
    assert written == ['1_evidence.png', '2_field.png', '3_confidence.png',
                       '4_compensation.png', '5_blocks.png']
    assert all((out / 'synthetic_translation_f0000' / n).stat().st_size > 0
               for n in written)


def test_cli_rejects_a_frame_past_the_end(tmp_path):
    with pytest.raises(SystemExit):
        vm.main(['--synthetic', 'translation:0,2', '--synthetic-size', '96x96',
                 '--frame', '900', '--device', 'cpu', '--out', str(tmp_path)])


def test_cli_subsets_figures(tmp_path):
    out = tmp_path / 'one'
    rc = vm.main(['--synthetic', 'static_noise', '--synthetic-size', '96x96',
                  '--figures', 'field', '--device', 'cpu', '--dpi', '50',
                  '--out', str(out)])
    assert rc == 0
    assert [p.name for p in (out / 'synthetic_static_noise_f0000').iterdir()] == \
           ['2_field.png']


# ------------------------------------------------- a moving object, not a moving camera

def _square_scene(vy, vx, heuristic='diamond', size='256x192', offset=2):
    """A scene from `--synthetic square:vy,vx`, plus its ground-truth vector."""
    import tempfile
    argv = ['--synthetic', f'square:{vy},{vx}', '--synthetic-size', size,
            '-b', '32', '--heuristic', heuristic, '--me-offset', str(offset),
            '--device', 'cpu']
    args = vm.build_viz_parser().parse_args(argv)
    tmp = tempfile.TemporaryDirectory()
    truth = vm.setup_synthetic(args, tmp.name)
    scene = vm.build_scene(args, 0, torch.device('cpu'))
    scene.gt = truth
    tmp.cleanup()
    return scene


def test_moving_square_geometry_and_colour():
    from validation.synthetic import BLACK_YUV, SOLID_COLOURS_YUV, gen_moving_square
    y, uv, gt = gen_moving_square(192, 256, 2, vy=0, vx=-2)
    assert len(y) == len(uv) == 2
    assert gt['mv_dy'] == 0 and gt['mv_dx'] == 2      # the negation of the velocity
    yc, uc, vc = SOLID_COLOURS_YUV['red']
    assert set(np.unique(y[0])) == {float(BLACK_YUV[0]), float(yc)}
    assert set(np.unique(uv[0][0])) == {float(BLACK_YUV[1]), float(uc)}
    assert set(np.unique(uv[0][1])) == {float(BLACK_YUV[2]), float(vc)}
    # The square really translates: its leading column moves two pixels left.
    col = lambda f: int(np.where(y[f] == yc)[1].min())
    assert col(1) == col(0) - 2


def test_moving_square_refuses_to_leave_the_frame():
    from validation.synthetic import gen_moving_square
    with pytest.raises(ValueError, match='leaves the'):
        gen_moving_square(96, 96, 40, vy=0, vx=-8)


@pytest.mark.parametrize('vy,vx,heuristic', [
    (0, -2, 'diamond'),    # left
    (0, 2, 'diamond'),     # right
    (-2, 0, 'diamond'),    # up
    (2, 0, 'diamond'),     # down
    (2, 2, 'square'),      # down-right
    (-2, -2, 'square'),    # up-left
    (-2, 2, 'square'),     # up-right
    (2, -2, 'square'),     # down-left
])
def test_every_direction_is_recovered_exactly_where_the_block_can_know_it(vy, vx, heuristic):
    """A moving object pins the sign convention from the opposite side to a camera pan.

    The estimate must be the *negation* of the object's velocity. Only blocks with a
    non-zero decision margin are held to it: the square's interior and the background
    are flat, so every candidate ties there and the vector reported is the tie-break.
    """
    scene = _square_scene(vy, vx, heuristic)
    decisive = scene.margin > 1e-6
    assert decisive.any(), 'no block saw an edge; the test case is degenerate'
    assert not decisive.all(), 'flat regions should be ambiguous on this clip'
    assert np.array_equal(np.unique(scene.mv[0][decisive]), np.array([-vy], float))
    assert np.array_equal(np.unique(scene.mv[1][decisive]), np.array([-vx], float))
    assert scene.gt == (-vy, -vx)


def test_diamond_cannot_represent_diagonal_object_motion():
    """Negative control for the pattern-shape trade-off, in object-motion terms.

    The axis-only diamond has no diagonal candidate, so it splits a diagonal into blocks
    voting pure-vertical and blocks voting pure-horizontal. The field is visibly
    incoherent, and both the residual and MVC say so.
    """
    good = _square_scene(2, 2, 'square')
    bad = _square_scene(2, 2, 'diamond')
    decisive = bad.margin > 1e-6
    found = {(int(a), int(b)) for a, b in zip(bad.mv[0][decisive], bad.mv[1][decisive])}
    assert found == {(-2, 0), (0, -2)}, 'diamond should split the diagonal onto its axes'
    # The residual is the signal that something went wrong: the square pattern predicts
    # the frame exactly, the diamond cannot.
    assert good.metrics['TC_SAD'] == 0.0 < bad.metrics['TC_SAD']
    # MVC, however, ranks the *wrong* field as the smoother one, and is recorded here so
    # it is not mistaken for an accuracy measure. It averages |Laplacian of the MV
    # field|: the correct answer puts a step of 2 in both components at the object
    # boundary, while the split answer puts a step in only one component per block. A
    # true motion discontinuity is genuinely rough, so roughness cannot score accuracy.
    assert good.metrics['MVC'] > bad.metrics['MVC']


def test_flat_regions_report_a_zero_margin_rather_than_a_confident_wrong_vector():
    """The property the whole confidence layer exists to surface."""
    scene = _square_scene(0, -2, 'diamond')
    flat = scene.margin <= 1e-6
    assert flat.mean() > 0.5
    # Every candidate ties at zero cost there, so the winning cost is zero too.
    assert np.allclose(scene.sad[flat], 0.0)
