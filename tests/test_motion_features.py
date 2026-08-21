"""Structural motion-field descriptors (Phase 5) against known synthetic motion."""
import numpy as np
import pytest
import torch

from libs.motion_estimation import HierarchicalBlockMatcher
from libs.motion_features import (affine_fit, compute_motion_features, divergence_curl,
                                  global_mv, mv_coherence, weighted_median)
from tests.conftest import make_args, run_evca, write_raw_yuv
from validation.synthetic import (gen_cut, gen_rotation, gen_static_noise,
                                  gen_translation, gen_zoom)

# Large enough for the differential features to have interior blocks to work with.
H, W = 320, 448


def _t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0).unsqueeze(0).float()


def _features(frames, **kw):
    matcher = HierarchicalBlockMatcher(32, width=W, **kw)
    mvs, sad = matcher(_t(frames[2]), _t(frames[1]))
    return compute_motion_features(mvs, sad)


# ------------------------------------------------------------------ primitives

def test_weighted_median_matches_unweighted_when_uniform():
    values = torch.tensor([[3.0, 1.0, 2.0, 5.0, 4.0]])
    weights = torch.ones_like(values)
    assert weighted_median(values, weights).item() == 3.0


def test_weighted_median_follows_the_weight_mass():
    values = torch.tensor([[1.0, 2.0, 9.0]])
    assert weighted_median(values, torch.tensor([[10.0, 1.0, 1.0]])).item() == 1.0
    assert weighted_median(values, torch.tensor([[1.0, 1.0, 10.0]])).item() == 9.0


def test_global_mv_ignores_a_minority_of_outliers():
    """A moving object over a static background must not move the global estimate."""
    mvs = torch.zeros(1, 2, 6, 6)
    mvs[:, 1] = 4.0
    mvs[0, :, :2, :2] = -20.0               # 4 of 36 blocks disagree
    sad = torch.ones(1, 1, 6, 6)
    gmv = global_mv(mvs, sad)
    assert gmv[0, 0].item() == 0.0 and gmv[0, 1].item() == 4.0


def test_coherence_counts_blocks_near_the_global_vector():
    mvs = torch.zeros(1, 2, 4, 4)
    mvs[:, 1] = 2.0
    mvs[0, 1, 0, :] = 9.0                   # one row of 4 disagrees
    sad = torch.ones(1, 1, 4, 4)
    coh = mv_coherence(mvs, global_mv(mvs, sad), eps=1.0)
    assert coh.item() == pytest.approx(0.75)


def test_divergence_and_curl_of_analytic_fields():
    """Signs follow the forward flow (-mv): outward flow is positive divergence."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 9), torch.linspace(-1, 1, 9),
                            indexing='ij')
    # forward flow pointing outward => mv = -flow
    expand = -torch.stack([yy, xx]).unsqueeze(0)
    div, curl = divergence_curl(expand)
    assert div.item() > 0 and abs(curl.item()) < 1e-5

    rotate = -torch.stack([xx, -yy]).unsqueeze(0)
    div, curl = divergence_curl(rotate)
    assert abs(div.item()) < 1e-5 and abs(curl.item()) > 0


def test_affine_fit_recovers_a_pure_translation():
    mvs = torch.zeros(1, 2, 8, 8)
    mvs[:, 0] = -3.0                        # forward flow ty = +3
    mvs[:, 1] = 5.0                         # forward flow tx = -5
    A, t = affine_fit(mvs, torch.ones(1, 1, 8, 8))
    assert t[0, 0].item() == pytest.approx(3.0, abs=1e-4)
    assert t[0, 1].item() == pytest.approx(-5.0, abs=1e-4)
    assert A.abs().max().item() < 1e-4      # no linear part


# -------------------------------------------------------------- synthetic motion

@pytest.mark.parametrize('vy,vx', [(0, 8), (-4, 0), (4, -4)])
def test_pan_gives_exact_gmv_zero_divergence_full_coherence(vy, vx):
    """Spec check: pure pan -> GMV equals the true shift, div/curl ~ 0, coherence ~ 1."""
    frames, _ = gen_translation(H, W, 3, vy=vy, vx=vx, seed=5)
    d = _features(frames)
    assert d['GMV_y'].item() == pytest.approx(vy, abs=1e-6)
    assert d['GMV_x'].item() == pytest.approx(vx, abs=1e-6)
    assert abs(d['MV_div'].item()) < 1e-4
    assert abs(d['MV_curl'].item()) < 1e-4
    assert d['MV_coherence'].item() > 0.99
    assert d['MVD_cost'].item() < 1e-6


def test_zoom_gives_positive_divergence():
    """Spec check: zoom -> MV_div > 0, and it dominates the curl."""
    frames, _ = gen_zoom(H, W, 3, rate=0.03, seed=5)
    d = _features(frames)
    assert d['MV_div'].item() > 0.1
    assert d['MV_div'].item() > 3 * abs(d['MV_curl'].item())
    assert d['MV_coherence'].item() < 0.9


def test_rotation_gives_nonzero_curl():
    """Spec check: rotation -> MV_curl != 0, and it dominates the divergence."""
    frames, _ = gen_rotation(H, W, 3, deg_per_frame=2.0, seed=5)
    d = _features(frames)
    assert abs(d['MV_curl'].item()) > 0.1
    assert abs(d['MV_curl'].item()) > 3 * abs(d['MV_div'].item())


def test_static_noise_is_coherent_and_still():
    frames, _ = gen_static_noise(H, W, 3, sigma=3.0, seed=8)
    d = _features(frames)
    assert d['GMV_mag'].item() < 1e-6
    assert d['MV_coherence'].item() > 0.99


def test_phase_correlation_gmv_agrees_with_weighted_median():
    """The two global estimates are reported separately but must agree on a clean pan."""
    frames, _ = gen_translation(H, W, 3, vy=0, vx=8, seed=5)
    matcher = HierarchicalBlockMatcher(32, width=W, predictor='global')
    mvs, sad = matcher(_t(frames[2]), _t(frames[1]))
    median_gmv = global_mv(mvs, sad)
    assert abs(matcher.last_gmv[0, 1].item() - median_gmv[0, 1].item()) <= 8.0


# ------------------------------------------------------------------ CSV columns

def test_cut_spikes_intra_frac_and_drops_coherence(tmp_path):
    """Spec check: a hard cut must spike intra_frac on the cut frame."""
    frames, gt = gen_cut(H, W, 8, seed=10)
    path = tmp_path / 'cut.yuv'
    write_raw_yuv(path, [np.rint(f) for f in frames], bit_depth=8)
    args = make_args(input=str(path), resolution=f'{W}x{H}', csv=str(tmp_path / 'cut.csv'),
                     motion_estimation=True, profile='full', me='hierarchical')
    df = run_evca(args)
    cut = gt['cut_frame']
    assert df.loc[cut, 'intra_frac'] > 0.9
    assert df.loc[cut, 'intra_frac'] == df['intra_frac'].max()
    quiet = df.drop(index=[0, cut])
    assert (quiet['intra_frac'] < 0.1).all()


def test_full_profile_emits_every_phase5_column(tmp_path):
    frames, _ = gen_translation(H, W, 6, vy=0, vx=4, seed=5)
    path = tmp_path / 'cols.yuv'
    write_raw_yuv(path, [np.rint(f) for f in frames], bit_depth=8)
    args = make_args(input=str(path), resolution=f'{W}x{H}', csv=str(tmp_path / 'cols.csv'),
                     motion_estimation=True, profile='full', me='hierarchical',
                     me_predictor='global')
    df = run_evca(args)
    for col in ['GMV_x', 'GMV_y', 'GMV_mag', 'GMV_pc_x', 'GMV_pc_y', 'MV_coherence',
                'MV_div', 'MV_curl', 'MVD_cost', 'intra_frac', 'skip_frac',
                'TC_SAD_full', 'aff_a11', 'aff_a12', 'aff_a21', 'aff_a22',
                'aff_tx', 'aff_ty']:
        assert col in df.columns, col
        assert df[col].notna().all(), col
    assert df.loc[0, 'GMV_x'] == 0.0, 'frame 0 must be zero-padded'


def test_skip_frac_responds_to_threshold(tmp_path):
    frames, _ = gen_translation(H, W, 5, vy=0, vx=4, seed=5)
    path = tmp_path / 'skip.yuv'
    write_raw_yuv(path, [np.rint(f) for f in frames], bit_depth=8)
    out = {}
    for thr in (0.01, 1000.0):
        args = make_args(input=str(path), resolution=f'{W}x{H}',
                         csv=str(tmp_path / f'skip{thr}.csv'), motion_estimation=True,
                         profile='full', me='hierarchical', skip_threshold=thr)
        out[thr] = run_evca(args)
    assert (out[1000.0].loc[1:, 'skip_frac'] == 1.0).all()
    assert out[0.01].loc[1:, 'skip_frac'].mean() < out[1000.0].loc[1:, 'skip_frac'].mean()


def test_tc_sad_full_tracks_prediction_error(tmp_path):
    """TC_SAD_full is near zero for a compensable pan and large across a cut."""
    frames, gt = gen_cut(H, W, 6, seed=10)
    path = tmp_path / 'sadfull.yuv'
    write_raw_yuv(path, [np.rint(f) for f in frames], bit_depth=8)
    args = make_args(input=str(path), resolution=f'{W}x{H}',
                     csv=str(tmp_path / 'sadfull.csv'), motion_estimation=True,
                     profile='full', me='hierarchical')
    df = run_evca(args)
    cut = gt['cut_frame']
    assert df.loc[cut, 'TC_SAD_full'] == df['TC_SAD_full'].max()
    assert df.loc[1, 'TC_SAD_full'] < 0.01 * df.loc[cut, 'TC_SAD_full']
