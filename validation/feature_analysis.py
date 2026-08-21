"""Univariate and multivariate analysis of the motion features against TC_gt (Phase 5).

    python validation/feature_analysis.py --results validation/results/<run>

Univariate: frame-level PCC/SRCC of every emitted feature against the Low-Delay-P
frame bits, per QP, with bootstrap CIs.

Multivariate: ridge regression predicting log(bits) from the feature block, scored by
leave-one-sequence-out cross-validation. Holding out whole sequences is the honest
protocol here: frames within a sequence are highly correlated, so a random split would
report a score that says nothing about generalising to unseen content.

Ridge is implemented directly in numpy (closed form) to avoid adding scikit-learn as a
dependency for one estimator.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from validation.report import format_markdown  # noqa: E402
from validation.stats import correlation_record  # noqa: E402

# Feature block for the multivariate fit, per the Phase 5 specification.
RIDGE_FEATURES = ['TC_MC', 'TC_SAD', 'MVC', 'GMV_mag', 'MV_coherence', 'MV_div',
                  'intra_frac', 'skip_frac']


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float):
    """Closed-form ridge on standardised features with an unpenalised intercept."""
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    Xs = (X - mu) / sigma
    y_mean = y.mean()
    n_feat = Xs.shape[1]
    w = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(n_feat), Xs.T @ (y - y_mean))
    return {'w': w, 'mu': mu, 'sigma': sigma, 'y_mean': y_mean}


def ridge_predict(model, X: np.ndarray) -> np.ndarray:
    return ((X - model['mu']) / model['sigma']) @ model['w'] + model['y_mean']


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def leave_one_sequence_out(df: pd.DataFrame, features: list, target: str,
                           alpha: float = 1.0) -> pd.DataFrame:
    """Ridge with each sequence held out in turn, scored on the held-out frames.

    Two R^2 values are reported because they answer different questions:

    * `R2` is the raw score. It asks whether the model predicts the held-out sequence's
      bitrate in absolute terms, and it is brutal: a constant offset between the
      training sequences' bit levels and the held-out one's is charged against a
      within-sequence variance that is comparatively tiny, so it goes strongly negative
      whenever absolute transfer fails.
    * `R2_centered` removes the held-out sequence's mean from both prediction and
      target first, so it measures only whether the model tracks frame-to-frame
      variation inside that sequence. That is what a per-frame complexity feature is
      actually for; an ABR ladder is fitted per title anyway.
    """
    rows = []
    for held in sorted(df['seq_name'].unique()):
        train, test = df[df['seq_name'] != held], df[df['seq_name'] == held]
        if len(train) < len(features) + 2 or len(test) < 3:
            continue
        model = ridge_fit(train[features].to_numpy(float),
                          train[target].to_numpy(float), alpha)
        pred = ridge_predict(model, test[features].to_numpy(float))
        actual = test[target].to_numpy(float)
        rows.append({'held_out': held, 'n_test': len(test),
                     'R2': r2(actual, pred),
                     'R2_centered': r2(actual - actual.mean(), pred - pred.mean()),
                     'PCC': float(np.corrcoef(actual, pred)[0, 1])})
    out = pd.DataFrame(rows)
    if not out.empty:
        out.loc[len(out)] = {'held_out': 'MEAN', 'n_test': int(out['n_test'].sum()),
                             'R2': out['R2'].mean(),
                             'R2_centered': out['R2_centered'].mean(),
                             'PCC': out['PCC'].mean()}
    return out


def univariate(df: pd.DataFrame, features: list, gt_col: str, qps: list,
               n_boot: int) -> pd.DataFrame:
    rows = []
    for qp in qps:
        sub = df[df['qp'] == qp]
        if len(sub) < 3:
            continue
        for feat in features:
            if feat not in sub.columns:
                continue
            x = sub[feat].to_numpy(float)
            y = sub[gt_col].to_numpy(float)
            keep = np.isfinite(x) & np.isfinite(y)
            if keep.sum() < 3 or np.ptp(x[keep]) == 0:
                continue
            groups = [(g[feat].to_numpy(float), g[gt_col].to_numpy(float))
                      for _, g in sub[keep].groupby('seq_name') if len(g) >= 3]
            rec = correlation_record(feat, 'pooled', x[keep], y[keep],
                                     groups=groups, n_boot=n_boot)
            per_seq = [np.corrcoef(a, b)[0, 1] for a, b in groups if np.ptp(a) > 0]
            rows.append({'QP': qp, 'feature': feat, 'PCC': rec['PCC'],
                         'PCC_lo': rec['PCC_lo'], 'PCC_hi': rec['PCC_hi'],
                         'SRCC': rec['SRCC'], 'PCC_log': rec['PCC_log'],
                         'perseq_PCC': float(np.mean(per_seq)) if per_seq else np.nan})
    return pd.DataFrame(rows)


def load_run(results_dir: Path):
    """Merges a run's per-frame features with its per-frame ground truth."""
    evca = pd.read_csv(results_dir / 'evca_frames.csv')
    gt = pd.read_csv(results_dir / 'ground_truth_frames.csv')
    # Strip the `<profile>_` prefix the benchmark adds, keeping the full profile.
    renames = {c: c.split('_', 1)[1] for c in evca.columns
               if c.startswith('full_')}
    evca = evca.rename(columns=renames)
    merged = evca.merge(gt, on=['seq_name', 'frame_idx'], how='inner')
    merged = merged[(merged['frame_idx'] > 0) & (merged['pict_type'] == 'P')].copy()
    merged['log_bits'] = np.log(merged['bits_ldp'].clip(lower=1))
    return merged


def main() -> int:
    p = argparse.ArgumentParser(description='Phase 5 feature analysis.')
    p.add_argument('--results', required=True, help='a results directory')
    p.add_argument('--alpha', type=float, default=1.0, help='ridge penalty')
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--out', default=None, help='where to write CSVs (default: --results)')
    args = p.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out) if args.out else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_run(results_dir)
    qps = sorted(df['qp'].unique())
    print(f'{len(df)} frame/QP rows from {df["seq_name"].nunique()} sequences, QPs {qps}')

    exclude = {'seq_name', 'frame_idx', 'qp', 'pict_type', 'bits_ai', 'bits_ldp',
               't_enc_ldp', 'log_bits'}
    features = [c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    uni = univariate(df, features, 'bits_ldp', qps, args.n_boot)
    uni.to_csv(out_dir / 'univariate_features.csv', index=False)
    print('\n=== Univariate vs TC_gt (frame level, mean over QPs) ===')
    summary = (uni.groupby('feature')
               .agg(PCC=('PCC', 'mean'), PCC_lo=('PCC_lo', 'mean'),
                    SRCC=('SRCC', 'mean'), PCC_log=('PCC_log', 'mean'),
                    perseq_PCC=('perseq_PCC', 'mean'))
               .sort_values('PCC', key=abs, ascending=False).reset_index())
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / 'univariate_summary.csv', index=False)

    available = [f for f in RIDGE_FEATURES if f in df.columns]
    missing = [f for f in RIDGE_FEATURES if f not in df.columns]
    if missing:
        print(f'\nNote: features absent from this run, excluded: {missing}')

    print('\n=== Ridge, leave-one-sequence-out (target: log bits) ===')
    ridge_rows = []
    for qp in qps:
        sub = df[df['qp'] == qp]
        loso = leave_one_sequence_out(sub, available, 'log_bits', args.alpha)
        loso.insert(0, 'QP', qp)
        ridge_rows.append(loso)
        print(f'\nQP {qp}:')
        print(loso.to_string(index=False))
    ridge = pd.concat(ridge_rows, ignore_index=True)
    ridge.to_csv(out_dir / 'ridge_loso.csv', index=False)

    # Single-feature TC_MC reference, to show what the extra features buy.
    base_rows = []
    for qp in qps:
        sub = df[df['qp'] == qp]
        loso = leave_one_sequence_out(sub, ['TC_MC'], 'log_bits', args.alpha)
        loso.insert(0, 'QP', qp)
        base_rows.append(loso)
    base = pd.concat(base_rows, ignore_index=True)
    base.to_csv(out_dir / 'ridge_loso_tcmc_only.csv', index=False)

    def mean_of(frame, qp, col):
        return frame[(frame.QP == qp) & (frame.held_out == 'MEAN')][col].iloc[0]

    print('\n=== Mean held-out score: full feature block vs TC_MC alone ===')
    comp = pd.DataFrame({
        'QP': qps,
        'R2_all_features': [mean_of(ridge, q, 'R2') for q in qps],
        'R2_TC_MC_only': [mean_of(base, q, 'R2') for q in qps],
        'R2c_all_features': [mean_of(ridge, q, 'R2_centered') for q in qps],
        'R2c_TC_MC_only': [mean_of(base, q, 'R2_centered') for q in qps],
        'PCC_all_features': [mean_of(ridge, q, 'PCC') for q in qps],
        'PCC_TC_MC_only': [mean_of(base, q, 'PCC') for q in qps],
    })
    print(comp.to_string(index=False))
    comp.to_csv(out_dir / 'ridge_comparison.csv', index=False)

    with open(out_dir / 'feature_analysis_meta.json', 'w') as f:
        json.dump({'features': available, 'excluded': missing, 'alpha': args.alpha,
                   'qps': [int(q) for q in qps],
                   'sequences': sorted(df['seq_name'].unique().tolist())}, f, indent=2)
    print(f'\nWritten to {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
