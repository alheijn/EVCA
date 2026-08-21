"""Per-sequence frame-level scatter plots for the final report.

    python validation/plots.py --old <results_dir> --new <results_dir> --out png/report

Draws, for each sequence, the frame-level relationship between a temporal metric and
the Low-Delay-P frame bits at one QP, with the old and new configurations side by side
so the change in fit is visible per sequence rather than only in a pooled number.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))


def load(results_dir: Path, metric: str) -> pd.DataFrame:
    evca = pd.read_csv(results_dir / 'evca_frames.csv')
    gt = pd.read_csv(results_dir / 'ground_truth_frames.csv')
    if metric not in evca.columns:
        raise KeyError(f'{metric} not in {results_dir}; have {list(evca.columns)}')
    df = evca[['seq_name', 'frame_idx', metric]].merge(gt, on=['seq_name', 'frame_idx'])
    return df[(df['frame_idx'] > 0) & (df['pict_type'] == 'P')]


def scatter_grid(datasets: dict, metric: str, qp: int, out_path: Path,
                 dpi: int = 120) -> Path:
    """One column per sequence, one row per configuration."""
    sequences = sorted(next(iter(datasets.values()))['seq_name'].unique())
    n_rows, n_cols = len(datasets), len(sequences)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.0 * n_rows),
                             squeeze=False)

    for r, (label, df) in enumerate(datasets.items()):
        sub_qp = df[df['qp'] == qp]
        for c, seq in enumerate(sequences):
            ax = axes[r][c]
            g = sub_qp[sub_qp['seq_name'] == seq]
            x, y = g[metric].to_numpy(float), g['bits_ldp'].to_numpy(float) / 1000.0
            ax.scatter(x, y, s=9, alpha=0.55, edgecolors='none')
            if len(g) > 2 and np.ptp(x) > 0:
                pcc = np.corrcoef(x, y)[0, 1]
                slope, icpt = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, slope * xs + icpt, lw=1.4, color='crimson')
                ax.set_title(f'{seq}\n{label}: PCC {pcc:.3f}', fontsize=9)
            else:
                ax.set_title(f'{seq}\n{label}: n/a', fontsize=9)
            if c == 0:
                ax.set_ylabel('P-frame bits (kbit)', fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel(metric, fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(f'Frame-level {metric} vs Low-Delay-P bits at QP {qp}', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--old', required=True, help='results dir for the old configuration')
    p.add_argument('--new', required=True, help='results dir for the new configuration')
    p.add_argument('--old-label', default='iter4')
    p.add_argument('--new-label', default='new default')
    p.add_argument('--metric', default='full_TC_MC')
    p.add_argument('--also-metric', default='baseline_TC',
                   help='second metric plotted from the old run for reference')
    p.add_argument('--qps', default='27,32')
    p.add_argument('--out', default='png/report')
    p.add_argument('--dpi', type=int, default=120)
    args = p.parse_args()

    out_dir = Path(args.out)
    written = []
    for qp in [int(q) for q in args.qps.split(',')]:
        datasets = {args.old_label: load(Path(args.old), args.metric),
                    args.new_label: load(Path(args.new), args.metric)}
        written.append(scatter_grid(datasets, args.metric, qp,
                                    out_dir / f'scatter_{args.metric}_qp{qp}.png',
                                    args.dpi))
        if args.also_metric:
            try:
                ref = {f'{args.also_metric} ({args.old_label})':
                       load(Path(args.old), args.also_metric)}
                written.append(scatter_grid(ref, args.also_metric, qp,
                                            out_dir / f'scatter_{args.also_metric}_qp{qp}.png',
                                            args.dpi))
            except KeyError as exc:
                print(f'skipping {args.also_metric}: {exc}')

    for path in written:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
