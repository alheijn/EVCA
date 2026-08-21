# EVCA

EVCA is an open-source video complexity analyzer with the following functionalities:

- EVCA is an advanced tool that integrates the functionalities of both VCA and SITI approaches,
- EVCA is developed in Python, ensuring compatibility with GPU processing,
- EVCA enhances the definition of temporal complexity that was originally used in VCA.

## Installation

You may install dependencies using the following command:

```
pip3 install -r requirements.txt
```

## Command Line Options

Run `python main.py --help` for the full list. Upstream options are documented at
https://github.com/cd-athena/EVCA/wiki/EVCA

## Temporal complexity extension

This fork reworks the motion-estimation and motion-compensation path behind `-me`.
The defaults were chosen from measured correlation against x265 output; see
[`validation/REPORT.md`](validation/REPORT.md) for the study and
[`validation/RESULTS.md`](validation/RESULTS.md) for the run-by-run ledger.

```bash
python main.py -i input.yuv -r 1920x1080 -me --profile full
```

That runs a hierarchical pyramid search with a neighbour-merge pass, a Gaussian-smoothed
dense warp, and ungated residual energy including the DC term. `--preset iter4` restores
the previous configuration. Selected flags:

| flag | default | meaning |
|---|---|---|
| `--me {hierarchical,pattern}` | `hierarchical` | pyramid search, or the older sparse ±6 px pattern |
| `--me-subpel {0,1,2}` | `0` | integer / half-pel / quarter-pel refinement |
| `--me-merge` / `--no-me-merge` | on | re-test each block against its neighbours' vectors |
| `--me-criterion {sad,satd}` | `sad` | block cost function |
| `--mc {dense_smooth,dense,block,obmc}` | `dense_smooth` | motion-compensation strategy |
| `--gate {none,intra}` | `none` | cap the residual energy at the block's intra energy |
| `--residual-dc` / `--no-residual-dc` | on | keep the residual's DC coefficient |
| `--rho` | off | emit `rho_qp22..rho_qp37` coefficient counts |

Validation harness:

```bash
python validation/run_benchmark.py --subset fast --label myrun
python -m pytest tests/ -q
```

## Citation

If this work is helpful for your research, please consider citing EVCA.

```
@inproceedings{amirpour_evca_2024,
author = {Amirpour, Hadi and Ghasempour, Mohammad and Qu, Lingfeng and Hamidouche, Wassim and Timmerer, Christian},
title = {{EVCA: Enhanced Video Complexity Analyzer}},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
booktitle = {Proceedings of the 15th ACM Multimedia Systems Conference},
series = {MMSys '24} }
```
