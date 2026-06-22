# Reproducibility code — real-data evaluation of adaptive multi-sensor fusion

All analysis code for the paper. Every table, figure, statistic, and significance test is
reproduced by running these scripts on the cited public datasets. No data is bundled here
(datasets are large and public); download them from the DOIs below.

## Datasets (public)
- **IPIN 2020 Competition Track 3** — Zenodo, DOI 10.5281/zenodo.4314992 (CC BY 4.0).
- **OutFin** — figshare, DOI 10.6084/m9.figshare.12069993 (CC BY 4.0).
- **GSDC 2023** (Google Smartphone Decimeter Challenge) — Kaggle: kaggle.com/competitions/smartphone-decimeter-2023.

## Environment
Python 3.9+. `pip install numpy scipy scikit-learn pandas matplotlib pymupdf`.

## Files
| Script | Reproduces |
|---|---|
| `ahgf_method.py` | The fusion method + baselines (EKF, AHGF NIS weighting, Sage-Husa / Innovation-AEKF / Huber). |
| `ipin.py` | IPIN logfile parser + ground-truth track. |
| `ipin_fingerprint.py` | IPIN Wi-Fi fingerprinting positioner (Table 1 indoor result). |
| `outfin_fuse.py` | OutFin per-modality accuracy (Table 1) + 4-modality fusion (Table 2) + paired significance. |
| `gsdc.py` | GSDC GNSS WLS engine (validated vs ground truth + official baseline). |
| `gsdc_fuse.py` | Per-constellation fusion experiment. |
| `gsdc_robust.py` | Innovation-robust WLS. |
| `gsdc_robust_ekf.py` | Robust + innovation-gated EKF (Table 3), trip-level significance, satellite-count segmentation + official-baseline comparison (Table 4); saves arrays for figures. |
| `gsdc_learn.py` | Learned reliability model (leave-trip-out). |
| `figures.py` | Figure 1 (GNSS error CDFs). |

## Run
Point the dataset-path constants at your local copies, then e.g.:
`python outfin_fuse.py`, `python gsdc_robust_ekf.py`, `python gsdc_learn.py`, `python figures.py`.

## Integrity
All reported results are produced by executing this code on the public datasets above.
No data, results, or citations are fabricated; negative/tie results are reported as such.
