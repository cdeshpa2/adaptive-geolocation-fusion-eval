# Adaptive Geolocation Fusion — Reproducible Real-Data Evaluation

Analysis and reproducibility code for the paper:

> Chaitanya Ravindra Deshpande. *Reassessing Innovation-Driven Adaptive Sensor Fusion for
> Heterogeneous Geolocation: A Reproducible Real-Data Evaluation.* Under review,
> PeerJ Computer Science, 2026.

Archived release: Zenodo DOI [10.5281/zenodo.20790513](https://doi.org/10.5281/zenodo.20790513).

## Description

This repository contains the complete analysis pipeline used to evaluate innovation-driven
adaptive multi-sensor fusion for geolocation **exclusively on real, public benchmark data**.
Every table, figure, statistic, and significance test in the paper is reproduced by running
the scripts here on the three public datasets listed below. The code implements the
signal-to-position pipelines (raw Wi-Fi/BLE/cellular RSSI and GNSS pseudoranges → position),
the baselines (inverse-variance fusion, all-satellite weighted least squares, Huber robust
estimation), the corrected **pre-update normalized-innovation-squared (NIS)** weighting, an
innovation-robust GNSS pipeline, and a learned reliability model.

No datasets are bundled (they are large and third-party); download them from the sources below.

## Dataset information

| Dataset | Use in this study | Source / accession |
|---|---|---|
| IPIN 2020 Competition Track 3 | Indoor Wi-Fi/inertial trajectories | Zenodo, DOI 10.5281/zenodo.4314992 — <https://doi.org/10.5281/zenodo.4314992> (CC BY 4.0) |
| OutFin | Outdoor static multi-modal (Wi-Fi/BLE/cellular/GNSS) | figshare, DOI 10.6084/m9.figshare.12069993 — <https://doi.org/10.6084/m9.figshare.12069993> (CC BY 4.0) |
| GSDC 2023 | Smartphone raw GNSS drives | Google Smartphone Decimeter Challenge 2023, Kaggle — <https://www.kaggle.com/competitions/smartphone-decimeter-2023> |

After downloading, point the dataset-path constant(s) at the top of each script at your local
copies (see *Usage instructions*). Reproduction is subject to acceptance of each dataset's
license/terms. This repository does not redistribute any dataset.

## Code information

| Script | Reproduces |
|---|---|
| `ahgf_method.py` | Fusion method + baselines: EKF, pre-update-NIS adaptive weighting, Sage–Husa / innovation-AEKF / Huber. |
| `ipin.py` | IPIN logfile parser + ground-truth track reconstruction. |
| `ipin_fingerprint.py` | IPIN Wi-Fi fingerprinting positioner (Table 1, indoor result). |
| `outfin_fuse.py` | OutFin per-modality accuracy (Table 1) + 4-modality fusion (Table 2) + paired Wilcoxon significance. |
| `gsdc.py` | GSDC GNSS weighted-least-squares engine (validated vs. ground truth and the official baseline). |
| `gsdc_fuse.py` | Per-constellation fusion experiment (divergence analysis). |
| `gsdc_robust.py` | Innovation-robust WLS (Huber weighting + gross-outlier rejection). |
| `gsdc_robust_ekf.py` | Robust + innovation-gated EKF (Table 3); trip-level significance; satellite-count segmentation + official-baseline comparison (Table 4); saves arrays for figures. |
| `gsdc_learn.py` | Learned gradient-boosted reliability model (leave-trip-out). |
| `figures.py` | Figure 1 (GNSS horizontal-error CDFs). |
| `CITATION.cff` | Machine-readable citation metadata. |

## Requirements

- Python 3.9 or newer.
- Python packages listed in `requirements.txt`. Install with:

  ```bash
  pip install -r requirements.txt
  ```

  (numpy, scipy, scikit-learn, pandas, matplotlib, pymupdf)

## Usage instructions

1. Create and activate a virtual environment, then install requirements (above).
2. Download the three datasets from the sources in *Dataset information* into local folders.
3. Edit the dataset-path constant(s) near the top of the script you want to run so they point
   at your local copies.
4. Run a script, e.g.:

   ```bash
   python ipin_fingerprint.py     # Table 1 (indoor)
   python outfin_fuse.py          # Tables 1–2 (OutFin) + significance
   python gsdc.py                 # validate the GNSS WLS engine
   python gsdc_robust_ekf.py      # Tables 3–4 + trip-level significance
   python gsdc_learn.py           # learned reliability model
   python figures.py              # Figure 1
   ```

   Each script prints the numbers that appear in the corresponding table and, where relevant,
   writes intermediate arrays consumed by `figures.py`.

## Methodology

The evaluation deliberately avoids simulation. Each modality's raw signal is first converted
to a position observation (Wi-Fi/BLE/cellular via inverse-distance-weighted *k*-nearest-neighbour
fingerprinting; GNSS via weighted least squares over corrected pseudoranges), then fused.
Adaptive weights are derived from the **pre-update** normalized innovation squared to remove the
posterior self-reinforcement bias common in residual-weighted filters. All learned components
use leave-trip-out / leave-point-out splits, and ground truth is never an input feature.
Difficulty segmentation for the GNSS experiments uses a method-independent proxy (satellite
count) for the headline comparison, and statistical significance is assessed at the trip level
to respect within-trip temporal correlation. See the paper's Methods section for full detail.

## Citations

If you use this code, please cite the paper and the datasets:

- Deshpande, C. R. (2026). *Reassessing Innovation-Driven Adaptive Sensor Fusion for
  Heterogeneous Geolocation: A Reproducible Real-Data Evaluation.* Under review, PeerJ Computer
  Science. Code archive: Zenodo, DOI 10.5281/zenodo.20790513.
- Torres-Sospedra, J. et al. (2020). *IPIN 2020 Competition Track 3 datasets and supporting
  materials.* Zenodo. DOI 10.5281/zenodo.4314992.
- Alhomayani, F.; Mahoor, M. H. (2021). *OutFin: A multi-device and multi-modal dataset for
  outdoor localization based on Wi-Fi, Bluetooth, cellular and GPS signals.* Scientific Data 8, 66.
  DOI 10.1038/s41597-021-00832-y.
- Google Smartphone Decimeter Challenge 2023, Kaggle.

A machine-readable citation is provided in `CITATION.cff`.

## License & contribution guidelines

- **Code license:** MIT (see `LICENSE`).
- **Datasets:** governed by their own licenses at the sources above (IPIN and OutFin are
  CC BY 4.0; GSDC is under the Kaggle competition terms). This repository redistributes none of them.
- **Contributions:** issues and pull requests are welcome via the GitHub repository. For
  questions or to report a problem, open an issue or contact the author (cdeshpa2@gmail.com).

## Integrity

All reported results are produced by executing this code on the public datasets above. No data,
results, or citations are fabricated; negative and tie results are reported as such.
