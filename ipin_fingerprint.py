#!/usr/bin/env python3
"""
IPIN 2020 — WiFi fingerprinting positioner (the raw signal->position step the
paper hand-waves; answers Reviewer 1 #3 for the WiFi modality).

Builds a WiFi radiomap by pairing each training scan's RSSI vector with the
piecewise-linear-interpolated POSI ground truth, then positions held-out
validation scans by weighted k-NN in RSSI space. Reports REAL horizontal error
(median / RMSE / IPIN 75th-pct) and floor accuracy on real walked trajectories.

This is the WiFi *measurement* model. The EKF/AHGF fusion (WiFi-position + PDR)
builds on top of it in ipin_fuse.py.
"""
import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipin import parse_logfile, make_gt, latlon_to_local, TRAIN_DIR, VAL_DIR  # noqa: E402

FILL = -100.0  # dBm sentinel for an AP not heard in a scan


def _origin(files):
    """Global local-frame origin = min lat/lon over all POSI in the given files."""
    lats, lons = [], []
    for f in files:
        for _, la, lo, _ in parse_logfile(f)["posi"]:
            lats.append(la); lons.append(lo)
    return min(lats), min(lons)


def build_samples(files, lat0, lon0):
    """List of (scan: {mac: best_rss}, east_m, north_m, floor) over all files."""
    out = []
    for f in files:
        d = parse_logfile(f)
        if not d["posi"]:
            continue
        gt = make_gt(d["posi"])
        for ts, obs in d["wifi"].items():
            p = gt(ts)
            if p is None:
                continue
            lat, lon, floor = p
            e, n = latlon_to_local(lat, lon, lat0, lon0)
            sd = {}
            for mac, _freq, rss in obs:
                if mac not in sd or rss > sd[mac]:
                    sd[mac] = rss
            if sd:
                out.append((sd, e, n, floor))
    return out


def to_matrix(samples, ap_index):
    X = np.full((len(samples), len(ap_index)), FILL, dtype=np.float32)
    Y = np.zeros((len(samples), 2), dtype=np.float64)
    F = np.zeros(len(samples), dtype=int)
    for i, (sd, e, n, fl) in enumerate(samples):
        for mac, rss in sd.items():
            j = ap_index.get(mac)
            if j is not None:
                X[i, j] = rss
        Y[i] = (e, n); F[i] = fl
    return X, Y, F


def knn_predict(Xtr, Ytr, Ftr, Xq, k=5):
    pos = np.zeros((len(Xq), 2)); fl = np.zeros(len(Xq), dtype=int)
    for i in range(len(Xq)):
        d = np.sqrt(((Xtr - Xq[i]) ** 2).sum(1))
        idx = np.argpartition(d, k)[:k]
        w = 1.0 / (d[idx] + 1e-6)
        pos[i] = (w[:, None] * Ytr[idx]).sum(0) / w.sum()
        fl[i] = np.bincount(Ftr[idx], weights=w).argmax()
    return pos, fl


def main():
    train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.txt")))
    val_files = sorted(glob.glob(os.path.join(VAL_DIR, "*.txt")))
    print(f"train files={len(train_files)}  val files={len(val_files)}")

    lat0, lon0 = _origin(train_files + val_files)
    tr = build_samples(train_files, lat0, lon0)
    va = build_samples(val_files, lat0, lon0)
    print(f"train scans={len(tr)}  val scans={len(va)}")

    ap_index = {}
    for sd, *_ in tr:
        for mac in sd:
            ap_index.setdefault(mac, len(ap_index))
    print(f"radiomap APs={len(ap_index)}")

    Xtr, Ytr, Ftr = to_matrix(tr, ap_index)
    Xva, Yva, Fva = to_matrix(va, ap_index)

    for k in (1, 3, 5, 9):
        pred, pfl = knn_predict(Xtr, Ytr, Ftr, Xva, k=k)
        err = np.sqrt(((pred - Yva) ** 2).sum(1))         # 2D horizontal error (m)
        floor_acc = (pfl == Fva).mean()
        ipin_metric = np.percentile(err + 15.0 * np.abs(pfl - Fva), 75)
        print(f"  k={k:>2}: median={np.median(err):5.2f}m  mean={err.mean():5.2f}m  "
              f"RMSE={np.sqrt((err**2).mean()):5.2f}m  p75={np.percentile(err,75):5.2f}m  "
              f"floor_acc={floor_acc*100:4.1f}%  IPIN_score(p75 w/floor)={ipin_metric:5.2f}m")


if __name__ == "__main__":
    main()
