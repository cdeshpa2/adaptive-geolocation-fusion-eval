#!/usr/bin/env python3
"""
Increment 2: add a TEMPORAL innovation-gated filter on top of per-pseudorange
robust positioning. Two-level "innovation-driven adaptive" idea:
  (1) measurement level  -> robust IRLS rejects NLOS/multipath pseudoranges (gsdc_robust);
  (2) trajectory level    -> a constant-velocity EKF whose position-update covariance
                             is inflated by the fix's normalized innovation (down-weights
                             bad epochs the per-epoch robustifier couldn't fully clean).

Compares, per difficulty segment: plain WLS (snapshot) | robust WLS (snapshot) |
robust + innovation-gated EKF (the proposed method). Same satellites/corrections
throughout; honest segmentation by post-fit residual RMS.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsdc import geodetic2ecef, enu_error, TRAIN  # noqa
from gsdc_robust import wls_solve, epoch_arrays  # noqa


def to_enu(ecef, o, lat, lon):
    d = ecef - o
    la, lo = np.radians(lat), np.radians(lon)
    e = -np.sin(lo) * d[0] + np.cos(lo) * d[1]
    n = -np.sin(la) * np.cos(lo) * d[0] - np.sin(la) * np.sin(lo) * d[1] + np.cos(la) * d[2]
    return np.array([e, n])


class CVFilter:
    """2D constant-velocity EKF with innovation-gated position updates."""
    def __init__(self, p0, q=4.0):
        self.x = np.array([p0[0], p0[1], 0.0, 0.0])
        self.P = np.diag([25.0, 25.0, 100.0, 100.0])
        self.q = q
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)

    def step(self, z, sigma, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        g = self.q ** 2
        Q = g * np.array([[dt**4/4, 0, dt**3/2, 0], [0, dt**4/4, 0, dt**3/2],
                          [dt**3/2, 0, dt**2, 0], [0, dt**3/2, 0, dt**2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        R = np.eye(2) * sigma ** 2
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        nis = float(y @ np.linalg.solve(S, y))
        if nis > 13.8:                      # chi2(2dof, 99.9%) -> inflate R for a suspect fix
            R = R * (nis / 13.8)
            S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2]


def main(n_trips=8):
    trips = [t for t in sorted(glob.glob(TRAIN + "/*/*/"))
             if os.path.exists(t + "device_gnss.csv") and os.path.exists(t + "ground_truth.csv")]
    pick = trips[:: max(1, len(trips) // n_trips)][:n_trips]
    rows = []  # difficulty, plain, robust, robust+ekf, trip_idx

    for ti, trip in enumerate(pick):
        g = pd.read_csv(os.path.join(trip, "device_gnss.csv"), low_memory=False)
        gt = pd.read_csv(os.path.join(trip, "ground_truth.csv"))
        gts = gt["UnixTimeMillis"].to_numpy()
        glat = gt["LatitudeDegrees"].to_numpy(); glon = gt["LongitudeDegrees"].to_numpy()
        galt = gt["AltitudeMeters"].to_numpy()
        o_ecef = geodetic2ecef(glat[0], glon[0], galt[0])
        x0 = None; filt = None; t_prev = None
        for t_ms, ep in g.groupby("utcTimeMillis"):
            arr = epoch_arrays(ep)
            if arr is None:
                continue
            sat, pr, w, ep_x0 = arr
            if x0 is None:
                x0 = ep_x0
            if x0 is None:
                continue
            xp, _, _ = wls_solve(sat, pr, w, x0, robust=False)
            xr, pf, _ = wls_solve(sat, pr, w, x0, robust=True)
            x0 = np.append(xr[:3], x0[3])
            i = int(np.argmin(np.abs(gts - t_ms)))
            if abs(gts[i] - t_ms) > 700:
                continue
            gtl = geodetic2ecef(glat[i], glon[i], galt[i])
            gten = to_enu(gtl, o_ecef, glat[0], glon[0])
            rob_en = to_enu(xr[:3], o_ecef, glat[0], glon[0])
            pln_en = to_enu(xp[:3], o_ecef, glat[0], glon[0])
            # temporal filter on the robust fix
            dt = 1.0 if t_prev is None else max(0.05, (t_ms - t_prev) / 1000.0)
            t_prev = t_ms
            sigma = float(np.clip(pf * 0.1, 1.5, 40.0))
            if filt is None:
                filt = CVFilter(rob_en)
                fused = rob_en
            else:
                fused = filt.step(rob_en, sigma, dt)
            prov_en = to_enu(ep_x0[:3], o_ecef, glat[0], glon[0])
            rows.append((pf,
                         np.linalg.norm(pln_en - gten),
                         np.linalg.norm(rob_en - gten),
                         np.linalg.norm(fused - gten),
                         ti,
                         len(sat),
                         np.linalg.norm(prov_en - gten)))

    a = np.array(rows)
    diff = a[:, 0]
    q1, q2 = np.percentile(diff, [50, 80])
    segs = {"all": np.ones(len(a), bool), "clean": diff <= q1,
            "moderate": (diff > q1) & (diff <= q2), "DEGRADED (top 20%)": diff > q2}

    def st(e):
        return f"med={np.median(e):6.2f}  RMSE={np.sqrt(np.mean(e**2)):6.2f}  p95={np.percentile(e,95):6.2f}"

    print(f"Trips={len(pick)}  epochs={len(a)}\n")
    for name, m in segs.items():
        pl, rb, rk = a[m, 1], a[m, 2], a[m, 3]
        ip = (1 - np.median(rk) / np.median(pl)) * 100
        print(f"[{name}]  n={m.sum()}")
        print(f"   plain WLS         {st(pl)}")
        print(f"   robust WLS        {st(rb)}")
        print(f"   robust+EKF (ours) {st(rk)}   (median {ip:+.1f}% vs plain WLS)\n")

    print("=== Significance (ours=robust+EKF vs plain WLS), paired per-epoch ===")
    rng = np.random.default_rng(0)
    for name, m in segs.items():
        pl, rk = a[m, 1], a[m, 3]
        try:
            _, p = wilcoxon(pl, rk)
        except ValueError:
            p = float("nan")
        idx = np.arange(len(pl))
        boots = []
        for _ in range(2000):
            b = rng.choice(idx, size=len(idx), replace=True)
            boots.append((1 - np.median(rk[b]) / np.median(pl[b])) * 100)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ip = (1 - np.median(rk) / np.median(pl)) * 100
        print(f"   [{name:<18}] median improvement {ip:+5.1f}%  95% CI [{lo:+.1f}, {hi:+.1f}]  "
              f"Wilcoxon p={p:.2e}  (n={len(pl)})")

    print("\n=== Trip-level significance (degraded; honest re within-trip correlation) ===")
    from math import comb
    ti_all = a[:, 4].astype(int)
    deg = segs["DEGRADED (top 20%)"]
    per_trip = []
    for t in np.unique(ti_all):
        m = deg & (ti_all == t)
        if m.sum() < 20:
            continue
        per_trip.append((1 - np.median(a[m, 3]) / np.median(a[m, 1])) * 100)
    per_trip = np.array(per_trip)
    n = len(per_trip); npos = int((per_trip > 0).sum())
    psign = sum(comb(n, k) for k in range(npos, n + 1)) / 2 ** n if n else float("nan")
    try:
        _, ptw = wilcoxon(per_trip)
    except ValueError:
        ptw = float("nan")
    print(f"   per-trip degraded improvement: {npos}/{n} trips positive, "
          f"mean {per_trip.mean():+.1f}%, median {np.median(per_trip):+.1f}%, "
          f"range [{per_trip.min():+.1f}, {per_trip.max():+.1f}]%")
    print(f"   sign-test p={psign:.4f}   trip-level Wilcoxon p={ptw:.4f}   (n={n} trips)")

    print("\n=== INDEPENDENT segmentation by satellite count (rebuts residual-circularity, OBJ-056) ===")
    print("    + dataset's official WLS baseline as extra comparator (OBJ-058)")
    nsat = a[:, 5]
    thr = np.percentile(nsat, 30)
    def s(e):
        return f"{np.median(e):5.2f}/{np.sqrt(np.mean(e**2)):5.2f}/{np.percentile(e,95):5.2f}"
    print("   (median / RMSE / p95, metres)")
    for label, m in [(f"few-sats (<= {thr:.0f}, geometry-poor)", nsat <= thr),
                     (f"many-sats (> {thr:.0f})", nsat > thr)]:
        pl, rk, pv = a[m, 1], a[m, 3], a[m, 6]
        print(f"   [{label:<28}] n={int(m.sum()):5d}   plain {s(pl)}   official {s(pv)}   ours {s(rk)}")
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsdc_errors.npy"), a)
    print("\n[saved gsdc_errors.npy for figures]")


if __name__ == "__main__":
    main()
