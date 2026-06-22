#!/usr/bin/env python3
"""
Test the genuine-contribution hypothesis: innovation-driven ROBUST weighting at
the raw-pseudorange level (NLOS/multipath rejection) beats plain all-sats WLS in
degraded (urban-canyon) epochs, while tying it in clean conditions.

Plain WLS = current best baseline. Robust WLS = same satellites, but each
pseudorange is reweighted by its normalized innovation (Huber + hard-reject of
gross outliers, IRLS) — the paper's "innovation-driven adaptive" idea applied
where it actually helps. Segmented by difficulty (post-fit residual RMS) so we
see if the win lands where it should.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsdc import (corrected_pr, geodetic2ecef, enu_error, TRAIN, C, OMEGA_E)  # noqa


def _resid(x, sat, pr):
    rng0 = np.linalg.norm(sat - x[:3], axis=1)
    th = OMEGA_E * rng0 / C
    ct, st = np.cos(th), np.sin(th)
    sx = ct * sat[:, 0] + st * sat[:, 1]
    sy = -st * sat[:, 0] + ct * sat[:, 1]
    satr = np.column_stack([sx, sy, sat[:, 2]])
    dvec = x[:3] - satr
    rng = np.linalg.norm(dvec, axis=1)
    return pr - (rng + x[3]), dvec, rng


def wls_solve(sat, pr, w0, x0, robust=False, huber_k=2.5, reject_k=6.0, iters=12):
    x = np.array(x0, float)
    rejected = 0
    for it in range(iters):
        res, dvec, rng = _resid(x, sat, pr)
        weff = w0.copy()
        if robust and it >= 1:
            s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6  # robust scale (MAD)
            nr = np.abs(res) / s
            rw = np.ones_like(res)
            m = nr > huber_k
            rw[m] = huber_k / nr[m]          # Huber down-weight
            rw[nr > reject_k] = 1e-4         # hard-reject gross NLOS outliers
            rejected = int((nr > reject_k).sum())
            weff = w0 * rw
            if (weff > 1e-3).sum() < 5:       # keep the problem solvable
                weff = w0.copy(); rejected = 0
        H = np.column_stack([dvec / rng[:, None], np.ones(len(sat))])
        HtW = H.T * weff
        try:
            dx = np.linalg.solve(HtW @ H, HtW @ res)
        except np.linalg.LinAlgError:
            break
        x = x + dx
        if np.linalg.norm(dx[:3]) < 1e-3 and it >= 1:
            break
    postfit = np.sqrt(np.mean(_resid(x, sat, pr)[0] ** 2))
    return x, postfit, rejected


def epoch_arrays(ep):
    ep = ep.dropna(subset=["RawPseudorangeMeters", "SvPositionXEcefMeters",
                           "SvPositionYEcefMeters", "SvPositionZEcefMeters"])
    ep = ep[ep["RawPseudorangeUncertaintyMeters"].between(0, 50, inclusive="neither")]
    if len(ep) < 6:
        return None
    sat = ep[["SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters"]].to_numpy()
    pr = corrected_pr(ep).to_numpy()
    w = 1.0 / ep["RawPseudorangeUncertaintyMeters"].to_numpy() ** 2
    wp = ep[["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]].dropna()
    x0 = np.append(wp.iloc[0].to_numpy(), 0.0) if len(wp) else None
    return sat, pr, w, x0


def main(n_trips=8):
    trips = [t for t in sorted(glob.glob(TRAIN + "/*/*/"))
             if os.path.exists(t + "device_gnss.csv") and os.path.exists(t + "ground_truth.csv")]
    pick = trips[:: max(1, len(trips) // n_trips)][:n_trips]

    rows = []  # (difficulty, plain_err, robust_err, rejected)
    for trip in pick:
        g = pd.read_csv(os.path.join(trip, "device_gnss.csv"))
        gt = pd.read_csv(os.path.join(trip, "ground_truth.csv"))
        gts = gt["UnixTimeMillis"].to_numpy()
        glat = gt["LatitudeDegrees"].to_numpy(); glon = gt["LongitudeDegrees"].to_numpy()
        galt = gt["AltitudeMeters"].to_numpy()
        x0 = None
        for t_ms, ep in g.groupby("utcTimeMillis"):
            arr = epoch_arrays(ep)
            if arr is None:
                continue
            sat, pr, w, ep_x0 = arr
            if x0 is None:
                x0 = ep_x0
            if x0 is None:
                continue
            xp, pf, _ = wls_solve(sat, pr, w, x0, robust=False)
            xr, _, rej = wls_solve(sat, pr, w, x0, robust=True)
            x0 = np.append(xp[:3], x0[3])
            i = int(np.argmin(np.abs(gts - t_ms)))
            if abs(gts[i] - t_ms) > 700:
                continue
            gtec = geodetic2ecef(glat[i], glon[i], galt[i])
            ep_plain = enu_error(xp[:3], gtec, glat[i], glon[i])
            ep_rob = enu_error(xr[:3], gtec, glat[i], glon[i])
            rows.append((pf, ep_plain, ep_rob, rej))

    a = np.array(rows)
    diff, plain, rob = a[:, 0], a[:, 1], a[:, 2]
    # difficulty terciles by post-fit residual RMS (method-independent multipath proxy)
    q1, q2 = np.percentile(diff, [50, 80])
    segs = {"all": np.ones(len(a), bool),
            "clean (low resid)": diff <= q1,
            "moderate": (diff > q1) & (diff <= q2),
            "DEGRADED (top 20% multipath)": diff > q2}

    def st(e):
        return f"med={np.median(e):6.2f}  mean={e.mean():6.2f}  RMSE={np.sqrt(np.mean(e**2)):7.2f}  p95={np.percentile(e,95):7.2f}"

    print(f"Trips={len(pick)}  epochs={len(a)}  total outliers rejected={int(a[:,3].sum())}\n")
    print(f"{'Segment':<32}{'n':>6}   plain WLS  /  ROBUST WLS (innovation-reweighted)")
    print("-" * 92)
    for name, m in segs.items():
        if m.sum() == 0:
            continue
        impr = (1 - np.median(rob[m]) / np.median(plain[m])) * 100
        print(f"{name:<32}{m.sum():>6}")
        print(f"{'    plain ':<32}        {st(plain[m])}")
        print(f"{'    robust':<32}        {st(rob[m])}   (median {impr:+.1f}%)")


if __name__ == "__main__":
    main()
