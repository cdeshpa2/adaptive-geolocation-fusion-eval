#!/usr/bin/env python3
"""
Learned per-measurement reliability model (the novelty lever).

A gradient-boosted regressor predicts each pseudorange's error from GT-FREE
features (Cn0, elevation, raw uncertainty, MultipathIndicator, constellation,
and the measurement's own plain-WLS post-fit residual = its "innovation").
The predicted error becomes the measurement's WLS weight — a data-driven,
innovation-driven reliability model replacing fixed Huber. This keeps the
paper's gradient-boosted-tree element, but repurposed onto a REAL, useful task.

Integrity:
 - Ground truth is used ONLY to label TRAINING measurements; at test time the
   model sees features only.
 - Evaluation is LEAVE-TRIP-OUT (2-fold over disjoint trips) so a win reflects
   generalization, not memorization.
Compares plain WLS | fixed-Huber robust WLS | learned-weight WLS, by difficulty.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsdc import geodetic2ecef, enu_error, corrected_pr, TRAIN, C, OMEGA_E  # noqa
from gsdc_robust import wls_solve, _resid  # noqa

FEATS = ["cn0", "elev", "unc", "mp", "constel", "innov"]


def geom_range(sat, rx):
    rng0 = np.linalg.norm(sat - rx, axis=1)
    th = OMEGA_E * rng0 / C
    ct, st = np.cos(th), np.sin(th)
    sx = ct * sat[:, 0] + st * sat[:, 1]
    sy = -st * sat[:, 0] + ct * sat[:, 1]
    return np.linalg.norm(np.column_stack([sx, sy, sat[:, 2]]) - rx, axis=1)


def extract_trip(trip):
    g = pd.read_csv(os.path.join(trip, "device_gnss.csv"), low_memory=False)
    gt = pd.read_csv(os.path.join(trip, "ground_truth.csv"))
    gts = gt["UnixTimeMillis"].to_numpy()
    glat = gt["LatitudeDegrees"].to_numpy(); glon = gt["LongitudeDegrees"].to_numpy()
    galt = gt["AltitudeMeters"].to_numpy()
    recs = []
    x0 = None
    for t_ms, ep in g.groupby("utcTimeMillis"):
        ep = ep.dropna(subset=["RawPseudorangeMeters", "SvPositionXEcefMeters",
                               "SvPositionYEcefMeters", "SvPositionZEcefMeters",
                               "Cn0DbHz", "SvElevationDegrees"])
        ep = ep[ep["RawPseudorangeUncertaintyMeters"].between(0, 50, inclusive="neither")]
        if len(ep) < 6:
            continue
        sat = ep[["SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters"]].to_numpy()
        pr = corrected_pr(ep).to_numpy()
        unc = ep["RawPseudorangeUncertaintyMeters"].to_numpy()
        cn0 = ep["Cn0DbHz"].to_numpy(); elev = ep["SvElevationDegrees"].to_numpy()
        mp = ep["MultipathIndicator"].fillna(0).to_numpy()
        constel = ep["ConstellationType"].to_numpy()
        wp = ep[["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]].dropna()
        if x0 is None:
            x0 = np.append(wp.iloc[0].to_numpy(), 0.0) if len(wp) else None
        if x0 is None:
            continue
        # plain WLS -> innovation feature (GT-free)
        xp, pf, _ = wls_solve(sat, pr, 1.0 / unc ** 2, x0, robust=False)
        innov = np.abs(_resid(xp, sat, pr)[0])
        x0 = np.append(xp[:3], x0[3])
        # GT-referenced per-measurement error label (per-constellation clock removed)
        i = int(np.argmin(np.abs(gts - t_ms)))
        if abs(gts[i] - t_ms) > 700:
            continue
        gtec = geodetic2ecef(glat[i], glon[i], galt[i])
        gr = geom_range(sat, gtec)
        label = np.empty(len(sat))
        for c in np.unique(constel):
            m = constel == c
            clk = np.median(pr[m] - gr[m])
            label[m] = np.abs(pr[m] - gr[m] - clk)
        feats = np.column_stack([cn0, elev, unc, mp, constel, innov])
        recs.append(dict(sat=sat, pr=pr, unc=unc, x0=x0.copy(), gtec=gtec,
                         glat=glat[i], glon=glon[i], feats=feats, label=label, pf=pf))
    return recs


def eval_trip(recs, model):
    out = {"plain": [], "robust": [], "learned": [], "diff": []}
    for r in recs:
        sat, pr, unc, x0 = r["sat"], r["pr"], r["unc"], r["x0"]
        xp, _, _ = wls_solve(sat, pr, 1.0 / unc ** 2, x0, robust=False)
        xr, _, _ = wls_solve(sat, pr, 1.0 / unc ** 2, x0, robust=True)
        pred = np.clip(model.predict(r["feats"]), 0.5, None)   # predicted |error| (m)
        xl, _, _ = wls_solve(sat, pr, 1.0 / pred ** 2, x0, robust=False)
        for x, k in ((xp, "plain"), (xr, "robust"), (xl, "learned")):
            out[k].append(enu_error(x[:3], r["gtec"], r["glat"], r["glon"]))
        out["diff"].append(r["pf"])
    return out


def main(n_trips=8):
    trips = [t for t in sorted(glob.glob(TRAIN + "/*/*/"))
             if os.path.exists(t + "device_gnss.csv") and os.path.exists(t + "ground_truth.csv")]
    pick = trips[:: max(1, len(trips) // n_trips)][:n_trips]
    print(f"extracting {len(pick)} trips...")
    per_trip = [extract_trip(t) for t in pick]
    per_trip = [r for r in per_trip if len(r) > 20]
    print(f"usable trips={len(per_trip)}  total epochs={sum(len(r) for r in per_trip)}")

    folds = [[i for i in range(len(per_trip)) if i % 2 == 0],
             [i for i in range(len(per_trip)) if i % 2 == 1]]
    agg = {"plain": [], "robust": [], "learned": [], "diff": []}
    for test_idx in folds:
        train_idx = [i for i in range(len(per_trip)) if i not in test_idx]
        X = np.vstack([rec["feats"] for i in train_idx for rec in per_trip[i]])
        y = np.concatenate([rec["label"] for i in train_idx for rec in per_trip[i]])
        model = HistGradientBoostingRegressor(max_iter=300, max_depth=4,
                                              learning_rate=0.08, l2_regularization=1.0)
        model.fit(X, np.log1p(y))
        wrap = type("M", (), {"predict": lambda self, Z, m=model: np.expm1(m.predict(Z))})()
        for i in test_idx:
            r = eval_trip(per_trip[i], wrap)
            for k in agg:
                agg[k] += r[k]

    a = {k: np.array(v) for k, v in agg.items()}
    diff = a["diff"]
    q1, q2 = np.percentile(diff, [50, 80])
    segs = {"all": np.ones(len(diff), bool), "clean": diff <= q1,
            "moderate": (diff > q1) & (diff <= q2), "DEGRADED (top 20%)": diff > q2}

    def st(e):
        return f"med={np.median(e):6.2f}  RMSE={np.sqrt(np.mean(e**2)):6.2f}  p95={np.percentile(e,95):6.2f}"

    print(f"\nLEAVE-TRIP-OUT results (epochs={len(diff)})\n")
    for name, m in segs.items():
        pl, rb, ln = a["plain"][m], a["robust"][m], a["learned"][m]
        ip = (1 - np.median(ln) / np.median(pl)) * 100
        ir = (1 - np.median(ln) / np.median(rb)) * 100
        print(f"[{name}]  n={m.sum()}")
        print(f"   plain WLS            {st(pl)}")
        print(f"   robust WLS (Huber)   {st(rb)}")
        print(f"   LEARNED (ours)       {st(ln)}   ({ip:+.1f}% vs plain, {ir:+.1f}% vs Huber)\n")


if __name__ == "__main__":
    main()
