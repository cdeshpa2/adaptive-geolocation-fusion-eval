#!/usr/bin/env python3
"""
GSDC — the decisive test of AHGF's adaptive weighting on REAL trajectories.

Per epoch we compute an independent WLS position for EACH constellation
(GPS / Galileo / GLONASS) -> multiple position "sensors" that degrade
DIFFERENTLY as the vehicle moves through open sky vs. urban canyons. We then
run AHGF (pure NIS weighting, no backwards context prior) as a TEMPORAL filter
over the trajectory and compare against fixed/equal fusion and the standard
all-sats WLS. Epochs are auto-segmented into 'clean' vs 'degraded' (by inter-
constellation disagreement) so we can see whether adaptivity pays off precisely
where conditions are bad — the only place it should.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsdc import (wls, corrected_pr, geodetic2ecef, TRAIN, CONSTEL, C, OMEGA_E)  # noqa
import ahgf_method as M  # noqa

SLOT = {1: 0, 6: 1, 3: 2}  # GPS->0, Galileo->1, GLONASS->2  (context disabled anyway)


def to_enu(ecef, o_ecef, lat, lon):
    d = ecef - o_ecef
    la, lo = np.radians(lat), np.radians(lon)
    e = -np.sin(lo) * d[0] + np.cos(lo) * d[1]
    n = -np.sin(la) * np.cos(lo) * d[0] - np.sin(la) * np.sin(lo) * d[1] + np.cos(la) * d[2]
    return np.array([e, n])


def const_wls(ep, x0):
    """Per-constellation WLS. Returns {ct: (ecef3, resid_rms, nsat)}."""
    out = {}
    ep = ep.dropna(subset=["RawPseudorangeMeters", "SvPositionXEcefMeters",
                           "SvPositionYEcefMeters", "SvPositionZEcefMeters"])
    ep = ep[ep["RawPseudorangeUncertaintyMeters"].between(0, 50, inclusive="neither")]
    for ct, sub in ep.groupby("ConstellationType"):
        if ct not in SLOT or len(sub) < 5:
            continue
        sat = sub[["SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters"]].to_numpy()
        pr = corrected_pr(sub).to_numpy()
        w = 1.0 / sub["RawPseudorangeUncertaintyMeters"].to_numpy() ** 2
        sol = wls(sat, pr, w, x0)
        rng = np.linalg.norm(sat - sol[:3], axis=1)
        res = pr - (rng + sol[3])
        rms = float(np.sqrt(np.mean(res ** 2)))
        out[ct] = (sol[:3], rms, len(sub))
    return out


def gt_lookup(gt):
    ts = gt["UnixTimeMillis"].to_numpy()
    lat = gt["LatitudeDegrees"].to_numpy(); lon = gt["LongitudeDegrees"].to_numpy()
    alt = gt["AltitudeMeters"].to_numpy()

    def f(t_ms):
        i = int(np.argmin(np.abs(ts - t_ms)))
        if abs(ts[i] - t_ms) > 700:
            return None
        return lat[i], lon[i], alt[i]
    return f


def ekf_step(method, f, sensors, context):
    f.predict()
    if method.startswith("AHGF"):
        f.step(sensors, context)
    elif method == "Fixed-R EKF":
        for z, sd, sid in sensors:
            f.update(z, np.eye(2) * sd ** 2)
    elif method in ("Sage-Husa AEKF",):
        for z, sd, sid in sensors:
            f.adaptive_update(z, sid)
    elif method == "Huber Robust EKF":
        for z, sd, sid in sensors:
            f.robust_update(z, sid)
    return f.pos()


def new_filter(method, p0):
    base = "AHGF" if method.startswith("AHGF") else method
    f = {"AHGF": M.AHGF, "Fixed-R EKF": M.EKF, "Sage-Husa AEKF": M.SageHusaEKF,
         "Huber Robust EKF": M.HuberEKF}[base](dt=1.0)
    f.x[:2] = p0; f.x[2:] = 0.0; f.P = np.eye(4) * 25.0
    return f


def run_trip(trip):
    g = pd.read_csv(os.path.join(trip, "device_gnss.csv"))
    gt = pd.read_csv(os.path.join(trip, "ground_truth.csv"))
    glook = gt_lookup(gt)
    o_lat, o_lon, o_alt = gt.iloc[0][["LatitudeDegrees", "LongitudeDegrees", "AltitudeMeters"]]
    o_ecef = geodetic2ecef(o_lat, o_lon, o_alt)

    epochs = []  # (t, {ct:(enu, std)}, gt_enu, allsats_enu)
    x0 = None
    for t_ms, ep in g.groupby("utcTimeMillis"):
        wp = ep[["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]].dropna()
        if x0 is None:
            x0 = np.append(wp.iloc[0].to_numpy(), 0.0) if len(wp) else np.array([-2.7e6, -4.3e6, 3.85e6, 0.])
        cw = const_wls(ep, x0)
        if not cw:
            continue
        # update init from the best (most sats) constellation
        best = max(cw.values(), key=lambda v: v[2])
        x0 = np.append(best[0], x0[3])
        gll = glook(t_ms)
        if gll is None:
            continue
        gt_enu = to_enu(geodetic2ecef(*gll), o_ecef, o_lat, o_lon)
        sens = {ct: (to_enu(e, o_ecef, o_lat, o_lon), max(rms * 0.1, 1.0)) for ct, (e, rms, n) in cw.items()}
        allsats = wp.iloc[0].to_numpy() if len(wp) else None
        allsats_enu = to_enu(allsats, o_ecef, o_lat, o_lon) if allsats is not None else None
        epochs.append((t_ms, sens, gt_enu, allsats_enu))
    return epochs


def main(n_trips=8):
    trips = [t for t in sorted(glob.glob(TRAIN + "/*/*/"))
             if os.path.exists(t + "device_gnss.csv") and os.path.exists(t + "ground_truth.csv")]
    # pick a spread across locations
    pick = trips[:: max(1, len(trips) // n_trips)][:n_trips]
    methods = ["Best-single (GPS)", "All-sats WLS", "Inverse-Variance",
               "Fixed-R EKF", "Sage-Husa AEKF", "Huber Robust EKF", "AHGF (NIS, no ctx)"]
    err = {m: {"clean": [], "degraded": []} for m in methods}

    for trip in pick:
        eps = run_trip(trip)
        if len(eps) < 30:
            continue
        filt = {m: None for m in ("Fixed-R EKF", "Sage-Husa AEKF", "Huber Robust EKF", "AHGF (NIS, no ctx)")}
        for (t, sens, gtl, allsats) in eps:
            # degradation = spread among per-constellation positions
            ps = np.array([v[0] for v in sens.values()])
            spread = float(np.linalg.norm(ps - ps.mean(0), axis=1).max()) if len(ps) > 1 else 0.0
            seg = "degraded" if spread > 10.0 else "clean"
            sensors = [(v[0], v[1], SLOT[ct]) for ct, v in sens.items()]
            # static / snapshot baselines
            err["Best-single (GPS)"][seg].append(np.linalg.norm(sens[1][0] - gtl) if 1 in sens else np.nan)
            if allsats is not None:
                err["All-sats WLS"][seg].append(np.linalg.norm(allsats - gtl))
            w = {ct: 1.0 / v[1] ** 2 for ct, v in sens.items()}
            iv = sum(w[ct] * sens[ct][0] for ct in sens) / sum(w.values())
            err["Inverse-Variance"][seg].append(np.linalg.norm(iv - gtl))
            # temporal EKF filters
            for m in filt:
                if filt[m] is None:
                    filt[m] = new_filter(m, np.mean([v[0] for v in sens.values()], axis=0))
                p = ekf_step(m, filt[m], sensors, context=None)
                err[m][seg].append(np.linalg.norm(p - gtl))

    def stat(a):
        a = np.array([x for x in a if np.isfinite(x)])
        return (np.median(a), np.sqrt(np.mean(a ** 2)), len(a)) if len(a) else (np.nan, np.nan, 0)

    print(f"\nTrips used: {len(pick)}")
    nclean = stat(err['All-sats WLS']['clean'])[2]; ndeg = stat(err['All-sats WLS']['degraded'])[2]
    print(f"Epochs: clean={nclean}  degraded={ndeg} (inter-constellation spread > 10 m)\n")
    print(f"{'Method':<22}{'clean med':>10}{'clean RMSE':>12}{'DEGRADED med':>14}{'DEGRADED RMSE':>15}")
    print("-" * 73)
    for m in methods:
        cm, cr, _ = stat(err[m]["clean"]); dm, dr, _ = stat(err[m]["degraded"])
        print(f"{m:<22}{cm:10.2f}{cr:12.2f}{dm:14.2f}{dr:15.2f}")


if __name__ == "__main__":
    main()
