#!/usr/bin/env python3
"""
GSDC 2023 — GNSS weighted-least-squares (WLS) positioning engine + validation.

Computes a receiver position per epoch from raw pseudoranges using the provided
satellite ECEF positions + clock/iono/tropo corrections. Step A here VALIDATES
the engine against ground truth and the dataset's own WLS baseline before any
fusion. Per-constellation positioning (for the AHGF multi-sensor test) builds on
this in gsdc_fuse.py.
"""
import os
import glob
import numpy as np
import pandas as pd

TRAIN = "/Users/chaitanyadeshpande/Downloads/paper run/sdc2023/train"
C = 299792458.0
OMEGA_E = 7.2921151467e-5
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
CONSTEL = {1: "GPS", 3: "GLONASS", 5: "BeiDou", 6: "Galileo", 4: "QZSS"}


def geodetic2ecef(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - WGS84_E2) + h) * np.sin(lat)
    return np.array([x, y, z])


def ecef2geodetic(x, y, z):
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(5):
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - WGS84_E2 * N / (N + h)))
    return np.degrees(lat), np.degrees(lon), h


def enu_error(est_ecef, gt_ecef, gt_lat, gt_lon):
    """Horizontal (East,North) error magnitude in meters."""
    d = est_ecef - gt_ecef
    la, lo = np.radians(gt_lat), np.radians(gt_lon)
    e = -np.sin(lo) * d[0] + np.cos(lo) * d[1]
    n = -np.sin(la) * np.cos(lo) * d[0] - np.sin(la) * np.sin(lo) * d[1] + np.cos(la) * d[2]
    return np.hypot(e, n)


def wls(sat_xyz, pr, w, x0, iters=8):
    """Gauss-Newton WLS for [x,y,z,clock]. Includes Sagnac earth-rotation correction."""
    x = np.array(x0, dtype=float)
    for _ in range(iters):
        # rotate satellite ECEF by earth rotation over signal travel time
        rng0 = np.linalg.norm(sat_xyz - x[:3], axis=1)
        theta = OMEGA_E * rng0 / C
        ct, st = np.cos(theta), np.sin(theta)
        sx = ct * sat_xyz[:, 0] + st * sat_xyz[:, 1]
        sy = -st * sat_xyz[:, 0] + ct * sat_xyz[:, 1]
        sat_rot = np.column_stack([sx, sy, sat_xyz[:, 2]])
        dvec = x[:3] - sat_rot
        rng = np.linalg.norm(dvec, axis=1)
        res = pr - (rng + x[3])
        H = np.column_stack([dvec / rng[:, None], np.ones(len(sat_xyz))])
        HtW = H.T * w
        try:
            dx = np.linalg.solve(HtW @ H, HtW @ res)
        except np.linalg.LinAlgError:
            break
        x = x + dx
        if np.linalg.norm(dx[:3]) < 1e-3:
            break
    return x


def corrected_pr(df):
    return (df["RawPseudorangeMeters"] + df["SvClockBiasMeters"]
            - df["IsrbMeters"].fillna(0) - df["IonosphericDelayMeters"].fillna(0)
            - df["TroposphericDelayMeters"].fillna(0))


def epoch_wls(ep, x0):
    """WLS over all valid sats in one epoch. Returns (ecef[3], nsats) or (None,0)."""
    ep = ep.dropna(subset=["RawPseudorangeMeters", "SvPositionXEcefMeters",
                           "SvPositionYEcefMeters", "SvPositionZEcefMeters"])
    ep = ep[ep["RawPseudorangeUncertaintyMeters"].between(0, 50, inclusive="neither")]
    if len(ep) < 5:
        return None, 0
    sat = ep[["SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters"]].to_numpy()
    pr = corrected_pr(ep).to_numpy()
    w = 1.0 / ep["RawPseudorangeUncertaintyMeters"].to_numpy() ** 2
    sol = wls(sat, pr, w, x0)
    return sol[:3], len(ep)


def load_trip(trip_dir):
    g = pd.read_csv(os.path.join(trip_dir, "device_gnss.csv"))
    gt = pd.read_csv(os.path.join(trip_dir, "ground_truth.csv"))
    return g, gt


def gt_ecef_at(gt, t_ms):
    """Nearest-GT ecef + (lat,lon) for a given utc ms."""
    i = (gt["UnixTimeMillis"] - t_ms).abs().idxmin()
    r = gt.loc[i]
    lat, lon, h = r["LatitudeDegrees"], r["LongitudeDegrees"], r["AltitudeMeters"]
    return geodetic2ecef(lat, lon, h), lat, lon, abs(gt.loc[i, "UnixTimeMillis"] - t_ms)


def main():
    trip = sorted(glob.glob(TRAIN + "/*/*/"))[0]
    for t in sorted(glob.glob(TRAIN + "/*/*/")):
        if os.path.exists(t + "device_gnss.csv") and os.path.exists(t + "ground_truth.csv"):
            trip = t; break
    print("TRIP:", trip.replace(TRAIN, ""))
    g, gt = load_trip(trip)
    print("constellations:", {CONSTEL.get(k, k): int(v) for k, v in g["ConstellationType"].value_counts().items()})

    mine, provided = [], []
    x0 = None
    for t_ms, ep in g.groupby("utcTimeMillis"):
        # init from provided WLS baseline (or previous solution)
        wp = ep[["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]].dropna()
        if x0 is None:
            x0 = np.append(wp.iloc[0].to_numpy(), 0.0) if len(wp) else np.array([-2700000., -4300000., 3850000., 0.])
        est, ns = epoch_wls(ep, x0)
        if est is None:
            continue
        x0 = np.append(est, x0[3])
        gtec, glat, glon, dt = gt_ecef_at(gt, t_ms)
        if dt > 700:
            continue
        mine.append(enu_error(est, gtec, glat, glon))
        if len(wp):
            provided.append(enu_error(wp.iloc[0].to_numpy(), gtec, glat, glon))

    mine = np.array(mine); provided = np.array(provided)
    print(f"\nepochs solved: {len(mine)}")
    print(f"MY WLS    vs GT:  median={np.median(mine):6.2f}m  mean={mine.mean():6.2f}m  "
          f"p75={np.percentile(mine,75):6.2f}m  p95={np.percentile(mine,95):6.2f}m")
    print(f"PROVIDED  vs GT:  median={np.median(provided):6.2f}m  mean={provided.mean():6.2f}m  "
          f"p75={np.percentile(provided,75):6.2f}m  p95={np.percentile(provided,95):6.2f}m")
    print("(my WLS should track the provided baseline closely -> engine validated)")


if __name__ == "__main__":
    main()
