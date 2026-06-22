#!/usr/bin/env python3
"""
OutFin — real 4-modality fusion test of AHGF vs. baselines.

OutFin carries GPS + WiFi + BLE + cellular at 122 surveyed reference points
(NAD83 meters ground truth) — mapping exactly onto AHGF's four sensor slots
(GPS=0, WiFi=1, BLE=2, cellular=3). For each RP we derive one 2D position
observation per modality (GPS = the phone fix directly; WiFi/BLE/cellular =
leave-RP-out kNN fingerprint positioners — the real signal->position step the
paper hand-waves), then fuse the four with AHGF and with the paper's baselines.

This is the real-data test of the CORE NOVELTY (the per-sensor NIS/context
weighting). Static dataset -> single-shot fusion per RP; the temporal NIS-history
component is exercised on the trajectory datasets (IPIN/GSDC), noted honestly.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ahgf_method as M  # EKF, AHGF, SageHusaEKF, InnovationAEKF, HuberEKF

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "data", "outfin", "OutFin")
MEAS = os.path.join(ROOT, "Measurements")
COORD = os.path.join(ROOT, "Coordinates")
IMAP = os.path.join(ROOT, "Interactive_Map")

SID = {"gps": 0, "wifi": 1, "ble": 2, "cell": 3}
FILL_RF = -100.0
FILL_CELL = -130.0


# ---------- ground truth + GPS ----------
def load_truth():
    dfs = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(COORD, "Site*_NAD83.csv")))]
    df = pd.concat(dfs, ignore_index=True)
    return {int(r): np.array([x, y]) for r, x, y in zip(df.RP_ID, df.X, df.Y)}


def load_gps():
    acc = {}
    for f in (os.path.join(IMAP, "Galaxy_S10_NAD83.csv"), os.path.join(IMAP, "Pixel_4_NAD83.csv")):
        df = pd.read_csv(f)
        for rp, x, y in zip(df.RP_ID, df.x, df.y):
            if x and y:
                acc.setdefault(int(rp), []).append((x, y))
    return {rp: np.mean(v, axis=0) for rp, v in acc.items()}


# ---------- per-RP RF fingerprints (pooled over both phones) ----------
def load_wifi(rp):
    vals = {}
    for p in (1, 2):
        f = os.path.join(MEAS, f"Phone{p}_WiFi_{rp}.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        rss = df[[f"RSS_{i}" for i in range(9)]].mean(axis=1, skipna=True)
        for b, r in zip(df.BSSID, rss):
            if pd.notna(r):
                vals.setdefault(int(b), []).append(float(r))
    return {b: float(np.mean(v)) for b, v in vals.items()}


def load_ble(rp):
    vals = {}
    for p in (1, 2):
        f = os.path.join(MEAS, f"Phone{p}_Bluetooth_{rp}.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        df = df[df.Protocol == "BLE"]
        for m, r in df.groupby("MAC_address").RSS.mean().items():
            vals.setdefault(int(m), []).append(float(r))
    return {m: float(np.mean(v)) for m, v in vals.items()}


def load_cell(rp):
    vals = {}
    for p in (1, 2):
        f = os.path.join(MEAS, f"Phone{p}_Cellular_{rp}.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f).dropna(subset=["ECI", "RSRP"])
        if df.empty:
            continue
        df = df.assign(ECI=df.ECI.astype(float).astype(int))
        for e, r in df.groupby("ECI").RSRP.mean().items():
            vals.setdefault(int(e), []).append(float(r))
    return {e: float(np.mean(v)) for e, v in vals.items()}


# ---------- leave-RP-out kNN positioner ----------
def loro_positions(vecs_by_rp, true_local, fill, k=5):
    rps = sorted(vecs_by_rp)
    vocab = sorted({key for rp in rps for key in vecs_by_rp[rp]})
    idx = {key: i for i, key in enumerate(vocab)}
    X = np.full((len(rps), len(vocab)), fill, dtype=np.float32)
    Y = np.array([true_local[rp] for rp in rps])
    for r, rp in enumerate(rps):
        for key, v in vecs_by_rp[rp].items():
            X[r, idx[key]] = v
    pred, std = {}, {}
    for r, rp in enumerate(rps):
        mask = np.ones(len(rps), bool); mask[r] = False
        d = np.sqrt(((X[mask] - X[r]) ** 2).sum(1))
        order = np.argsort(d)[:k]
        Ytr = Y[mask][order]; w = 1.0 / (d[order] + 1e-6)
        pred[rp] = (w[:, None] * Ytr).sum(0) / w.sum()
        std[rp] = float(np.sqrt(((Ytr - pred[rp]) ** 2).sum(1).mean()) + 1e-3)
    return pred, std, len(vocab)


def rmse(errs):
    return float(np.sqrt(np.mean(np.square(errs))))


# ---------- single-shot fusion with a given filter ----------
def fuse(method, obs, stds, context):
    if method == "AHGF":
        f = M.AHGF(dt=1.0)
    elif method == "Fixed-R EKF":
        f = M.EKF(dt=1.0)
    elif method == "Sage-Husa AEKF":
        f = M.SageHusaEKF(dt=1.0)
    elif method == "Innovation AEKF":
        f = M.InnovationAEKF(dt=1.0)
    elif method == "Huber Robust EKF":
        f = M.HuberEKF(dt=1.0)
    else:
        raise ValueError(method)
    prior = np.mean([obs[s] for s in obs], axis=0)
    f.x[:2] = prior; f.x[2:] = 0.0; f.P = np.eye(4) * 100.0
    f.predict()
    sensors = [(obs[s], stds[s], s) for s in sorted(obs)]
    if method == "AHGF":
        f.step(sensors, context)
    elif method == "Fixed-R EKF":
        for z, sd, sid in sensors:
            f.update(z, np.eye(2) * sd ** 2)
    elif method in ("Sage-Husa AEKF", "Innovation AEKF"):
        for z, sd, sid in sensors:
            f.adaptive_update(z, sid)
    elif method == "Huber Robust EKF":
        for z, sd, sid in sensors:
            f.robust_update(z, sid)
    return f.pos()


def main():
    truth = load_truth()
    gps = load_gps()
    origin = np.array([min(p[0] for p in truth.values()),
                       min(p[1] for p in truth.values())])
    true_local = {rp: truth[rp] - origin for rp in truth}
    rps = sorted(truth)
    print(f"RPs={len(rps)}  GPS RPs={len(gps)}")

    # per-modality fingerprints
    wifi_v = {rp: load_wifi(rp) for rp in rps if load_wifi(rp)}
    ble_v = {rp: load_ble(rp) for rp in rps if load_ble(rp)}
    cell_v = {rp: load_cell(rp) for rp in rps if load_cell(rp)}

    wifi_p, wifi_s, nwifi = loro_positions(wifi_v, true_local, FILL_RF)
    ble_p, ble_s, nble = loro_positions(ble_v, true_local, FILL_RF)
    cell_p, cell_s, ncell = loro_positions(cell_v, true_local, FILL_CELL)
    gps_local = {rp: gps[rp] - origin for rp in gps}

    # ---- per-modality real accuracy ----
    print(f"\nAPs: WiFi={nwifi} BLE={nble} Cells={ncell}")
    print("\n--- Per-modality real positioning error (m) vs NAD83 truth ---")
    nominal = {}
    for name, pred in [("GPS", gps_local), ("WiFi", wifi_p), ("BLE", ble_p), ("Cellular", cell_p)]:
        errs = np.array([np.linalg.norm(pred[rp] - true_local[rp]) for rp in pred])
        nominal[name] = rmse(errs)
        print(f"  {name:<9} n={len(errs):3d}  median={np.median(errs):6.2f}  "
              f"mean={errs.mean():6.2f}  RMSE={rmse(errs):6.2f}  p75={np.percentile(errs,75):6.2f}")

    # nominal per-sensor std (real, fair to all methods)
    REAL_STD = {SID["gps"]: nominal["GPS"], SID["wifi"]: nominal["WiFi"],
                SID["ble"]: nominal["BLE"], SID["cell"]: nominal["Cellular"]}
    M.fixed_R = lambda sid: np.eye(2) * REAL_STD[sid] ** 2  # fair real R for baselines

    # ---- fusion over RPs having all four modalities ----
    methods = ["Inverse-Variance", "Fixed-R EKF", "Sage-Husa AEKF",
               "Innovation AEKF", "Huber Robust EKF",
               "AHGF (no ctx prior)", "AHGF (ctx prior=published)"]
    agg = {m: [] for m in methods}
    common = [rp for rp in rps if rp in gps_local and rp in wifi_p and rp in ble_p and rp in cell_p]
    print(f"\n--- Fusion on {len(common)} RPs with all 4 modalities (single-shot, outdoor) ---")
    for rp in common:
        obs = {SID["gps"]: gps_local[rp], SID["wifi"]: wifi_p[rp],
               SID["ble"]: ble_p[rp], SID["cell"]: cell_p[rp]}
        stds = {SID["gps"]: max(REAL_STD[0], 1.0), SID["wifi"]: max(wifi_s[rp], 1.0),
                SID["ble"]: max(ble_s[rp], 1.0), SID["cell"]: max(cell_s[rp], 1.0)}
        t = true_local[rp]
        # inverse-variance optimal linear baseline
        w = {s: 1.0 / stds[s] ** 2 for s in obs}
        iv = sum(w[s] * obs[s] for s in obs) / sum(w.values())
        agg["Inverse-Variance"].append(np.linalg.norm(iv - t))
        for m in ("Fixed-R EKF", "Sage-Husa AEKF", "Innovation AEKF", "Huber Robust EKF"):
            agg[m].append(np.linalg.norm(fuse(m, obs, stds, context=0) - t))
        agg["AHGF (no ctx prior)"].append(np.linalg.norm(fuse("AHGF", obs, stds, context=None) - t))
        agg["AHGF (ctx prior=published)"].append(np.linalg.norm(fuse("AHGF", obs, stds, context=0) - t))

    print(f"\n{'Method':<20}{'median':>9}{'mean':>9}{'RMSE':>9}{'p75':>9}")
    print("-" * 56)
    for m in methods:
        e = np.array(agg[m])
        print(f"{m:<20}{np.median(e):9.2f}{e.mean():9.2f}{rmse(e):9.2f}{np.percentile(e,75):9.2f}")

    print("\n=== OutFin paired significance (per-RP, vs inverse-variance) ===")
    from scipy.stats import wilcoxon
    iv = np.array(agg["Inverse-Variance"])
    for m in ["Fixed-R EKF", "AHGF (no ctx prior)", "AHGF (ctx prior=published)"]:
        e = np.array(agg[m])
        try:
            _, p = wilcoxon(iv, e)
        except ValueError:
            p = float("nan")
        verdict = "inv-var better" if np.median(e) > np.median(iv) else "method better"
        print(f"   inv-variance vs {m:<28}: paired Wilcoxon p={p:.2e}  ({verdict}, n={len(iv)})")


if __name__ == "__main__":
    main()
