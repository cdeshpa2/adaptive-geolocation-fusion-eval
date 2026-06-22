#!/usr/bin/env python3
"""
IPIN 2020 Competition Track 3 — logfile parser + ground-truth track.

Parses GetSensorData ';'-delimited logfiles into per-sensor streams and builds a
piecewise-linear ground-truth position track from the sparse POSI checkpoints.
Format per the verified spec (see project notes). All times use AppTimestamp(s)
(token index 1) as the master clock except where noted.

Pure stdlib — no third-party deps — so it can run anywhere.
"""
import os
import bisect
import math
from collections import defaultdict

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "ipin", "2020_IPIN_Competition_Track03",
)
TRAIN_DIR = os.path.join(DATA_ROOT, "01-Logfiles", "01-Training", "01a-Regular")
VAL_DIR = os.path.join(DATA_ROOT, "01-Logfiles", "02-Validation")


def _f(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return float("nan")


def parse_logfile(path):
    """Return a dict of per-sensor streams. Tuples use AppTimestamp first."""
    acce, gyro, magn, pres, gnss, posi = [], [], [], [], [], []
    wifi = defaultdict(list)  # app_ts -> [(mac, freq, rss), ...] (one scan per ts)
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            t = line.split(";")
            rt = t[0]
            try:
                if rt == "ACCE":
                    acce.append((_f(t[1]), _f(t[3]), _f(t[4]), _f(t[5])))
                elif rt == "GYRO":
                    gyro.append((_f(t[1]), _f(t[3]), _f(t[4]), _f(t[5])))
                elif rt == "MAGN":
                    magn.append((_f(t[1]), _f(t[3]), _f(t[4]), _f(t[5])))
                elif rt == "PRES":
                    pres.append((_f(t[1]), _f(t[3])))                  # app_ts, mbar
                elif rt == "GNSS":
                    gnss.append((_f(t[1]), _f(t[3]), _f(t[4]), _f(t[7])))  # app, lat, lon, acc
                elif rt == "WIFI":
                    wifi[round(_f(t[1]), 3)].append((t[4], _f(t[5]), _f(t[6])))  # mac, freq, rss
                elif rt == "POSI":
                    posi.append((_f(t[1]), _f(t[3]), _f(t[4]), int(float(t[5]))))  # app, lat, lon, floor
            except (IndexError, ValueError):
                continue
    posi.sort()
    return {
        "acce": acce, "gyro": gyro, "magn": magn, "pres": pres, "gnss": gnss,
        "wifi": dict(sorted(wifi.items())), "posi": posi,
    }


def latlon_to_local(lat, lon, lat0, lon0):
    """Equirectangular approx -> local meters (east, north). Fine at building scale."""
    east = math.radians(lon - lon0) * 6378137.0 * math.cos(math.radians(lat0))
    north = math.radians(lat - lat0) * 6378137.0
    return east, north


def make_gt(posi):
    """Build gt(app_ts) -> (lat, lon, floor) via piecewise-linear time interpolation.
    Returns None outside the [first, last] POSI window (no extrapolation)."""
    ts = [p[0] for p in posi]

    def gt(t):
        if not posi or t < ts[0] or t > ts[-1]:
            return None
        i = bisect.bisect_right(ts, t) - 1
        if i >= len(posi) - 1:
            _, la, lo, fl = posi[-1]
            return (la, lo, fl)
        t0, la0, lo0, f0 = posi[i]
        t1, la1, lo1, _ = posi[i + 1]
        a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return (la0 + a * (la1 - la0), lo0 + a * (lo1 - lo0), f0)

    return gt


def _self_test(path):
    d = parse_logfile(path)
    wifi_rows = sum(len(v) for v in d["wifi"].values())
    print(f"file: {os.path.basename(path)}")
    print(f"  ACCE={len(d['acce'])} GYRO={len(d['gyro'])} MAGN={len(d['magn'])} "
          f"PRES={len(d['pres'])} GNSS={len(d['gnss'])}")
    print(f"  WIFI scans={len(d['wifi'])} rows={wifi_rows}  POSI={len(d['posi'])}")
    if d["acce"]:
        dur = d["acce"][-1][0] - d["acce"][0][0]
        print(f"  duration={dur:.1f}s  ACCE rate={len(d['acce'])/dur:.1f} Hz")
    if d["wifi"]:
        sizes = [len(v) for v in d["wifi"].values()]
        print(f"  APs/scan: min={min(sizes)} mean={sum(sizes)/len(sizes):.1f} max={max(sizes)}")
        uniq = {m for v in d["wifi"].values() for (m, _, _) in v}
        print(f"  unique APs (MACs): {len(uniq)}")
    if d["posi"]:
        print(f"  POSI[:3]={[(round(a,1), round(la,6), round(lo,6), fl) for a,la,lo,fl in d['posi'][:3]]}")
        floors = sorted({p[3] for p in d["posi"]})
        print(f"  floors visited: {floors}")
        # sanity: GT interpolation midway
        gt = make_gt(d["posi"])
        mid = (d["posi"][0][0] + d["posi"][-1][0]) / 2
        print(f"  gt(mid={mid:.1f}) = {gt(mid)}")


if __name__ == "__main__":
    _self_test(os.path.join(TRAIN_DIR, "T01_01.txt"))
