#!/usr/bin/env python3
"""Generate manuscript figures from saved real-data error arrays (OBJ-059)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "..", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})


def cdf(ax, e, label, **kw):
    d = np.sort(e[np.isfinite(e)])
    ax.plot(d, np.linspace(0, 1, len(d)), label=label, **kw)


a = np.load(os.path.join(HERE, "gsdc_errors.npy"))
plain, fused, official, nsat = a[:, 1], a[:, 3], a[:, 6], a[:, 5]

# Fig 1: GNSS error CDF — naive vs official vs ours (the key honest result: ours ~ official)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
for ax, mask, title in [(axes[0], np.ones(len(a), bool), "All epochs"),
                        (axes[1], nsat <= np.percentile(nsat, 30), "Few satellites (poor geometry)")]:
    cdf(ax, plain[mask], "Naïve WLS", color="#9e9e9e", lw=1.4)
    cdf(ax, official[mask], "Official WLS", color="#1f77b4", lw=1.4)
    cdf(ax, fused[mask], "Ours (robust+EKF)", color="#2ca02c", lw=1.6, ls="--")
    ax.set_xlim(0, 12); ax.set_xlabel("Horizontal error (m)"); ax.set_title(title)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("CDF"); axes[0].legend(loc="lower right", fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_gnss_cdf.pdf"))
fig.savefig(os.path.join(FIG, "fig_gnss_cdf.png"))
plt.close(fig)

print("saved:", os.listdir(FIG))
