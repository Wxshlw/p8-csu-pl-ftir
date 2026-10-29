"""exp29 -- where the level-dependent spectral change lies, band by band.

Why this file exists
--------------------
The audit quantities (CNR, curvature) are scalars, and the manuscript uses them to rank
seven matrix-adulterant systems without ever naming the spectral region that carries the
contrast.  This script measures that region from the deposited spectra so that the text
can describe the contrast chemically instead of statistically.

Definition
----------
For every dataset the level-mean absorbance at the highest prepared concentration is
differenced against the level-mean absorbance at 0 % (pure matrix), on the stored
absorbance scale and with the same spectra the audit uses.  A linear baseline fitted
through two windows that carry no analyte band in these materials (1870-1830 and
2500-2450 cm^-1) is removed from the difference, and the difference is smoothed with a
Savitzky-Golay filter (window 21 channels, order 2) before the extrema are located.
Reported positions are therefore measured band positions, not assumed assignments.

Output
------
out/exp29_band_contrast.csv   dataset, level_high, n_high, n_zero, wavenumber,
                              signed amplitude (absorbance), rank, sign
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

from unmix_ftir import data

DATASETS = ["D1", "D2", "D3", "D7d", "D7p", "D7w", "D8"]
BASELINE_WINDOWS = [(1870.0, 1830.0), (2500.0, 2450.0)]
MIN_SEP_CM = 20.0
TOP = 6


def remove_baseline(d, wvn):
    """Subtract the straight line through the two band-free windows."""
    xs, ys = [], []
    for lo, hi in BASELINE_WINDOWS:
        m = (wvn >= lo) & (wvn <= hi)
        if m.any():
            xs.append(wvn[m].mean())
            ys.append(d[m].mean())
    if len(xs) < 2:
        return d
    a, b = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
    return d - (a * wvn + b)


def extrema(d, wvn, step, lo=400.0, hi=3600.0):
    """Signed extrema at least MIN_SEP_CM apart, inside [lo, hi], ranked by magnitude."""
    near = int(max(1, round(MIN_SEP_CM / max(step, 1e-9))))
    peaks, _ = find_peaks(d, distance=near)
    troughs, _ = find_peaks(-d, distance=near)
    cand = [(wvn[i], float(d[i])) for i in peaks] + [(wvn[i], float(d[i])) for i in troughs]
    cand = [t for t in cand if lo <= t[0] <= hi]
    cand.sort(key=lambda t: -abs(t[1]))
    kept = []
    for w, v in cand:
        if all(abs(w - w2) >= MIN_SEP_CM for w2, _ in kept):
            kept.append((w, v))
        if len(kept) == TOP:
            break
    return kept


def main():
    rows = []
    print("dataset  level   n_high n_zero  step    top extrema (cm^-1, signed absorbance)")
    for nm in DATASETS:
        d = data.load(nm)
        X, y, wvn = d["X"], np.asarray(d["y"], float), np.asarray(d["wvn"], float)
        levels = np.unique(y)
        hi, zero = levels.max(), levels.min()
        if zero != 0.0:
            raise RuntimeError(f"{nm}: lowest prepared level is {zero}, not 0")
        m_hi, m_0 = y == hi, y == 0.0
        d_hi = X[m_hi].mean(axis=0)
        d_0 = X[m_0].mean(axis=0)
        diff = savgol_filter(d_hi - d_0, window_length=21, polyorder=2)
        diff = remove_baseline(diff, wvn)
        step = float(np.median(np.abs(np.diff(wvn))))
        kept = extrema(diff, wvn, step)
        print(f"{nm:7s} {hi:5.1f}%  {int(m_hi.sum()):5d}  {int(m_0.sum()):6d}  {step:5.3f}   "
              + "  ".join(f"{w:7.1f}({v:+.4f})" for w, v in kept))
        for r, (w, v) in enumerate(kept, 1):
            rows.append(dict(dataset=nm, level_high=hi, n_high=int(m_hi.sum()),
                             n_zero=int(m_0.sum()), wavenumber=round(w, 1),
                             amplitude=round(v, 5), rank=r,
                             sign="up" if v > 0 else "down"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp29_band_contrast.csv"), index=False)
    print("\nwrote out/exp29_band_contrast.csv  rows:", len(df))


if __name__ == "__main__":
    main()
