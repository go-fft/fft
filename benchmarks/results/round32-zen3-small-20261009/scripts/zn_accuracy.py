#!/usr/bin/env python3
"""zn_accuracy.py DIR: numpy.fft on the exact bytes TestZnExport wrote.

For every exported length: go-fft's FFT, IFFT, RFFT and IRFFT against
numpy's on the same input bytes (the IRFFT on go-fft's own RFFT output, the
bytes go-fft read). Prints the largest error relative to the largest output
magnitude per transform, and fails above 1e-14 (the parity gate's rtol is
1e-9)."""
import os
import sys

import numpy as np

d = sys.argv[1]
TOL = 1e-14


def rd(name, cplx):
    x = np.fromfile(os.path.join(d, name), dtype="<f8")
    return x[0::2] + 1j * x[1::2] if cplx else x


def err(got, want):
    return float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-300))


worst = {}
bad = 0
rows = 0
for line in open(os.path.join(d, "lengths.txt")):
    n, zn = line.split()
    n = int(n)
    x = rd(f"c{n}.in", True)
    r = rd(f"r{2*n}.in", False)
    spec = rd(f"r{2*n}.rfft", True)
    checks = {
        "fft": err(rd(f"c{n}.fft", True), np.fft.fft(x)),
        "ifft": err(rd(f"c{n}.ifft", True), np.fft.ifft(x)),
        "rfft": err(spec, np.fft.rfft(r)),
        "irfft": err(rd(f"r{2*n}.irfft", False), np.fft.irfft(spec, 2 * n)),
    }
    rows += 1
    for k, e in checks.items():
        if e > worst.get(k, (0, 0))[0]:
            worst[k] = (e, n)
        if e > TOL:
            bad += 1
            print(f"FAIL n={n} (kernel {zn}) {k}: {e:.3e}")
print(f"{rows} lengths (complex n, real 2n), numpy {np.__version__}")
for k, (e, n) in sorted(worst.items()):
    print(f"  {k:6} worst relative error {e:.3e} at n={n}")
print("PASS" if bad == 0 else f"FAIL: {bad}")
sys.exit(1 if bad else 0)
