#!/usr/bin/env python3
"""r32_accuracy.py DIR [plant]: numpy.fft on the exact bytes TestR32Export wrote
(Round 36's zm_accuracy.py, unchanged but for this line: the same format).

Bluestein FFT/IFFT (b<n>.*), composite FFT/IFFT (c<n>.*), RFFT and IRFFT
(r<n>.*: the IRFFT of go-fft's own RFFT output, the bytes go-fft read) and
2-D FFT2/IFFT2 (d<n>.*): the largest error relative to the largest output
magnitude per transform. Fails above TOL_B (Bluestein: two FFTs of m >= 2n-1
points and three pointwise products) or TOL (the others). With "plant",
flips one high mantissa byte in a copy of each output first and requires
every check to FAIL (the checker can see a defect)."""
import os
import sys

import numpy as np

d = sys.argv[1]
plant = len(sys.argv) > 2 and sys.argv[2] == "plant"
TOL_B, TOL = 1e-13, 1e-14


def rd(name):
    x = np.fromfile(os.path.join(d, name), dtype="<f8")
    if plant and not name.endswith(".in"):
        x = x.copy()
        b = x.view(np.uint8)
        k = 8 * (2 * (len(x) // 6)) + 6  # a high mantissa byte of a real part
        b[k] ^= 0x08
    return x[0::2] + 1j * x[1::2]


def err(got, want):
    return float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-300))


worst, bad, rows, checks = {}, 0, 0, 0
for line in open(os.path.join(d, "list.txt")):
    kind, n, extra = line.split()
    n = int(n)
    x = rd(f"{kind}{n}.in")
    tol = TOL
    if kind in "bc":
        want = {kind + "fft": ("fft", np.fft.fft(x)), kind + "ifft": ("ifft", np.fft.ifft(x))}
        if kind == "b":
            tol = TOL_B
    elif kind == "r":
        spec = rd(f"r{n}.rfft")
        want = {"rfft": ("rfft", np.fft.rfft(x.real)), "irfft": ("irfft", np.fft.irfft(spec, n) + 0j)}
    else:
        x = x.reshape(n, n)
        want = {"fft2": ("fft", np.fft.fft2(x).ravel()), "ifft2": ("ifft", np.fft.ifft2(x).ravel())}
    rows += 1
    for k, (suffix, w) in want.items():
        e = err(rd(f"{kind}{n}.{suffix}"), w)
        checks += 1
        if e > worst.get(k, (0, 0))[0]:
            worst[k] = (e, n)
        if e > tol:
            bad += 1
            if not plant:
                print(f"FAIL {kind} n={n} ({extra}) {k}: {e:.3e}")
print(f"{rows} transforms, {checks} checks, numpy {np.__version__}{' (planted)' if plant else ''}")
for k, (e, n) in sorted(worst.items()):
    print(f"  {k:6} worst relative error {e:.3e} at n={n}")
if plant:
    print("PLANTED: all caught" if bad == checks else f"PLANTED: only {bad} of {checks} caught")
    sys.exit(0 if bad == checks else 1)
print("PASS" if bad == 0 else f"FAIL: {bad}")
sys.exit(1 if bad else 0)
