#!/usr/bin/env python3
"""Numerical-correctness gate for the parity benchmark.

Reads the JSON dumped by ``verify/main.go`` (go-fft outputs on the benchmark
inputs) and checks every transform against ``numpy.fft`` within tolerance. A
fast wrong answer does not count, so run.sh runs this BEFORE the timing harness
and aborts on failure.

Usage: go run ./verify | python3 verify_correctness.py
"""
import json
import sys

import numpy as np

RTOL = 1e-9
ATOL = 1e-7


def cmplx(n):
    i = np.arange(n)
    return (((i * 7 + 1) % 13) * 0.1 + 1j * (((i * 3 + 2) % 11) * 0.1)).astype(np.complex128)


def realv(n):
    i = np.arange(n)
    return (((i * 7 + 1) % 13) * 0.1).astype(np.float64)


def halfspec(n):
    i = np.arange(n // 2 + 1)
    return (((i * 7 + 1) % 13) * 0.1 + 1j * (((i * 3 + 2) % 11) * 0.1)).astype(np.complex128)


def to_np(pairs):
    return np.array([p["Re"] + 1j * p["Im"] for p in pairs], dtype=np.complex128)


def main():
    data = json.load(sys.stdin)
    failures = 0
    checked = 0

    for k, pairs in data["complex"].items():
        n = int(k)
        got = to_np(pairs)
        ref = np.fft.fft(cmplx(n))
        ok = np.allclose(got, ref, rtol=RTOL, atol=ATOL)
        checked += 1
        if not ok:
            failures += 1
            err = np.max(np.abs(got - ref))
            print(f"FAIL  complex N={n}  max|err|={err:.3e}", file=sys.stderr)

    for k, pairs in data["real"].items():
        n = int(k)
        got = to_np(pairs)
        ref = np.fft.rfft(realv(n))
        ok = np.allclose(got, ref, rtol=RTOL, atol=ATOL)
        checked += 1
        if not ok:
            failures += 1
            err = np.max(np.abs(got - ref))
            print(f"FAIL  rfft N={n}  max|err|={err:.3e}", file=sys.stderr)

    for k, vals in data.get("ireal", {}).items():
        n = int(k)
        got = np.asarray(vals, dtype=np.float64)
        # numpy.fft.irfft of the same half spectrum the FFTW c2r harness inverts;
        # numpy discards the imaginary part of the DC/Nyquist bins, exactly as the
        # packed inverse does, so the two agree to tolerance for any half spectrum.
        ref = np.fft.irfft(halfspec(n), n)
        ok = np.allclose(got, ref, rtol=RTOL, atol=ATOL)
        checked += 1
        if not ok:
            failures += 1
            err = np.max(np.abs(got - ref))
            print(f"FAIL  irfft N={n}  max|err|={err:.3e}", file=sys.stderr)

    for k, pairs in data["fft2"].items():
        a, b = (int(v) for v in k.split("x"))
        got = to_np(pairs).reshape((a, b))
        ref = np.fft.fft2(cmplx(a * b).reshape((a, b)))
        ok = np.allclose(got, ref, rtol=RTOL, atol=ATOL)
        checked += 1
        if not ok:
            failures += 1
            err = np.max(np.abs(got - ref))
            print(f"FAIL  fft2 {k}  max|err|={err:.3e}", file=sys.stderr)

    if failures:
        print(f"\nCORRECTNESS: {failures}/{checked} transforms FAILED", file=sys.stderr)
        sys.exit(1)
    print(f"CORRECTNESS: all {checked} go-fft transforms match numpy.fft "
          f"(rtol={RTOL}, atol={ATOL})", file=sys.stderr)


if __name__ == "__main__":
    main()
