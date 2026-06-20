#!/usr/bin/env python3
"""Head-to-head FFT timing for numpy.fft and scipy.fft, mirroring the Go
BenchmarkH2H* matrix so go-fft can be compared on identical transforms/sizes on
the same host. Reports ns/op (median of repeated timed runs) for the steady
state, i.e. plan reuse where the backend supports it (scipy caches plans
internally; numpy has no explicit plan).

Usage: python3 scripts/fftbench.py
"""
import time
import numpy as np

try:
    import scipy.fft as spfft
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

COMPLEX_SIZES = [64, 256, 1024, 4096, 65536, 1000, 1296, 10007, 9973, 5003, 2017]
REAL_SIZES = [64, 256, 1024, 4096, 65536, 1000, 1296]
SHAPES = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]


def cmplx(n):
    i = np.arange(n)
    return ((i * 7 + 1) % 13) * 0.1 + 1j * (((i * 3 + 2) % 11) * 0.1)


def realv(n):
    i = np.arange(n)
    return ((i * 7 + 1) % 13) * 0.1


def time_ns(fn, inner_target=0.2):
    # Warm up (lets scipy build & cache its plan).
    fn(); fn()
    # Auto-scale iteration count to ~inner_target seconds, take the best.
    iters = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        dt = time.perf_counter() - t0
        if dt >= inner_target or iters > 1 << 24:
            break
        iters *= 2
    best = dt / iters
    for _ in range(4):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        dt = (time.perf_counter() - t0) / iters
        if dt < best:
            best = dt
    return best * 1e9


def main():
    print("# numpy", np.__version__, "scipy", (spfft.__name__ if HAVE_SCIPY else "n/a"))
    print("\n## Complex FFT (ns/op)")
    print(f"{'N':>8} {'numpy':>14} {'scipy':>14}")
    for n in COMPLEX_SIZES:
        x = cmplx(n)
        npn = time_ns(lambda: np.fft.fft(x))
        spn = time_ns(lambda: spfft.fft(x)) if HAVE_SCIPY else float('nan')
        print(f"{n:>8} {npn:>14.0f} {spn:>14.0f}")

    print("\n## Real RFFT (ns/op)")
    print(f"{'N':>8} {'numpy':>14} {'scipy':>14}")
    for n in REAL_SIZES:
        x = realv(n)
        npn = time_ns(lambda: np.fft.rfft(x))
        spn = time_ns(lambda: spfft.rfft(x)) if HAVE_SCIPY else float('nan')
        print(f"{n:>8} {npn:>14.0f} {spn:>14.0f}")

    print("\n## 2-D FFT2 (ns/op)")
    print(f"{'shape':>10} {'numpy':>14} {'scipy':>14}")
    for s in SHAPES:
        x = cmplx(s[0] * s[1]).reshape(s)
        npn = time_ns(lambda: np.fft.fft2(x))
        spn = time_ns(lambda: spfft.fft2(x, workers=1)) if HAVE_SCIPY else float('nan')
        print(f"{s[0]}x{s[1]:>6} {npn:>14.0f} {spn:>14.0f}")


if __name__ == "__main__":
    main()
