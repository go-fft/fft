#!/usr/bin/env python3
"""Reference-FFT timing harness for the go-fft performance-parity report.

Times three references on the SAME host, SAME input arrays, SAME sizes as the
Go ``benchmarks/h2h_test.go`` matrix, single-threaded, with plan reuse measured
in steady state (planning cost reported separately):

* **FFTW** (the gold-standard C library) via ``pyfftw`` — explicit plan, reused.
* **numpy.fft** — no explicit plan (pocketfft, replans each call internally).
* **scipy.fft** — pocketfft with an internal plan cache, ``workers=1``.

Inputs are bit-identical to the Go generators in ``h2h_test.go`` so the two
sides transform the same data. Output is a JSON document on stdout (the report
generator consumes it) plus a human table on stderr.

Single-thread pinning: set before importing numpy/scipy ::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 fft_reference.py

pyfftw is always told ``threads=1``.

Metric: ns/op (best of repeated timed batches, each auto-scaled to ~0.2 s) and
GFLOP/s using the standard 5*N*log2(N) convention for a complex N-point FFT (a
real N-point rfft is counted as half, 2.5*N*log2(N), the usual convention).
"""
import json
import math
import sys
import time

import numpy as np

try:
    import scipy.fft as spfft
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False

try:
    import pyfftw
    HAVE_FFTW = True
except Exception:  # pragma: no cover
    HAVE_FFTW = False

# Size taxonomy — mirrors benchmarks/h2h_test.go exactly.
COMPLEX_SIZES = [256, 1024, 4096, 65536, 1048576,
                 1000, 1080, 1920, 1009, 1296, 10007]
REAL_SIZES = [256, 1024, 4096, 65536, 1048576, 1000, 1080, 1920]
SHAPES = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]


def cmplx(n):
    """Identical to h2hComplex in the Go harness."""
    i = np.arange(n)
    return (((i * 7 + 1) % 13) * 0.1 + 1j * (((i * 3 + 2) % 11) * 0.1)).astype(np.complex128)


def realv(n):
    """Identical to h2hReal in the Go harness."""
    i = np.arange(n)
    return (((i * 7 + 1) % 13) * 0.1).astype(np.float64)


def gflops(n, ns_per_op, real=False):
    if ns_per_op <= 0 or n <= 1:
        return 0.0
    flops = 5.0 * n * math.log2(n)
    if real:
        flops *= 0.5
    return flops / (ns_per_op * 1e-9) / 1e9


def time_ns(fn, inner_target=0.2, batches=6):
    """Best-of-`batches` ns/op; each batch auto-scaled to ~inner_target seconds.

    Returns (best_ns, median_ns).
    """
    fn(); fn()  # warm up (also lets a plan cache populate)
    iters = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        dt = time.perf_counter() - t0
        if dt >= inner_target or iters > (1 << 26):
            break
        iters *= 2
    samples = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        samples.append((time.perf_counter() - t0) / iters)
    samples.sort()
    best = samples[0] * 1e9
    median = samples[len(samples) // 2] * 1e9
    return best, median


# ---- FFTW (pyfftw) plan builders: plan once, reuse the call object ----------

def fftw_complex_caller(x):
    a = pyfftw.empty_aligned(x.shape, dtype="complex128")
    a[:] = x
    plan = pyfftw.builders.fft(a, threads=1, planner_effort="FFTW_MEASURE",
                               overwrite_input=False, auto_align_input=True)
    return plan


def fftw_real_caller(x):
    a = pyfftw.empty_aligned(x.shape, dtype="float64")
    a[:] = x
    plan = pyfftw.builders.rfft(a, threads=1, planner_effort="FFTW_MEASURE",
                                overwrite_input=False, auto_align_input=True)
    return plan


def fftw_fft2_caller(x):
    a = pyfftw.empty_aligned(x.shape, dtype="complex128")
    a[:] = x
    plan = pyfftw.builders.fft2(a, threads=1, planner_effort="FFTW_MEASURE",
                                overwrite_input=False, auto_align_input=True)
    return plan


def measure_plan_ns(build_fn, x):
    """Wall time to construct one plan (FFTW planning cost), best of 3."""
    best = math.inf
    for _ in range(3):
        t0 = time.perf_counter()
        build_fn(x)
        best = min(best, time.perf_counter() - t0)
    return best * 1e9


def run():
    out = {
        "host": {},
        "versions": {
            "numpy": np.__version__,
            "scipy": (scipy_version() if HAVE_SCIPY else None),
            "pyfftw": (pyfftw.__version__ if HAVE_FFTW else None),
            "fftw": (fftw_version() if HAVE_FFTW else None),
        },
        "complex": [], "real": [], "fft2": [],
    }

    log = lambda *a: print(*a, file=sys.stderr)
    log("# numpy", np.__version__, "| scipy",
        (scipy_version() if HAVE_SCIPY else "n/a"),
        "| pyfftw", (pyfftw.__version__ if HAVE_FFTW else "n/a"),
        "| FFTW", (fftw_version() if HAVE_FFTW else "n/a"))

    # ---------- Complex 1-D ----------
    log("\n## Complex FFT (ns/op | GFLOP/s)")
    log(f"{'N':>9} {'FFTW':>22} {'numpy':>22} {'scipy':>22}")
    for n in COMPLEX_SIZES:
        x = cmplx(n)
        row = {"n": n}
        if HAVE_FFTW:
            plan = fftw_complex_caller(x)
            b, m = time_ns(lambda: plan())
            row["fftw_ns"] = b; row["fftw_gflops"] = gflops(n, b)
            row["fftw_plan_ns"] = measure_plan_ns(fftw_complex_caller, x)
        b, m = time_ns(lambda: np.fft.fft(x))
        row["numpy_ns"] = b; row["numpy_gflops"] = gflops(n, b)
        if HAVE_SCIPY:
            b, m = time_ns(lambda: spfft.fft(x, workers=1))
            row["scipy_ns"] = b; row["scipy_gflops"] = gflops(n, b)
        out["complex"].append(row)
        log(f"{n:>9} {fmt(row,'fftw'):>22} {fmt(row,'numpy'):>22} {fmt(row,'scipy'):>22}")

    # ---------- Real 1-D ----------
    log("\n## Real RFFT (ns/op | GFLOP/s)")
    log(f"{'N':>9} {'FFTW':>22} {'numpy':>22} {'scipy':>22}")
    for n in REAL_SIZES:
        x = realv(n)
        row = {"n": n}
        if HAVE_FFTW:
            plan = fftw_real_caller(x)
            b, m = time_ns(lambda: plan())
            row["fftw_ns"] = b; row["fftw_gflops"] = gflops(n, b, real=True)
            row["fftw_plan_ns"] = measure_plan_ns(fftw_real_caller, x)
        b, m = time_ns(lambda: np.fft.rfft(x))
        row["numpy_ns"] = b; row["numpy_gflops"] = gflops(n, b, real=True)
        if HAVE_SCIPY:
            b, m = time_ns(lambda: spfft.rfft(x, workers=1))
            row["scipy_ns"] = b; row["scipy_gflops"] = gflops(n, b, real=True)
        out["real"].append(row)
        log(f"{n:>9} {fmt(row,'fftw'):>22} {fmt(row,'numpy'):>22} {fmt(row,'scipy'):>22}")

    # ---------- 2-D complex ----------
    log("\n## 2-D FFT2 (ns/op | GFLOP/s)")
    log(f"{'shape':>9} {'FFTW':>22} {'numpy':>22} {'scipy':>22}")
    for s in SHAPES:
        x = cmplx(s[0] * s[1]).reshape(s)
        nn = s[0] * s[1]
        # 2-D complex FFT2 flop convention: 5*N*log2(N) with N = total points.
        row = {"shape": f"{s[0]}x{s[1]}", "n": nn}
        if HAVE_FFTW:
            plan = fftw_fft2_caller(x)
            b, m = time_ns(lambda: plan())
            row["fftw_ns"] = b; row["fftw_gflops"] = gflops(nn, b)
            row["fftw_plan_ns"] = measure_plan_ns(fftw_fft2_caller, x)
        b, m = time_ns(lambda: np.fft.fft2(x))
        row["numpy_ns"] = b; row["numpy_gflops"] = gflops(nn, b)
        if HAVE_SCIPY:
            b, m = time_ns(lambda: spfft.fft2(x, workers=1))
            row["scipy_ns"] = b; row["scipy_gflops"] = gflops(nn, b)
        out["fft2"].append(row)
        log(f"{row['shape']:>9} {fmt(row,'fftw'):>22} {fmt(row,'numpy'):>22} {fmt(row,'scipy'):>22}")

    json.dump(out, sys.stdout)
    sys.stdout.write("\n")


def fmt(row, key):
    ns = row.get(key + "_ns")
    if ns is None:
        return "—"
    g = row.get(key + "_gflops", 0.0)
    return f"{ns:,.0f} ({g:.2f})"


def scipy_version():
    import scipy
    return scipy.__version__


def fftw_version():
    # pyfftw doesn't expose the linked FFTW version directly; query the dylib.
    import ctypes
    import ctypes.util
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("fftw3") or "libfftw3.dylib")
        lib.fftw_version  # noqa
        return None  # symbol form differs by build; left as build-string below
    except Exception:
        return None


if __name__ == "__main__":
    run()
