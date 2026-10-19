# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3, cfarm420).
- **Toolchains**: go1.27.1 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128** (3.3.10 built from source by setup.sh); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
- **Single-threaded** for the apples-to-apples core comparison: FFTW planned with `threads=1`; numpy/scipy pinned via `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`; scipy `workers=1`; Go benchmarks are single-goroutine for 1-D. The 2-D rows are the one place go-fft uses its multicore path (the others are all 1-core).
- **Plan reuse / steady state**: go-fft via its cached `Plan` API (`NewPlan(n).FFT`, `NewRealPlan(n).RFFT`); gonum via its reused `CmplxFFT`/`FFT` object; FFTW via a reused `FFTW_MEASURE` plan; scipy via its internal plan cache. Each number is the **steady-state transform** cost, not planning — plan/setup cost is reported separately below.
- **Iterations**: Go uses `-benchtime=1s` (auto-scaled `b.N`); the C and Python harnesses auto-scale each batch to ~0.2 s and take the **best of 6** batches after warm-up. Lower ns/op is better.
- **Metric**: ns/op and **GFLOP/s** using the standard `5·N·log2(N)` flop convention for a complex N-point FFT (real rfft counted at half, `2.5·N·log2(N)`; 2-D at `5·N·log2(N)` with N = total points).
- **Inputs**: bit-identical across all four implementations (`((i·7+1)%13)·0.1 + i·((i·3+2)%11)·0.1` for complex; `((i·7+1)%13)·0.1` for real).
- **Note on pyfftw**: the FFTW column is the **native FFTW called directly from C** (`benchmarks/cbench/fftw_bench.c`), not pyfftw: the pip-wheel FFTW bundled with pyfftw planned a 2-D transform about 4× slower than the native library on Apple Silicon.

## Complex 1-D FFT (`complex128`)

ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW (lower is better; ≤1.05 = parity).

| N | go-fft | FFTW | numpy.fft | scipy.fft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 292 (35.1) | 6,894 (1.5) | 5,178 (2.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,550 (33.0) | 13,200 (3.9) | 9,407 (5.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,686 (23.0) | 44,007 (5.6) | 33,396 (7.4) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 324,410 (16.2) | 2,079,205 (2.5) | 991,250 (5.3) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,200,520 (9.4) | 26,648,411 (3.9) | 14,332,667 (7.3) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,031 (24.5) | 14,422 (3.5) | 9,778 (5.1) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,351 (23.1) | 14,849 (3.7) | 11,113 (4.9) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,892 (26.9) | 23,188 (4.5) | 16,732 (6.3) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,521 (2.5) | 61,840 (0.8) | 39,462 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,063 (21.9) | 17,311 (3.9) | 12,917 (5.2) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 220,069 (3.0) | 722,841 (0.9) | 489,950 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 223 (22.9) | 7,048 (0.7) | 7,063 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,080 (23.7) | 10,350 (2.5) | 8,366 (3.1) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,279 (23.3) | 47,334 (2.6) | 30,744 (4.0) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 177,627 (14.8) | 493,606 (5.3) | 524,895 (5.0) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,897,301 (13.5) | 8,628,409 (6.1) | 10,246,673 (5.1) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,302 (19.1) | 10,875 (2.3) | 8,853 (2.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,463 (18.6) | 10,926 (2.5) | 8,899 (3.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,374 (22.1) | 15,137 (3.5) | 12,942 (4.0) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 318 (16.1) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,197 (11.7) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,887 (15.6) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 193,744 (13.5) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,153,717 (12.6) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,597 (15.6) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,655 (16.4) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,809 (18.6) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 11,228 (21.9) | 49,003 (5.0) | 33,008 (7.4) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 64,916 (17.7) | 155,987 (7.4) | 128,684 (8.9) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 284,764 (18.4) | 608,196 (8.6) | 452,318 (11.6) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,354,687 (17.4) | 2,997,164 (7.9) | 2,053,714 (11.5) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,713,950 (15.6) | 13,231,353 (7.9) | 9,455,654 (11.1) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,816,633 |
| 1,024 | — | — | 131,085,930 |
| 4,096 | — | — | 243,608,200 |
| 65,536 | — | — | 2,325,037,126 |
| 1,048,576 | — | — | 4,355,420,370 |
| 1,000 | — | — | 153,258,869 |
| 1,080 | — | — | 366,556,286 |
| 1,920 | — | — | 616,561,241 |
| 1,009 | — | — | 146,825,796 |
| 1,296 | — | — | 279,604,678 |
| 10,007 | — | — | 1,318,709,319 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
