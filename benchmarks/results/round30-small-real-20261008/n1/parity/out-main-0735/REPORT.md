# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Neoverse-N1 (cfarm424), core 40.
- **Toolchains**: go1.27.1 linux/arm64 (cross-compiled); native **FFTW fftw-3.3.10-neon** (3.3.10 built from source by setup.sh); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 1,115 (9.2) | 1,156 (8.9) | 9,523 (1.1) | 7,687 (1.3) | 5,838 (1.8) | 0.96× | **≥ parity** |
| 1,024 (2¹⁰) | 5,247 (9.8) | 6,403 (8.0) | 18,833 (2.7) | 14,594 (3.5) | 30,308 (1.7) | 0.82× | **≥ parity** |
| 4,096 (2¹²) | 27,571 (8.9) | 41,076 (6.0) | 60,699 (4.0) | 47,909 (5.1) | 147,399 (1.7) | 0.67× | **≥ parity** |
| 65,536 (2¹⁶) | 718,817 (7.3) | 1,225,599 (4.3) | 2,289,293 (2.3) | 1,525,630 (3.4) | 3,401,412 (1.5) | 0.59× | **≥ parity** |
| 1,048,576 (2²⁰) | 20,468,578 (5.1) | 44,838,682 (2.3) | 48,061,242 (2.2) | 31,595,187 (3.3) | 74,694,495 (1.4) | 0.46× | **≥ parity** |
| 1,000 (2³·5³) | 6,076 (8.2) | 7,240 (6.9) | 19,193 (2.6) | 15,101 (3.3) | 33,686 (1.5) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,885 (7.9) | 7,759 (7.0) | 21,821 (2.5) | 16,834 (3.2) | 41,444 (1.3) | 0.89× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,966 (8.8) | 13,599 (7.7) | 32,884 (3.2) | 25,903 (4.0) | 69,874 (1.5) | 0.88× | **≥ parity** |
| 1,009 (prime) | 21,491 (2.3) | 48,235 (1.0) | 96,734 (0.5) | 57,062 (0.9) | 2,216,078 (0.0) | 0.45× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,681 (7.7) | 11,433 (5.9) | 24,922 (2.7) | 19,736 (3.4) | 48,674 (1.4) | 0.76× | **≥ parity** |
| 10,007 (prime) | 418,930 (1.6) | 596,562 (1.1) | 1,040,834 (0.6) | 776,648 (0.9) | 217,149,029 (0.0) | 0.70× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 771 (6.6) | 722 (7.1) | 8,745 (0.6) | 7,755 (0.7) | 2,989 (1.7) | 1.07× | lags FFTW 1.07× |
| 1,024 (2¹⁰) | 3,272 (7.8) | 4,246 (6.0) | 14,350 (1.8) | 12,395 (2.1) | 15,158 (1.7) | 0.77× | **≥ parity** |
| 4,096 (2¹²) | 15,540 (7.9) | 19,925 (6.2) | 37,233 (3.3) | 32,809 (3.7) | 68,291 (1.8) | 0.78× | **≥ parity** |
| 65,536 (2¹⁶) | 395,987 (6.6) | 548,138 (4.8) | 738,645 (3.5) | 761,683 (3.4) | 1,744,424 (1.5) | 0.72× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,275,324 (5.1) | 18,196,177 (2.9) | 20,593,391 (2.5) | 18,652,332 (2.8) | 43,155,900 (1.2) | 0.56× | **≥ parity** |
| 1,000 (2³·5³) | 3,904 (6.4) | 3,960 (6.3) | 15,386 (1.6) | 12,617 (2.0) | 14,858 (1.7) | 0.99× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,372 (6.2) | 4,234 (6.4) | 15,964 (1.7) | 14,119 (1.9) | 17,474 (1.6) | 1.03× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,114 (7.4) | 7,443 (7.0) | 21,849 (2.4) | 19,028 (2.8) | 30,597 (1.7) | 0.96× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 778 (6.6) | 815 (6.3) | 0.96× | **≥ parity** |
| 1,024 (2¹⁰) | 3,229 (7.9) | 4,467 (5.7) | 0.72× | **≥ parity** |
| 4,096 (2¹²) | 15,517 (7.9) | 21,194 (5.8) | 0.73× | **≥ parity** |
| 65,536 (2¹⁶) | 399,185 (6.6) | 650,605 (4.0) | 0.61× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,307,131 (5.1) | 21,319,125 (2.5) | 0.48× | **≥ parity** |
| 1,000 (2³·5³) | 4,061 (6.1) | 4,243 (5.9) | 0.96× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,453 (6.1) | 4,428 (6.1) | 1.01× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,222 (7.2) | 7,826 (6.7) | 0.92× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 27,967 (8.8) | 33,170 (7.4) | 30,977 (7.9) | 72,063 (3.4) | 51,001 (4.8) | 0.90× | **≥ parity** |
| 128x128 | 125,400 (9.1) | 151,113 (7.6) | 191,626 (6.0) | 255,435 (4.5) | 201,133 (5.7) | 0.65× | **≥ parity** |
| 256x256 | 680,454 (7.7) | 757,029 (6.9) | 1,305,920 (4.0) | 1,110,390 (4.7) | 930,762 (5.6) | 0.52× | **≥ parity** |
| 512x512 | 3,627,162 (6.5) | 4,275,033 (5.5) | 7,031,556 (3.4) | 5,424,877 (4.3) | 3,822,381 (6.2) | 0.52× | **≥ parity** |
| 1024x1024 | 18,700,020 (5.6) | 19,072,256 (5.5) | 43,863,647 (2.4) | 26,563,092 (3.9) | 20,845,528 (5.0) | 0.43× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 4,460,460 |
| 1,024 | — | — | 3,373,446 |
| 4,096 | — | — | 6,299,565 |
| 65,536 | — | — | 21,904,377 |
| 1,048,576 | — | — | 19,990,349 |
| 1,000 | — | — | 4,767,265 |
| 1,080 | — | — | 12,726,773 |
| 1,920 | — | — | 17,392,434 |
| 1,009 | — | — | 6,089,842 |
| 1,296 | — | — | 10,621,024 |
| 10,007 | — | — | 35,595,402 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 23/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.07×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
