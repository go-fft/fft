# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Ampere Altra (Neoverse-N1), cfarm424, 64 cores, pinned to one core.
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
| 256 (2⁸) | 1,111 (9.2) | 1,158 (8.8) | 9,636 (1.1) | 7,775 (1.3) | 5,851 (1.8) | 0.96× | **≥ parity** |
| 1,024 (2¹⁰) | 5,264 (9.7) | 6,317 (8.1) | 19,089 (2.7) | 14,706 (3.5) | 30,390 (1.7) | 0.83× | **≥ parity** |
| 4,096 (2¹²) | 27,868 (8.8) | 41,049 (6.0) | 60,773 (4.0) | 48,028 (5.1) | 147,124 (1.7) | 0.68× | **≥ parity** |
| 65,536 (2¹⁶) | 721,279 (7.3) | 1,228,385 (4.3) | 2,123,847 (2.5) | 1,434,630 (3.7) | 3,389,743 (1.5) | 0.59× | **≥ parity** |
| 1,048,576 (2²⁰) | 20,130,779 (5.2) | 44,269,336 (2.4) | 44,403,435 (2.4) | 32,268,204 (3.2) | 75,229,513 (1.4) | 0.45× | **≥ parity** |
| 1,000 (2³·5³) | 7,342 (6.8) | 7,164 (7.0) | 19,484 (2.6) | 15,135 (3.3) | 33,799 (1.5) | 1.02× | **≥ parity** |
| 1,080 (2³·3³·5) | 8,711 (6.2) | 7,713 (7.1) | 21,664 (2.5) | 17,012 (3.2) | 41,472 (1.3) | 1.13× | lags FFTW 1.13× |
| 1,920 (2⁷·3·5) | 15,099 (6.9) | 13,562 (7.7) | 32,687 (3.2) | 26,007 (4.0) | 70,203 (1.5) | 1.11× | lags FFTW 1.11× |
| 1,009 (prime) | 21,638 (2.3) | 48,238 (1.0) | 96,251 (0.5) | 57,397 (0.9) | 2,214,103 (0.0) | 0.45× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 11,678 (5.7) | 11,407 (5.9) | 24,705 (2.7) | 19,728 (3.4) | 49,470 (1.4) | 1.02× | **≥ parity** |
| 10,007 (prime) | 468,607 (1.4) | 614,812 (1.1) | 1,034,675 (0.6) | 764,943 (0.9) | 219,833,693 (0.0) | 0.76× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 839 (6.1) | 720 (7.1) | 8,793 (0.6) | 7,875 (0.7) | 3,002 (1.7) | 1.17× | lags FFTW 1.17× |
| 1,024 (2¹⁰) | 3,599 (7.1) | 4,213 (6.1) | 14,372 (1.8) | 12,533 (2.0) | 15,277 (1.7) | 0.85× | **≥ parity** |
| 4,096 (2¹²) | 16,644 (7.4) | 19,851 (6.2) | 37,405 (3.3) | 33,391 (3.7) | 68,949 (1.8) | 0.84× | **≥ parity** |
| 65,536 (2¹⁶) | 413,605 (6.3) | 556,857 (4.7) | 758,957 (3.5) | 772,139 (3.4) | 1,736,490 (1.5) | 0.74× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,689,725 (4.9) | 19,203,977 (2.7) | 21,416,699 (2.4) | 19,315,265 (2.7) | 42,825,936 (1.2) | 0.56× | **≥ parity** |
| 1,000 (2³·5³) | 4,733 (5.3) | 3,978 (6.3) | 15,528 (1.6) | 12,694 (2.0) | 14,851 (1.7) | 1.19× | lags FFTW 1.19× |
| 1,080 (2³·3³·5) | 5,519 (4.9) | 4,249 (6.4) | 15,916 (1.7) | 13,997 (1.9) | 17,423 (1.6) | 1.30× | lags FFTW 1.30× |
| 1,920 (2⁷·3·5) | 8,983 (5.8) | 7,360 (7.1) | 22,473 (2.3) | 19,025 (2.8) | 30,756 (1.7) | 1.22× | lags FFTW 1.22× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 961 (5.3) | 812 (6.3) | 1.18× | lags FFTW 1.18× |
| 1,024 (2¹⁰) | 3,942 (6.5) | 4,488 (5.7) | 0.88× | **≥ parity** |
| 4,096 (2¹²) | 17,936 (6.9) | 21,063 (5.8) | 0.85× | **≥ parity** |
| 65,536 (2¹⁶) | 437,159 (6.0) | 648,957 (4.0) | 0.67× | **≥ parity** |
| 1,048,576 (2²⁰) | 11,302,626 (4.6) | 23,873,086 (2.2) | 0.47× | **≥ parity** |
| 1,000 (2³·5³) | 5,095 (4.9) | 4,252 (5.9) | 1.20× | lags FFTW 1.20× |
| 1,080 (2³·3³·5) | 5,890 (4.6) | 4,426 (6.1) | 1.33× | lags FFTW 1.33× |
| 1,920 (2⁷·3·5) | 9,545 (5.5) | 7,833 (6.7) | 1.22× | lags FFTW 1.22× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 34,407 (7.1) | 41,185 (6.0) | 30,915 (8.0) | 73,278 (3.4) | 51,265 (4.8) | 1.11× | lags FFTW 1.11× |
| 128x128 | 149,002 (7.7) | 175,732 (6.5) | 199,171 (5.8) | 259,769 (4.4) | 199,701 (5.7) | 0.75× | **≥ parity** |
| 256x256 | 745,913 (7.0) | 843,286 (6.2) | 1,326,163 (4.0) | 1,140,124 (4.6) | 868,014 (6.0) | 0.56× | **≥ parity** |
| 512x512 | 3,399,213 (6.9) | 3,980,271 (5.9) | 7,341,626 (3.2) | 5,860,452 (4.0) | 4,176,480 (5.6) | 0.46× | **≥ parity** |
| 1024x1024 | 18,200,866 (5.8) | 18,983,113 (5.5) | 46,244,175 (2.3) | 26,236,941 (4.0) | 23,008,661 (4.6) | 0.39× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 12,356 | 9,692 | 4,727,134 |
| 1,024 | 44,709 | 34,457 | 3,674,922 |
| 4,096 | 162,233 | 130,338 | 6,264,431 |
| 65,536 | 2,731,905 | 1,956,913 | 21,756,252 |
| 1,048,576 | 39,637,917 | 32,352,017 | 20,221,513 |
| 1,000 | 43,202 | 34,304 | 4,864,736 |
| 1,080 | 50,423 | 37,034 | 13,096,312 |
| 1,920 | 80,562 | 64,181 | 17,769,964 |
| 1,009 | 137,915 | 40,242 | 6,102,150 |
| 1,296 | 60,815 | 44,123 | 10,443,481 |
| 10,007 | 1,425,269 | 402,863 | 37,011,828 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 17/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 1,080 (2³·3³·5): 1.30×
- real 1,920 (2⁷·3·5): 1.22×
- real 1,000 (2³·5³): 1.19×
- real 256 (2⁸): 1.17×
- complex 1,080 (2³·3³·5): 1.13×
- complex 1,920 (2⁷·3·5): 1.11×
- 2-D 64x64: 1.11×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
