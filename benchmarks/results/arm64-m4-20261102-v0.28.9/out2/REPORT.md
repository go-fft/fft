# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Apple M4 Max (16-core: 12P+4E), macOS 27.2, arm64, GOMAXPROCS=1 (not pinned).
- **Toolchains**: go1.27.3 darwin/arm64; native **FFTW fftw-3.3.11-neon** (FFTW 3.3.11, pkgx bottle (go-pkgx/packages#399), NEON, static libfftw3.a linked from C); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 279 (36.7) | 305 (33.5) | 2,531 (4.0) | 1,970 (5.2) | 1,696 (6.0) | 0.91× | **≥ parity** |
| 1,024 (2¹⁰) | 1,260 (40.6) | 1,306 (39.2) | 5,352 (9.6) | 4,099 (12.5) | 8,810 (5.8) | 0.96× | **≥ parity** |
| 4,096 (2¹²) | 6,632 (37.1) | 7,146 (34.4) | 20,031 (12.3) | 15,376 (16.0) | 39,222 (6.3) | 0.93× | **≥ parity** |
| 65,536 (2¹⁶) | 263,410 (19.9) | 201,445 (26.0) | 502,331 (10.4) | 473,483 (11.1) | 977,889 (5.4) | 1.31× | lags FFTW 1.31× |
| 1,048,576 (2²⁰) | 4,777,291 (21.9) | 5,561,578 (18.9) | 10,399,146 (10.1) | 7,605,312 (13.8) | 21,724,406 (4.8) | 0.86× | **≥ parity** |
| 1,000 (2³·5³) | 1,519 (32.8) | 1,616 (30.8) | 5,641 (8.8) | 4,555 (10.9) | 9,272 (5.4) | 0.94× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,655 (32.9) | 1,789 (30.4) | 6,359 (8.6) | 4,840 (11.2) | 10,891 (5.0) | 0.93× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,861 (36.6) | 3,366 (31.1) | 9,863 (10.6) | 8,480 (12.3) | 18,619 (5.6) | 0.85× | **≥ parity** |
| 1,009 (prime) | 6,096 (8.3) | 10,791 (4.7) | 30,290 (1.7) | 18,022 (2.8) | 507,020 (0.1) | 0.56× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 2,032 (33.0) | 2,289 (29.3) | 7,245 (9.2) | 5,875 (11.4) | 12,907 (5.2) | 0.89× | **≥ parity** |
| 10,007 (prime) | 116,761 (5.7) | 136,312 (4.9) | 309,566 (2.1) | 265,306 (2.5) | 69,009,286 (0.0) | 0.86× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 202 (25.3) | 176 (29.1) | 2,139 (2.4) | 2,016 (2.5) | 776 (6.6) | 1.15× | lags FFTW 1.15× |
| 1,024 (2¹⁰) | 802 (31.9) | 791 (32.4) | 3,824 (6.7) | 3,287 (7.8) | 3,988 (6.4) | 1.01× | **≥ parity** |
| 4,096 (2¹²) | 3,586 (34.3) | 3,863 (31.8) | 11,234 (10.9) | 9,734 (12.6) | 17,542 (7.0) | 0.93× | **≥ parity** |
| 65,536 (2¹⁶) | 120,647 (21.7) | 99,770 (26.3) | 213,511 (12.3) | 242,064 (10.8) | 425,147 (6.2) | 1.21× | lags FFTW 1.21× |
| 1,048,576 (2²⁰) | 2,492,240 (21.0) | 2,200,664 (23.8) | 4,380,643 (12.0) | 5,865,645 (8.9) | 9,232,891 (5.7) | 1.13× | lags FFTW 1.13× |
| 1,000 (2³·5³) | 955 (26.1) | 944 (26.4) | 3,947 (6.3) | 3,249 (7.7) | 4,105 (6.1) | 1.01× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,040 (26.2) | 1,034 (26.3) | 4,142 (6.6) | 3,740 (7.3) | 4,729 (5.8) | 1.01× | **≥ parity** |
| 1,920 (2⁷·3·5) | 1,730 (30.3) | 1,768 (29.6) | 5,842 (9.0) | 5,243 (10.0) | 8,194 (6.4) | 0.98× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 206 (24.8) | 209 (24.5) | 0.99× | **≥ parity** |
| 1,024 (2¹⁰) | 801 (32.0) | 890 (28.8) | 0.90× | **≥ parity** |
| 4,096 (2¹²) | 3,569 (34.4) | 4,784 (25.7) | 0.75× | **≥ parity** |
| 65,536 (2¹⁶) | 136,990 (19.1) | 109,574 (23.9) | 1.25× | lags FFTW 1.25× |
| 1,048,576 (2²⁰) | 2,947,186 (17.8) | 2,475,617 (21.2) | 1.19× | lags FFTW 1.19× |
| 1,000 (2³·5³) | 948 (26.3) | 1,031 (24.2) | 0.92× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,039 (26.2) | 1,090 (25.0) | 0.95× | **≥ parity** |
| 1,920 (2⁷·3·5) | 1,717 (30.5) | 1,898 (27.6) | 0.90× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 6,469 (38.0) | 7,538 (32.6) | 5,065 (48.5) | 18,317 (13.4) | 13,129 (18.7) | 1.28× | lags FFTW 1.28× |
| 128x128 | 45,895 (25.0) | 49,296 (23.3) | 36,679 (31.3) | 75,101 (15.3) | 61,649 (18.6) | 1.25× | lags FFTW 1.25× |
| 256x256 | 239,674 (21.9) | 253,143 (20.7) | 192,112 (27.3) | 337,038 (15.6) | 199,201 (26.3) | 1.25× | lags FFTW 1.25× |
| 512x512 | 863,420 (27.3) | 899,918 (26.2) | 856,398 (27.5) | 1,573,976 (15.0) | 925,966 (25.5) | 1.01× | **≥ parity** |
| 1024x1024 | 4,836,414 (21.7) | 4,942,863 (21.2) | 3,942,141 (26.6) | 7,807,747 (13.4) | 5,226,642 (20.1) | 1.23× | lags FFTW 1.23× |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 3,673 | 2,417 | 3,864,000 |
| 1,024 | 13,090 | 8,638 | 5,999,000 |
| 4,096 | 51,748 | 32,330 | 21,502,000 |
| 65,536 | 690,234 | 462,036 | 446,768,000 |
| 1,048,576 | 9,960,707 | 6,253,551 | 1,656,987,000 |
| 1,000 | 13,094 | 8,258 | 7,841,000 |
| 1,080 | 14,703 | 9,020 | 18,469,000 |
| 1,920 | 24,543 | 15,442 | 31,439,000 |
| 1,009 | 25,558 | 9,513 | 12,452,000 |
| 1,296 | 17,495 | 10,717 | 14,958,000 |
| 10,007 | 397,329 | 89,341 | 206,823,000 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 16/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 23/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 65,536 (2¹⁶): 1.31×
- 2-D 64x64: 1.28×
- 2-D 128x128: 1.25×
- 2-D 256x256: 1.25×
- 2-D 1024x1024: 1.23×
- real 65,536 (2¹⁶): 1.21×
- real 256 (2⁸): 1.15×
- real 1,048,576 (2²⁰): 1.13×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
