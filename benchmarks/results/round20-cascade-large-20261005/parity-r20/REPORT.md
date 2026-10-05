# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon (Cascade Lake), 8 vCPUs, GCC Compile Farm cfarm151.
- **Toolchains**: go1.27.1 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512** (Homebrew arm64 bottle, NEON, linked from C); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 502 (20.4) | 370 (27.7) | 8,999 (1.1) | 6,365 (1.6) | 5,545 (1.8) | 1.36× | lags FFTW 1.36× |
| 1,024 (2¹⁰) | 2,744 (18.7) | 2,108 (24.3) | 16,298 (3.1) | 11,713 (4.4) | 29,004 (1.8) | 1.30× | lags FFTW 1.30× |
| 4,096 (2¹²) | 13,120 (18.7) | 12,056 (20.4) | 59,717 (4.1) | 44,103 (5.6) | 133,483 (1.8) | 1.09× | lags FFTW 1.09× |
| 65,536 (2¹⁶) | 464,999 (11.3) | 391,792 (13.4) | 1,911,638 (2.7) | 1,383,767 (3.8) | 3,717,712 (1.4) | 1.19× | lags FFTW 1.19× |
| 1,048,576 (2²⁰) | 21,149,064 (5.0) | 25,444,376 (4.1) | 51,170,709 (2.0) | 40,735,986 (2.6) | 98,925,235 (1.1) | 0.83× | **≥ parity** |
| 1,000 (2³·5³) | 3,399 (14.7) | 2,731 (18.2) | 17,116 (2.9) | 13,801 (3.6) | 30,485 (1.6) | 1.24× | lags FFTW 1.24× |
| 1,080 (2³·3³·5) | 4,165 (13.1) | 3,010 (18.1) | 20,089 (2.7) | 13,011 (4.2) | 35,610 (1.5) | 1.38× | lags FFTW 1.38× |
| 1,920 (2⁷·3·5) | 7,200 (14.5) | 4,852 (21.6) | 31,234 (3.4) | 21,801 (4.8) | 62,312 (1.7) | 1.48× | lags FFTW 1.48× |
| 1,009 (prime) | 18,756 (2.7) | 30,505 (1.6) | 78,417 (0.6) | 54,563 (0.9) | 1,452,325 (0.0) | 0.61× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 5,485 (12.2) | 3,646 (18.4) | 21,356 (3.1) | 15,273 (4.4) | 42,519 (1.6) | 1.50× | lags FFTW 1.50× |
| 10,007 (prime) | 334,347 (2.0) | 365,867 (1.8) | 854,413 (0.8) | 728,122 (0.9) | 142,403,188 (0.0) | 0.91× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 470 (10.9) | 368 (13.9) | 6,557 (0.8) | 7,089 (0.7) | 2,647 (1.9) | 1.28× | lags FFTW 1.28× |
| 1,024 (2¹⁰) | 1,764 (14.5) | 1,516 (16.9) | 13,077 (2.0) | 11,609 (2.2) | 13,981 (1.8) | 1.16× | lags FFTW 1.16× |
| 4,096 (2¹²) | 9,102 (13.5) | 7,402 (16.6) | 32,673 (3.8) | 31,361 (3.9) | 61,242 (2.0) | 1.23× | lags FFTW 1.23× |
| 65,536 (2¹⁶) | 244,108 (10.7) | 182,520 (14.4) | 596,393 (4.4) | 694,655 (3.8) | 1,418,149 (1.8) | 1.34× | lags FFTW 1.34× |
| 1,048,576 (2²⁰) | 9,752,189 (5.4) | 10,551,617 (5.0) | 25,145,379 (2.1) | 21,514,993 (2.4) | 46,231,383 (1.1) | 0.92× | **≥ parity** |
| 1,000 (2³·5³) | 2,218 (11.2) | 1,854 (13.4) | 14,257 (1.7) | 12,018 (2.1) | 14,170 (1.8) | 1.20× | lags FFTW 1.20× |
| 1,080 (2³·3³·5) | 2,559 (10.6) | 2,185 (12.5) | 13,396 (2.0) | 12,466 (2.2) | 16,437 (1.7) | 1.17× | lags FFTW 1.17× |
| 1,920 (2⁷·3·5) | 4,143 (12.6) | 3,561 (14.7) | 18,187 (2.9) | 17,256 (3.0) | 28,883 (1.8) | 1.16× | lags FFTW 1.16× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 500 (10.2) | 531 (9.6) | 0.94× | **≥ parity** |
| 1,024 (2¹⁰) | 1,772 (14.4) | 1,834 (14.0) | 0.97× | **≥ parity** |
| 4,096 (2¹²) | 9,349 (13.1) | 8,468 (14.5) | 1.10× | lags FFTW 1.10× |
| 65,536 (2¹⁶) | 252,790 (10.4) | 212,873 (12.3) | 1.19× | lags FFTW 1.19× |
| 1,048,576 (2²⁰) | 11,081,301 (4.7) | 9,849,658 (5.3) | 1.13× | lags FFTW 1.13× |
| 1,000 (2³·5³) | 2,258 (11.0) | 2,257 (11.0) | 1.00× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,566 (10.6) | 2,504 (10.9) | 1.02× | **≥ parity** |
| 1,920 (2⁷·3·5) | 4,169 (12.6) | 3,862 (13.6) | 1.08× | lags FFTW 1.08× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,005 (14.5) | 30,095 (8.2) | 16,673 (14.7) | 54,138 (4.5) | 44,740 (5.5) | 1.02× | **≥ parity** |
| 128x128 | 86,937 (13.2) | 141,274 (8.1) | 71,963 (15.9) | 211,202 (5.4) | 180,545 (6.4) | 1.21× | lags FFTW 1.21× |
| 256x256 | 487,789 (10.7) | 678,082 (7.7) | 475,374 (11.0) | 929,074 (5.6) | 757,952 (6.9) | 1.03× | **≥ parity** |
| 512x512 | 2,528,027 (9.3) | 4,498,780 (5.2) | 2,605,001 (9.1) | 6,463,724 (3.7) | 4,027,749 (5.9) | 0.97× | **≥ parity** |
| 1024x1024 | 15,730,661 (6.7) | 19,004,510 (5.5) | 23,312,782 (4.5) | 30,619,575 (3.4) | 27,084,908 (3.9) | 0.67× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 68,968,372 |
| 1,024 | — | — | 126,281,150 |
| 4,096 | — | — | 277,047,148 |
| 65,536 | — | — | 3,022,788,215 |
| 1,048,576 | — | — | 7,820,280,109 |
| 1,000 | — | — | 142,835,503 |
| 1,080 | — | — | 343,141,084 |
| 1,920 | — | — | 567,451,494 |
| 1,009 | — | — | 135,266,800 |
| 1,296 | — | — | 257,672,142 |
| 10,007 | — | — | 1,502,033,882 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 8/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 1,296 (2⁴·3⁴): 1.50×
- complex 1,920 (2⁷·3·5): 1.48×
- complex 1,080 (2³·3³·5): 1.38×
- complex 256 (2⁸): 1.36×
- real 65,536 (2¹⁶): 1.34×
- complex 1,024 (2¹⁰): 1.30×
- real 256 (2⁸): 1.28×
- complex 1,000 (2³·5³): 1.24×
- real 4,096 (2¹²): 1.23×
- 2-D 128x128: 1.21×
- real 1,000 (2³·5³): 1.20×
- complex 65,536 (2¹⁶): 1.19×
- real 1,080 (2³·3³·5): 1.17×
- real 1,024 (2¹⁰): 1.16×
- real 1,920 (2⁷·3·5): 1.16×
- complex 4,096 (2¹²): 1.09×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
