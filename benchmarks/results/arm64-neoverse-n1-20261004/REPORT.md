# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: ARM Neoverse-N1, 64 cores, GCC Compile Farm cfarm424.
- **Toolchains**: go1.26.4 linux/arm64 (cross-compiled); native **FFTW fftw-3.3.10-neon** (3.3.10 built from source with --enable-neon); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 1,857 (5.5) | 1,162 (8.8) | 9,774 (1.0) | 7,777 (1.3) | 10,271 (1.0) | 1.60× | lags FFTW 1.60× |
| 1,024 (2¹⁰) | 9,069 (5.6) | 6,332 (8.1) | 19,090 (2.7) | 14,618 (3.5) | 55,385 (0.9) | 1.43× | lags FFTW 1.43× |
| 4,096 (2¹²) | 44,177 (5.6) | 41,306 (6.0) | 59,804 (4.1) | 47,622 (5.2) | 279,379 (0.9) | 1.07× | lags FFTW 1.07× |
| 65,536 (2¹⁶) | 1,127,891 (4.6) | 1,287,248 (4.1) | 2,128,874 (2.5) | 1,407,683 (3.7) | 6,489,827 (0.8) | 0.88× | **≥ parity** |
| 1,048,576 (2²⁰) | 27,509,147 (3.8) | 44,238,458 (2.4) | 43,438,677 (2.4) | 32,363,357 (3.2) | 147,310,022 (0.7) | 0.62× | **≥ parity** |
| 1,000 (2³·5³) | 11,161 (4.5) | 7,180 (6.9) | 19,320 (2.6) | 15,090 (3.3) | 55,330 (0.9) | 1.55× | lags FFTW 1.55× |
| 1,080 (2³·3³·5) | 12,636 (4.3) | 7,711 (7.1) | 21,968 (2.5) | 16,994 (3.2) | 66,788 (0.8) | 1.64× | lags FFTW 1.64× |
| 1,920 (2⁷·3·5) | 21,828 (4.8) | 13,540 (7.7) | 33,630 (3.1) | 26,465 (4.0) | 118,866 (0.9) | 1.61× | lags FFTW 1.61× |
| 1,009 (prime) | 27,438 (1.8) | 48,539 (1.0) | 97,681 (0.5) | 58,156 (0.9) | 2,343,464 (0.0) | 0.57× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 15,801 (4.2) | 11,395 (5.9) | 25,265 (2.7) | 19,986 (3.4) | 78,926 (0.8) | 1.39× | lags FFTW 1.39× |
| 10,007 (prime) | 821,129 (0.8) | 587,476 (1.1) | 1,011,481 (0.7) | 753,379 (0.9) | 227,781,164 (0.0) | 1.40× | lags FFTW 1.40× |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 1,190 (4.3) | 722 (7.1) | 8,831 (0.6) | 7,893 (0.6) | 6,311 (0.8) | 1.65× | lags FFTW 1.65× |
| 1,024 (2¹⁰) | 5,175 (4.9) | 4,273 (6.0) | 14,366 (1.8) | 12,370 (2.1) | 30,327 (0.8) | 1.21× | lags FFTW 1.21× |
| 4,096 (2¹²) | 25,967 (4.7) | 19,924 (6.2) | 37,447 (3.3) | 32,891 (3.7) | 133,124 (0.9) | 1.30× | lags FFTW 1.30× |
| 65,536 (2¹⁶) | 603,386 (4.3) | 543,889 (4.8) | 728,365 (3.6) | 765,077 (3.4) | 3,228,027 (0.8) | 1.11× | lags FFTW 1.11× |
| 1,048,576 (2²⁰) | 14,992,257 (3.5) | 19,326,860 (2.7) | 21,712,886 (2.4) | 19,274,562 (2.7) | 76,612,707 (0.7) | 0.78× | **≥ parity** |
| 1,000 (2³·5³) | 6,110 (4.1) | 3,984 (6.3) | 15,768 (1.6) | 12,659 (2.0) | 28,370 (0.9) | 1.53× | lags FFTW 1.53× |
| 1,080 (2³·3³·5) | 6,940 (3.9) | 4,213 (6.5) | 16,109 (1.7) | 14,140 (1.9) | 31,663 (0.9) | 1.65× | lags FFTW 1.65× |
| 1,920 (2⁷·3·5) | 12,657 (4.1) | 7,363 (7.1) | 22,335 (2.3) | 19,011 (2.8) | 57,776 (0.9) | 1.72× | lags FFTW 1.72× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 1,302 (3.9) | 810 (6.3) | 1.61× | lags FFTW 1.61× |
| 1,024 (2¹⁰) | 5,611 (4.6) | 4,464 (5.7) | 1.26× | lags FFTW 1.26× |
| 4,096 (2¹²) | 27,352 (4.5) | 21,422 (5.7) | 1.28× | lags FFTW 1.28× |
| 65,536 (2¹⁶) | 638,014 (4.1) | 673,284 (3.9) | 0.95× | **≥ parity** |
| 1,048,576 (2²⁰) | 14,536,141 (3.6) | 24,316,430 (2.2) | 0.60× | **≥ parity** |
| 1,000 (2³·5³) | 6,493 (3.8) | 4,240 (5.9) | 1.53× | lags FFTW 1.53× |
| 1,080 (2³·3³·5) | 7,630 (3.6) | 4,420 (6.2) | 1.73× | lags FFTW 1.73× |
| 1,920 (2⁷·3·5) | 13,390 (3.9) | 7,835 (6.7) | 1.71× | lags FFTW 1.71× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 62,549 (3.9) | 158,856 (1.5) | 30,982 (7.9) | 71,510 (3.4) | 50,740 (4.8) | 2.02× | lags FFTW 2.02× |
| 128x128 | 456,082 (2.5) | 548,814 (2.1) | 193,261 (5.9) | 259,818 (4.4) | 201,830 (5.7) | 2.36× | lags FFTW 2.36× |
| 256x256 | 928,023 (5.6) | 1,143,472 (4.6) | 1,264,972 (4.1) | 1,134,936 (4.6) | 903,724 (5.8) | 0.73× | **≥ parity** |
| 512x512 | 2,299,652 (10.3) | 3,033,616 (7.8) | 7,068,074 (3.3) | 5,745,346 (4.1) | 4,106,600 (5.7) | 0.33× | **≥ parity** |
| 1024x1024 | 6,155,016 (17.0) | 8,018,557 (13.1) | 44,695,444 (2.3) | 26,422,459 (4.0) | 20,231,095 (5.2) | 0.14× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 31,134 | 28,363 | 7,833,349 |
| 1,024 | 127,974 | 106,352 | 3,294,285 |
| 4,096 | 438,889 | 404,404 | 6,640,892 |
| 65,536 | 6,737,478 | 5,902,948 | 21,863,544 |
| 1,048,576 | 77,485,622 | 63,892,660 | 20,441,845 |
| 1,000 | 110,234 | 105,150 | 4,689,146 |
| 1,080 | 120,842 | 114,958 | 12,473,093 |
| 1,920 | 206,152 | 197,479 | 17,371,722 |
| 1,009 | 348,794 | 122,710 | 6,216,206 |
| 1,296 | 150,342 | 138,017 | 10,583,427 |
| 10,007 | 4,927,912 | 1,171,748 | 36,387,067 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 7/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 23/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 21/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- 2-D 128x128: 2.36×
- 2-D 64x64: 2.02×
- real 1,920 (2⁷·3·5): 1.72×
- real 256 (2⁸): 1.65×
- real 1,080 (2³·3³·5): 1.65×
- complex 1,080 (2³·3³·5): 1.64×
- complex 1,920 (2⁷·3·5): 1.61×
- complex 256 (2⁸): 1.60×
- complex 1,000 (2³·5³): 1.55×
- real 1,000 (2³·5³): 1.53×
- complex 1,024 (2¹⁰): 1.43×
- complex 10,007 (prime): 1.40×
- complex 1,296 (2⁴·3⁴): 1.39×
- real 4,096 (2¹²): 1.30×
- real 1,024 (2¹⁰): 1.21×
- real 65,536 (2¹⁶): 1.11×
- complex 4,096 (2¹²): 1.07×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
