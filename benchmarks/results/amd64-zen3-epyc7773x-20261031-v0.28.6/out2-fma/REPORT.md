# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3, cfarm420), one pinned core.
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
| 256 (2⁸) | 293 (35.0) | 304 (33.7) | 6,776 (1.5) | 5,248 (2.0) | 5,402 (1.9) | 0.96× | **≥ parity** |
| 1,024 (2¹⁰) | 1,717 (29.8) | 1,528 (33.5) | 13,120 (3.9) | 9,579 (5.3) | 27,837 (1.8) | 1.12× | lags FFTW 1.12× |
| 4,096 (2¹²) | 9,085 (27.1) | 10,362 (23.7) | 45,938 (5.3) | 33,187 (7.4) | 133,808 (1.8) | 0.88× | **≥ parity** |
| 65,536 (2¹⁶) | 240,977 (21.8) | 305,853 (17.1) | 2,030,602 (2.6) | 936,630 (5.6) | 2,833,516 (1.9) | 0.79× | **≥ parity** |
| 1,048,576 (2²⁰) | 5,547,345 (18.9) | 11,457,043 (9.2) | 22,554,937 (4.6) | 13,720,462 (7.6) | 57,897,591 (1.8) | 0.48× | **≥ parity** |
| 1,000 (2³·5³) | 2,039 (24.4) | 2,101 (23.7) | 13,724 (3.6) | 9,899 (5.0) | 29,926 (1.7) | 0.97× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,592 (21.0) | 2,312 (23.5) | 14,273 (3.8) | 10,578 (5.1) | 35,664 (1.5) | 1.12× | lags FFTW 1.12× |
| 1,920 (2⁷·3·5) | 4,007 (26.1) | 3,983 (26.3) | 21,972 (4.8) | 16,076 (6.5) | 61,110 (1.7) | 1.01× | **≥ parity** |
| 1,009 (prime) | 13,075 (3.9) | 20,931 (2.4) | 59,954 (0.8) | 38,263 (1.3) | 1,155,752 (0.0) | 0.62× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 2,787 (24.0) | 3,030 (22.1) | 16,226 (4.1) | 12,192 (5.5) | 43,280 (1.5) | 0.92× | **≥ parity** |
| 10,007 (prime) | 180,722 (3.7) | 216,008 (3.1) | 648,768 (1.0) | 473,390 (1.4) | 117,830,974 (0.0) | 0.84× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 234 (21.8) | 224 (22.8) | 6,332 (0.8) | 5,280 (1.0) | 2,550 (2.0) | 1.04× | **≥ parity** |
| 1,024 (2¹⁰) | 943 (27.2) | 994 (25.8) | 9,640 (2.7) | 8,173 (3.1) | 12,584 (2.0) | 0.95× | **≥ parity** |
| 4,096 (2¹²) | 5,477 (22.4) | 5,480 (22.4) | 25,700 (4.8) | 21,924 (5.6) | 59,346 (2.1) | 1.00× | **≥ parity** |
| 65,536 (2¹⁶) | 128,626 (20.4) | 152,516 (17.2) | 399,363 (6.6) | 495,376 (5.3) | 1,301,686 (2.0) | 0.84× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,690,915 (19.5) | 3,661,497 (14.3) | 8,491,845 (6.2) | 10,099,292 (5.2) | 26,469,442 (2.0) | 0.73× | **≥ parity** |
| 1,000 (2³·5³) | 1,132 (22.0) | 1,240 (20.1) | 10,514 (2.4) | 8,673 (2.9) | 13,470 (1.8) | 0.91× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,339 (20.3) | 1,337 (20.3) | 10,959 (2.5) | 8,887 (3.1) | 15,766 (1.7) | 1.00× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,361 (22.2) | 2,301 (22.8) | 14,337 (3.7) | 12,129 (4.3) | 27,816 (1.9) | 1.03× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 248 (20.7) | 294 (17.4) | 0.84× | **≥ parity** |
| 1,024 (2¹⁰) | 989 (25.9) | 1,143 (22.4) | 0.86× | **≥ parity** |
| 4,096 (2¹²) | 5,469 (22.5) | 5,749 (21.4) | 0.95× | **≥ parity** |
| 65,536 (2¹⁶) | 131,710 (19.9) | 168,244 (15.6) | 0.78× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,870,454 (18.3) | 4,128,648 (12.7) | 0.70× | **≥ parity** |
| 1,000 (2³·5³) | 1,198 (20.8) | 1,432 (17.4) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,383 (19.7) | 1,523 (17.9) | 0.91× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,491 (21.0) | 2,706 (19.3) | 0.92× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 9,220 (26.7) | 12,427 (19.8) | 10,590 (23.2) | 47,262 (5.2) | 33,208 (7.4) | 0.87× | **≥ parity** |
| 128x128 | 50,600 (22.7) | 59,662 (19.2) | 60,044 (19.1) | 146,646 (7.8) | 127,042 (9.0) | 0.84× | **≥ parity** |
| 256x256 | 261,647 (20.0) | 307,758 (17.0) | 290,107 (18.1) | 590,248 (8.9) | 455,106 (11.5) | 0.90× | **≥ parity** |
| 512x512 | 1,194,663 (19.7) | 1,360,452 (17.3) | 1,307,800 (18.0) | 2,913,711 (8.1) | 1,874,154 (12.6) | 0.91× | **≥ parity** |
| 1024x1024 | 5,683,382 (18.4) | 6,681,022 (15.7) | 6,751,928 (15.5) | 13,028,262 (8.0) | 9,212,363 (11.4) | 0.84× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 11,583 | 7,204 | 71,273,802 |
| 1,024 | 40,626 | 27,195 | 133,031,341 |
| 4,096 | 161,410 | 104,048 | 240,909,109 |
| 65,536 | 2,370,642 | 1,574,480 | 2,213,060,666 |
| 1,048,576 | 43,538,950 | 31,141,565 | 4,086,393,637 |
| 1,000 | 41,254 | 26,619 | 153,465,929 |
| 1,080 | 44,536 | 28,505 | 371,146,679 |
| 1,920 | 77,894 | 49,865 | 585,191,596 |
| 1,009 | 74,531 | 31,419 | 145,696,086 |
| 1,296 | 54,666 | 34,697 | 274,497,245 |
| 10,007 | 988,368 | 295,904 | 1,262,389,445 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 22/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 1,024 (2¹⁰): 1.12×
- complex 1,080 (2³·3³·5): 1.12×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
