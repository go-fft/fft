# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon Cascade Lake (cfarm151, KVM, AVX-512).
- **Toolchains**: go1.27.1 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512** (3.3.10 built from source by setup.sh); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 461 (22.2) | 358 (28.6) | 8,957 (1.1) | 6,486 (1.6) | 5,510 (1.9) | 1.29× | lags FFTW 1.29× |
| 1,024 (2¹⁰) | 2,544 (20.1) | 2,106 (24.3) | 16,705 (3.1) | 11,919 (4.3) | 29,401 (1.7) | 1.21× | lags FFTW 1.21× |
| 4,096 (2¹²) | 12,474 (19.7) | 13,129 (18.7) | 62,432 (3.9) | 51,891 (4.7) | 134,516 (1.8) | 0.95× | **≥ parity** |
| 65,536 (2¹⁶) | 465,151 (11.3) | 405,843 (12.9) | 1,837,948 (2.9) | 1,343,861 (3.9) | 3,065,967 (1.7) | 1.15× | lags FFTW 1.15× |
| 1,048,576 (2²⁰) | 21,317,081 (4.9) | 26,762,596 (3.9) | 50,601,541 (2.1) | 40,516,075 (2.6) | 100,763,115 (1.0) | 0.80× | **≥ parity** |
| 1,000 (2³·5³) | 3,357 (14.8) | 2,719 (18.3) | 17,148 (2.9) | 13,996 (3.6) | 29,983 (1.7) | 1.23× | lags FFTW 1.23× |
| 1,080 (2³·3³·5) | 4,246 (12.8) | 3,018 (18.0) | 20,892 (2.6) | 13,379 (4.1) | 34,896 (1.6) | 1.41× | lags FFTW 1.41× |
| 1,920 (2⁷·3·5) | 6,445 (16.2) | 5,190 (20.2) | 31,937 (3.3) | 23,761 (4.4) | 61,819 (1.7) | 1.24× | lags FFTW 1.24× |
| 1,009 (prime) | 18,786 (2.7) | 30,717 (1.6) | 80,964 (0.6) | 57,288 (0.9) | 1,431,527 (0.0) | 0.61× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 4,426 (15.1) | 3,690 (18.2) | 21,452 (3.1) | 15,983 (4.2) | 42,276 (1.6) | 1.20× | lags FFTW 1.20× |
| 10,007 (prime) | 337,737 (2.0) | 360,083 (1.8) | 1,090,637 (0.6) | 858,640 (0.8) | 139,403,434 (0.0) | 0.94× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 387 (13.2) | 359 (14.3) | 8,076 (0.6) | 7,886 (0.6) | 2,915 (1.8) | 1.08× | lags FFTW 1.08× |
| 1,024 (2¹⁰) | 1,444 (17.7) | 1,569 (16.3) | 14,735 (1.7) | 12,805 (2.0) | 13,144 (1.9) | 0.92× | **≥ parity** |
| 4,096 (2¹²) | 8,531 (14.4) | 7,796 (15.8) | 41,013 (3.0) | 35,885 (3.4) | 61,102 (2.0) | 1.09× | lags FFTW 1.09× |
| 65,536 (2¹⁶) | 215,218 (12.2) | 190,911 (13.7) | 707,918 (3.7) | 818,189 (3.2) | 1,432,123 (1.8) | 1.13× | lags FFTW 1.13× |
| 1,048,576 (2²⁰) | 9,928,282 (5.3) | 6,985,496 (7.5) | 20,352,466 (2.6) | 21,437,851 (2.4) | 41,833,280 (1.3) | 1.42× | lags FFTW 1.42× |
| 1,000 (2³·5³) | 2,036 (12.2) | 1,931 (12.9) | 15,996 (1.6) | 13,590 (1.8) | 14,624 (1.7) | 1.05× | lags FFTW 1.05× |
| 1,080 (2³·3³·5) | 2,198 (12.4) | 2,229 (12.2) | 17,223 (1.6) | 14,118 (1.9) | 16,440 (1.7) | 0.99× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,664 (14.3) | 3,480 (15.0) | 23,213 (2.3) | 19,694 (2.7) | 28,959 (1.8) | 1.05× | lags FFTW 1.05× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 406 (12.6) | 516 (9.9) | 0.79× | **≥ parity** |
| 1,024 (2¹⁰) | 1,618 (15.8) | 1,872 (13.7) | 0.86× | **≥ parity** |
| 4,096 (2¹²) | 8,984 (13.7) | 8,678 (14.2) | 1.04× | **≥ parity** |
| 65,536 (2¹⁶) | 248,172 (10.6) | 217,437 (12.1) | 1.14× | lags FFTW 1.14× |
| 1,048,576 (2²⁰) | 10,742,848 (4.9) | 9,006,281 (5.8) | 1.19× | lags FFTW 1.19× |
| 1,000 (2³·5³) | 2,106 (11.8) | 2,277 (10.9) | 0.92× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,252 (12.1) | 2,471 (11.0) | 0.91× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,799 (13.8) | 3,965 (13.2) | 0.96× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,555 (14.0) | 30,715 (8.0) | 17,083 (14.4) | 68,825 (3.6) | 51,467 (4.8) | 1.03× | **≥ parity** |
| 128x128 | 79,810 (14.4) | 137,220 (8.4) | 82,335 (13.9) | 236,657 (4.8) | 210,475 (5.4) | 0.97× | **≥ parity** |
| 256x256 | 449,041 (11.7) | 665,923 (7.9) | 571,585 (9.2) | 1,009,328 (5.2) | 864,931 (6.1) | 0.79× | **≥ parity** |
| 512x512 | 2,508,129 (9.4) | 4,663,230 (5.1) | 2,290,103 (10.3) | 5,627,385 (4.2) | 4,077,722 (5.8) | 1.10× | lags FFTW 1.10× |
| 1024x1024 | 17,033,250 (6.2) | 20,785,197 (5.0) | 13,657,412 (7.7) | 33,543,339 (3.1) | 27,962,636 (3.7) | 1.25× | lags FFTW 1.25× |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 22,813 | 10,068 | 69,238,905 |
| 1,024 | 79,403 | 36,599 | 127,508,488 |
| 4,096 | 307,843 | 138,457 | 279,539,630 |
| 65,536 | 4,145,878 | 2,174,353 | 3,109,774,124 |
| 1,048,576 | 74,614,526 | 35,405,800 | 8,227,649,469 |
| 1,000 | 64,484 | 34,873 | 144,958,613 |
| 1,080 | 72,907 | 38,510 | 349,513,482 |
| 1,920 | 119,343 | 66,350 | 565,691,305 |
| 1,009 | 185,404 | 43,607 | 137,711,942 |
| 1,296 | 83,867 | 45,672 | 260,078,624 |
| 10,007 | 2,172,361 | 418,050 | 1,510,494,467 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 9/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 1,048,576 (2²⁰): 1.42×
- complex 1,080 (2³·3³·5): 1.41×
- complex 256 (2⁸): 1.29×
- 2-D 1024x1024: 1.25×
- complex 1,920 (2⁷·3·5): 1.24×
- complex 1,000 (2³·5³): 1.23×
- complex 1,024 (2¹⁰): 1.21×
- complex 1,296 (2⁴·3⁴): 1.20×
- complex 65,536 (2¹⁶): 1.15×
- real 65,536 (2¹⁶): 1.13×
- 2-D 512x512: 1.10×
- real 4,096 (2¹²): 1.09×
- real 256 (2⁸): 1.08×
- real 1,000 (2³·5³): 1.05×
- real 1,920 (2⁷·3·5): 1.05×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
