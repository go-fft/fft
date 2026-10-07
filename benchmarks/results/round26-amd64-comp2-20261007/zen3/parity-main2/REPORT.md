# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3), cfarm420, one pinned core.
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
| 256 (2⁸) | 358 (28.6) | 293 (34.9) | 6,676 (1.5) | 5,347 (1.9) | 5,469 (1.9) | 1.22× | lags FFTW 1.22× |
| 1,024 (2¹⁰) | 1,720 (29.8) | 1,505 (34.0) | 13,063 (3.9) | 9,814 (5.2) | 28,532 (1.8) | 1.14× | lags FFTW 1.14× |
| 4,096 (2¹²) | 9,067 (27.1) | 10,601 (23.2) | 45,241 (5.4) | 33,052 (7.4) | 131,314 (1.9) | 0.86× | **≥ parity** |
| 65,536 (2¹⁶) | 247,042 (21.2) | 328,789 (15.9) | 1,927,625 (2.7) | 898,462 (5.8) | 2,847,217 (1.8) | 0.75× | **≥ parity** |
| 1,048,576 (2²⁰) | 6,137,359 (17.1) | 11,435,646 (9.2) | 23,872,167 (4.4) | 13,766,551 (7.6) | 57,966,463 (1.8) | 0.54× | **≥ parity** |
| 1,000 (2³·5³) | 2,356 (21.1) | 2,000 (24.9) | 13,726 (3.6) | 10,054 (5.0) | 30,033 (1.7) | 1.18× | lags FFTW 1.18× |
| 1,080 (2³·3³·5) | 2,786 (19.5) | 2,476 (22.0) | 14,472 (3.8) | 10,956 (5.0) | 35,037 (1.6) | 1.13× | lags FFTW 1.13× |
| 1,920 (2⁷·3·5) | 4,433 (23.6) | 4,008 (26.1) | 22,064 (4.7) | 16,775 (6.2) | 60,539 (1.7) | 1.11× | lags FFTW 1.11× |
| 1,009 (prime) | 12,880 (3.9) | 20,066 (2.5) | 61,391 (0.8) | 39,049 (1.3) | 1,155,558 (0.0) | 0.64× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,544 (18.9) | 3,173 (21.1) | 17,087 (3.9) | 12,666 (5.3) | 42,077 (1.6) | 1.12× | lags FFTW 1.12× |
| 10,007 (prime) | 213,561 (3.1) | 213,948 (3.1) | 651,525 (1.0) | 470,602 (1.4) | 119,099,377 (0.0) | 1.00× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 308 (16.6) | 226 (22.7) | 6,165 (0.8) | 5,378 (1.0) | 2,618 (2.0) | 1.36× | lags FFTW 1.36× |
| 1,024 (2¹⁰) | 1,121 (22.8) | 986 (26.0) | 9,596 (2.7) | 8,650 (3.0) | 12,935 (2.0) | 1.14× | lags FFTW 1.14× |
| 4,096 (2¹²) | 5,909 (20.8) | 5,358 (22.9) | 25,486 (4.8) | 22,839 (5.4) | 60,093 (2.0) | 1.10× | lags FFTW 1.10× |
| 65,536 (2¹⁶) | 139,066 (18.9) | 151,869 (17.3) | 408,422 (6.4) | 507,475 (5.2) | 1,316,145 (2.0) | 0.92× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,914,750 (18.0) | 3,775,123 (13.9) | 8,969,918 (5.8) | 10,580,653 (5.0) | 26,983,965 (1.9) | 0.77× | **≥ parity** |
| 1,000 (2³·5³) | 1,491 (16.7) | 1,278 (19.5) | 10,413 (2.4) | 9,133 (2.7) | 13,575 (1.8) | 1.17× | lags FFTW 1.17× |
| 1,080 (2³·3³·5) | 1,733 (15.7) | 1,384 (19.7) | 12,703 (2.1) | 9,063 (3.0) | 15,980 (1.7) | 1.25× | lags FFTW 1.25× |
| 1,920 (2⁷·3·5) | 2,893 (18.1) | 2,361 (22.2) | 14,635 (3.6) | 12,846 (4.1) | 28,132 (1.9) | 1.23× | lags FFTW 1.23× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 331 (15.5) | 312 (16.4) | 1.06× | lags FFTW 1.06× |
| 1,024 (2¹⁰) | 1,171 (21.9) | 1,229 (20.8) | 0.95× | **≥ parity** |
| 4,096 (2¹²) | 6,172 (19.9) | 5,752 (21.4) | 1.07× | lags FFTW 1.07× |
| 65,536 (2¹⁶) | 139,099 (18.8) | 165,288 (15.9) | 0.84× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,924,117 (17.9) | 4,129,244 (12.7) | 0.71× | **≥ parity** |
| 1,000 (2³·5³) | 1,507 (16.5) | 1,476 (16.9) | 1.02× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,732 (15.7) | 1,597 (17.0) | 1.08× | lags FFTW 1.08× |
| 1,920 (2⁷·3·5) | 2,979 (17.6) | 2,672 (19.6) | 1.11× | lags FFTW 1.11× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 10,960 (22.4) | 14,694 (16.7) | 11,197 (21.9) | 48,087 (5.1) | 33,252 (7.4) | 0.98× | **≥ parity** |
| 128x128 | 53,689 (21.4) | 67,809 (16.9) | 62,474 (18.4) | 150,065 (7.6) | 127,541 (9.0) | 0.86× | **≥ parity** |
| 256x256 | 275,172 (19.1) | 336,247 (15.6) | 283,721 (18.5) | 645,548 (8.1) | 456,441 (11.5) | 0.97× | **≥ parity** |
| 512x512 | 1,217,288 (19.4) | 1,453,079 (16.2) | 1,312,134 (18.0) | 2,917,503 (8.1) | 1,924,191 (12.3) | 0.93× | **≥ parity** |
| 1024x1024 | 6,084,468 (17.2) | 7,616,026 (13.8) | 6,493,997 (16.1) | 13,437,344 (7.8) | 9,342,400 (11.2) | 0.94× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 11,470 | 7,293 | 72,418,623 |
| 1,024 | 40,688 | 27,235 | 133,864,181 |
| 4,096 | 163,517 | 109,047 | 246,335,572 |
| 65,536 | 2,479,839 | 1,595,077 | 2,310,612,110 |
| 1,048,576 | 44,486,003 | 26,723,581 | 4,044,017,528 |
| 1,000 | 35,127 | 27,607 | 152,095,210 |
| 1,080 | 38,205 | 29,594 | 365,383,086 |
| 1,920 | 68,497 | 52,045 | 570,052,459 |
| 1,009 | 75,121 | 31,289 | 141,548,525 |
| 1,296 | 45,056 | 36,839 | 269,128,472 |
| 10,007 | 994,907 | 309,896 | 1,319,984,650 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 12/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.36×
- real 1,080 (2³·3³·5): 1.25×
- real 1,920 (2⁷·3·5): 1.23×
- complex 256 (2⁸): 1.22×
- complex 1,000 (2³·5³): 1.18×
- real 1,000 (2³·5³): 1.17×
- complex 1,024 (2¹⁰): 1.14×
- real 1,024 (2¹⁰): 1.14×
- complex 1,080 (2³·3³·5): 1.13×
- complex 1,296 (2⁴·3⁴): 1.12×
- complex 1,920 (2⁷·3·5): 1.11×
- real 4,096 (2¹²): 1.10×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
