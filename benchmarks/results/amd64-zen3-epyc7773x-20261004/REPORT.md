# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3), GCC Compile Farm cfarm420.
- **Toolchains**: go1.26.4 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128** (3.3.10 built from source with --enable-sse2 --enable-avx --enable-avx2 --enable-fma); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 464 (22.1) | 273 (37.6) | 6,857 (1.5) | 5,451 (1.9) | 15,709 (0.7) | 1.70× | lags FFTW 1.70× |
| 1,024 (2¹⁰) | 2,213 (23.1) | 1,509 (33.9) | 12,935 (4.0) | 9,592 (5.3) | 60,991 (0.8) | 1.47× | lags FFTW 1.47× |
| 4,096 (2¹²) | 15,877 (15.5) | 10,368 (23.7) | 43,443 (5.7) | 33,293 (7.4) | 290,762 (0.8) | 1.53× | lags FFTW 1.53× |
| 65,536 (2¹⁶) | 314,412 (16.7) | 293,420 (17.9) | 1,885,343 (2.8) | 934,184 (5.6) | 6,297,020 (0.8) | 1.07× | lags FFTW 1.07× |
| 1,048,576 (2²⁰) | 6,956,929 (15.1) | 11,882,651 (8.8) | 39,677,839 (2.6) | 13,967,766 (7.5) | 131,976,338 (0.8) | 0.59× | **≥ parity** |
| 1,000 (2³·5³) | 2,823 (17.7) | 2,145 (23.2) | 13,688 (3.6) | 10,075 (4.9) | 67,304 (0.7) | 1.32× | lags FFTW 1.32× |
| 1,080 (2³·3³·5) | 3,295 (16.5) | 2,218 (24.5) | 14,324 (3.8) | 10,821 (5.0) | 70,959 (0.8) | 1.49× | lags FFTW 1.49× |
| 1,920 (2⁷·3·5) | 5,469 (19.1) | 3,978 (26.3) | 22,071 (4.7) | 16,501 (6.3) | 129,151 (0.8) | 1.37× | lags FFTW 1.37× |
| 1,009 (prime) | 14,326 (3.5) | 20,599 (2.4) | 61,638 (0.8) | 39,132 (1.3) | 1,453,834 (0.0) | 0.70× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 4,446 (15.1) | 2,936 (22.8) | 16,341 (4.1) | 12,552 (5.3) | 86,924 (0.8) | 1.51× | lags FFTW 1.51× |
| 10,007 (prime) | 340,479 (2.0) | 207,119 (3.2) | 644,150 (1.0) | 472,621 (1.4) | 152,775,887 (0.0) | 1.64× | lags FFTW 1.64× |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 345 (14.8) | 225 (22.7) | 6,177 (0.8) | 5,411 (0.9) | 5,136 (1.0) | 1.53× | lags FFTW 1.53× |
| 1,024 (2¹⁰) | 1,341 (19.1) | 979 (26.2) | 9,513 (2.7) | 8,274 (3.1) | 24,748 (1.0) | 1.37× | lags FFTW 1.37× |
| 4,096 (2¹²) | 8,363 (14.7) | 5,321 (23.1) | 26,196 (4.7) | 22,173 (5.5) | 112,864 (1.1) | 1.57× | lags FFTW 1.57× |
| 65,536 (2¹⁶) | 176,168 (14.9) | 156,197 (16.8) | 397,069 (6.6) | 493,761 (5.3) | 2,373,714 (1.1) | 1.13× | lags FFTW 1.13× |
| 1,048,576 (2²⁰) | 3,878,571 (13.5) | 3,816,128 (13.7) | 8,287,924 (6.3) | 10,043,065 (5.2) | 46,357,757 (1.1) | 1.02× | **≥ parity** |
| 1,000 (2³·5³) | 1,704 (14.6) | 1,357 (18.4) | 10,403 (2.4) | 8,818 (2.8) | 26,568 (0.9) | 1.26× | lags FFTW 1.26× |
| 1,080 (2³·3³·5) | 1,914 (14.2) | 1,448 (18.8) | 10,985 (2.5) | 9,131 (3.0) | 27,404 (1.0) | 1.32× | lags FFTW 1.32× |
| 1,920 (2⁷·3·5) | 3,117 (16.8) | 2,325 (22.5) | 14,346 (3.6) | 12,465 (4.2) | 50,339 (1.0) | 1.34× | lags FFTW 1.34× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 354 (14.4) | 300 (17.0) | 1.18× | lags FFTW 1.18× |
| 1,024 (2¹⁰) | 1,336 (19.2) | 1,259 (20.3) | 1.06× | lags FFTW 1.06× |
| 4,096 (2¹²) | 8,407 (14.6) | 5,916 (20.8) | 1.42× | lags FFTW 1.42× |
| 65,536 (2¹⁶) | 179,493 (14.6) | 164,806 (15.9) | 1.09× | lags FFTW 1.09× |
| 1,048,576 (2²⁰) | 3,797,061 (13.8) | 4,088,409 (12.8) | 0.93× | **≥ parity** |
| 1,000 (2³·5³) | 1,733 (14.4) | 1,438 (17.3) | 1.21× | lags FFTW 1.21× |
| 1,080 (2³·3³·5) | 1,927 (14.1) | 1,590 (17.1) | 1.21× | lags FFTW 1.21× |
| 1,920 (2⁷·3·5) | 3,221 (16.3) | 2,682 (19.5) | 1.20× | lags FFTW 1.20× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 23,383 (10.5) | 40,137 (6.1) | 11,619 (21.2) | 47,984 (5.1) | 32,805 (7.5) | 2.01× | lags FFTW 2.01× |
| 128x128 | 113,049 (10.1) | 201,093 (5.7) | 61,978 (18.5) | 147,094 (7.8) | 123,967 (9.3) | 1.82× | lags FFTW 1.82× |
| 256x256 | 403,123 (13.0) | 520,817 (10.1) | 284,471 (18.4) | 597,734 (8.8) | 464,156 (11.3) | 1.42× | lags FFTW 1.42× |
| 512x512 | 883,738 (26.7) | 1,691,436 (13.9) | 1,358,744 (17.4) | 2,860,149 (8.2) | 1,927,852 (12.2) | 0.65× | **≥ parity** |
| 1024x1024 | 2,679,536 (39.1) | 5,260,705 (19.9) | 6,681,912 (15.7) | 13,377,563 (7.8) | 9,262,235 (11.3) | 0.40× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 28,319 | 12,188 | 71,855,503 |
| 1,024 | 103,532 | 47,151 | 137,039,302 |
| 4,096 | 360,677 | 181,845 | 248,360,233 |
| 65,536 | 4,442,519 | 3,292,240 | 2,139,103,862 |
| 1,048,576 | 56,859,432 | 37,456,344 | 4,278,018,144 |
| 1,000 | 91,005 | 51,394 | 151,848,119 |
| 1,080 | 106,231 | 62,995 | 364,302,336 |
| 1,920 | 164,807 | 94,393 | 576,926,163 |
| 1,009 | 124,239 | 56,845 | 141,567,234 |
| 1,296 | 126,087 | 68,364 | 277,997,986 |
| 10,007 | 1,762,927 | 534,371 | 1,219,918,384 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 5/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- 2-D 64x64: 2.01×
- 2-D 128x128: 1.82×
- complex 256 (2⁸): 1.70×
- complex 10,007 (prime): 1.64×
- real 4,096 (2¹²): 1.57×
- real 256 (2⁸): 1.53×
- complex 4,096 (2¹²): 1.53×
- complex 1,296 (2⁴·3⁴): 1.51×
- complex 1,080 (2³·3³·5): 1.49×
- complex 1,024 (2¹⁰): 1.47×
- 2-D 256x256: 1.42×
- complex 1,920 (2⁷·3·5): 1.37×
- real 1,024 (2¹⁰): 1.37×
- real 1,920 (2⁷·3·5): 1.34×
- real 1,080 (2³·3³·5): 1.32×
- complex 1,000 (2³·5³): 1.32×
- real 1,000 (2³·5³): 1.26×
- real 65,536 (2¹⁶): 1.13×
- complex 65,536 (2¹⁶): 1.07×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
