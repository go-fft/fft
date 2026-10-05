# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon (Cascade Lake), 8 vCPUs, GCC Compile Farm cfarm151.
- **Toolchains**: go1.27.1 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512** (3.3.10 built from source with SSE2/AVX/AVX2/AVX-512/FMA); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 506 (20.2) | 373 (27.4) | 8,813 (1.2) | 6,566 (1.6) | 5,548 (1.8) | 1.36× | lags FFTW 1.36× |
| 1,024 (2¹⁰) | 2,765 (18.5) | 2,307 (22.2) | 16,318 (3.1) | 12,187 (4.2) | 29,198 (1.8) | 1.20× | lags FFTW 1.20× |
| 4,096 (2¹²) | 13,256 (18.5) | 13,184 (18.6) | 59,676 (4.1) | 46,152 (5.3) | 133,257 (1.8) | 1.01× | **≥ parity** |
| 65,536 (2¹⁶) | 614,506 (8.5) | 401,443 (13.1) | 2,020,579 (2.6) | 1,408,327 (3.7) | 3,780,052 (1.4) | 1.53× | lags FFTW 1.53× |
| 1,048,576 (2²⁰) | 28,575,615 (3.7) | 26,274,590 (4.0) | 52,316,517 (2.0) | 41,122,196 (2.5) | 100,772,881 (1.0) | 1.09× | lags FFTW 1.09× |
| 1,000 (2³·5³) | 3,389 (14.7) | 2,752 (18.1) | 16,880 (3.0) | 14,108 (3.5) | 30,299 (1.6) | 1.23× | lags FFTW 1.23× |
| 1,080 (2³·3³·5) | 4,141 (13.1) | 3,017 (18.0) | 20,156 (2.7) | 13,124 (4.1) | 35,269 (1.5) | 1.37× | lags FFTW 1.37× |
| 1,920 (2⁷·3·5) | 7,248 (14.4) | 4,985 (21.0) | 30,804 (3.4) | 23,407 (4.5) | 62,335 (1.7) | 1.45× | lags FFTW 1.45× |
| 1,009 (prime) | 18,738 (2.7) | 30,523 (1.6) | 82,143 (0.6) | 56,989 (0.9) | 1,447,068 (0.0) | 0.61× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 5,606 (12.0) | 3,687 (18.2) | 20,887 (3.2) | 15,877 (4.2) | 42,317 (1.6) | 1.52× | lags FFTW 1.52× |
| 10,007 (prime) | 336,791 (2.0) | 361,047 (1.8) | 960,318 (0.7) | 734,594 (0.9) | 141,361,856 (0.0) | 0.93× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 476 (10.7) | 364 (14.1) | 6,786 (0.8) | 6,938 (0.7) | 2,668 (1.9) | 1.31× | lags FFTW 1.31× |
| 1,024 (2¹⁰) | 1,753 (14.6) | 1,497 (17.1) | 13,023 (2.0) | 10,165 (2.5) | 13,664 (1.9) | 1.17× | lags FFTW 1.17× |
| 4,096 (2¹²) | 9,089 (13.5) | 7,438 (16.5) | 33,631 (3.7) | 31,403 (3.9) | 61,101 (2.0) | 1.22× | lags FFTW 1.22× |
| 65,536 (2¹⁶) | 257,281 (10.2) | 193,294 (13.6) | 607,084 (4.3) | 698,752 (3.8) | 1,449,498 (1.8) | 1.33× | lags FFTW 1.33× |
| 1,048,576 (2²⁰) | 14,507,761 (3.6) | 9,464,891 (5.5) | 33,531,421 (1.6) | 22,599,165 (2.3) | 49,884,958 (1.1) | 1.53× | lags FFTW 1.53× |
| 1,000 (2³·5³) | 2,210 (11.3) | 1,880 (13.3) | 14,024 (1.8) | 12,061 (2.1) | 14,280 (1.7) | 1.18× | lags FFTW 1.18× |
| 1,080 (2³·3³·5) | 2,563 (10.6) | 2,032 (13.4) | 13,492 (2.0) | 12,528 (2.2) | 17,076 (1.6) | 1.26× | lags FFTW 1.26× |
| 1,920 (2⁷·3·5) | 4,173 (12.5) | 3,453 (15.2) | 18,725 (2.8) | 17,297 (3.0) | 29,390 (1.8) | 1.21× | lags FFTW 1.21× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 499 (10.3) | 535 (9.6) | 0.93× | **≥ parity** |
| 1,024 (2¹⁰) | 1,809 (14.2) | 1,866 (13.7) | 0.97× | **≥ parity** |
| 4,096 (2¹²) | 9,378 (13.1) | 8,703 (14.1) | 1.08× | lags FFTW 1.08× |
| 65,536 (2¹⁶) | 261,249 (10.0) | 216,207 (12.1) | 1.21× | lags FFTW 1.21× |
| 1,048,576 (2²⁰) | 15,518,795 (3.4) | 10,319,166 (5.1) | 1.50× | lags FFTW 1.50× |
| 1,000 (2³·5³) | 2,274 (11.0) | 2,291 (10.9) | 0.99× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,558 (10.6) | 2,343 (11.6) | 1.09× | lags FFTW 1.09× |
| 1,920 (2⁷·3·5) | 4,167 (12.6) | 3,947 (13.3) | 1.06× | lags FFTW 1.06× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,987 (13.7) | 30,679 (8.0) | 14,806 (16.6) | 55,648 (4.4) | 45,646 (5.4) | 1.21× | lags FFTW 1.21× |
| 128x128 | 86,215 (13.3) | 145,025 (7.9) | 71,124 (16.1) | 207,045 (5.5) | 181,449 (6.3) | 1.21× | lags FFTW 1.21× |
| 256x256 | 486,557 (10.8) | 762,958 (6.9) | 455,352 (11.5) | 981,202 (5.3) | 793,981 (6.6) | 1.07× | lags FFTW 1.07× |
| 512x512 | 3,956,199 (6.0) | 5,900,679 (4.0) | 2,572,734 (9.2) | 7,081,572 (3.3) | 5,223,995 (4.5) | 1.54× | lags FFTW 1.54× |
| 1024x1024 | 16,226,482 (6.5) | 19,869,204 (5.3) | 22,871,676 (4.6) | 31,646,437 (3.3) | 28,330,303 (3.7) | 0.71× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 19,439 | 10,733 | 68,713,725 |
| 1,024 | 69,767 | 38,673 | 127,123,198 |
| 4,096 | 259,903 | 151,105 | 281,129,869 |
| 65,536 | 4,502,781 | 2,402,045 | 3,136,051,526 |
| 1,048,576 | 79,027,520 | 35,866,114 | 8,350,245,956 |
| 1,000 | 70,515 | 38,131 | 143,407,434 |
| 1,080 | 80,094 | 41,310 | 346,403,738 |
| 1,920 | 132,648 | 71,020 | 570,048,280 |
| 1,009 | 183,036 | 45,813 | 136,228,654 |
| 1,296 | 92,870 | 48,978 | 258,388,862 |
| 10,007 | 2,235,371 | 445,168 | 1,513,807,567 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 4/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- 2-D 512x512: 1.54×
- real 1,048,576 (2²⁰): 1.53×
- complex 65,536 (2¹⁶): 1.53×
- complex 1,296 (2⁴·3⁴): 1.52×
- complex 1,920 (2⁷·3·5): 1.45×
- complex 1,080 (2³·3³·5): 1.37×
- complex 256 (2⁸): 1.36×
- real 65,536 (2¹⁶): 1.33×
- real 256 (2⁸): 1.31×
- real 1,080 (2³·3³·5): 1.26×
- complex 1,000 (2³·5³): 1.23×
- real 4,096 (2¹²): 1.22×
- 2-D 64x64: 1.21×
- 2-D 128x128: 1.21×
- real 1,920 (2⁷·3·5): 1.21×
- complex 1,024 (2¹⁰): 1.20×
- real 1,000 (2³·5³): 1.18×
- real 1,024 (2¹⁰): 1.17×
- complex 1,048,576 (2²⁰): 1.09×
- 2-D 256x256: 1.07×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
