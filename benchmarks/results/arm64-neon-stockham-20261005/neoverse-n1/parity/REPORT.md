# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Neoverse-N1 (cfarm424), perf-arm64-neon.
- **Toolchains**: go1.27.1 linux/arm64 (cross-compiled); native **FFTW fftw-3.3.10-neon** (3.3.10 built from source with --enable-neon); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 1,445 (7.1) | 1,151 (8.9) | 9,367 (1.1) | 7,657 (1.3) | 6,049 (1.7) | 1.26× | lags FFTW 1.26× |
| 1,024 (2¹⁰) | 7,040 (7.3) | 6,312 (8.1) | 18,917 (2.7) | 14,623 (3.5) | 30,297 (1.7) | 1.12× | lags FFTW 1.12× |
| 4,096 (2¹²) | 34,087 (7.2) | 40,865 (6.0) | 59,655 (4.1) | 47,636 (5.2) | 148,739 (1.7) | 0.83× | **≥ parity** |
| 65,536 (2¹⁶) | 815,171 (6.4) | 1,223,530 (4.3) | 2,300,408 (2.3) | 1,511,052 (3.5) | 3,566,263 (1.5) | 0.67× | **≥ parity** |
| 1,048,576 (2²⁰) | 22,505,663 (4.7) | 44,372,311 (2.4) | 48,392,375 (2.2) | 31,268,249 (3.4) | 78,471,867 (1.3) | 0.51× | **≥ parity** |
| 1,000 (2³·5³) | 7,641 (6.5) | 7,156 (7.0) | 19,322 (2.6) | 15,236 (3.3) | 33,919 (1.5) | 1.07× | lags FFTW 1.07× |
| 1,080 (2³·3³·5) | 9,145 (6.0) | 7,663 (7.1) | 21,620 (2.5) | 16,824 (3.2) | 41,417 (1.3) | 1.19× | lags FFTW 1.19× |
| 1,920 (2⁷·3·5) | 15,745 (6.7) | 13,492 (7.8) | 32,691 (3.2) | 25,986 (4.0) | 72,872 (1.4) | 1.17× | lags FFTW 1.17× |
| 1,009 (prime) | 23,356 (2.2) | 48,037 (1.0) | 96,036 (0.5) | 57,098 (0.9) | 2,321,839 (0.0) | 0.49× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 11,517 (5.8) | 11,314 (5.9) | 24,889 (2.7) | 19,788 (3.4) | 48,705 (1.4) | 1.02× | **≥ parity** |
| 10,007 (prime) | 527,937 (1.3) | 591,258 (1.1) | 1,042,148 (0.6) | 784,284 (0.8) | 227,908,718 (0.0) | 0.89× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 967 (5.3) | 719 (7.1) | 9,270 (0.6) | 8,385 (0.6) | 3,136 (1.6) | 1.34× | lags FFTW 1.34× |
| 1,024 (2¹⁰) | 4,146 (6.2) | 4,199 (6.1) | 14,658 (1.7) | 12,691 (2.0) | 15,107 (1.7) | 0.99× | **≥ parity** |
| 4,096 (2¹²) | 20,350 (6.0) | 19,800 (6.2) | 37,180 (3.3) | 33,376 (3.7) | 72,423 (1.7) | 1.03× | **≥ parity** |
| 65,536 (2¹⁶) | 483,162 (5.4) | 548,397 (4.8) | 733,056 (3.6) | 751,738 (3.5) | 1,836,265 (1.4) | 0.88× | **≥ parity** |
| 1,048,576 (2²⁰) | 12,022,117 (4.4) | 20,135,567 (2.6) | 22,317,009 (2.3) | 18,391,511 (2.9) | 44,765,172 (1.2) | 0.60× | **≥ parity** |
| 1,000 (2³·5³) | 4,799 (5.2) | 3,944 (6.3) | 15,217 (1.6) | 12,751 (2.0) | 15,581 (1.6) | 1.22× | lags FFTW 1.22× |
| 1,080 (2³·3³·5) | 5,767 (4.7) | 4,190 (6.5) | 15,963 (1.7) | 14,102 (1.9) | 18,259 (1.5) | 1.38× | lags FFTW 1.38× |
| 1,920 (2⁷·3·5) | 8,904 (5.9) | 7,303 (7.2) | 21,710 (2.4) | 19,016 (2.8) | 30,450 (1.7) | 1.22× | lags FFTW 1.22× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 1,051 (4.9) | 808 (6.3) | 1.30× | lags FFTW 1.30× |
| 1,024 (2¹⁰) | 4,741 (5.4) | 4,451 (5.8) | 1.07× | lags FFTW 1.07× |
| 4,096 (2¹²) | 21,752 (5.6) | 21,017 (5.8) | 1.03× | **≥ parity** |
| 65,536 (2¹⁶) | 511,944 (5.1) | 636,258 (4.1) | 0.80× | **≥ parity** |
| 1,048,576 (2²⁰) | 13,364,791 (3.9) | 25,935,733 (2.0) | 0.52× | **≥ parity** |
| 1,000 (2³·5³) | 5,258 (4.7) | 4,245 (5.9) | 1.24× | lags FFTW 1.24× |
| 1,080 (2³·3³·5) | 6,134 (4.4) | 4,418 (6.2) | 1.39× | lags FFTW 1.39× |
| 1,920 (2⁷·3·5) | 9,948 (5.3) | 7,845 (6.7) | 1.27× | lags FFTW 1.27× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 53,789 (4.6) | 134,189 (1.8) | 30,935 (7.9) | 71,528 (3.4) | 50,308 (4.9) | 1.74× | lags FFTW 1.74× |
| 128x128 | 393,855 (2.9) | 431,000 (2.7) | 197,991 (5.8) | 251,970 (4.6) | 198,322 (5.8) | 1.99× | lags FFTW 1.99× |
| 256x256 | 845,653 (6.2) | 1,017,437 (5.2) | 1,313,881 (4.0) | 1,089,263 (4.8) | 901,686 (5.8) | 0.64× | **≥ parity** |
| 512x512 | 2,075,061 (11.4) | 2,594,231 (9.1) | 7,766,802 (3.0) | 5,953,796 (4.0) | 4,250,378 (5.6) | 0.27× | **≥ parity** |
| 1024x1024 | 5,853,728 (17.9) | 7,145,597 (14.7) | 44,428,331 (2.4) | 25,670,433 (4.1) | 19,994,544 (5.2) | 0.13× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 36,987 | 26,266 | 7,652,765 |
| 1,024 | 142,550 | 97,591 | 3,251,692 |
| 4,096 | 551,094 | 372,773 | 6,148,979 |
| 65,536 | 6,932,897 | 5,930,574 | 21,496,628 |
| 1,048,576 | 77,974,908 | 68,314,134 | 19,977,322 |
| 1,000 | 135,329 | 96,294 | 4,697,874 |
| 1,080 | 158,157 | 106,632 | 12,485,763 |
| 1,920 | 259,307 | 177,065 | 17,371,239 |
| 1,009 | 334,800 | 111,889 | 6,079,817 |
| 1,296 | 188,200 | 124,918 | 10,477,170 |
| 10,007 | 3,983,603 | 1,076,379 | 35,264,168 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 13/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 23/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 22/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- 2-D 128x128: 1.99×
- 2-D 64x64: 1.74×
- real 1,080 (2³·3³·5): 1.38×
- real 256 (2⁸): 1.34×
- complex 256 (2⁸): 1.26×
- real 1,920 (2⁷·3·5): 1.22×
- real 1,000 (2³·5³): 1.22×
- complex 1,080 (2³·3³·5): 1.19×
- complex 1,920 (2⁷·3·5): 1.17×
- complex 1,024 (2¹⁰): 1.12×
- complex 1,000 (2³·5³): 1.07×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
