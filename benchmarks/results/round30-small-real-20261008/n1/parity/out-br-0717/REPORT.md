# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Neoverse-N1 (cfarm424), core 40.
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
| 256 (2⁸) | 1,013 (10.1) | 1,152 (8.9) | 9,433 (1.1) | 7,699 (1.3) | 5,881 (1.7) | 0.88× | **≥ parity** |
| 1,024 (2¹⁰) | 4,915 (10.4) | 6,330 (8.1) | 19,209 (2.7) | 14,652 (3.5) | 30,620 (1.7) | 0.78× | **≥ parity** |
| 4,096 (2¹²) | 23,921 (10.3) | 41,093 (6.0) | 59,469 (4.1) | 47,958 (5.1) | 147,515 (1.7) | 0.58× | **≥ parity** |
| 65,536 (2¹⁶) | 736,687 (7.1) | 1,290,294 (4.1) | 2,193,878 (2.4) | 1,402,015 (3.7) | 3,397,675 (1.5) | 0.57× | **≥ parity** |
| 1,048,576 (2²⁰) | 20,886,418 (5.0) | 45,363,293 (2.3) | 44,270,961 (2.4) | 31,275,982 (3.4) | 75,030,319 (1.4) | 0.46× | **≥ parity** |
| 1,000 (2³·5³) | 6,097 (8.2) | 7,146 (7.0) | 20,395 (2.4) | 15,507 (3.2) | 34,096 (1.5) | 0.85× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,887 (7.9) | 7,666 (7.1) | 21,958 (2.5) | 17,014 (3.2) | 41,740 (1.3) | 0.90× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,945 (8.8) | 13,495 (7.8) | 32,645 (3.2) | 25,891 (4.0) | 70,388 (1.5) | 0.89× | **≥ parity** |
| 1,009 (prime) | 21,502 (2.3) | 48,501 (1.0) | 97,169 (0.5) | 57,619 (0.9) | 2,215,856 (0.0) | 0.44× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,603 (7.8) | 11,432 (5.9) | 24,848 (2.7) | 19,757 (3.4) | 48,503 (1.4) | 0.75× | **≥ parity** |
| 10,007 (prime) | 422,888 (1.6) | 603,152 (1.1) | 1,016,395 (0.7) | 759,897 (0.9) | 219,901,786 (0.0) | 0.70× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 751 (6.8) | 719 (7.1) | 8,562 (0.6) | 7,877 (0.6) | 3,005 (1.7) | 1.05× | **≥ parity** |
| 1,024 (2¹⁰) | 3,141 (8.2) | 4,196 (6.1) | 14,272 (1.8) | 12,654 (2.0) | 15,251 (1.7) | 0.75× | **≥ parity** |
| 4,096 (2¹²) | 14,832 (8.3) | 19,882 (6.2) | 37,204 (3.3) | 32,953 (3.7) | 69,665 (1.8) | 0.75× | **≥ parity** |
| 65,536 (2¹⁶) | 403,694 (6.5) | 567,518 (4.6) | 727,137 (3.6) | 758,131 (3.5) | 1,741,288 (1.5) | 0.71× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,565,758 (5.0) | 19,326,364 (2.7) | 19,965,424 (2.6) | 18,489,684 (2.8) | 42,729,360 (1.2) | 0.55× | **≥ parity** |
| 1,000 (2³·5³) | 3,922 (6.4) | 3,942 (6.3) | 15,468 (1.6) | 12,907 (1.9) | 14,949 (1.7) | 0.99× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,431 (6.1) | 4,215 (6.5) | 16,086 (1.7) | 14,324 (1.9) | 17,469 (1.6) | 1.05× | lags FFTW 1.05× |
| 1,920 (2⁷·3·5) | 7,100 (7.4) | 7,358 (7.1) | 22,052 (2.4) | 19,203 (2.7) | 30,696 (1.7) | 0.96× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 768 (6.7) | 821 (6.2) | 0.93× | **≥ parity** |
| 1,024 (2¹⁰) | 3,087 (8.3) | 4,439 (5.8) | 0.70× | **≥ parity** |
| 4,096 (2¹²) | 14,595 (8.4) | 20,994 (5.9) | 0.70× | **≥ parity** |
| 65,536 (2¹⁶) | 393,106 (6.7) | 668,696 (3.9) | 0.59× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,226,239 (5.1) | 23,366,376 (2.2) | 0.44× | **≥ parity** |
| 1,000 (2³·5³) | 3,869 (6.4) | 4,251 (5.9) | 0.91× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,383 (6.2) | 4,446 (6.1) | 0.99× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,063 (7.4) | 7,864 (6.7) | 0.90× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 24,334 (10.1) | 31,225 (7.9) | 31,126 (7.9) | 70,899 (3.5) | 51,360 (4.8) | 0.78× | **≥ parity** |
| 128x128 | 126,071 (9.1) | 155,604 (7.4) | 193,003 (5.9) | 258,257 (4.4) | 203,259 (5.6) | 0.65× | **≥ parity** |
| 256x256 | 682,450 (7.7) | 791,042 (6.6) | 1,242,463 (4.2) | 1,174,655 (4.5) | 908,475 (5.8) | 0.55× | **≥ parity** |
| 512x512 | 3,170,531 (7.4) | 3,908,217 (6.0) | 6,788,895 (3.5) | 5,479,374 (4.3) | 4,252,946 (5.5) | 0.47× | **≥ parity** |
| 1024x1024 | 18,310,594 (5.7) | 18,181,966 (5.8) | 42,806,147 (2.5) | 26,311,587 (4.0) | 20,151,619 (5.2) | 0.43× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 4,213,877 |
| 1,024 | — | — | 3,273,428 |
| 4,096 | — | — | 6,403,696 |
| 65,536 | — | — | 22,136,958 |
| 1,048,576 | — | — | 19,918,818 |
| 1,000 | — | — | 4,686,602 |
| 1,080 | — | — | 12,494,793 |
| 1,920 | — | — | 17,303,556 |
| 1,009 | — | — | 6,169,617 |
| 1,296 | — | — | 10,544,976 |
| 10,007 | — | — | 36,454,491 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 23/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 1,080 (2³·3³·5): 1.05×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
