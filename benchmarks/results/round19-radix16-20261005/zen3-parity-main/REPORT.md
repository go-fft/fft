# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3), cfarm420, one pinned core.
- **Toolchains**: go1.27.1 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128** (3.3.10 built from source with --enable-sse2 --enable-avx --enable-avx2 --enable-fma); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 422 (24.3) | 273 (37.5) | 6,695 (1.5) | 5,527 (1.9) | 6,798 (1.5) | 1.54× | lags FFTW 1.54× |
| 1,024 (2¹⁰) | 2,031 (25.2) | 1,529 (33.5) | 13,081 (3.9) | 9,830 (5.2) | 28,759 (1.8) | 1.33× | lags FFTW 1.33× |
| 4,096 (2¹²) | 12,844 (19.1) | 10,983 (22.4) | 48,190 (5.1) | 34,335 (7.2) | 130,956 (1.9) | 1.17× | lags FFTW 1.17× |
| 65,536 (2¹⁶) | 292,696 (17.9) | 304,039 (17.2) | 1,948,712 (2.7) | 906,441 (5.8) | 2,841,199 (1.8) | 0.96× | **≥ parity** |
| 1,048,576 (2²⁰) | 6,189,224 (16.9) | 11,481,393 (9.1) | 23,347,903 (4.5) | 14,237,966 (7.4) | 57,878,558 (1.8) | 0.54× | **≥ parity** |
| 1,000 (2³·5³) | 2,338 (21.3) | 1,970 (25.3) | 13,464 (3.7) | 10,234 (4.9) | 30,264 (1.6) | 1.19× | lags FFTW 1.19× |
| 1,080 (2³·3³·5) | 2,762 (19.7) | 2,252 (24.2) | 14,243 (3.8) | 10,758 (5.1) | 35,423 (1.5) | 1.23× | lags FFTW 1.23× |
| 1,920 (2⁷·3·5) | 4,668 (22.4) | 3,975 (26.3) | 21,550 (4.9) | 16,503 (6.3) | 61,352 (1.7) | 1.17× | lags FFTW 1.17× |
| 1,009 (prime) | 12,857 (3.9) | 20,291 (2.5) | 60,901 (0.8) | 38,524 (1.3) | 1,150,583 (0.0) | 0.63× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,545 (18.9) | 2,934 (22.8) | 16,167 (4.1) | 12,513 (5.4) | 41,132 (1.6) | 1.21× | lags FFTW 1.21× |
| 10,007 (prime) | 259,682 (2.6) | 207,042 (3.2) | 654,633 (1.0) | 477,789 (1.4) | 113,619,555 (0.0) | 1.25× | lags FFTW 1.25× |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 316 (16.2) | 227 (22.6) | 6,225 (0.8) | 5,615 (0.9) | 2,566 (2.0) | 1.39× | lags FFTW 1.39× |
| 1,024 (2¹⁰) | 1,240 (20.6) | 980 (26.1) | 9,969 (2.6) | 8,572 (3.0) | 12,828 (2.0) | 1.27× | lags FFTW 1.27× |
| 4,096 (2¹²) | 6,774 (18.1) | 5,218 (23.5) | 24,991 (4.9) | 22,666 (5.4) | 59,179 (2.1) | 1.30× | lags FFTW 1.30× |
| 65,536 (2¹⁶) | 175,872 (14.9) | 154,814 (16.9) | 397,219 (6.6) | 494,027 (5.3) | 1,337,363 (2.0) | 1.14× | lags FFTW 1.14× |
| 1,048,576 (2²⁰) | 3,320,492 (15.8) | 3,727,502 (14.1) | 8,234,509 (6.4) | 10,175,162 (5.2) | 27,343,512 (1.9) | 0.89× | **≥ parity** |
| 1,000 (2³·5³) | 1,480 (16.8) | 1,262 (19.7) | 10,340 (2.4) | 8,865 (2.8) | 13,504 (1.8) | 1.17× | lags FFTW 1.17× |
| 1,080 (2³·3³·5) | 1,694 (16.1) | 1,345 (20.2) | 10,872 (2.5) | 9,197 (3.0) | 16,022 (1.7) | 1.26× | lags FFTW 1.26× |
| 1,920 (2⁷·3·5) | 2,821 (18.6) | 2,304 (22.7) | 14,344 (3.6) | 12,940 (4.0) | 27,953 (1.9) | 1.22× | lags FFTW 1.22× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 326 (15.7) | 286 (17.9) | 1.14× | lags FFTW 1.14× |
| 1,024 (2¹⁰) | 1,318 (19.4) | 1,186 (21.6) | 1.11× | lags FFTW 1.11× |
| 4,096 (2¹²) | 6,808 (18.0) | 5,901 (20.8) | 1.15× | lags FFTW 1.15× |
| 65,536 (2¹⁶) | 164,250 (16.0) | 167,498 (15.7) | 0.98× | **≥ parity** |
| 1,048,576 (2²⁰) | 3,386,215 (15.5) | 4,041,447 (13.0) | 0.84× | **≥ parity** |
| 1,000 (2³·5³) | 1,516 (16.4) | 1,505 (16.6) | 1.01× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,743 (15.6) | 1,564 (17.4) | 1.11× | lags FFTW 1.11× |
| 1,920 (2⁷·3·5) | 2,833 (18.5) | 2,688 (19.5) | 1.05× | lags FFTW 1.05× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 11,127 (22.1) | 14,439 (17.0) | 10,690 (23.0) | 47,191 (5.2) | 33,893 (7.3) | 1.04× | **≥ parity** |
| 128x128 | 57,246 (20.0) | 70,440 (16.3) | 62,409 (18.4) | 145,959 (7.9) | 127,035 (9.0) | 0.92× | **≥ parity** |
| 256x256 | 289,735 (18.1) | 348,718 (15.0) | 288,975 (18.1) | 628,347 (8.3) | 457,328 (11.5) | 1.00× | **≥ parity** |
| 512x512 | 1,276,938 (18.5) | 1,574,542 (15.0) | 1,331,275 (17.7) | 2,943,259 (8.0) | 1,901,774 (12.4) | 0.96× | **≥ parity** |
| 1024x1024 | 6,020,363 (17.4) | 7,343,944 (14.3) | 6,321,463 (16.6) | 13,301,898 (7.9) | 9,501,360 (11.0) | 0.95× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 72,107,883 |
| 1,024 | — | — | 130,994,850 |
| 4,096 | — | — | 257,114,497 |
| 65,536 | — | — | 2,277,988,725 |
| 1,048,576 | — | — | 4,129,259,356 |
| 1,000 | — | — | 151,243,219 |
| 1,080 | — | — | 372,655,470 |
| 1,920 | — | — | 584,588,736 |
| 1,009 | — | — | 141,562,454 |
| 1,296 | — | — | 278,270,246 |
| 10,007 | — | — | 1,249,149,038 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 9/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 256 (2⁸): 1.54×
- real 256 (2⁸): 1.39×
- complex 1,024 (2¹⁰): 1.33×
- real 4,096 (2¹²): 1.30×
- real 1,024 (2¹⁰): 1.27×
- real 1,080 (2³·3³·5): 1.26×
- complex 10,007 (prime): 1.25×
- complex 1,080 (2³·3³·5): 1.23×
- real 1,920 (2⁷·3·5): 1.22×
- complex 1,296 (2⁴·3⁴): 1.21×
- complex 1,000 (2³·5³): 1.19×
- complex 1,920 (2⁷·3·5): 1.17×
- real 1,000 (2³·5³): 1.17×
- complex 4,096 (2¹²): 1.17×
- real 65,536 (2¹⁶): 1.14×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
