# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon (Cascade Lake), cfarm151, 8 vCPUs, pinned to one core.
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
| 256 (2⁸) | 489 (20.9) | 373 (27.5) | 8,704 (1.2) | 6,425 (1.6) | 5,509 (1.9) | 1.31× | lags FFTW 1.31× |
| 1,024 (2¹⁰) | 2,729 (18.8) | 2,091 (24.5) | 16,217 (3.2) | 11,958 (4.3) | 28,336 (1.8) | 1.31× | lags FFTW 1.31× |
| 4,096 (2¹²) | 13,109 (18.7) | 13,202 (18.6) | 58,109 (4.2) | 46,921 (5.2) | 131,057 (1.9) | 0.99× | **≥ parity** |
| 65,536 (2¹⁶) | 467,823 (11.2) | 417,330 (12.6) | 2,014,480 (2.6) | 1,401,208 (3.7) | 3,420,358 (1.5) | 1.12× | lags FFTW 1.12× |
| 1,048,576 (2²⁰) | 22,919,015 (4.6) | 27,575,506 (3.8) | 52,338,134 (2.0) | 43,105,189 (2.4) | 99,651,242 (1.1) | 0.83× | **≥ parity** |
| 1,000 (2³·5³) | 3,367 (14.8) | 2,738 (18.2) | 17,049 (2.9) | 14,145 (3.5) | 30,000 (1.7) | 1.23× | lags FFTW 1.23× |
| 1,080 (2³·3³·5) | 4,124 (13.2) | 2,948 (18.5) | 20,712 (2.6) | 13,597 (4.0) | 34,944 (1.6) | 1.40× | lags FFTW 1.40× |
| 1,920 (2⁷·3·5) | 7,241 (14.5) | 4,898 (21.4) | 31,340 (3.3) | 23,437 (4.5) | 61,869 (1.7) | 1.48× | lags FFTW 1.48× |
| 1,009 (prime) | 18,787 (2.7) | 30,420 (1.7) | 80,715 (0.6) | 56,444 (0.9) | 1,433,060 (0.0) | 0.62× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 5,531 (12.1) | 3,642 (18.4) | 20,792 (3.2) | 15,767 (4.2) | 41,332 (1.6) | 1.52× | lags FFTW 1.52× |
| 10,007 (prime) | 329,166 (2.0) | 363,634 (1.8) | 937,993 (0.7) | 729,300 (0.9) | 139,550,561 (0.0) | 0.91× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 472 (10.8) | 363 (14.1) | 6,495 (0.8) | 7,099 (0.7) | 2,728 (1.9) | 1.30× | lags FFTW 1.30× |
| 1,024 (2¹⁰) | 1,762 (14.5) | 1,495 (17.1) | 12,555 (2.0) | 10,718 (2.4) | 13,426 (1.9) | 1.18× | lags FFTW 1.18× |
| 4,096 (2¹²) | 9,187 (13.4) | 7,876 (15.6) | 32,588 (3.8) | 31,267 (3.9) | 61,217 (2.0) | 1.17× | lags FFTW 1.17× |
| 65,536 (2¹⁶) | 248,460 (10.6) | 199,532 (13.1) | 596,264 (4.4) | 689,082 (3.8) | 1,433,159 (1.8) | 1.25× | lags FFTW 1.25× |
| 1,048,576 (2²⁰) | 12,788,699 (4.1) | 9,472,484 (5.5) | 32,667,645 (1.6) | 22,275,789 (2.4) | 50,251,542 (1.0) | 1.35× | lags FFTW 1.35× |
| 1,000 (2³·5³) | 2,197 (11.3) | 1,872 (13.3) | 14,164 (1.8) | 12,359 (2.0) | 14,359 (1.7) | 1.17× | lags FFTW 1.17× |
| 1,080 (2³·3³·5) | 2,534 (10.7) | 2,041 (13.3) | 13,398 (2.0) | 12,757 (2.1) | 16,473 (1.7) | 1.24× | lags FFTW 1.24× |
| 1,920 (2⁷·3·5) | 4,143 (12.6) | 3,439 (15.2) | 18,217 (2.9) | 17,373 (3.0) | 29,372 (1.8) | 1.20× | lags FFTW 1.20× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 496 (10.3) | 500 (10.2) | 0.99× | **≥ parity** |
| 1,024 (2¹⁰) | 1,758 (14.6) | 1,808 (14.2) | 0.97× | **≥ parity** |
| 4,096 (2¹²) | 9,200 (13.4) | 8,412 (14.6) | 1.09× | lags FFTW 1.09× |
| 65,536 (2¹⁶) | 260,127 (10.1) | 213,233 (12.3) | 1.22× | lags FFTW 1.22× |
| 1,048,576 (2²⁰) | 13,991,138 (3.7) | 11,104,296 (4.7) | 1.26× | lags FFTW 1.26× |
| 1,000 (2³·5³) | 2,234 (11.2) | 2,220 (11.2) | 1.01× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,532 (10.7) | 2,594 (10.5) | 0.98× | **≥ parity** |
| 1,920 (2⁷·3·5) | 4,151 (12.6) | 3,943 (13.3) | 1.05× | lags FFTW 1.05× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,657 (13.9) | 31,059 (7.9) | 14,861 (16.5) | 53,398 (4.6) | 45,153 (5.4) | 1.19× | lags FFTW 1.19× |
| 128x128 | 84,996 (13.5) | 151,997 (7.5) | 81,394 (14.1) | 207,174 (5.5) | 174,332 (6.6) | 1.04× | **≥ parity** |
| 256x256 | 502,595 (10.4) | 808,902 (6.5) | 473,270 (11.1) | 867,739 (6.0) | 757,709 (6.9) | 1.06× | lags FFTW 1.06× |
| 512x512 | 4,044,109 (5.8) | 6,131,487 (3.8) | 3,007,871 (7.8) | 5,835,368 (4.0) | 4,156,261 (5.7) | 1.34× | lags FFTW 1.34× |
| 1024x1024 | 17,711,508 (5.9) | 21,522,190 (4.9) | 24,370,331 (4.3) | 28,358,946 (3.7) | 27,926,328 (3.8) | 0.73× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 17,857 | 10,419 | 69,343,776 |
| 1,024 | 67,143 | 37,150 | 128,103,720 |
| 4,096 | 246,959 | 142,196 | 278,319,374 |
| 65,536 | 4,188,037 | 2,288,180 | 3,154,517,010 |
| 1,048,576 | 75,782,974 | 35,737,001 | 8,635,551,862 |
| 1,000 | 64,644 | 36,631 | 145,388,760 |
| 1,080 | 72,315 | 39,207 | 346,623,883 |
| 1,920 | 120,690 | 68,833 | 565,537,905 |
| 1,009 | 179,793 | 44,995 | 135,148,024 |
| 1,296 | 85,002 | 47,667 | 257,069,424 |
| 10,007 | 2,153,336 | 436,500 | 1,510,761,479 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 6/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 1,296 (2⁴·3⁴): 1.52×
- complex 1,920 (2⁷·3·5): 1.48×
- complex 1,080 (2³·3³·5): 1.40×
- real 1,048,576 (2²⁰): 1.35×
- 2-D 512x512: 1.34×
- complex 256 (2⁸): 1.31×
- complex 1,024 (2¹⁰): 1.31×
- real 256 (2⁸): 1.30×
- real 65,536 (2¹⁶): 1.25×
- real 1,080 (2³·3³·5): 1.24×
- complex 1,000 (2³·5³): 1.23×
- real 1,920 (2⁷·3·5): 1.20×
- 2-D 64x64: 1.19×
- real 1,024 (2¹⁰): 1.18×
- real 1,000 (2³·5³): 1.17×
- real 4,096 (2¹²): 1.17×
- complex 65,536 (2¹⁶): 1.12×
- 2-D 256x256: 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
