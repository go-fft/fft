# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon (Cascade Lake), cfarm151 VM, one pinned core.
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
| 256 (2⁸) | 500 (20.5) | 375 (27.3) | 8,746 (1.2) | 6,517 (1.6) | 5,505 (1.9) | 1.33× | lags FFTW 1.33× |
| 1,024 (2¹⁰) | 2,795 (18.3) | 2,150 (23.8) | 16,210 (3.2) | 11,880 (4.3) | 28,648 (1.8) | 1.30× | lags FFTW 1.30× |
| 4,096 (2¹²) | 13,315 (18.5) | 12,851 (19.1) | 59,081 (4.2) | 45,836 (5.4) | 131,731 (1.9) | 1.04× | **≥ parity** |
| 65,536 (2¹⁶) | 453,231 (11.6) | 398,695 (13.2) | 1,852,548 (2.8) | 1,318,179 (4.0) | 3,412,838 (1.5) | 1.14× | lags FFTW 1.14× |
| 1,048,576 (2²⁰) | 21,586,390 (4.9) | 26,063,906 (4.0) | 50,253,412 (2.1) | 41,899,341 (2.5) | 99,153,126 (1.1) | 0.83× | **≥ parity** |
| 1,000 (2³·5³) | 3,351 (14.9) | 2,706 (18.4) | 17,193 (2.9) | 13,917 (3.6) | 30,127 (1.7) | 1.24× | lags FFTW 1.24× |
| 1,080 (2³·3³·5) | 4,167 (13.1) | 3,036 (17.9) | 20,200 (2.7) | 13,383 (4.1) | 34,953 (1.6) | 1.37× | lags FFTW 1.37× |
| 1,920 (2⁷·3·5) | 6,385 (16.4) | 5,035 (20.8) | 30,837 (3.4) | 22,960 (4.6) | 61,591 (1.7) | 1.27× | lags FFTW 1.27× |
| 1,009 (prime) | 18,737 (2.7) | 30,170 (1.7) | 78,399 (0.6) | 55,328 (0.9) | 1,434,669 (0.0) | 0.62× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 4,399 (15.2) | 3,634 (18.4) | 23,410 (2.9) | 16,182 (4.1) | 41,564 (1.6) | 1.21× | lags FFTW 1.21× |
| 10,007 (prime) | 333,435 (2.0) | 353,338 (1.9) | 869,264 (0.8) | 737,037 (0.9) | 140,094,726 (0.0) | 0.94× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 483 (10.6) | 348 (14.7) | 7,197 (0.7) | 6,916 (0.7) | 2,687 (1.9) | 1.39× | lags FFTW 1.39× |
| 1,024 (2¹⁰) | 1,788 (14.3) | 1,513 (16.9) | 11,860 (2.2) | 10,310 (2.5) | 13,287 (1.9) | 1.18× | lags FFTW 1.18× |
| 4,096 (2¹²) | 9,265 (13.3) | 7,559 (16.3) | 32,383 (3.8) | 31,607 (3.9) | 62,087 (2.0) | 1.23× | lags FFTW 1.23× |
| 65,536 (2¹⁶) | 249,850 (10.5) | 188,024 (13.9) | 596,030 (4.4) | 683,712 (3.8) | 1,438,803 (1.8) | 1.33× | lags FFTW 1.33× |
| 1,048,576 (2²⁰) | 11,598,102 (4.5) | 9,655,631 (5.4) | 20,662,586 (2.5) | 19,880,818 (2.6) | 42,099,151 (1.2) | 1.20× | lags FFTW 1.20× |
| 1,000 (2³·5³) | 2,204 (11.3) | 1,900 (13.1) | 13,909 (1.8) | 11,998 (2.1) | 14,370 (1.7) | 1.16× | lags FFTW 1.16× |
| 1,080 (2³·3³·5) | 2,434 (11.2) | 2,259 (12.0) | 13,396 (2.0) | 12,483 (2.2) | 16,736 (1.6) | 1.08× | lags FFTW 1.08× |
| 1,920 (2⁷·3·5) | 4,019 (13.0) | 3,483 (15.0) | 20,332 (2.6) | 17,276 (3.0) | 28,950 (1.8) | 1.15× | lags FFTW 1.15× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 508 (10.1) | 540 (9.5) | 0.94× | **≥ parity** |
| 1,024 (2¹⁰) | 1,777 (14.4) | 1,865 (13.7) | 0.95× | **≥ parity** |
| 4,096 (2¹²) | 9,291 (13.2) | 8,572 (14.3) | 1.08× | lags FFTW 1.08× |
| 65,536 (2¹⁶) | 257,507 (10.2) | 213,860 (12.3) | 1.20× | lags FFTW 1.20× |
| 1,048,576 (2²⁰) | 11,010,891 (4.8) | 9,566,449 (5.5) | 1.15× | lags FFTW 1.15× |
| 1,000 (2³·5³) | 2,273 (11.0) | 2,210 (11.3) | 1.03× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,399 (11.3) | 2,433 (11.2) | 0.99× | **≥ parity** |
| 1,920 (2⁷·3·5) | 4,103 (12.8) | 3,793 (13.8) | 1.08× | lags FFTW 1.08× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,708 (13.9) | 30,473 (8.1) | 16,906 (14.5) | 54,059 (4.5) | 44,669 (5.5) | 1.05× | **≥ parity** |
| 128x128 | 84,788 (13.5) | 144,325 (7.9) | 81,937 (14.0) | 210,516 (5.4) | 181,020 (6.3) | 1.03× | **≥ parity** |
| 256x256 | 473,684 (11.1) | 728,233 (7.2) | 494,607 (10.6) | 869,178 (6.0) | 745,541 (7.0) | 0.96× | **≥ parity** |
| 512x512 | 2,728,564 (8.6) | 4,089,965 (5.8) | 2,589,173 (9.1) | 4,113,648 (5.7) | 3,407,766 (6.9) | 1.05× | lags FFTW 1.05× |
| 1024x1024 | 15,824,282 (6.6) | 18,903,295 (5.5) | 17,189,934 (6.1) | 26,936,508 (3.9) | 24,754,530 (4.2) | 0.92× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 17,335 | 10,112 | 69,029,042 |
| 1,024 | 64,712 | 36,254 | 126,421,014 |
| 4,096 | 240,668 | 137,978 | 276,166,819 |
| 65,536 | 4,185,278 | 2,213,014 | 3,115,588,253 |
| 1,048,576 | 73,962,387 | 35,402,533 | 8,347,034,143 |
| 1,000 | 66,836 | 35,215 | 144,582,104 |
| 1,080 | 78,455 | 38,982 | 348,122,789 |
| 1,920 | 117,253 | 67,112 | 570,762,740 |
| 1,009 | 178,963 | 43,624 | 135,531,440 |
| 1,296 | 82,580 | 45,524 | 259,125,052 |
| 10,007 | 2,286,668 | 416,851 | 1,500,963,038 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 8/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.39×
- complex 1,080 (2³·3³·5): 1.37×
- complex 256 (2⁸): 1.33×
- real 65,536 (2¹⁶): 1.33×
- complex 1,024 (2¹⁰): 1.30×
- complex 1,920 (2⁷·3·5): 1.27×
- complex 1,000 (2³·5³): 1.24×
- real 4,096 (2¹²): 1.23×
- complex 1,296 (2⁴·3⁴): 1.21×
- real 1,048,576 (2²⁰): 1.20×
- real 1,024 (2¹⁰): 1.18×
- real 1,000 (2³·5³): 1.16×
- real 1,920 (2⁷·3·5): 1.15×
- complex 65,536 (2¹⁶): 1.14×
- real 1,080 (2³·3³·5): 1.08×
- 2-D 512x512: 1.05×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
