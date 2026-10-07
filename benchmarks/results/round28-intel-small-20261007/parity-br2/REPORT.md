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
| 256 (2⁸) | 462 (22.2) | 365 (28.1) | 8,849 (1.2) | 6,352 (1.6) | 5,521 (1.9) | 1.27× | lags FFTW 1.27× |
| 1,024 (2¹⁰) | 2,493 (20.5) | 2,109 (24.3) | 16,304 (3.1) | 12,101 (4.2) | 29,043 (1.8) | 1.18× | lags FFTW 1.18× |
| 4,096 (2¹²) | 11,942 (20.6) | 12,828 (19.2) | 57,891 (4.2) | 45,768 (5.4) | 132,665 (1.9) | 0.93× | **≥ parity** |
| 65,536 (2¹⁶) | 457,875 (11.5) | 400,240 (13.1) | 1,844,171 (2.8) | 1,363,184 (3.8) | 3,549,357 (1.5) | 1.14× | lags FFTW 1.14× |
| 1,048,576 (2²⁰) | 21,571,351 (4.9) | 26,037,623 (4.0) | 50,245,144 (2.1) | 42,013,715 (2.5) | 99,546,142 (1.1) | 0.83× | **≥ parity** |
| 1,000 (2³·5³) | 3,354 (14.9) | 2,723 (18.3) | 17,014 (2.9) | 13,874 (3.6) | 30,031 (1.7) | 1.23× | lags FFTW 1.23× |
| 1,080 (2³·3³·5) | 4,192 (13.0) | 3,026 (18.0) | 20,313 (2.7) | 13,872 (3.9) | 34,940 (1.6) | 1.39× | lags FFTW 1.39× |
| 1,920 (2⁷·3·5) | 6,449 (16.2) | 4,887 (21.4) | 31,434 (3.3) | 22,226 (4.7) | 61,637 (1.7) | 1.32× | lags FFTW 1.32× |
| 1,009 (prime) | 18,736 (2.7) | 30,034 (1.7) | 80,530 (0.6) | 56,820 (0.9) | 1,428,089 (0.0) | 0.62× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 4,441 (15.1) | 3,587 (18.7) | 20,758 (3.2) | 15,650 (4.3) | 41,928 (1.6) | 1.24× | lags FFTW 1.24× |
| 10,007 (prime) | 344,044 (1.9) | 357,803 (1.9) | 878,541 (0.8) | 736,130 (0.9) | 139,737,668 (0.0) | 0.96× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 435 (11.8) | 360 (14.2) | 6,497 (0.8) | 6,487 (0.8) | 2,663 (1.9) | 1.21× | lags FFTW 1.21× |
| 1,024 (2¹⁰) | 1,673 (15.3) | 1,516 (16.9) | 13,100 (2.0) | 10,272 (2.5) | 13,181 (1.9) | 1.10× | lags FFTW 1.10× |
| 4,096 (2¹²) | 9,178 (13.4) | 7,640 (16.1) | 32,569 (3.8) | 31,441 (3.9) | 62,050 (2.0) | 1.20× | lags FFTW 1.20× |
| 65,536 (2¹⁶) | 254,541 (10.3) | 186,699 (14.0) | 592,952 (4.4) | 682,752 (3.8) | 1,438,358 (1.8) | 1.36× | lags FFTW 1.36× |
| 1,048,576 (2²⁰) | 9,763,710 (5.4) | 9,924,578 (5.3) | 17,422,845 (3.0) | 20,909,599 (2.5) | 41,952,695 (1.2) | 0.98× | **≥ parity** |
| 1,000 (2³·5³) | 2,215 (11.2) | 1,854 (13.4) | 12,692 (2.0) | 10,720 (2.3) | 14,208 (1.8) | 1.19× | lags FFTW 1.19× |
| 1,080 (2³·3³·5) | 2,373 (11.5) | 1,995 (13.6) | 13,428 (2.0) | 10,924 (2.5) | 16,482 (1.7) | 1.19× | lags FFTW 1.19× |
| 1,920 (2⁷·3·5) | 3,936 (13.3) | 3,379 (15.5) | 17,965 (2.9) | 17,509 (3.0) | 28,949 (1.8) | 1.16× | lags FFTW 1.16× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 456 (11.2) | 502 (10.2) | 0.91× | **≥ parity** |
| 1,024 (2¹⁰) | 1,737 (14.7) | 1,913 (13.4) | 0.91× | **≥ parity** |
| 4,096 (2¹²) | 9,052 (13.6) | 8,380 (14.7) | 1.08× | lags FFTW 1.08× |
| 65,536 (2¹⁶) | 248,898 (10.5) | 213,591 (12.3) | 1.17× | lags FFTW 1.17× |
| 1,048,576 (2²⁰) | 11,766,175 (4.5) | 9,201,246 (5.7) | 1.28× | lags FFTW 1.28× |
| 1,000 (2³·5³) | 2,261 (11.0) | 2,270 (11.0) | 1.00× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,372 (11.5) | 2,324 (11.7) | 1.02× | **≥ parity** |
| 1,920 (2⁷·3·5) | 4,071 (12.9) | 4,039 (13.0) | 1.01× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,544 (14.0) | 30,544 (8.0) | 17,140 (14.3) | 54,722 (4.5) | 45,883 (5.4) | 1.02× | **≥ parity** |
| 128x128 | 79,554 (14.4) | 136,478 (8.4) | 79,329 (14.5) | 212,902 (5.4) | 181,062 (6.3) | 1.00× | **≥ parity** |
| 256x256 | 471,750 (11.1) | 674,067 (7.8) | 452,062 (11.6) | 861,663 (6.1) | 752,781 (7.0) | 1.04× | **≥ parity** |
| 512x512 | 2,126,357 (11.1) | 4,747,621 (5.0) | 2,434,624 (9.7) | 4,328,751 (5.5) | 3,324,983 (7.1) | 0.87× | **≥ parity** |
| 1024x1024 | 15,863,499 (6.6) | 19,612,861 (5.3) | 16,905,520 (6.2) | 26,884,234 (3.9) | 27,141,627 (3.9) | 0.94× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 22,811 | 9,985 | 68,990,398 |
| 1,024 | 79,966 | 37,006 | 126,439,046 |
| 4,096 | 305,947 | 141,363 | 277,354,380 |
| 65,536 | 4,430,942 | 2,275,987 | 3,056,001,235 |
| 1,048,576 | 75,120,720 | 37,007,554 | 8,283,702,126 |
| 1,000 | 64,048 | 36,767 | 142,848,119 |
| 1,080 | 74,693 | 38,343 | 345,866,436 |
| 1,920 | 119,488 | 67,053 | 566,181,773 |
| 1,009 | 185,207 | 43,547 | 135,021,271 |
| 1,296 | 82,464 | 45,183 | 258,479,590 |
| 10,007 | 2,158,905 | 417,906 | 1,506,913,934 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 10/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 1,080 (2³·3³·5): 1.39×
- real 65,536 (2¹⁶): 1.36×
- complex 1,920 (2⁷·3·5): 1.32×
- complex 256 (2⁸): 1.27×
- complex 1,296 (2⁴·3⁴): 1.24×
- complex 1,000 (2³·5³): 1.23×
- real 256 (2⁸): 1.21×
- real 4,096 (2¹²): 1.20×
- real 1,000 (2³·5³): 1.19×
- real 1,080 (2³·3³·5): 1.19×
- complex 1,024 (2¹⁰): 1.18×
- real 1,920 (2⁷·3·5): 1.16×
- complex 65,536 (2¹⁶): 1.14×
- real 1,024 (2¹⁰): 1.10×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
