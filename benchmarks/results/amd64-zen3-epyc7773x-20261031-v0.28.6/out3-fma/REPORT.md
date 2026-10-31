# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3, cfarm420), one pinned core.
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
| 256 (2⁸) | 294 (34.9) | 297 (34.5) | 6,969 (1.5) | 5,431 (1.9) | 5,775 (1.8) | 0.99× | **≥ parity** |
| 1,024 (2¹⁰) | 1,721 (29.8) | 1,492 (34.3) | 13,161 (3.9) | 9,509 (5.4) | 28,461 (1.8) | 1.15× | lags FFTW 1.15× |
| 4,096 (2¹²) | 9,207 (26.7) | 10,803 (22.8) | 46,338 (5.3) | 32,753 (7.5) | 132,614 (1.9) | 0.85× | **≥ parity** |
| 65,536 (2¹⁶) | 258,441 (20.3) | 290,506 (18.0) | 1,938,313 (2.7) | 898,793 (5.8) | 2,868,800 (1.8) | 0.89× | **≥ parity** |
| 1,048,576 (2²⁰) | 5,250,388 (20.0) | 11,257,232 (9.3) | 22,794,658 (4.6) | 13,519,229 (7.8) | 58,339,899 (1.8) | 0.47× | **≥ parity** |
| 1,000 (2³·5³) | 1,923 (25.9) | 2,027 (24.6) | 14,596 (3.4) | 9,997 (5.0) | 29,763 (1.7) | 0.95× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,494 (21.8) | 2,208 (24.6) | 14,407 (3.8) | 10,475 (5.2) | 35,323 (1.5) | 1.13× | lags FFTW 1.13× |
| 1,920 (2⁷·3·5) | 3,837 (27.3) | 4,036 (25.9) | 21,787 (4.8) | 16,061 (6.5) | 61,184 (1.7) | 0.95× | **≥ parity** |
| 1,009 (prime) | 12,331 (4.1) | 20,450 (2.5) | 60,465 (0.8) | 38,035 (1.3) | 1,130,990 (0.0) | 0.60× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 2,729 (24.6) | 2,940 (22.8) | 16,393 (4.1) | 12,198 (5.5) | 41,396 (1.6) | 0.93× | **≥ parity** |
| 10,007 (prime) | 169,507 (3.9) | 204,230 (3.3) | 640,049 (1.0) | 471,708 (1.4) | 116,082,041 (0.0) | 0.83× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 237 (21.6) | 226 (22.7) | 6,141 (0.8) | 5,322 (1.0) | 2,514 (2.0) | 1.05× | lags FFTW 1.05× |
| 1,024 (2¹⁰) | 1,000 (25.6) | 979 (26.2) | 9,786 (2.6) | 9,674 (2.6) | 13,854 (1.8) | 1.02× | **≥ parity** |
| 4,096 (2¹²) | 5,426 (22.6) | 5,263 (23.3) | 25,725 (4.8) | 22,471 (5.5) | 59,989 (2.0) | 1.03× | **≥ parity** |
| 65,536 (2¹⁶) | 132,685 (19.8) | 152,997 (17.1) | 429,484 (6.1) | 488,320 (5.4) | 1,300,986 (2.0) | 0.87× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,775,079 (18.9) | 3,514,035 (14.9) | 8,359,611 (6.3) | 10,131,431 (5.2) | 26,370,063 (2.0) | 0.79× | **≥ parity** |
| 1,000 (2³·5³) | 1,134 (22.0) | 1,237 (20.1) | 10,379 (2.4) | 8,850 (2.8) | 14,224 (1.8) | 0.92× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,398 (19.5) | 1,326 (20.5) | 10,806 (2.5) | 8,968 (3.0) | 15,669 (1.7) | 1.05× | lags FFTW 1.05× |
| 1,920 (2⁷·3·5) | 2,336 (22.4) | 2,338 (22.4) | 14,520 (3.6) | 12,807 (4.1) | 28,488 (1.8) | 1.00× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 247 (20.8) | 356 (14.4) | 0.69× | **≥ parity** |
| 1,024 (2¹⁰) | 984 (26.0) | 1,179 (21.7) | 0.83× | **≥ parity** |
| 4,096 (2¹²) | 5,454 (22.5) | 5,664 (21.7) | 0.96× | **≥ parity** |
| 65,536 (2¹⁶) | 132,131 (19.8) | 165,935 (15.8) | 0.80× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,795,500 (18.8) | 4,075,200 (12.9) | 0.69× | **≥ parity** |
| 1,000 (2³·5³) | 1,177 (21.2) | 1,453 (17.1) | 0.81× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,378 (19.7) | 1,562 (17.4) | 0.88× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,462 (21.3) | 2,640 (19.8) | 0.93× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 9,105 (27.0) | 12,612 (19.5) | 11,170 (22.0) | 46,817 (5.2) | 34,812 (7.1) | 0.82× | **≥ parity** |
| 128x128 | 49,067 (23.4) | 60,200 (19.1) | 68,148 (16.8) | 152,444 (7.5) | 135,041 (8.5) | 0.72× | **≥ parity** |
| 256x256 | 262,765 (20.0) | 309,734 (16.9) | 297,005 (17.7) | 648,057 (8.1) | 458,414 (11.4) | 0.88× | **≥ parity** |
| 512x512 | 1,182,785 (19.9) | 1,357,360 (17.4) | 1,316,567 (17.9) | 2,959,979 (8.0) | 1,909,368 (12.4) | 0.90× | **≥ parity** |
| 1024x1024 | 5,606,237 (18.7) | 6,635,120 (15.8) | 6,308,668 (16.6) | 13,569,293 (7.7) | 9,370,406 (11.2) | 0.89× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 11,966 | 7,227 | 71,471,074 |
| 1,024 | 40,443 | 27,273 | 132,884,060 |
| 4,096 | 161,697 | 107,379 | 237,216,627 |
| 65,536 | 2,454,231 | 1,602,544 | 2,113,867,791 |
| 1,048,576 | 43,233,593 | 24,119,743 | 4,029,303,231 |
| 1,000 | 42,923 | 26,348 | 150,769,119 |
| 1,080 | 50,634 | 29,007 | 364,912,636 |
| 1,920 | 76,156 | 50,252 | 565,856,678 |
| 1,009 | 73,313 | 31,851 | 141,488,065 |
| 1,296 | 54,738 | 34,330 | 263,510,728 |
| 10,007 | 980,693 | 300,454 | 1,248,770,367 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 20/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 1,024 (2¹⁰): 1.15×
- complex 1,080 (2³·3³·5): 1.13×
- real 1,080 (2³·3³·5): 1.05×
- real 256 (2⁸): 1.05×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
