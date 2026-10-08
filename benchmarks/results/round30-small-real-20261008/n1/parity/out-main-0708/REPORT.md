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
| 256 (2⁸) | 1,113 (9.2) | 1,151 (8.9) | 9,354 (1.1) | 7,690 (1.3) | 5,870 (1.7) | 0.97× | **≥ parity** |
| 1,024 (2¹⁰) | 5,280 (9.7) | 6,328 (8.1) | 18,759 (2.7) | 14,600 (3.5) | 30,448 (1.7) | 0.83× | **≥ parity** |
| 4,096 (2¹²) | 27,851 (8.8) | 40,888 (6.0) | 60,561 (4.1) | 47,481 (5.2) | 145,720 (1.7) | 0.68× | **≥ parity** |
| 65,536 (2¹⁶) | 714,308 (7.3) | 1,221,095 (4.3) | 2,185,206 (2.4) | 1,463,463 (3.6) | 3,445,806 (1.5) | 0.58× | **≥ parity** |
| 1,048,576 (2²⁰) | 20,250,375 (5.2) | 44,094,208 (2.4) | 46,663,209 (2.2) | 32,261,818 (3.3) | 74,519,288 (1.4) | 0.46× | **≥ parity** |
| 1,000 (2³·5³) | 6,064 (8.2) | 7,177 (6.9) | 19,219 (2.6) | 15,238 (3.3) | 33,854 (1.5) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,868 (7.9) | 7,674 (7.1) | 21,808 (2.5) | 16,755 (3.2) | 41,395 (1.3) | 0.90× | **≥ parity** |
| 1,920 (2⁷·3·5) | 12,011 (8.7) | 13,514 (7.7) | 32,706 (3.2) | 25,831 (4.1) | 69,784 (1.5) | 0.89× | **≥ parity** |
| 1,009 (prime) | 21,468 (2.3) | 48,007 (1.0) | 96,222 (0.5) | 57,754 (0.9) | 2,221,524 (0.0) | 0.45× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,716 (7.7) | 11,347 (5.9) | 24,861 (2.7) | 19,678 (3.4) | 48,049 (1.4) | 0.77× | **≥ parity** |
| 10,007 (prime) | 416,622 (1.6) | 603,320 (1.1) | 1,046,969 (0.6) | 774,843 (0.9) | 217,344,456 (0.0) | 0.69× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 772 (6.6) | 719 (7.1) | 8,667 (0.6) | 7,877 (0.7) | 2,990 (1.7) | 1.07× | lags FFTW 1.07× |
| 1,024 (2¹⁰) | 3,344 (7.7) | 4,224 (6.1) | 14,264 (1.8) | 12,387 (2.1) | 15,225 (1.7) | 0.79× | **≥ parity** |
| 4,096 (2¹²) | 15,585 (7.9) | 20,037 (6.1) | 37,377 (3.3) | 32,644 (3.8) | 70,091 (1.8) | 0.78× | **≥ parity** |
| 65,536 (2¹⁶) | 398,735 (6.6) | 537,833 (4.9) | 735,018 (3.6) | 754,573 (3.5) | 1,763,884 (1.5) | 0.74× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,434,733 (5.0) | 18,512,311 (2.8) | 22,884,198 (2.3) | 18,681,299 (2.8) | 42,747,660 (1.2) | 0.56× | **≥ parity** |
| 1,000 (2³·5³) | 3,894 (6.4) | 3,950 (6.3) | 15,081 (1.7) | 12,594 (2.0) | 14,964 (1.7) | 0.99× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,412 (6.2) | 4,223 (6.4) | 15,976 (1.7) | 14,227 (1.9) | 17,464 (1.6) | 1.04× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,199 (7.3) | 7,323 (7.1) | 21,826 (2.4) | 19,093 (2.7) | 30,640 (1.7) | 0.98× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 773 (6.6) | 811 (6.3) | 0.95× | **≥ parity** |
| 1,024 (2¹⁰) | 3,280 (7.8) | 4,489 (5.7) | 0.73× | **≥ parity** |
| 4,096 (2¹²) | 15,336 (8.0) | 21,235 (5.8) | 0.72× | **≥ parity** |
| 65,536 (2¹⁶) | 389,982 (6.7) | 641,089 (4.1) | 0.61× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,121,458 (5.2) | 23,184,870 (2.3) | 0.44× | **≥ parity** |
| 1,000 (2³·5³) | 3,939 (6.3) | 4,283 (5.8) | 0.92× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,351 (6.3) | 4,460 (6.1) | 0.98× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,203 (7.3) | 7,889 (6.6) | 0.91× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 27,755 (8.9) | 33,126 (7.4) | 30,926 (7.9) | 70,197 (3.5) | 50,981 (4.8) | 0.90× | **≥ parity** |
| 128x128 | 127,756 (9.0) | 149,418 (7.7) | 192,744 (6.0) | 251,293 (4.6) | 198,796 (5.8) | 0.66× | **≥ parity** |
| 256x256 | 672,095 (7.8) | 755,793 (6.9) | 1,279,049 (4.1) | 1,168,257 (4.5) | 888,932 (5.9) | 0.53× | **≥ parity** |
| 512x512 | 3,481,934 (6.8) | 4,395,102 (5.4) | 6,888,753 (3.4) | 5,384,215 (4.4) | 4,178,945 (5.6) | 0.51× | **≥ parity** |
| 1024x1024 | 18,590,682 (5.6) | 18,506,970 (5.7) | 43,639,481 (2.4) | 25,600,683 (4.1) | 20,053,601 (5.2) | 0.43× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 4,514,430 |
| 1,024 | — | — | 3,292,613 |
| 4,096 | — | — | 6,320,901 |
| 65,536 | — | — | 21,663,899 |
| 1,048,576 | — | — | 32,740,595 |
| 1,000 | — | — | 4,834,395 |
| 1,080 | — | — | 12,626,078 |
| 1,920 | — | — | 17,323,393 |
| 1,009 | — | — | 6,124,696 |
| 1,296 | — | — | 10,444,045 |
| 10,007 | — | — | 35,382,595 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 23/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.07×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
