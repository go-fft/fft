# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon (Cascade Lake), cfarm151, 1 pinned core.
- **Toolchains**: go1.27.2 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512** (3.3.10 built from source with AVX-512 (setup.sh)); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 389 (26.3) | 378 (27.1) | 8,863 (1.2) | 6,461 (1.6) | 5,515 (1.9) | 1.03× | **≥ parity** |
| 1,024 (2¹⁰) | 2,494 (20.5) | 2,104 (24.3) | 16,517 (3.1) | 12,035 (4.3) | 29,744 (1.7) | 1.19× | lags FFTW 1.19× |
| 4,096 (2¹²) | 12,209 (20.1) | 13,293 (18.5) | 59,618 (4.1) | 45,759 (5.4) | 135,206 (1.8) | 0.92× | **≥ parity** |
| 65,536 (2¹⁶) | 430,455 (12.2) | 407,677 (12.9) | 1,956,870 (2.7) | 1,365,766 (3.8) | 3,035,472 (1.7) | 1.06× | lags FFTW 1.06× |
| 1,048,576 (2²⁰) | 17,854,251 (5.9) | 24,993,196 (4.2) | 49,663,366 (2.1) | 40,295,113 (2.6) | 99,513,784 (1.1) | 0.71× | **≥ parity** |
| 1,000 (2³·5³) | 2,999 (16.6) | 2,744 (18.2) | 17,160 (2.9) | 14,210 (3.5) | 30,405 (1.6) | 1.09× | lags FFTW 1.09× |
| 1,080 (2³·3³·5) | 3,399 (16.0) | 3,024 (18.0) | 20,607 (2.6) | 13,383 (4.1) | 35,371 (1.5) | 1.12× | lags FFTW 1.12× |
| 1,920 (2⁷·3·5) | 5,406 (19.4) | 5,084 (20.6) | 31,620 (3.3) | 22,694 (4.6) | 62,449 (1.7) | 1.06× | lags FFTW 1.06× |
| 1,009 (prime) | 17,883 (2.8) | 30,371 (1.7) | 80,806 (0.6) | 56,937 (0.9) | 1,443,809 (0.0) | 0.59× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,901 (17.2) | 3,620 (18.5) | 21,028 (3.2) | 15,786 (4.2) | 42,382 (1.6) | 1.08× | lags FFTW 1.08× |
| 10,007 (prime) | 330,298 (2.0) | 368,610 (1.8) | 889,404 (0.7) | 728,076 (0.9) | 140,443,860 (0.0) | 0.90× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 352 (14.6) | 371 (13.8) | 6,573 (0.8) | 6,677 (0.8) | 2,658 (1.9) | 0.95× | **≥ parity** |
| 1,024 (2¹⁰) | 1,207 (21.2) | 1,518 (16.9) | 13,056 (2.0) | 10,524 (2.4) | 13,229 (1.9) | 0.80× | **≥ parity** |
| 4,096 (2¹²) | 7,872 (15.6) | 7,683 (16.0) | 32,798 (3.7) | 32,043 (3.8) | 61,150 (2.0) | 1.02× | **≥ parity** |
| 65,536 (2¹⁶) | 207,379 (12.6) | 191,889 (13.7) | 607,049 (4.3) | 696,093 (3.8) | 1,418,069 (1.8) | 1.08× | lags FFTW 1.08× |
| 1,048,576 (2²⁰) | 7,483,210 (7.0) | 7,211,276 (7.3) | 18,073,435 (2.9) | 20,534,037 (2.6) | 42,435,209 (1.2) | 1.04× | **≥ parity** |
| 1,000 (2³·5³) | 1,787 (13.9) | 1,913 (13.0) | 14,246 (1.7) | 12,265 (2.0) | 14,285 (1.7) | 0.93× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,066 (13.2) | 2,010 (13.5) | 13,313 (2.0) | 12,852 (2.1) | 16,546 (1.6) | 1.03× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,399 (15.4) | 3,502 (15.0) | 17,902 (2.9) | 17,380 (3.0) | 29,103 (1.8) | 0.97× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 365 (14.0) | 494 (10.4) | 0.74× | **≥ parity** |
| 1,024 (2¹⁰) | 1,320 (19.4) | 1,974 (13.0) | 0.67× | **≥ parity** |
| 4,096 (2¹²) | 8,044 (15.3) | 8,631 (14.2) | 0.93× | **≥ parity** |
| 65,536 (2¹⁶) | 247,078 (10.6) | 221,524 (11.8) | 1.12× | lags FFTW 1.12× |
| 1,048,576 (2²⁰) | 9,990,809 (5.2) | 9,364,443 (5.6) | 1.07× | lags FFTW 1.07× |
| 1,000 (2³·5³) | 1,917 (13.0) | 2,287 (10.9) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,230 (12.2) | 2,543 (10.7) | 0.88× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,726 (14.1) | 4,013 (13.0) | 0.93× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,820 (13.8) | 30,681 (8.0) | 17,209 (14.3) | 54,292 (4.5) | 45,028 (5.5) | 1.04× | **≥ parity** |
| 128x128 | 80,482 (14.3) | 139,415 (8.2) | 73,960 (15.5) | 211,224 (5.4) | 180,596 (6.4) | 1.09× | lags FFTW 1.09× |
| 256x256 | 479,334 (10.9) | 689,960 (7.6) | 483,185 (10.9) | 922,813 (5.7) | 760,327 (6.9) | 0.99× | **≥ parity** |
| 512x512 | 2,135,964 (11.0) | 4,450,448 (5.3) | 2,366,196 (10.0) | 4,904,124 (4.8) | 3,380,035 (7.0) | 0.90× | **≥ parity** |
| 1024x1024 | 16,165,375 (6.5) | 18,921,704 (5.5) | 16,939,813 (6.2) | 31,357,942 (3.3) | 26,812,319 (3.9) | 0.95× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 22,798 | 9,536 | 69,010,921 |
| 1,024 | 80,147 | 35,830 | 127,413,401 |
| 4,096 | 304,960 | 133,160 | 276,445,920 |
| 65,536 | 3,811,043 | 2,100,894 | 3,104,427,846 |
| 1,048,576 | 70,696,249 | 35,462,256 | 7,940,594,742 |
| 1,000 | 60,783 | 34,819 | 141,919,292 |
| 1,080 | 65,028 | 36,951 | 343,392,661 |
| 1,920 | 138,421 | 66,170 | 563,847,808 |
| 1,009 | 175,701 | 43,433 | 135,748,804 |
| 1,296 | 77,075 | 44,872 | 258,474,657 |
| 10,007 | 2,102,199 | 416,008 | 1,508,175,159 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 16/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 1,024 (2¹⁰): 1.19×
- complex 1,080 (2³·3³·5): 1.12×
- complex 1,000 (2³·5³): 1.09×
- 2-D 128x128: 1.09×
- real 65,536 (2¹⁶): 1.08×
- complex 1,296 (2⁴·3⁴): 1.08×
- complex 1,920 (2⁷·3·5): 1.06×
- complex 65,536 (2¹⁶): 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
