# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3), cfarm420.
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
| 256 (2⁸) | — (0.0) | 280 (36.6) | 7,475 (1.4) | 5,757 (1.8) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,504 (34.0) | 13,574 (3.8) | 10,053 (5.1) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,849 (22.7) | 46,808 (5.3) | 35,924 (6.8) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 304,586 (17.2) | 1,976,405 (2.7) | 957,133 (5.5) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,439,069 (9.2) | 43,063,038 (2.4) | 14,495,923 (7.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,038 (24.5) | 14,451 (3.4) | 10,409 (4.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,286 (23.8) | 15,184 (3.6) | 11,842 (4.6) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,850 (27.2) | 24,108 (4.3) | 18,838 (5.6) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 21,340 (2.4) | 64,551 (0.8) | 42,897 (1.2) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,068 (21.8) | 17,308 (3.9) | 13,105 (5.1) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 213,934 (3.1) | 673,884 (1.0) | 476,826 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 225 (22.8) | 6,074 (0.8) | 5,308 (1.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 982 (26.1) | 9,807 (2.6) | 8,448 (3.0) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,409 (22.7) | 25,698 (4.8) | 23,899 (5.1) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 155,282 (16.9) | 407,998 (6.4) | 507,435 (5.2) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,922,139 (13.4) | 8,966,902 (5.8) | 10,550,316 (5.0) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,344 (18.5) | 10,741 (2.3) | 9,340 (2.7) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,375 (19.8) | 11,057 (2.5) | 9,220 (3.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,358 (22.2) | 16,093 (3.3) | 12,514 (4.2) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 357 (14.4) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,249 (20.5) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 6,114 (20.1) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 175,732 (14.9) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,214,264 (12.4) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,459 (17.1) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,698 (16.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,784 (18.8) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,971 (22.4) | 48,945 (5.0) | 35,534 (6.9) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 65,932 (17.4) | 145,385 (7.9) | 128,019 (9.0) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 282,885 (18.5) | 696,515 (7.5) | 461,396 (11.4) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,340,524 (17.6) | 2,997,746 (7.9) | 1,989,944 (11.9) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,782,039 (15.5) | 13,851,932 (7.6) | 9,654,083 (10.9) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 75,983,454 |
| 1,024 | — | — | 132,398,060 |
| 4,096 | — | — | 262,770,360 |
| 65,536 | — | — | 2,305,959,698 |
| 1,048,576 | — | — | 4,532,450,750 |
| 1,000 | — | — | 156,261,761 |
| 1,080 | — | — | 393,396,399 |
| 1,920 | — | — | 603,562,574 |
| 1,009 | — | — | 144,990,726 |
| 1,296 | — | — | 276,259,166 |
| 10,007 | — | — | 1,280,425,672 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
