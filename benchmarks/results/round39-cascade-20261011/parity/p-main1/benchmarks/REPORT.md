# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon (Cascade Lake), cfarm151, 1 pinned core.
- **Toolchains**: go1.27.1 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512** (3.3.10 built from source with AVX-512 (setup.sh)); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | — (0.0) | 361 (28.3) | 8,876 (1.2) | 6,398 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,053 (24.9) | 16,134 (3.2) | 11,860 (4.3) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 12,036 (20.4) | 57,351 (4.3) | 43,603 (5.6) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 411,721 (12.7) | 1,949,347 (2.7) | 1,335,214 (3.9) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 25,789,123 (4.1) | 48,875,751 (2.1) | 39,881,645 (2.6) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,809 (17.7) | 19,193 (2.6) | 13,846 (3.6) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 3,035 (17.9) | 20,412 (2.7) | 13,248 (4.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,747 (22.1) | 31,704 (3.3) | 23,068 (4.5) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 29,970 (1.7) | 80,166 (0.6) | 56,958 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,584 (18.7) | 20,889 (3.2) | 17,698 (3.8) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 348,213 (1.9) | 880,387 (0.8) | 749,185 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 366 (14.0) | 6,764 (0.8) | 7,272 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,544 (16.6) | 13,141 (1.9) | 10,500 (2.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,197 (17.1) | 33,069 (3.7) | 31,329 (3.9) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 186,344 (14.1) | 600,469 (4.4) | 696,796 (3.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 9,333,174 (5.6) | 19,754,242 (2.7) | 19,953,806 (2.6) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,827 (13.6) | 14,273 (1.7) | 12,257 (2.0) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,126 (12.8) | 13,532 (2.0) | 12,539 (2.2) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,483 (15.0) | 18,072 (2.9) | 17,333 (3.0) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 533 (9.6) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,910 (13.4) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,614 (14.3) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 209,229 (12.5) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 9,451,999 (5.5) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,312 (10.8) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,304 (11.8) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,948 (13.3) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 17,035 (14.4) | 54,498 (4.5) | 45,118 (5.4) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 82,414 (13.9) | 210,189 (5.5) | 181,159 (6.3) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 466,837 (11.2) | 874,693 (6.0) | 763,413 (6.9) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 2,481,173 (9.5) | 4,499,179 (5.2) | 3,333,800 (7.1) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 23,349,133 (4.5) | 26,450,390 (4.0) | 25,838,092 (4.1) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 68,788,376 |
| 1,024 | — | — | 126,708,809 |
| 4,096 | — | — | 278,935,796 |
| 65,536 | — | — | 3,084,644,580 |
| 1,048,576 | — | — | 8,155,954,158 |
| 1,000 | — | — | 143,374,393 |
| 1,080 | — | — | 345,370,550 |
| 1,920 | — | — | 565,730,488 |
| 1,009 | — | — | 134,771,358 |
| 1,296 | — | — | 256,856,680 |
| 10,007 | — | — | 1,487,129,839 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
