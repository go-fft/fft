# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Xeon Cascade Lake (cfarm151).
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
| 256 (2⁸) | — (0.0) | 372 (27.5) | 8,635 (1.2) | 6,340 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,121 (24.1) | 16,133 (3.2) | 12,014 (4.3) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 12,889 (19.1) | 62,139 (4.0) | 47,606 (5.2) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 396,395 (13.2) | 2,017,512 (2.6) | 1,394,608 (3.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 25,248,432 (4.2) | 51,279,412 (2.0) | 39,776,674 (2.6) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,796 (17.8) | 16,877 (3.0) | 13,881 (3.6) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,954 (18.4) | 20,401 (2.7) | 13,382 (4.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,853 (21.6) | 31,277 (3.3) | 23,452 (4.5) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 30,582 (1.6) | 82,512 (0.6) | 56,696 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 4,103 (16.3) | 20,898 (3.2) | 15,861 (4.2) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 362,396 (1.8) | 955,246 (0.7) | 729,162 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 373 (13.7) | 6,550 (0.8) | 6,290 (0.8) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,530 (16.7) | 13,045 (2.0) | 10,213 (2.5) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,784 (15.8) | 33,563 (3.7) | 30,986 (4.0) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 188,328 (13.9) | 596,397 (4.4) | 694,752 (3.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 6,472,381 (8.1) | 19,793,897 (2.6) | 21,600,044 (2.4) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,874 (13.3) | 14,048 (1.8) | 10,761 (2.3) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,083 (13.1) | 13,888 (2.0) | 12,416 (2.2) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,587 (14.6) | 18,310 (2.9) | 17,547 (3.0) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 549 (9.3) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,918 (13.3) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,751 (14.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 222,872 (11.8) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 9,926,708 (5.3) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,257 (11.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,311 (11.8) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,987 (13.1) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 14,939 (16.5) | 54,939 (4.5) | 44,964 (5.5) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 85,224 (13.5) | 214,752 (5.3) | 180,603 (6.4) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 491,583 (10.7) | 918,979 (5.7) | 761,748 (6.9) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 2,414,924 (9.8) | 4,664,946 (5.1) | 3,542,382 (6.7) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 18,309,770 (5.7) | 27,069,469 (3.9) | 25,399,467 (4.1) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 69,169,598 |
| 1,024 | — | — | 127,627,863 |
| 4,096 | — | — | 279,769,030 |
| 65,536 | — | — | 3,102,711,249 |
| 1,048,576 | — | — | 7,983,528,607 |
| 1,000 | — | — | 143,887,514 |
| 1,080 | — | — | 341,940,012 |
| 1,920 | — | — | 566,448,731 |
| 1,009 | — | — | 136,442,978 |
| 1,296 | — | — | 258,041,812 |
| 10,007 | — | — | 1,507,734,271 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
