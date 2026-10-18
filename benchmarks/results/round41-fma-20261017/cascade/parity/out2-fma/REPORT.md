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
| 256 (2⁸) | — (0.0) | 399 (25.7) | 8,799 (1.2) | 6,382 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,229 (23.0) | 16,499 (3.1) | 12,037 (4.3) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 13,026 (18.9) | 62,625 (3.9) | 48,538 (5.1) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 408,888 (12.8) | 1,942,001 (2.7) | 1,418,045 (3.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 23,856,455 (4.4) | 61,966,529 (1.7) | 47,403,803 (2.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,710 (18.4) | 17,157 (2.9) | 14,338 (3.5) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,938 (18.5) | 20,286 (2.7) | 13,509 (4.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,819 (21.7) | 31,287 (3.3) | 23,385 (4.5) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 30,373 (1.7) | 86,606 (0.6) | 56,134 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,624 (18.5) | 21,137 (3.2) | 16,068 (4.2) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 360,442 (1.8) | 919,336 (0.7) | 725,865 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 368 (13.9) | 6,657 (0.8) | 7,209 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,525 (16.8) | 11,570 (2.2) | 10,345 (2.5) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,516 (16.3) | 33,585 (3.7) | 31,722 (3.9) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 191,162 (13.7) | 597,679 (4.4) | 696,808 (3.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 6,121,861 (8.6) | 27,706,349 (1.9) | 24,071,372 (2.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,941 (12.8) | 12,746 (2.0) | 12,133 (2.1) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,014 (13.5) | 13,537 (2.0) | 11,170 (2.4) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,477 (15.1) | 18,079 (2.9) | 17,180 (3.0) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 530 (9.7) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,866 (13.7) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,497 (14.5) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 223,095 (11.8) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 8,862,124 (5.9) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,317 (10.8) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,356 (11.6) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,967 (13.2) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 17,062 (14.4) | 54,640 (4.5) | 45,777 (5.4) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 83,248 (13.8) | 205,372 (5.6) | 175,678 (6.5) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 448,184 (11.7) | 881,658 (5.9) | 759,708 (6.9) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 2,337,274 (10.1) | 7,602,847 (3.1) | 4,672,442 (5.0) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 10,820,955 (9.7) | 36,437,825 (2.9) | 29,761,768 (3.5) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 69,932,586 |
| 1,024 | — | — | 127,684,720 |
| 4,096 | — | — | 280,012,939 |
| 65,536 | — | — | 3,094,630,551 |
| 1,048,576 | — | — | 7,662,399,709 |
| 1,000 | — | — | 142,066,607 |
| 1,080 | — | — | 342,649,249 |
| 1,920 | — | — | 563,644,184 |
| 1,009 | — | — | 141,392,348 |
| 1,296 | — | — | 259,032,479 |
| 10,007 | — | — | 1,531,967,167 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
