# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon Cascade Lake, 8 cores (GCC Compile Farm cfarm151).
- **Toolchains**: go1.27.3 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512** (3.3.10 built from source by setup.sh); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | — (0.0) | 374 (27.4) | 9,107 (1.1) | 7,260 (1.4) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,339 (21.9) | 18,627 (2.7) | 13,325 (3.8) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 13,384 (18.4) | 67,945 (3.6) | 51,767 (4.7) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 422,482 (12.4) | 2,719,005 (1.9) | 1,933,718 (2.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 27,909,243 (3.8) | 64,176,726 (1.6) | 48,535,447 (2.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,809 (17.7) | 18,110 (2.8) | 14,621 (3.4) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 3,502 (15.5) | 21,399 (2.5) | 14,600 (3.7) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 5,470 (19.1) | 31,888 (3.3) | 23,594 (4.4) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 31,582 (1.6) | 86,468 (0.6) | 59,226 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,806 (17.6) | 20,983 (3.2) | 16,793 (4.0) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 383,341 (1.7) | 984,759 (0.7) | 746,399 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 378 (13.6) | 6,747 (0.8) | 7,451 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,577 (16.2) | 13,325 (1.9) | 10,667 (2.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,964 (15.4) | 35,774 (3.4) | 31,838 (3.9) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 201,861 (13.0) | 647,637 (4.0) | 747,635 (3.5) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,251,746 (4.7) | 39,867,341 (1.3) | 25,038,874 (2.1) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,957 (12.7) | 14,607 (1.7) | 12,358 (2.0) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,291 (11.9) | 14,030 (1.9) | 11,316 (2.4) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,485 (15.0) | 19,733 (2.7) | 17,839 (2.9) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 516 (9.9) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,947 (13.2) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 9,193 (13.4) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 239,759 (10.9) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 12,883,679 (4.1) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,437 (10.2) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,714 (10.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,387 (11.9) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 16,124 (15.2) | 59,223 (4.1) | 46,420 (5.3) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 77,051 (14.9) | 210,937 (5.4) | 185,164 (6.2) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 494,649 (10.6) | 1,126,617 (4.7) | 824,018 (6.4) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 4,136,940 (5.7) | 7,686,918 (3.1) | 5,707,342 (4.1) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 24,855,488 (4.2) | 35,578,203 (2.9) | 28,834,908 (3.6) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 72,183,880 |
| 1,024 | — | — | 129,359,336 |
| 4,096 | — | — | 286,900,019 |
| 65,536 | — | — | 3,372,232,509 |
| 1,048,576 | — | — | 9,030,155,571 |
| 1,000 | — | — | 148,861,957 |
| 1,080 | — | — | 357,764,562 |
| 1,920 | — | — | 598,161,309 |
| 1,009 | — | — | 152,848,856 |
| 1,296 | — | — | 274,695,963 |
| 10,007 | — | — | 1,677,300,066 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
