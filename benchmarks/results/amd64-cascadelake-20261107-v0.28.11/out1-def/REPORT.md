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
| 256 (2⁸) | — (0.0) | 372 (27.5) | 8,717 (1.2) | 6,383 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,112 (24.2) | 17,855 (2.9) | 11,688 (4.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 13,122 (18.7) | 61,930 (4.0) | 47,900 (5.1) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 405,366 (12.9) | 1,972,115 (2.7) | 1,314,437 (4.0) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 27,036,046 (3.9) | 51,542,035 (2.0) | 41,478,838 (2.5) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,812 (17.7) | 17,464 (2.9) | 13,991 (3.6) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 3,024 (18.0) | 20,411 (2.7) | 13,569 (4.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 5,037 (20.8) | 31,676 (3.3) | 23,750 (4.4) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 30,777 (1.6) | 87,816 (0.6) | 58,256 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,721 (18.0) | 22,835 (2.9) | 17,205 (3.9) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 371,674 (1.8) | 915,641 (0.7) | 729,248 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 366 (14.0) | 6,708 (0.8) | 7,094 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,539 (16.6) | 13,290 (1.9) | 11,616 (2.2) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,858 (15.6) | 33,082 (3.7) | 31,667 (3.9) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 190,102 (13.8) | 593,297 (4.4) | 688,854 (3.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 9,478,008 (5.5) | 25,278,227 (2.1) | 20,989,318 (2.5) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,906 (13.1) | 14,106 (1.8) | 10,743 (2.3) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,026 (13.4) | 13,417 (2.0) | 12,553 (2.2) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,481 (15.0) | 18,172 (2.9) | 17,054 (3.1) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 537 (9.5) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,095 (12.2) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,608 (14.3) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 219,768 (11.9) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 10,310,642 (5.1) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,324 (10.7) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,350 (11.6) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,000 (13.1) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 17,748 (13.8) | 55,203 (4.5) | 46,546 (5.3) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 85,775 (13.4) | 204,094 (5.6) | 183,064 (6.3) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 476,559 (11.0) | 892,450 (5.9) | 760,223 (6.9) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 2,680,663 (8.8) | 6,095,084 (3.9) | 3,810,899 (6.2) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 22,665,223 (4.6) | 30,100,259 (3.5) | 25,999,400 (4.0) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 70,219,749 |
| 1,024 | — | — | 128,622,137 |
| 4,096 | — | — | 284,062,051 |
| 65,536 | — | — | 3,148,690,854 |
| 1,048,576 | — | — | 8,546,626,521 |
| 1,000 | — | — | 143,220,763 |
| 1,080 | — | — | 346,079,150 |
| 1,920 | — | — | 572,895,520 |
| 1,009 | — | — | 136,457,872 |
| 1,296 | — | — | 260,513,651 |
| 10,007 | — | — | 1,528,932,911 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
