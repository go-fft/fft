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
| 256 (2⁸) | — (0.0) | 377 (27.2) | 9,278 (1.1) | 6,487 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,220 (23.1) | 17,002 (3.0) | 11,988 (4.3) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 12,147 (20.2) | 60,214 (4.1) | 47,468 (5.2) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 401,972 (13.0) | 1,904,922 (2.8) | 1,323,195 (4.0) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 25,160,166 (4.2) | 45,932,704 (2.3) | 38,326,184 (2.7) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,695 (18.5) | 17,547 (2.8) | 13,807 (3.6) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,983 (18.2) | 20,723 (2.6) | 13,499 (4.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,879 (21.5) | 31,716 (3.3) | 23,614 (4.4) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 30,832 (1.6) | 80,834 (0.6) | 56,856 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,582 (18.7) | 21,516 (3.1) | 15,999 (4.2) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 362,505 (1.8) | 943,290 (0.7) | 729,335 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 362 (14.1) | 6,838 (0.7) | 6,379 (0.8) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,557 (16.4) | 13,337 (1.9) | 10,433 (2.5) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,428 (16.5) | 32,475 (3.8) | 30,990 (4.0) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 194,101 (13.5) | 589,153 (4.4) | 694,767 (3.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 6,755,864 (7.8) | 17,321,326 (3.0) | 19,465,006 (2.7) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,853 (13.4) | 14,489 (1.7) | 10,722 (2.3) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,001 (13.6) | 13,654 (2.0) | 11,076 (2.5) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,403 (15.4) | 18,260 (2.9) | 17,201 (3.0) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 536 (9.5) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,813 (14.1) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,452 (14.5) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 215,200 (12.2) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 8,923,945 (5.9) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,284 (10.9) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,284 (11.9) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,946 (13.3) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 16,913 (14.5) | 55,306 (4.4) | 44,749 (5.5) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 73,483 (15.6) | 205,963 (5.6) | 172,984 (6.6) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 495,561 (10.6) | 898,973 (5.8) | 761,907 (6.9) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 2,309,741 (10.2) | 4,724,096 (5.0) | 3,380,686 (7.0) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 15,131,224 (6.9) | 29,719,323 (3.5) | 25,905,116 (4.0) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 68,472,168 |
| 1,024 | — | — | 127,478,592 |
| 4,096 | — | — | 285,546,757 |
| 65,536 | — | — | 3,119,581,649 |
| 1,048,576 | — | — | 8,039,058,940 |
| 1,000 | — | — | 144,752,527 |
| 1,080 | — | — | 346,168,425 |
| 1,920 | — | — | 570,273,452 |
| 1,009 | — | — | 134,865,647 |
| 1,296 | — | — | 257,803,403 |
| 10,007 | — | — | 1,510,613,928 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
