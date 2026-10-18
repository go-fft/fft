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
| 256 (2⁸) | — (0.0) | 373 (27.5) | 8,932 (1.1) | 6,319 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,152 (23.8) | 16,628 (3.1) | 12,172 (4.2) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 13,186 (18.6) | 61,166 (4.0) | 46,159 (5.3) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 430,377 (12.2) | 1,848,291 (2.8) | 1,353,041 (3.9) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 29,628,324 (3.5) | 44,552,219 (2.4) | 39,605,004 (2.6) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,798 (17.8) | 17,332 (2.9) | 13,962 (3.6) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 3,061 (17.8) | 20,612 (2.6) | 13,222 (4.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 5,070 (20.7) | 31,571 (3.3) | 23,368 (4.5) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 30,734 (1.6) | 81,599 (0.6) | 56,683 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,721 (18.0) | 20,709 (3.2) | 16,445 (4.1) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 375,181 (1.8) | 937,680 (0.7) | 734,582 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 367 (14.0) | 6,608 (0.8) | 7,063 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,544 (16.6) | 11,954 (2.1) | 10,008 (2.6) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,952 (15.5) | 32,409 (3.8) | 31,204 (3.9) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 194,296 (13.5) | 589,547 (4.4) | 681,484 (3.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 10,375,785 (5.1) | 16,535,965 (3.2) | 18,469,672 (2.8) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,880 (13.3) | 14,290 (1.7) | 11,840 (2.1) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,014 (13.5) | 13,563 (2.0) | 10,844 (2.5) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,446 (15.2) | 18,426 (2.8) | 16,924 (3.1) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 487 (10.5) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,886 (13.6) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,529 (14.4) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 218,740 (12.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 10,606,951 (4.9) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,350 (10.6) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,371 (11.5) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,019 (13.0) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 17,186 (14.3) | 56,940 (4.3) | 44,213 (5.6) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 73,676 (15.6) | 209,341 (5.5) | 181,144 (6.3) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 461,965 (11.3) | 868,804 (6.0) | 763,073 (6.9) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 2,713,350 (8.7) | 4,381,515 (5.4) | 3,287,193 (7.2) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 24,313,840 (4.3) | 29,575,255 (3.5) | 21,975,506 (4.8) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 72,152,574 |
| 1,024 | — | — | 128,312,522 |
| 4,096 | — | — | 285,108,641 |
| 65,536 | — | — | 3,192,654,010 |
| 1,048,576 | — | — | 9,629,128,722 |
| 1,000 | — | — | 146,025,509 |
| 1,080 | — | — | 347,906,800 |
| 1,920 | — | — | 575,904,340 |
| 1,009 | — | — | 136,583,868 |
| 1,296 | — | — | 264,790,950 |
| 10,007 | — | — | 1,551,445,879 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
