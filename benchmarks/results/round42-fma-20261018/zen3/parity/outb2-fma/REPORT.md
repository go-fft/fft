# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3, cfarm420).
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
| 256 (2⁸) | — (0.0) | 273 (37.6) | 6,726 (1.5) | 5,273 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,538 (33.3) | 13,353 (3.8) | 9,605 (5.3) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,367 (23.7) | 43,990 (5.6) | 33,478 (7.3) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 296,342 (17.7) | 2,063,324 (2.5) | 914,746 (5.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,424,015 (9.2) | 22,944,860 (4.6) | 14,346,247 (7.3) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,083 (23.9) | 13,543 (3.7) | 10,062 (5.0) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,310 (23.6) | 14,601 (3.7) | 10,832 (5.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,118 (25.4) | 21,927 (4.8) | 16,398 (6.4) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 19,861 (2.5) | 61,383 (0.8) | 38,432 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,000 (22.3) | 16,277 (4.1) | 12,346 (5.4) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 205,350 (3.2) | 660,416 (1.0) | 473,199 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 225 (22.8) | 5,970 (0.9) | 5,346 (1.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 989 (25.9) | 9,649 (2.7) | 8,636 (3.0) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,271 (23.3) | 25,212 (4.9) | 22,519 (5.5) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 155,270 (16.9) | 406,960 (6.4) | 498,107 (5.3) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,617,548 (14.5) | 8,504,447 (6.2) | 10,311,483 (5.1) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,234 (20.2) | 10,183 (2.4) | 8,716 (2.9) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,333 (20.4) | 10,752 (2.5) | 9,155 (3.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,360 (22.2) | 14,630 (3.6) | 12,472 (4.2) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 355 (14.4) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,156 (22.2) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,745 (21.4) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 169,094 (15.5) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,069,563 (12.9) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,575 (15.8) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,546 (17.6) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,740 (19.1) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 11,761 (20.9) | 49,771 (4.9) | 36,076 (6.8) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 63,588 (18.0) | 153,470 (7.5) | 127,714 (9.0) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 284,758 (18.4) | 624,695 (8.4) | 454,337 (11.5) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,312,908 (18.0) | 2,749,648 (8.6) | 1,908,233 (12.4) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,732,967 (15.6) | 13,335,684 (7.9) | 9,446,088 (11.1) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,627,952 |
| 1,024 | — | — | 132,040,280 |
| 4,096 | — | — | 239,196,439 |
| 65,536 | — | — | 2,141,769,233 |
| 1,048,576 | — | — | 4,180,203,680 |
| 1,000 | — | — | 151,671,958 |
| 1,080 | — | — | 363,245,795 |
| 1,920 | — | — | 574,793,921 |
| 1,009 | — | — | 142,164,755 |
| 1,296 | — | — | 264,594,601 |
| 10,007 | — | — | 1,210,941,780 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
