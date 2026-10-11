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
| 256 (2⁸) | — (0.0) | 284 (36.1) | 6,785 (1.5) | 5,392 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,546 (33.1) | 12,922 (4.0) | 9,863 (5.2) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,823 (22.7) | 43,439 (5.7) | 33,696 (7.3) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 296,597 (17.7) | 1,942,235 (2.7) | 905,725 (5.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,964,296 (8.8) | 39,406,292 (2.7) | 13,956,549 (7.5) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,126 (23.4) | 13,854 (3.6) | 10,342 (4.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,332 (23.3) | 14,712 (3.7) | 10,888 (5.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,869 (27.1) | 22,143 (4.7) | 16,398 (6.4) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,526 (2.5) | 61,734 (0.8) | 38,636 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,316 (20.2) | 16,106 (4.2) | 12,413 (5.4) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 200,679 (3.3) | 638,143 (1.0) | 463,604 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 228 (22.5) | 5,959 (0.9) | 5,400 (0.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 963 (26.6) | 9,294 (2.8) | 8,401 (3.0) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,212 (23.6) | 24,902 (4.9) | 21,950 (5.6) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 166,610 (15.7) | 402,926 (6.5) | 485,681 (5.4) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,903,167 (13.4) | 8,594,476 (6.1) | 10,360,873 (5.1) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,410 (17.7) | 10,370 (2.4) | 8,816 (2.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,488 (18.3) | 10,870 (2.5) | 9,187 (3.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,503 (20.9) | 14,332 (3.7) | 12,920 (4.1) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 362 (14.1) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,270 (20.2) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,833 (21.1) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 170,441 (15.4) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,123,647 (12.7) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,468 (17.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,579 (17.2) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,806 (18.7) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 11,385 (21.6) | 49,164 (5.0) | 33,804 (7.3) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 61,623 (18.6) | 149,117 (7.7) | 129,916 (8.8) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 389,317 (13.5) | 673,249 (7.8) | 460,618 (11.4) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,315,516 (17.9) | 2,862,465 (8.2) | 1,922,365 (12.3) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,770,498 (15.5) | 14,152,206 (7.4) | 9,474,822 (11.1) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 72,316,313 |
| 1,024 | — | — | 131,220,010 |
| 4,096 | — | — | 248,280,242 |
| 65,536 | — | — | 2,138,370,392 |
| 1,048,576 | — | — | 4,436,116,276 |
| 1,000 | — | — | 152,083,399 |
| 1,080 | — | — | 374,467,591 |
| 1,920 | — | — | 566,657,327 |
| 1,009 | — | — | 143,041,275 |
| 1,296 | — | — | 265,711,350 |
| 10,007 | — | — | 1,210,014,110 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
