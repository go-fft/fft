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
| 256 (2⁸) | — (0.0) | 296 (34.6) | 6,740 (1.5) | 5,416 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,528 (33.5) | 13,139 (3.9) | 9,765 (5.2) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,188 (24.1) | 45,431 (5.4) | 33,385 (7.4) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 297,050 (17.6) | 1,893,127 (2.8) | 920,217 (5.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,376,919 (9.2) | 25,609,894 (4.1) | 13,652,238 (7.7) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,112 (23.6) | 13,653 (3.6) | 10,282 (4.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,240 (24.3) | 14,189 (3.8) | 10,835 (5.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,793 (27.6) | 21,992 (4.8) | 17,401 (6.0) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,569 (2.4) | 60,815 (0.8) | 39,090 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,125 (21.4) | 16,318 (4.1) | 12,473 (5.4) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 206,232 (3.2) | 645,532 (1.0) | 469,931 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 227 (22.6) | 6,100 (0.8) | 5,469 (0.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 980 (26.1) | 9,561 (2.7) | 8,465 (3.0) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,266 (23.3) | 25,360 (4.8) | 22,199 (5.5) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 161,090 (16.3) | 397,809 (6.6) | 496,084 (5.3) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,597,559 (14.6) | 8,394,802 (6.2) | 10,171,496 (5.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,364 (18.3) | 10,383 (2.4) | 8,899 (2.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,332 (20.4) | 11,064 (2.5) | 9,175 (3.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,930 (17.9) | 14,406 (3.6) | 12,265 (4.3) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 356 (14.4) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,178 (21.7) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,877 (20.9) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 170,903 (15.3) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,506,128 (11.6) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,451 (17.2) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,514 (18.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,688 (19.5) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,928 (22.5) | 49,086 (5.0) | 33,520 (7.3) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 60,438 (19.0) | 153,745 (7.5) | 125,219 (9.2) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 301,738 (17.4) | 697,311 (7.5) | 455,099 (11.5) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,280,911 (18.4) | 2,966,106 (8.0) | 1,913,735 (12.3) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,210,467 (16.9) | 13,101,029 (8.0) | 9,389,594 (11.2) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 72,241,942 |
| 1,024 | — | — | 132,547,221 |
| 4,096 | — | — | 237,924,008 |
| 65,536 | — | — | 2,178,234,969 |
| 1,048,576 | — | — | 4,066,353,608 |
| 1,000 | — | — | 153,487,879 |
| 1,080 | — | — | 372,372,489 |
| 1,920 | — | — | 597,792,052 |
| 1,009 | — | — | 148,715,407 |
| 1,296 | — | — | 270,483,023 |
| 10,007 | — | — | 1,246,608,247 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
