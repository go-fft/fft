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
| 256 (2⁸) | — (0.0) | 282 (36.4) | 6,747 (1.5) | 5,404 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,551 (33.0) | 12,771 (4.0) | 9,509 (5.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,755 (22.9) | 45,146 (5.4) | 33,362 (7.4) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 298,387 (17.6) | 1,880,265 (2.8) | 914,338 (5.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 12,088,202 (8.7) | 38,572,099 (2.7) | 13,737,284 (7.6) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,016 (24.7) | 13,721 (3.6) | 10,074 (4.9) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,308 (23.6) | 14,413 (3.8) | 10,903 (5.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,740 (22.1) | 21,683 (4.8) | 16,407 (6.4) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,115 (2.5) | 61,297 (0.8) | 38,420 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,014 (22.2) | 16,173 (4.1) | 12,488 (5.4) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 203,507 (3.3) | 638,390 (1.0) | 475,577 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 226 (22.7) | 6,032 (0.8) | 5,428 (0.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 955 (26.8) | 9,374 (2.7) | 8,277 (3.1) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,260 (23.4) | 24,941 (4.9) | 22,336 (5.5) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 150,988 (17.4) | 418,812 (6.3) | 502,512 (5.2) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,737,839 (14.0) | 8,514,785 (6.2) | 10,661,110 (4.9) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,325 (18.8) | 10,188 (2.4) | 8,786 (2.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,336 (20.4) | 10,862 (2.5) | 9,000 (3.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,387 (21.9) | 14,370 (3.6) | 12,187 (4.3) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 302 (17.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,169 (21.9) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,736 (21.4) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 165,614 (15.8) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,347,062 (12.1) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,482 (16.8) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,544 (17.6) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,708 (19.3) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,602 (23.2) | 46,576 (5.3) | 32,563 (7.5) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 62,490 (18.4) | 153,613 (7.5) | 128,612 (8.9) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 289,671 (18.1) | 711,295 (7.4) | 457,847 (11.5) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,320,941 (17.9) | 3,062,095 (7.7) | 1,891,046 (12.5) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,599,211 (15.9) | 13,535,943 (7.7) | 9,878,415 (10.6) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,855,343 |
| 1,024 | — | — | 130,890,239 |
| 4,096 | — | — | 240,196,830 |
| 65,536 | — | — | 2,250,494,292 |
| 1,048,576 | — | — | 4,278,041,444 |
| 1,000 | — | — | 151,479,369 |
| 1,080 | — | — | 365,121,916 |
| 1,920 | — | — | 585,834,936 |
| 1,009 | — | — | 143,780,315 |
| 1,296 | — | — | 275,203,035 |
| 10,007 | — | — | 1,206,809,288 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
