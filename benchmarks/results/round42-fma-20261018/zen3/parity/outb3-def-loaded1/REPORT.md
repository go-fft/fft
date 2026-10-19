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
| 256 (2⁸) | — (0.0) | 289 (35.4) | 6,988 (1.5) | 5,205 (2.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,645 (31.1) | 13,414 (3.8) | 9,509 (5.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 11,685 (21.0) | 46,090 (5.3) | 35,366 (6.9) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 304,036 (17.2) | 1,899,145 (2.8) | 927,210 (5.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 13,236,688 (7.9) | 23,587,436 (4.4) | 14,450,225 (7.3) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,107 (23.6) | 13,696 (3.6) | 9,739 (5.1) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,320 (23.5) | 14,402 (3.8) | 10,588 (5.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,279 (24.5) | 22,084 (4.7) | 16,228 (6.5) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 21,875 (2.3) | 61,254 (0.8) | 38,368 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,282 (20.4) | 17,211 (3.9) | 12,341 (5.4) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 250,070 (2.7) | 648,036 (1.0) | 479,487 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 222 (23.1) | 6,109 (0.8) | 5,317 (1.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,012 (25.3) | 9,614 (2.7) | 8,189 (3.1) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,242 (23.4) | 24,858 (4.9) | 22,083 (5.6) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 155,202 (16.9) | 438,724 (6.0) | 501,055 (5.2) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,625,417 (14.5) | 8,272,430 (6.3) | 9,991,363 (5.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,290 (19.3) | 10,749 (2.3) | 8,635 (2.9) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,326 (20.5) | 10,874 (2.5) | 9,027 (3.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,765 (18.9) | 14,274 (3.7) | 12,100 (4.3) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 299 (17.1) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,239 (20.7) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 6,119 (20.1) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 177,437 (14.8) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,101,391 (12.8) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,409 (17.7) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,598 (17.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,737 (19.1) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,723 (22.9) | 51,586 (4.8) | 32,502 (7.6) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 63,742 (18.0) | 149,845 (7.7) | 129,535 (8.9) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 286,673 (18.3) | 672,810 (7.8) | 457,868 (11.5) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,354,270 (17.4) | 3,159,261 (7.5) | 1,896,585 (12.4) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,670,730 (15.7) | 14,443,482 (7.3) | 9,304,595 (11.3) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 73,229,833 |
| 1,024 | — | — | 132,830,870 |
| 4,096 | — | — | 248,836,273 |
| 65,536 | — | — | 2,285,834,079 |
| 1,048,576 | — | — | 4,442,191,429 |
| 1,000 | — | — | 153,739,000 |
| 1,080 | — | — | 367,135,477 |
| 1,920 | — | — | 587,426,407 |
| 1,009 | — | — | 151,046,628 |
| 1,296 | — | — | 288,617,191 |
| 10,007 | — | — | 1,437,416,933 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
