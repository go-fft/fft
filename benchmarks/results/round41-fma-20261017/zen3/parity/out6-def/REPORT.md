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
| 256 (2⁸) | — (0.0) | 276 (37.1) | 6,872 (1.5) | 5,197 (2.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,595 (32.1) | 13,192 (3.9) | 9,502 (5.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,468 (23.5) | 44,380 (5.5) | 33,316 (7.4) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 304,452 (17.2) | 1,916,699 (2.7) | 973,230 (5.4) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,326,460 (9.3) | 23,473,855 (4.5) | 14,006,013 (7.5) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,279 (21.9) | 13,715 (3.6) | 11,213 (4.4) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,375 (22.9) | 14,483 (3.8) | 10,729 (5.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,047 (25.9) | 21,895 (4.8) | 16,155 (6.5) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,121 (2.5) | 60,624 (0.8) | 38,401 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,214 (20.8) | 16,382 (4.1) | 12,309 (5.4) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 204,604 (3.2) | 661,561 (1.0) | 480,066 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 223 (23.0) | 6,246 (0.8) | 5,407 (0.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 997 (25.7) | 9,765 (2.6) | 8,381 (3.1) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,264 (23.3) | 25,408 (4.8) | 22,221 (5.5) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 151,294 (17.3) | 434,535 (6.0) | 491,255 (5.3) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,835,875 (13.7) | 8,757,447 (6.0) | 10,220,621 (5.1) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,339 (18.6) | 10,475 (2.4) | 8,587 (2.9) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,320 (20.6) | 10,991 (2.5) | 9,020 (3.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,364 (22.1) | 14,259 (3.7) | 11,964 (4.4) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 347 (14.8) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,172 (21.8) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,850 (21.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 170,747 (15.4) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,063,444 (12.9) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,427 (17.5) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,619 (16.8) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,650 (19.8) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 11,084 (22.2) | 47,318 (5.2) | 33,696 (7.3) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 62,044 (18.5) | 145,042 (7.9) | 125,831 (9.1) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 280,929 (18.7) | 722,591 (7.3) | 454,102 (11.5) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,327,188 (17.8) | 3,177,705 (7.4) | 1,993,755 (11.8) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,529,536 (16.1) | 14,395,876 (7.3) | 9,714,500 (10.8) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,264,093 |
| 1,024 | — | — | 132,766,660 |
| 4,096 | — | — | 238,887,399 |
| 65,536 | — | — | 2,282,532,397 |
| 1,048,576 | — | — | 4,188,763,483 |
| 1,000 | — | — | 152,732,209 |
| 1,080 | — | — | 365,144,606 |
| 1,920 | — | — | 577,748,402 |
| 1,009 | — | — | 143,500,386 |
| 1,296 | — | — | 270,849,803 |
| 10,007 | — | — | 1,221,490,395 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
