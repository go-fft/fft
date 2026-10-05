# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3), cfarm420, one pinned core.
- **Toolchains**: go1.27.1 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128** (3.3.10 built from source with --enable-sse2 --enable-avx --enable-avx2 --enable-fma); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 392 (26.1) | 279 (36.7) | 6,762 (1.5) | 5,129 (2.0) | 6,506 (1.6) | 1.40× | lags FFTW 1.40× |
| 1,024 (2¹⁰) | 1,929 (26.5) | 1,513 (33.8) | 13,056 (3.9) | 9,383 (5.5) | 28,984 (1.8) | 1.27× | lags FFTW 1.27× |
| 4,096 (2¹²) | 13,465 (18.3) | 10,439 (23.5) | 43,873 (5.6) | 33,408 (7.4) | 134,432 (1.8) | 1.29× | lags FFTW 1.29× |
| 65,536 (2¹⁶) | 289,276 (18.1) | 294,760 (17.8) | 1,896,843 (2.8) | 914,724 (5.7) | 2,886,940 (1.8) | 0.98× | **≥ parity** |
| 1,048,576 (2²⁰) | 6,268,394 (16.7) | 11,732,105 (8.9) | 23,083,952 (4.5) | 13,759,221 (7.6) | 58,986,505 (1.8) | 0.53× | **≥ parity** |
| 1,000 (2³·5³) | 2,342 (21.3) | 2,032 (24.5) | 13,676 (3.6) | 9,718 (5.1) | 29,697 (1.7) | 1.15× | lags FFTW 1.15× |
| 1,080 (2³·3³·5) | 2,764 (19.7) | 2,389 (22.8) | 14,507 (3.8) | 10,348 (5.3) | 35,301 (1.5) | 1.16× | lags FFTW 1.16× |
| 1,920 (2⁷·3·5) | 4,727 (22.2) | 3,876 (27.0) | 21,852 (4.8) | 15,371 (6.8) | 63,361 (1.7) | 1.22× | lags FFTW 1.22× |
| 1,009 (prime) | 12,690 (4.0) | 20,223 (2.5) | 60,768 (0.8) | 38,194 (1.3) | 1,136,393 (0.0) | 0.63× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,533 (19.0) | 2,891 (23.2) | 16,203 (4.1) | 12,111 (5.5) | 41,909 (1.6) | 1.22× | lags FFTW 1.22× |
| 10,007 (prime) | 247,050 (2.7) | 200,754 (3.3) | 640,499 (1.0) | 471,001 (1.4) | 116,570,203 (0.0) | 1.23× | lags FFTW 1.23× |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 313 (16.4) | 225 (22.8) | 5,982 (0.9) | 5,122 (1.0) | 2,650 (1.9) | 1.39× | lags FFTW 1.39× |
| 1,024 (2¹⁰) | 1,251 (20.5) | 965 (26.5) | 9,564 (2.7) | 8,167 (3.1) | 12,637 (2.0) | 1.30× | lags FFTW 1.30× |
| 4,096 (2¹²) | 6,753 (18.2) | 5,326 (23.1) | 24,637 (5.0) | 21,969 (5.6) | 59,802 (2.1) | 1.27× | lags FFTW 1.27× |
| 65,536 (2¹⁶) | 160,913 (16.3) | 153,398 (17.1) | 397,907 (6.6) | 492,804 (5.3) | 1,314,221 (2.0) | 1.05× | **≥ parity** |
| 1,048,576 (2²⁰) | 3,306,441 (15.9) | 3,835,318 (13.7) | 8,087,968 (6.5) | 9,911,058 (5.3) | 26,249,260 (2.0) | 0.86× | **≥ parity** |
| 1,000 (2³·5³) | 1,484 (16.8) | 1,357 (18.4) | 10,541 (2.4) | 8,488 (2.9) | 13,365 (1.9) | 1.09× | lags FFTW 1.09× |
| 1,080 (2³·3³·5) | 1,691 (16.1) | 1,660 (16.4) | 10,942 (2.5) | 8,884 (3.1) | 15,593 (1.7) | 1.02× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,780 (18.8) | 2,477 (21.1) | 14,480 (3.6) | 11,979 (4.4) | 28,023 (1.9) | 1.12× | lags FFTW 1.12× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 352 (14.6) | 412 (12.4) | 0.85× | **≥ parity** |
| 1,024 (2¹⁰) | 1,288 (19.9) | 1,182 (21.7) | 1.09× | lags FFTW 1.09× |
| 4,096 (2¹²) | 6,713 (18.3) | 5,872 (20.9) | 1.14× | lags FFTW 1.14× |
| 65,536 (2¹⁶) | 165,091 (15.9) | 168,379 (15.6) | 0.98× | **≥ parity** |
| 1,048,576 (2²⁰) | 3,561,438 (14.7) | 4,152,693 (12.6) | 0.86× | **≥ parity** |
| 1,000 (2³·5³) | 1,520 (16.4) | 1,452 (17.2) | 1.05× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,723 (15.8) | 1,551 (17.5) | 1.11× | lags FFTW 1.11× |
| 1,920 (2⁷·3·5) | 2,987 (17.5) | 2,712 (19.3) | 1.10× | lags FFTW 1.10× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 11,089 (22.2) | 15,718 (15.6) | 11,235 (21.9) | 46,953 (5.2) | 31,959 (7.7) | 0.99× | **≥ parity** |
| 128x128 | 55,159 (20.8) | 70,291 (16.3) | 60,590 (18.9) | 147,585 (7.8) | 128,888 (8.9) | 0.91× | **≥ parity** |
| 256x256 | 279,317 (18.8) | 328,308 (16.0) | 281,353 (18.6) | 666,774 (7.9) | 451,205 (11.6) | 0.99× | **≥ parity** |
| 512x512 | 1,303,604 (18.1) | 1,621,838 (14.5) | 1,308,368 (18.0) | 3,138,327 (7.5) | 1,892,781 (12.5) | 1.00× | **≥ parity** |
| 1024x1024 | 6,119,465 (17.1) | 7,560,949 (13.9) | 6,487,431 (16.2) | 14,558,257 (7.2) | 9,283,255 (11.3) | 0.94× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,506,973 |
| 1,024 | — | — | 131,031,519 |
| 4,096 | — | — | 238,687,448 |
| 65,536 | — | — | 2,283,051,078 |
| 1,048,576 | — | — | 4,096,187,812 |
| 1,000 | — | — | 157,045,741 |
| 1,080 | — | — | 371,170,078 |
| 1,920 | — | — | 595,996,831 |
| 1,009 | — | — | 141,569,154 |
| 1,296 | — | — | 267,012,102 |
| 10,007 | — | — | 1,221,007,895 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 11/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 256 (2⁸): 1.40×
- real 256 (2⁸): 1.39×
- real 1,024 (2¹⁰): 1.30×
- complex 4,096 (2¹²): 1.29×
- complex 1,024 (2¹⁰): 1.27×
- real 4,096 (2¹²): 1.27×
- complex 10,007 (prime): 1.23×
- complex 1,296 (2⁴·3⁴): 1.22×
- complex 1,920 (2⁷·3·5): 1.22×
- complex 1,080 (2³·3³·5): 1.16×
- complex 1,000 (2³·5³): 1.15×
- real 1,920 (2⁷·3·5): 1.12×
- real 1,000 (2³·5³): 1.09×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
