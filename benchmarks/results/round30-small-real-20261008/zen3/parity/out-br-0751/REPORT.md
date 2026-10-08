# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3, cfarm420), core 40.
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
| 256 (2⁸) | 353 (29.0) | 285 (35.9) | 6,684 (1.5) | 5,518 (1.9) | 5,603 (1.8) | 1.24× | lags FFTW 1.24× |
| 1,024 (2¹⁰) | 1,733 (29.5) | 1,563 (32.8) | 12,853 (4.0) | 9,611 (5.3) | 28,867 (1.8) | 1.11× | lags FFTW 1.11× |
| 4,096 (2¹²) | 9,161 (26.8) | 10,264 (23.9) | 44,564 (5.5) | 33,085 (7.4) | 131,461 (1.9) | 0.89× | **≥ parity** |
| 65,536 (2¹⁶) | 261,386 (20.1) | 298,712 (17.6) | 1,869,327 (2.8) | 921,098 (5.7) | 2,845,444 (1.8) | 0.88× | **≥ parity** |
| 1,048,576 (2²⁰) | 6,013,757 (17.4) | 11,136,196 (9.4) | 22,590,235 (4.6) | 13,883,392 (7.6) | 58,047,541 (1.8) | 0.54× | **≥ parity** |
| 1,000 (2³·5³) | 2,185 (22.8) | 2,080 (24.0) | 13,751 (3.6) | 10,166 (4.9) | 31,796 (1.6) | 1.05× | lags FFTW 1.05× |
| 1,080 (2³·3³·5) | 2,594 (21.0) | 2,261 (24.1) | 14,540 (3.7) | 11,439 (4.8) | 36,774 (1.5) | 1.15× | lags FFTW 1.15× |
| 1,920 (2⁷·3·5) | 3,853 (27.2) | 3,713 (28.2) | 21,240 (4.9) | 16,506 (6.3) | 63,146 (1.7) | 1.04× | **≥ parity** |
| 1,009 (prime) | 13,035 (3.9) | 20,176 (2.5) | 60,658 (0.8) | 39,248 (1.3) | 1,159,845 (0.0) | 0.65× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,016 (22.2) | 3,532 (19.0) | 16,209 (4.1) | 12,774 (5.2) | 42,130 (1.6) | 0.85× | **≥ parity** |
| 10,007 (prime) | 208,575 (3.2) | 216,728 (3.1) | 636,252 (1.0) | 470,823 (1.4) | 124,028,095 (0.0) | 0.96× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 279 (18.3) | 236 (21.7) | 6,041 (0.8) | 5,524 (0.9) | 2,518 (2.0) | 1.18× | lags FFTW 1.18× |
| 1,024 (2¹⁰) | 1,045 (24.5) | 1,012 (25.3) | 9,628 (2.7) | 8,365 (3.1) | 12,681 (2.0) | 1.03× | **≥ parity** |
| 4,096 (2¹²) | 5,922 (20.7) | 5,435 (22.6) | 25,361 (4.8) | 22,890 (5.4) | 61,476 (2.0) | 1.09× | lags FFTW 1.09× |
| 65,536 (2¹⁶) | 139,940 (18.7) | 150,535 (17.4) | 455,666 (5.8) | 502,587 (5.2) | 1,337,350 (2.0) | 0.93× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,897,709 (18.1) | 3,807,252 (13.8) | 8,355,431 (6.3) | 10,112,933 (5.2) | 26,516,083 (2.0) | 0.76× | **≥ parity** |
| 1,000 (2³·5³) | 1,296 (19.2) | 1,255 (19.9) | 11,208 (2.2) | 9,315 (2.7) | 13,227 (1.9) | 1.03× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,476 (18.4) | 1,322 (20.6) | 11,455 (2.4) | 9,397 (2.9) | 16,049 (1.7) | 1.12× | lags FFTW 1.12× |
| 1,920 (2⁷·3·5) | 2,405 (21.8) | 2,290 (22.9) | 15,010 (3.5) | 13,461 (3.9) | 28,406 (1.8) | 1.05× | lags FFTW 1.05× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 300 (17.0) | 348 (14.7) | 0.86× | **≥ parity** |
| 1,024 (2¹⁰) | 1,114 (23.0) | 1,202 (21.3) | 0.93× | **≥ parity** |
| 4,096 (2¹²) | 5,992 (20.5) | 5,783 (21.2) | 1.04× | **≥ parity** |
| 65,536 (2¹⁶) | 136,163 (19.3) | 162,776 (16.1) | 0.84× | **≥ parity** |
| 1,048,576 (2²⁰) | 3,004,156 (17.5) | 4,162,154 (12.6) | 0.72× | **≥ parity** |
| 1,000 (2³·5³) | 1,393 (17.9) | 1,463 (17.0) | 0.95× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,599 (17.0) | 1,600 (17.0) | 1.00× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,577 (20.3) | 2,754 (19.0) | 0.94× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 11,912 (20.6) | 14,296 (17.2) | 11,507 (21.4) | 50,146 (4.9) | 32,932 (7.5) | 1.04× | **≥ parity** |
| 128x128 | 53,143 (21.6) | 64,422 (17.8) | 65,910 (17.4) | 142,132 (8.1) | 126,083 (9.1) | 0.81× | **≥ parity** |
| 256x256 | 276,647 (19.0) | 327,938 (16.0) | 283,824 (18.5) | 715,377 (7.3) | 464,287 (11.3) | 0.97× | **≥ parity** |
| 512x512 | 1,264,188 (18.7) | 1,491,245 (15.8) | 1,265,978 (18.6) | 2,831,169 (8.3) | 1,891,886 (12.5) | 1.00× | **≥ parity** |
| 1024x1024 | 6,077,566 (17.3) | 7,467,054 (14.0) | 6,289,000 (16.7) | 14,556,452 (7.2) | 9,289,554 (11.3) | 0.97× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,570,943 |
| 1,024 | — | — | 130,490,929 |
| 4,096 | — | — | 239,440,089 |
| 65,536 | — | — | 2,278,788,866 |
| 1,048,576 | — | — | 3,990,887,413 |
| 1,000 | — | — | 151,451,559 |
| 1,080 | — | — | 365,071,326 |
| 1,920 | — | — | 569,512,449 |
| 1,009 | — | — | 147,293,797 |
| 1,296 | — | — | 279,754,297 |
| 10,007 | — | — | 1,326,643,973 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 16/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 256 (2⁸): 1.24×
- real 256 (2⁸): 1.18×
- complex 1,080 (2³·3³·5): 1.15×
- real 1,080 (2³·3³·5): 1.12×
- complex 1,024 (2¹⁰): 1.11×
- real 4,096 (2¹²): 1.09×
- complex 1,000 (2³·5³): 1.05×
- real 1,920 (2⁷·3·5): 1.05×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
