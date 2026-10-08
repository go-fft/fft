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
| 256 (2⁸) | 345 (29.7) | 274 (37.4) | 7,063 (1.4) | 5,335 (1.9) | 5,410 (1.9) | 1.26× | lags FFTW 1.26× |
| 1,024 (2¹⁰) | 1,713 (29.9) | 1,537 (33.3) | 13,526 (3.8) | 9,562 (5.4) | 28,784 (1.8) | 1.11× | lags FFTW 1.11× |
| 4,096 (2¹²) | 9,720 (25.3) | 10,312 (23.8) | 44,600 (5.5) | 32,901 (7.5) | 131,633 (1.9) | 0.94× | **≥ parity** |
| 65,536 (2¹⁶) | 259,254 (20.2) | 292,016 (18.0) | 1,868,270 (2.8) | 913,689 (5.7) | 2,874,539 (1.8) | 0.89× | **≥ parity** |
| 1,048,576 (2²⁰) | 5,939,038 (17.7) | 11,334,112 (9.3) | 26,179,601 (4.0) | 15,173,396 (6.9) | 59,191,622 (1.8) | 0.52× | **≥ parity** |
| 1,000 (2³·5³) | 2,188 (22.8) | 2,014 (24.7) | 13,972 (3.6) | 10,333 (4.8) | 30,452 (1.6) | 1.09× | lags FFTW 1.09× |
| 1,080 (2³·3³·5) | 2,550 (21.3) | 2,382 (22.8) | 14,956 (3.6) | 11,106 (4.9) | 35,309 (1.5) | 1.07× | lags FFTW 1.07× |
| 1,920 (2⁷·3·5) | 3,869 (27.1) | 3,840 (27.3) | 21,992 (4.8) | 16,235 (6.4) | 61,246 (1.7) | 1.01× | **≥ parity** |
| 1,009 (prime) | 13,019 (3.9) | 20,272 (2.5) | 62,977 (0.8) | 38,482 (1.3) | 1,152,820 (0.0) | 0.64× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,116 (21.5) | 2,984 (22.4) | 17,181 (3.9) | 12,950 (5.2) | 41,458 (1.6) | 1.04× | **≥ parity** |
| 10,007 (prime) | 216,059 (3.1) | 215,063 (3.1) | 684,450 (1.0) | 470,831 (1.4) | 120,235,487 (0.0) | 1.00× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 278 (18.5) | 227 (22.6) | 6,175 (0.8) | 5,351 (1.0) | 2,574 (2.0) | 1.22× | lags FFTW 1.22× |
| 1,024 (2¹⁰) | 1,035 (24.7) | 993 (25.8) | 10,395 (2.5) | 8,569 (3.0) | 12,994 (2.0) | 1.04× | **≥ parity** |
| 4,096 (2¹²) | 5,641 (21.8) | 5,329 (23.1) | 25,022 (4.9) | 22,116 (5.6) | 62,886 (2.0) | 1.06× | lags FFTW 1.06× |
| 65,536 (2¹⁶) | 133,306 (19.7) | 163,042 (16.1) | 401,412 (6.5) | 495,309 (5.3) | 1,302,300 (2.0) | 0.82× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,798,178 (18.7) | 3,720,999 (14.1) | 8,742,824 (6.0) | 10,328,897 (5.1) | 26,567,251 (2.0) | 0.75× | **≥ parity** |
| 1,000 (2³·5³) | 1,322 (18.8) | 1,229 (20.3) | 10,574 (2.4) | 8,552 (2.9) | 13,629 (1.8) | 1.08× | lags FFTW 1.08× |
| 1,080 (2³·3³·5) | 1,500 (18.1) | 1,309 (20.8) | 11,426 (2.4) | 8,983 (3.0) | 16,528 (1.6) | 1.15× | lags FFTW 1.15× |
| 1,920 (2⁷·3·5) | 2,475 (21.2) | 2,300 (22.8) | 14,717 (3.6) | 12,252 (4.3) | 28,341 (1.8) | 1.08× | lags FFTW 1.08× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 292 (17.6) | 348 (14.7) | 0.84× | **≥ parity** |
| 1,024 (2¹⁰) | 1,109 (23.1) | 1,194 (21.4) | 0.93× | **≥ parity** |
| 4,096 (2¹²) | 5,777 (21.3) | 5,720 (21.5) | 1.01× | **≥ parity** |
| 65,536 (2¹⁶) | 136,388 (19.2) | 165,899 (15.8) | 0.82× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,859,125 (18.3) | 4,067,449 (12.9) | 0.70× | **≥ parity** |
| 1,000 (2³·5³) | 1,402 (17.8) | 1,423 (17.5) | 0.99× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,657 (16.4) | 1,534 (17.7) | 1.08× | lags FFTW 1.08× |
| 1,920 (2⁷·3·5) | 2,528 (20.7) | 2,782 (18.8) | 0.91× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 11,851 (20.7) | 15,044 (16.3) | 10,654 (23.1) | 50,121 (4.9) | 39,282 (6.3) | 1.11× | lags FFTW 1.11× |
| 128x128 | 52,671 (21.8) | 70,651 (16.2) | 62,349 (18.4) | 143,863 (8.0) | 125,115 (9.2) | 0.84× | **≥ parity** |
| 256x256 | 285,631 (18.4) | 319,385 (16.4) | 288,224 (18.2) | 720,633 (7.3) | 454,984 (11.5) | 0.99× | **≥ parity** |
| 512x512 | 1,315,207 (17.9) | 1,531,097 (15.4) | 1,347,993 (17.5) | 3,032,459 (7.8) | 1,909,039 (12.4) | 0.98× | **≥ parity** |
| 1024x1024 | 6,118,523 (17.1) | 7,272,664 (14.4) | 6,337,853 (16.5) | 13,897,244 (7.5) | 9,386,778 (11.2) | 0.97× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 72,030,393 |
| 1,024 | — | — | 137,713,013 |
| 4,096 | — | — | 245,486,852 |
| 65,536 | — | — | 2,154,160,409 |
| 1,048,576 | — | — | 4,062,388,286 |
| 1,000 | — | — | 155,792,441 |
| 1,080 | — | — | 368,142,917 |
| 1,920 | — | — | 579,209,713 |
| 1,009 | — | — | 141,118,464 |
| 1,296 | — | — | 272,136,394 |
| 10,007 | — | — | 1,286,269,905 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 14/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 256 (2⁸): 1.26×
- real 256 (2⁸): 1.22×
- real 1,080 (2³·3³·5): 1.15×
- complex 1,024 (2¹⁰): 1.11×
- 2-D 64x64: 1.11×
- complex 1,000 (2³·5³): 1.09×
- real 1,000 (2³·5³): 1.08×
- real 1,920 (2⁷·3·5): 1.08×
- complex 1,080 (2³·3³·5): 1.07×
- real 4,096 (2¹²): 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
