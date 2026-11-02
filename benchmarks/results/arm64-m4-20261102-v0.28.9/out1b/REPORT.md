# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Apple M4 Max (16-core: 12P+4E), macOS 27.2, arm64, GOMAXPROCS=1 (not pinned).
- **Toolchains**: go1.27.3 darwin/arm64; native **FFTW fftw-3.3.11-neon** (FFTW 3.3.11, pkgx bottle (go-pkgx/packages#399), NEON, static libfftw3.a linked from C); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 279 (36.7) | 289 (35.5) | 2,686 (3.8) | 1,951 (5.2) | 1,692 (6.1) | 0.97× | **≥ parity** |
| 1,024 (2¹⁰) | 1,259 (40.7) | 1,303 (39.3) | 5,341 (9.6) | 4,074 (12.6) | 8,785 (5.8) | 0.97× | **≥ parity** |
| 4,096 (2¹²) | 6,691 (36.7) | 7,326 (33.5) | 19,827 (12.4) | 15,290 (16.1) | 41,764 (5.9) | 0.91× | **≥ parity** |
| 65,536 (2¹⁶) | 263,163 (19.9) | 193,060 (27.2) | 516,494 (10.2) | 475,490 (11.0) | 986,438 (5.3) | 1.36× | lags FFTW 1.36× |
| 1,048,576 (2²⁰) | 5,658,909 (18.5) | 5,646,812 (18.6) | 10,489,405 (10.0) | 7,153,902 (14.7) | 21,629,277 (4.8) | 1.00× | **≥ parity** |
| 1,000 (2³·5³) | 1,519 (32.8) | 1,612 (30.9) | 5,611 (8.9) | 4,509 (11.1) | 9,243 (5.4) | 0.94× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,653 (32.9) | 1,936 (28.1) | 6,341 (8.6) | 4,820 (11.3) | 10,870 (5.0) | 0.85× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,859 (36.6) | 3,347 (31.3) | 10,309 (10.2) | 8,108 (12.9) | 18,625 (5.6) | 0.85× | **≥ parity** |
| 1,009 (prime) | 6,044 (8.3) | 10,855 (4.6) | 30,232 (1.7) | 17,948 (2.8) | 506,487 (0.1) | 0.56× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 2,026 (33.1) | 2,277 (29.4) | 7,259 (9.2) | 5,668 (11.8) | 12,886 (5.2) | 0.89× | **≥ parity** |
| 10,007 (prime) | 131,075 (5.1) | 132,296 (5.0) | 306,414 (2.2) | 264,836 (2.5) | 68,900,841 (0.0) | 0.99× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 202 (25.3) | 172 (29.8) | 2,149 (2.4) | 2,013 (2.5) | 771 (6.6) | 1.18× | lags FFTW 1.18× |
| 1,024 (2¹⁰) | 802 (31.9) | 786 (32.6) | 3,754 (6.8) | 3,236 (7.9) | 4,001 (6.4) | 1.02× | **≥ parity** |
| 4,096 (2¹²) | 3,590 (34.2) | 3,900 (31.5) | 11,159 (11.0) | 9,656 (12.7) | 17,572 (7.0) | 0.92× | **≥ parity** |
| 65,536 (2¹⁶) | 136,963 (19.1) | 105,943 (24.7) | 212,191 (12.4) | 251,985 (10.4) | 424,229 (6.2) | 1.29× | lags FFTW 1.29× |
| 1,048,576 (2²⁰) | 2,489,070 (21.1) | 2,242,727 (23.4) | 4,287,890 (12.2) | 5,801,956 (9.0) | 9,057,791 (5.8) | 1.11× | lags FFTW 1.11× |
| 1,000 (2³·5³) | 954 (26.1) | 946 (26.3) | 3,948 (6.3) | 3,261 (7.6) | 4,022 (6.2) | 1.01× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,038 (26.2) | 1,037 (26.2) | 4,101 (6.6) | 3,776 (7.2) | 4,709 (5.8) | 1.00× | **≥ parity** |
| 1,920 (2⁷·3·5) | 1,727 (30.3) | 1,780 (29.4) | 6,202 (8.4) | 5,288 (9.9) | 8,181 (6.4) | 0.97× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 206 (24.8) | 193 (26.6) | 1.07× | lags FFTW 1.07× |
| 1,024 (2¹⁰) | 803 (31.9) | 864 (29.6) | 0.93× | **≥ parity** |
| 4,096 (2¹²) | 3,629 (33.9) | 4,400 (27.9) | 0.82× | **≥ parity** |
| 65,536 (2¹⁶) | 136,497 (19.2) | 113,039 (23.2) | 1.21× | lags FFTW 1.21× |
| 1,048,576 (2²⁰) | 2,872,551 (18.3) | 2,441,000 (21.5) | 1.18× | lags FFTW 1.18× |
| 1,000 (2³·5³) | 949 (26.2) | 1,013 (24.6) | 0.94× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,039 (26.2) | 1,088 (25.0) | 0.96× | **≥ parity** |
| 1,920 (2⁷·3·5) | 1,725 (30.3) | 1,917 (27.3) | 0.90× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 6,603 (37.2) | 7,488 (32.8) | 5,060 (48.6) | 18,903 (13.0) | 13,220 (18.6) | 1.31× | lags FFTW 1.31× |
| 128x128 | 46,209 (24.8) | 49,256 (23.3) | 68,794 (16.7) | 77,055 (14.9) | 62,459 (18.4) | 0.67× | **≥ parity** |
| 256x256 | 248,690 (21.1) | 253,064 (20.7) | 190,370 (27.5) | 342,080 (15.3) | 199,151 (26.3) | 1.31× | lags FFTW 1.31× |
| 512x512 | 921,938 (25.6) | 904,384 (26.1) | 883,098 (26.7) | 1,592,350 (14.8) | 930,533 (25.4) | 1.04× | **≥ parity** |
| 1024x1024 | 4,699,131 (22.3) | 5,059,453 (20.7) | 3,898,672 (26.9) | 7,783,005 (13.5) | 5,216,794 (20.1) | 1.21× | lags FFTW 1.21× |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 3,865 | 2,428 | 3,933,000 |
| 1,024 | 13,547 | 8,779 | 6,052,000 |
| 4,096 | 51,157 | 32,392 | 21,287,000 |
| 65,536 | 692,975 | 470,325 | 446,532,000 |
| 1,048,576 | 10,102,053 | 6,268,646 | 1,664,329,000 |
| 1,000 | 13,360 | 8,316 | 7,836,000 |
| 1,080 | 15,218 | 9,066 | 18,815,000 |
| 1,920 | 25,285 | 15,577 | 31,262,000 |
| 1,009 | 25,499 | 9,496 | 12,426,000 |
| 1,296 | 17,605 | 10,832 | 14,813,000 |
| 10,007 | 389,819 | 87,373 | 206,365,000 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 17/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 23/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 65,536 (2¹⁶): 1.36×
- 2-D 256x256: 1.31×
- 2-D 64x64: 1.31×
- real 65,536 (2¹⁶): 1.29×
- 2-D 1024x1024: 1.21×
- real 256 (2⁸): 1.18×
- real 1,048,576 (2²⁰): 1.11×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
