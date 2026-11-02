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
| 256 (2⁸) | 280 (36.6) | 271 (37.8) | 2,500 (4.1) | 1,964 (5.2) | 1,692 (6.1) | 1.03× | **≥ parity** |
| 1,024 (2¹⁰) | 1,258 (40.7) | 1,285 (39.8) | 5,317 (9.6) | 4,063 (12.6) | 8,759 (5.8) | 0.98× | **≥ parity** |
| 4,096 (2¹²) | 6,694 (36.7) | 7,136 (34.4) | 19,748 (12.4) | 15,270 (16.1) | 41,310 (5.9) | 0.94× | **≥ parity** |
| 65,536 (2¹⁶) | 229,623 (22.8) | 194,300 (27.0) | 521,012 (10.1) | 454,495 (11.5) | 975,992 (5.4) | 1.18× | lags FFTW 1.18× |
| 1,048,576 (2²⁰) | 4,849,512 (21.6) | 5,569,859 (18.8) | 10,415,875 (10.1) | 7,195,952 (14.6) | 21,135,886 (5.0) | 0.87× | **≥ parity** |
| 1,000 (2³·5³) | 1,521 (32.8) | 1,632 (30.5) | 5,561 (9.0) | 4,531 (11.0) | 9,226 (5.4) | 0.93× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,655 (32.9) | 1,781 (30.6) | 6,309 (8.6) | 4,846 (11.2) | 10,849 (5.0) | 0.93× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,864 (36.6) | 3,015 (34.7) | 9,969 (10.5) | 8,124 (12.9) | 18,551 (5.6) | 0.95× | **≥ parity** |
| 1,009 (prime) | 6,045 (8.3) | 10,766 (4.7) | 30,178 (1.7) | 17,993 (2.8) | 525,967 (0.1) | 0.56× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 2,028 (33.0) | 2,286 (29.3) | 7,236 (9.3) | 5,684 (11.8) | 12,876 (5.2) | 0.89× | **≥ parity** |
| 10,007 (prime) | 115,378 (5.8) | 143,005 (4.6) | 306,781 (2.2) | 281,930 (2.4) | 69,458,130 (0.0) | 0.81× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 202 (25.4) | 172 (29.9) | 2,136 (2.4) | 1,992 (2.6) | 771 (6.6) | 1.18× | lags FFTW 1.18× |
| 1,024 (2¹⁰) | 801 (32.0) | 813 (31.5) | 3,805 (6.7) | 3,231 (7.9) | 3,996 (6.4) | 0.99× | **≥ parity** |
| 4,096 (2¹²) | 3,575 (34.4) | 3,824 (32.1) | 11,240 (10.9) | 9,671 (12.7) | 17,501 (7.0) | 0.93× | **≥ parity** |
| 65,536 (2¹⁶) | 121,870 (21.5) | 103,109 (25.4) | 224,343 (11.7) | 255,666 (10.3) | 424,269 (6.2) | 1.18× | lags FFTW 1.18× |
| 1,048,576 (2²⁰) | 2,495,219 (21.0) | 2,173,648 (24.1) | 4,395,457 (11.9) | 5,794,153 (9.0) | 9,107,621 (5.8) | 1.15× | lags FFTW 1.15× |
| 1,000 (2³·5³) | 954 (26.1) | 958 (26.0) | 3,953 (6.3) | 3,256 (7.7) | 4,061 (6.1) | 1.00× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,052 (25.9) | 1,035 (26.3) | 4,113 (6.6) | 3,730 (7.3) | 4,711 (5.8) | 1.02× | **≥ parity** |
| 1,920 (2⁷·3·5) | 1,749 (29.9) | 1,743 (30.0) | 5,853 (8.9) | 5,228 (10.0) | 8,176 (6.4) | 1.00× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 207 (24.7) | 191 (26.8) | 1.09× | lags FFTW 1.09× |
| 1,024 (2¹⁰) | 803 (31.9) | 861 (29.7) | 0.93× | **≥ parity** |
| 4,096 (2¹²) | 3,580 (34.3) | 4,581 (26.8) | 0.78× | **≥ parity** |
| 65,536 (2¹⁶) | 136,682 (19.2) | 110,093 (23.8) | 1.24× | lags FFTW 1.24× |
| 1,048,576 (2²⁰) | 2,908,403 (18.0) | 2,448,445 (21.4) | 1.19× | lags FFTW 1.19× |
| 1,000 (2³·5³) | 950 (26.2) | 1,029 (24.2) | 0.92× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,038 (26.2) | 1,083 (25.1) | 0.96× | **≥ parity** |
| 1,920 (2⁷·3·5) | 1,718 (30.5) | 1,987 (26.3) | 0.86× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 6,480 (37.9) | 7,491 (32.8) | 5,061 (48.6) | 18,218 (13.5) | 12,990 (18.9) | 1.28× | lags FFTW 1.28× |
| 128x128 | 45,779 (25.1) | 48,942 (23.4) | 39,040 (29.4) | 75,017 (15.3) | 62,337 (18.4) | 1.17× | lags FFTW 1.17× |
| 256x256 | 242,094 (21.7) | 253,489 (20.7) | 190,857 (27.5) | 339,280 (15.5) | 200,058 (26.2) | 1.27× | lags FFTW 1.27× |
| 512x512 | 862,072 (27.4) | 901,300 (26.2) | 840,930 (28.1) | 1,652,542 (14.3) | 939,808 (25.1) | 1.03× | **≥ parity** |
| 1024x1024 | 4,855,322 (21.6) | 5,286,708 (19.8) | 3,826,875 (27.4) | 8,077,833 (13.0) | 5,233,740 (20.0) | 1.27× | lags FFTW 1.27× |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 3,708 | 2,457 | 3,930,000 |
| 1,024 | 13,351 | 8,743 | 6,060,000 |
| 4,096 | 51,214 | 32,762 | 21,401,000 |
| 65,536 | 703,646 | 471,212 | 445,383,000 |
| 1,048,576 | 9,808,669 | 6,224,064 | 1,676,355,000 |
| 1,000 | 13,501 | 8,273 | 7,872,000 |
| 1,080 | 14,804 | 9,075 | 18,595,000 |
| 1,920 | 24,805 | 15,609 | 31,327,000 |
| 1,009 | 25,673 | 9,377 | 12,347,000 |
| 1,296 | 17,575 | 10,873 | 14,747,000 |
| 10,007 | 389,717 | 89,210 | 209,342,000 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 16/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 23/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- 2-D 64x64: 1.28×
- 2-D 1024x1024: 1.27×
- 2-D 256x256: 1.27×
- real 65,536 (2¹⁶): 1.18×
- complex 65,536 (2¹⁶): 1.18×
- real 256 (2⁸): 1.18×
- 2-D 128x128: 1.17×
- real 1,048,576 (2²⁰): 1.15×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
