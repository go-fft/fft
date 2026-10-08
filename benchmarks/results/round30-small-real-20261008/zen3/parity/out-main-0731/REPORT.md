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
| 256 (2⁸) | 361 (28.4) | 281 (36.4) | 6,806 (1.5) | 5,276 (1.9) | 5,560 (1.8) | 1.28× | lags FFTW 1.28× |
| 1,024 (2¹⁰) | 1,771 (28.9) | 1,501 (34.1) | 13,783 (3.7) | 10,476 (4.9) | 28,896 (1.8) | 1.18× | lags FFTW 1.18× |
| 4,096 (2¹²) | 9,698 (25.3) | 10,107 (24.3) | 43,402 (5.7) | 33,845 (7.3) | 131,389 (1.9) | 0.96× | **≥ parity** |
| 65,536 (2¹⁶) | 254,927 (20.6) | 295,337 (17.8) | 1,990,977 (2.6) | 905,262 (5.8) | 2,884,526 (1.8) | 0.86× | **≥ parity** |
| 1,048,576 (2²⁰) | 5,586,304 (18.8) | 12,594,214 (8.3) | 40,893,791 (2.6) | 14,300,828 (7.3) | 59,783,809 (1.8) | 0.44× | **≥ parity** |
| 1,000 (2³·5³) | 2,126 (23.4) | 2,141 (23.3) | 13,787 (3.6) | 9,854 (5.1) | 29,651 (1.7) | 0.99× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,557 (21.3) | 2,206 (24.7) | 14,220 (3.8) | 10,603 (5.1) | 36,031 (1.5) | 1.16× | lags FFTW 1.16× |
| 1,920 (2⁷·3·5) | 3,863 (27.1) | 3,980 (26.3) | 23,813 (4.4) | 16,887 (6.2) | 62,995 (1.7) | 0.97× | **≥ parity** |
| 1,009 (prime) | 12,985 (3.9) | 21,388 (2.4) | 64,429 (0.8) | 38,061 (1.3) | 1,127,511 (0.0) | 0.61× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 2,966 (22.6) | 3,030 (22.1) | 16,181 (4.1) | 12,633 (5.3) | 43,210 (1.6) | 0.98× | **≥ parity** |
| 10,007 (prime) | 216,446 (3.1) | 206,130 (3.2) | 653,563 (1.0) | 468,611 (1.4) | 119,033,855 (0.0) | 1.05× | lags FFTW 1.05× |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 306 (16.8) | 226 (22.7) | 6,153 (0.8) | 5,242 (1.0) | 2,534 (2.0) | 1.35× | lags FFTW 1.35× |
| 1,024 (2¹⁰) | 1,127 (22.7) | 987 (25.9) | 9,606 (2.7) | 8,532 (3.0) | 13,436 (1.9) | 1.14× | lags FFTW 1.14× |
| 4,096 (2¹²) | 5,894 (20.8) | 5,122 (24.0) | 25,894 (4.7) | 22,359 (5.5) | 59,538 (2.1) | 1.15× | lags FFTW 1.15× |
| 65,536 (2¹⁶) | 147,275 (17.8) | 153,568 (17.1) | 395,739 (6.6) | 491,084 (5.3) | 1,318,452 (2.0) | 0.96× | **≥ parity** |
| 1,048,576 (2²⁰) | 3,122,308 (16.8) | 3,911,052 (13.4) | 8,444,605 (6.2) | 10,098,614 (5.2) | 27,031,840 (1.9) | 0.80× | **≥ parity** |
| 1,000 (2³·5³) | 1,424 (17.5) | 1,219 (20.4) | 10,183 (2.4) | 8,543 (2.9) | 13,835 (1.8) | 1.17× | lags FFTW 1.17× |
| 1,080 (2³·3³·5) | 1,669 (16.3) | 1,366 (19.9) | 10,761 (2.5) | 8,882 (3.1) | 15,801 (1.7) | 1.22× | lags FFTW 1.22× |
| 1,920 (2⁷·3·5) | 2,657 (19.7) | 2,337 (22.4) | 14,913 (3.5) | 13,046 (4.0) | 27,872 (1.9) | 1.14× | lags FFTW 1.14× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 327 (15.7) | 350 (14.6) | 0.93× | **≥ parity** |
| 1,024 (2¹⁰) | 1,198 (21.4) | 1,220 (21.0) | 0.98× | **≥ parity** |
| 4,096 (2¹²) | 5,963 (20.6) | 5,896 (20.8) | 1.01× | **≥ parity** |
| 65,536 (2¹⁶) | 139,760 (18.8) | 172,770 (15.2) | 0.81× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,926,232 (17.9) | 4,040,914 (13.0) | 0.72× | **≥ parity** |
| 1,000 (2³·5³) | 1,538 (16.2) | 1,462 (17.0) | 1.05× | lags FFTW 1.05× |
| 1,080 (2³·3³·5) | 1,731 (15.7) | 1,709 (15.9) | 1.01× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,690 (19.5) | 2,884 (18.2) | 0.93× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 11,777 (20.9) | 15,265 (16.1) | 11,038 (22.3) | 51,551 (4.8) | 34,575 (7.1) | 1.07× | lags FFTW 1.07× |
| 128x128 | 50,959 (22.5) | 75,160 (15.3) | 65,192 (17.6) | 158,589 (7.2) | 126,173 (9.1) | 0.78× | **≥ parity** |
| 256x256 | 283,933 (18.5) | 366,353 (14.3) | 295,975 (17.7) | 705,599 (7.4) | 456,571 (11.5) | 0.96× | **≥ parity** |
| 512x512 | 1,380,181 (17.1) | 1,656,591 (14.2) | 1,375,579 (17.2) | 2,733,907 (8.6) | 1,893,507 (12.5) | 1.00× | **≥ parity** |
| 1024x1024 | 6,494,040 (16.1) | 8,392,498 (12.5) | 6,499,724 (16.1) | 13,846,651 (7.6) | 9,736,571 (10.8) | 1.00× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 74,115,874 |
| 1,024 | — | — | 130,710,159 |
| 4,096 | — | — | 241,992,190 |
| 65,536 | — | — | 2,203,597,091 |
| 1,048,576 | — | — | 4,452,092,933 |
| 1,000 | — | — | 151,661,749 |
| 1,080 | — | — | 371,453,479 |
| 1,920 | — | — | 606,825,816 |
| 1,009 | — | — | 150,310,199 |
| 1,296 | — | — | 271,117,723 |
| 10,007 | — | — | 1,257,943,611 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 13/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.35×
- complex 256 (2⁸): 1.28×
- real 1,080 (2³·3³·5): 1.22×
- complex 1,024 (2¹⁰): 1.18×
- real 1,000 (2³·5³): 1.17×
- complex 1,080 (2³·3³·5): 1.16×
- real 4,096 (2¹²): 1.15×
- real 1,024 (2¹⁰): 1.14×
- real 1,920 (2⁷·3·5): 1.14×
- 2-D 64x64: 1.07×
- complex 10,007 (prime): 1.05×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
