# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Neoverse-N1 (cfarm424), one pinned core.
- **Toolchains**: go1.27.1 linux/arm64 (cross-compiled); native **FFTW fftw-3.3.10-neon** (3.3.10 built from source by setup.sh); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 1,105 (9.3) | 1,159 (8.8) | 9,539 (1.1) | 7,671 (1.3) | 5,846 (1.8) | 0.95× | **≥ parity** |
| 1,024 (2¹⁰) | 5,235 (9.8) | 6,318 (8.1) | 18,881 (2.7) | 14,589 (3.5) | 30,546 (1.7) | 0.83× | **≥ parity** |
| 4,096 (2¹²) | 27,294 (9.0) | 40,628 (6.0) | 60,317 (4.1) | 48,147 (5.1) | 146,418 (1.7) | 0.67× | **≥ parity** |
| 65,536 (2¹⁶) | 730,603 (7.2) | 1,224,577 (4.3) | 2,094,941 (2.5) | 1,417,797 (3.7) | 3,257,120 (1.6) | 0.60× | **≥ parity** |
| 1,048,576 (2²⁰) | 22,297,242 (4.7) | 43,952,134 (2.4) | 43,606,431 (2.4) | 31,173,853 (3.4) | 75,656,431 (1.4) | 0.51× | **≥ parity** |
| 1,000 (2³·5³) | 6,025 (8.3) | 7,185 (6.9) | 19,293 (2.6) | 15,144 (3.3) | 34,069 (1.5) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,877 (7.9) | 7,666 (7.1) | 21,894 (2.5) | 16,792 (3.2) | 41,755 (1.3) | 0.90× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,995 (8.7) | 13,517 (7.7) | 33,010 (3.2) | 25,864 (4.0) | 69,800 (1.5) | 0.89× | **≥ parity** |
| 1,009 (prime) | 21,709 (2.3) | 48,018 (1.0) | 96,575 (0.5) | 57,088 (0.9) | 2,231,044 (0.0) | 0.45× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,695 (7.7) | 11,369 (5.9) | 25,068 (2.7) | 19,646 (3.4) | 49,137 (1.4) | 0.76× | **≥ parity** |
| 10,007 (prime) | 417,811 (1.6) | 620,681 (1.1) | 1,050,153 (0.6) | 783,869 (0.8) | 218,050,109 (0.0) | 0.67× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 833 (6.1) | 722 (7.1) | 8,695 (0.6) | 7,654 (0.7) | 2,994 (1.7) | 1.15× | lags FFTW 1.15× |
| 1,024 (2¹⁰) | 3,584 (7.1) | 4,187 (6.1) | 14,381 (1.8) | 12,324 (2.1) | 15,186 (1.7) | 0.86× | **≥ parity** |
| 4,096 (2¹²) | 16,683 (7.4) | 19,826 (6.2) | 38,149 (3.2) | 32,825 (3.7) | 69,608 (1.8) | 0.84× | **≥ parity** |
| 65,536 (2¹⁶) | 419,900 (6.2) | 543,676 (4.8) | 722,477 (3.6) | 744,976 (3.5) | 1,740,406 (1.5) | 0.77× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,301,568 (5.1) | 17,885,263 (2.9) | 20,119,755 (2.6) | 18,445,342 (2.8) | 41,838,605 (1.3) | 0.58× | **≥ parity** |
| 1,000 (2³·5³) | 4,123 (6.0) | 3,969 (6.3) | 15,221 (1.6) | 12,687 (2.0) | 14,989 (1.7) | 1.04× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,666 (5.8) | 4,201 (6.5) | 16,255 (1.7) | 13,989 (1.9) | 17,546 (1.6) | 1.11× | lags FFTW 1.11× |
| 1,920 (2⁷·3·5) | 7,670 (6.8) | 7,364 (7.1) | 22,070 (2.4) | 19,163 (2.7) | 30,797 (1.7) | 1.04× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 965 (5.3) | 809 (6.3) | 1.19× | lags FFTW 1.19× |
| 1,024 (2¹⁰) | 3,878 (6.6) | 4,480 (5.7) | 0.87× | **≥ parity** |
| 4,096 (2¹²) | 17,747 (6.9) | 21,230 (5.8) | 0.84× | **≥ parity** |
| 65,536 (2¹⁶) | 433,754 (6.0) | 714,578 (3.7) | 0.61× | **≥ parity** |
| 1,048,576 (2²⁰) | 11,065,385 (4.7) | 23,504,893 (2.2) | 0.47× | **≥ parity** |
| 1,000 (2³·5³) | 4,524 (5.5) | 4,283 (5.8) | 1.06× | lags FFTW 1.06× |
| 1,080 (2³·3³·5) | 5,037 (5.4) | 4,443 (6.1) | 1.13× | lags FFTW 1.13× |
| 1,920 (2⁷·3·5) | 8,303 (6.3) | 7,824 (6.7) | 1.06× | lags FFTW 1.06× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 34,046 (7.2) | 40,820 (6.0) | 30,936 (7.9) | 71,124 (3.5) | 50,564 (4.9) | 1.10× | lags FFTW 1.10× |
| 128x128 | 143,633 (8.0) | 165,414 (6.9) | 195,283 (5.9) | 257,069 (4.5) | 199,879 (5.7) | 0.74× | **≥ parity** |
| 256x256 | 763,964 (6.9) | 848,456 (6.2) | 1,315,510 (4.0) | 1,053,950 (5.0) | 879,000 (6.0) | 0.58× | **≥ parity** |
| 512x512 | 3,541,254 (6.7) | 4,243,465 (5.6) | 7,205,960 (3.3) | 5,249,736 (4.5) | 4,047,572 (5.8) | 0.49× | **≥ parity** |
| 1024x1024 | 17,717,234 (5.9) | 18,386,265 (5.7) | 43,781,288 (2.4) | 26,526,018 (4.0) | 22,316,871 (4.7) | 0.40× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 14,948 | 9,860 | 4,818,475 |
| 1,024 | 53,118 | 34,007 | 3,268,611 |
| 4,096 | 194,192 | 127,333 | 6,163,459 |
| 65,536 | 3,180,635 | 2,077,234 | 21,332,858 |
| 1,048,576 | 47,686,247 | 28,676,160 | 20,686,045 |
| 1,000 | 51,295 | 33,632 | 4,681,194 |
| 1,080 | 58,463 | 36,579 | 12,601,078 |
| 1,920 | 99,149 | 63,700 | 17,733,158 |
| 1,009 | 116,870 | 40,713 | 6,080,575 |
| 1,296 | 83,618 | 43,819 | 10,540,366 |
| 10,007 | 1,400,795 | 391,221 | 35,859,523 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 21/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.15×
- real 1,080 (2³·3³·5): 1.11×
- 2-D 64x64: 1.10×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
