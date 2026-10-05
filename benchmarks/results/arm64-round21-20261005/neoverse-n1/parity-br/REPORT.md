# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: ARM Neoverse-N1, 64 cores, GCC Compile Farm cfarm424 (perf-arm64-2).
- **Toolchains**: go1.27.1 linux/arm64 (cross-compiled); native **FFTW fftw-3.3.10-neon** (3.3.10 built from source with --enable-neon); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 1,171 (8.7) | 1,155 (8.9) | 9,542 (1.1) | 7,752 (1.3) | 5,807 (1.8) | 1.01× | **≥ parity** |
| 1,024 (2¹⁰) | 5,490 (9.3) | 6,352 (8.1) | 18,814 (2.7) | 14,523 (3.5) | 31,906 (1.6) | 0.86× | **≥ parity** |
| 4,096 (2¹²) | 27,706 (8.9) | 40,991 (6.0) | 58,883 (4.2) | 47,496 (5.2) | 149,039 (1.6) | 0.68× | **≥ parity** |
| 65,536 (2¹⁶) | 712,790 (7.4) | 1,233,581 (4.2) | 2,031,529 (2.6) | 1,419,109 (3.7) | 3,552,909 (1.5) | 0.58× | **≥ parity** |
| 1,048,576 (2²⁰) | 20,567,636 (5.1) | 44,957,810 (2.3) | 43,707,533 (2.4) | 31,562,419 (3.3) | 78,832,624 (1.3) | 0.46× | **≥ parity** |
| 1,000 (2³·5³) | 7,699 (6.5) | 7,159 (7.0) | 19,342 (2.6) | 15,109 (3.3) | 33,659 (1.5) | 1.08× | lags FFTW 1.08× |
| 1,080 (2³·3³·5) | 9,158 (5.9) | 7,663 (7.1) | 21,704 (2.5) | 16,624 (3.3) | 41,312 (1.3) | 1.20× | lags FFTW 1.20× |
| 1,920 (2⁷·3·5) | 15,783 (6.6) | 13,495 (7.8) | 32,720 (3.2) | 25,979 (4.0) | 70,020 (1.5) | 1.17× | lags FFTW 1.17× |
| 1,009 (prime) | 21,547 (2.3) | 48,124 (1.0) | 96,442 (0.5) | 57,092 (0.9) | 2,323,968 (0.0) | 0.45× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 12,142 (5.5) | 11,313 (5.9) | 24,702 (2.7) | 19,511 (3.4) | 51,019 (1.3) | 1.07× | lags FFTW 1.07× |
| 10,007 (prime) | 489,323 (1.4) | 603,875 (1.1) | 1,013,494 (0.7) | 758,396 (0.9) | 227,863,734 (0.0) | 0.81× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 873 (5.9) | 724 (7.1) | 8,583 (0.6) | 7,733 (0.7) | 3,048 (1.7) | 1.21× | lags FFTW 1.21× |
| 1,024 (2¹⁰) | 3,735 (6.9) | 4,210 (6.1) | 14,362 (1.8) | 12,477 (2.1) | 15,801 (1.6) | 0.89× | **≥ parity** |
| 4,096 (2¹²) | 17,449 (7.0) | 19,850 (6.2) | 37,363 (3.3) | 32,527 (3.8) | 67,982 (1.8) | 0.88× | **≥ parity** |
| 65,536 (2¹⁶) | 432,654 (6.1) | 550,126 (4.8) | 711,672 (3.7) | 741,999 (3.5) | 1,734,554 (1.5) | 0.79× | **≥ parity** |
| 1,048,576 (2²⁰) | 11,817,534 (4.4) | 18,941,246 (2.8) | 21,356,505 (2.5) | 18,256,820 (2.9) | 46,010,612 (1.1) | 0.62× | **≥ parity** |
| 1,000 (2³·5³) | 4,981 (5.0) | 3,939 (6.3) | 15,255 (1.6) | 12,762 (2.0) | 15,560 (1.6) | 1.26× | lags FFTW 1.26× |
| 1,080 (2³·3³·5) | 5,801 (4.7) | 4,200 (6.5) | 15,819 (1.7) | 14,015 (1.9) | 18,190 (1.5) | 1.38× | lags FFTW 1.38× |
| 1,920 (2⁷·3·5) | 9,197 (5.7) | 7,306 (7.2) | 21,901 (2.4) | 18,980 (2.8) | 32,171 (1.6) | 1.26× | lags FFTW 1.26× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 1,006 (5.1) | 811 (6.3) | 1.24× | lags FFTW 1.24× |
| 1,024 (2¹⁰) | 4,123 (6.2) | 4,454 (5.7) | 0.93× | **≥ parity** |
| 4,096 (2¹²) | 17,968 (6.8) | 21,078 (5.8) | 0.85× | **≥ parity** |
| 65,536 (2¹⁶) | 447,504 (5.9) | 638,608 (4.1) | 0.70× | **≥ parity** |
| 1,048,576 (2²⁰) | 11,578,102 (4.5) | 22,897,989 (2.3) | 0.51× | **≥ parity** |
| 1,000 (2³·5³) | 5,288 (4.7) | 4,245 (5.9) | 1.25× | lags FFTW 1.25× |
| 1,080 (2³·3³·5) | 5,972 (4.6) | 4,419 (6.2) | 1.35× | lags FFTW 1.35× |
| 1,920 (2⁷·3·5) | 9,503 (5.5) | 7,822 (6.7) | 1.21× | lags FFTW 1.21× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 34,708 (7.1) | 96,204 (2.6) | 30,814 (8.0) | 69,350 (3.5) | 50,053 (4.9) | 1.13× | lags FFTW 1.13× |
| 128x128 | 155,039 (7.4) | 407,290 (2.8) | 194,246 (5.9) | 260,171 (4.4) | 202,250 (5.7) | 0.80× | **≥ parity** |
| 256x256 | 709,961 (7.4) | 879,220 (6.0) | 1,249,232 (4.2) | 1,054,863 (5.0) | 841,328 (6.2) | 0.57× | **≥ parity** |
| 512x512 | 1,531,912 (15.4) | 2,152,661 (11.0) | 6,936,183 (3.4) | 5,847,853 (4.0) | 4,192,867 (5.6) | 0.22× | **≥ parity** |
| 1024x1024 | 3,856,620 (27.2) | 5,689,408 (18.4) | 43,737,163 (2.4) | 25,803,658 (4.1) | 22,640,532 (4.6) | 0.09× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 38,794 | 26,029 | 8,523,010 |
| 1,024 | 142,230 | 95,935 | 3,322,220 |
| 4,096 | 545,663 | 369,082 | 6,224,798 |
| 65,536 | 7,244,615 | 5,389,161 | 21,577,053 |
| 1,048,576 | 86,951,784 | 64,980,230 | 20,280,687 |
| 1,000 | 135,747 | 95,043 | 4,842,270 |
| 1,080 | 161,473 | 102,607 | 12,965,320 |
| 1,920 | 263,611 | 178,645 | 17,410,070 |
| 1,009 | 317,313 | 118,780 | 6,089,119 |
| 1,296 | 193,109 | 130,843 | 10,302,147 |
| 10,007 | 4,056,551 | 1,097,865 | 35,133,507 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 15/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 1,080 (2³·3³·5): 1.38×
- real 1,000 (2³·5³): 1.26×
- real 1,920 (2⁷·3·5): 1.26×
- real 256 (2⁸): 1.21×
- complex 1,080 (2³·3³·5): 1.20×
- complex 1,920 (2⁷·3·5): 1.17×
- 2-D 64x64: 1.13×
- complex 1,000 (2³·5³): 1.08×
- complex 1,296 (2⁴·3⁴): 1.07×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
