# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3), cfarm420.
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
| 256 (2⁸) | 368 (27.9) | 281 (36.5) | 6,977 (1.5) | 5,375 (1.9) | 6,497 (1.6) | 1.31× | lags FFTW 1.31× |
| 1,024 (2¹⁰) | 1,700 (30.1) | 1,572 (32.6) | 13,174 (3.9) | 9,859 (5.2) | 29,190 (1.8) | 1.08× | lags FFTW 1.08× |
| 4,096 (2¹²) | 9,242 (26.6) | 10,341 (23.8) | 45,598 (5.4) | 33,896 (7.3) | 133,594 (1.8) | 0.89× | **≥ parity** |
| 65,536 (2¹⁶) | 252,679 (20.7) | 298,073 (17.6) | 1,948,321 (2.7) | 927,204 (5.7) | 2,908,451 (1.8) | 0.85× | **≥ parity** |
| 1,048,576 (2²⁰) | 5,614,539 (18.7) | 11,531,470 (9.1) | 38,513,865 (2.7) | 15,063,678 (7.0) | 59,560,428 (1.8) | 0.49× | **≥ parity** |
| 1,000 (2³·5³) | 2,193 (22.7) | 2,036 (24.5) | 13,988 (3.6) | 10,054 (5.0) | 29,791 (1.7) | 1.08× | lags FFTW 1.08× |
| 1,080 (2³·3³·5) | 2,583 (21.1) | 2,415 (22.5) | 14,469 (3.8) | 10,861 (5.0) | 35,437 (1.5) | 1.07× | lags FFTW 1.07× |
| 1,920 (2⁷·3·5) | 3,857 (27.1) | 3,892 (26.9) | 21,921 (4.8) | 17,658 (5.9) | 61,864 (1.7) | 0.99× | **≥ parity** |
| 1,009 (prime) | 13,077 (3.8) | 20,222 (2.5) | 61,193 (0.8) | 38,699 (1.3) | 1,145,514 (0.0) | 0.65× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,018 (22.2) | 2,997 (22.4) | 16,677 (4.0) | 12,738 (5.3) | 44,014 (1.5) | 1.01× | **≥ parity** |
| 10,007 (prime) | 208,014 (3.2) | 217,835 (3.1) | 678,451 (1.0) | 476,224 (1.4) | 120,177,206 (0.0) | 0.95× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 287 (17.8) | 230 (22.3) | 6,199 (0.8) | 5,373 (1.0) | 2,551 (2.0) | 1.25× | lags FFTW 1.25× |
| 1,024 (2¹⁰) | 1,041 (24.6) | 1,062 (24.1) | 9,699 (2.6) | 8,579 (3.0) | 12,824 (2.0) | 0.98× | **≥ parity** |
| 4,096 (2¹²) | 5,550 (22.1) | 5,357 (22.9) | 25,193 (4.9) | 21,753 (5.6) | 59,932 (2.1) | 1.04× | **≥ parity** |
| 65,536 (2¹⁶) | 132,681 (19.8) | 152,954 (17.1) | 398,127 (6.6) | 512,138 (5.1) | 1,307,751 (2.0) | 0.87× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,947,279 (17.8) | 3,729,040 (14.1) | 8,482,562 (6.2) | 11,580,844 (4.5) | 26,474,331 (2.0) | 0.79× | **≥ parity** |
| 1,000 (2³·5³) | 1,323 (18.8) | 1,266 (19.7) | 10,549 (2.4) | 8,717 (2.9) | 13,106 (1.9) | 1.05× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,494 (18.2) | 1,352 (20.1) | 11,054 (2.5) | 8,933 (3.0) | 15,590 (1.7) | 1.11× | lags FFTW 1.11× |
| 1,920 (2⁷·3·5) | 2,425 (21.6) | 2,400 (21.8) | 14,277 (3.7) | 12,234 (4.3) | 28,876 (1.8) | 1.01× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 288 (17.8) | 288 (17.8) | 1.00× | **≥ parity** |
| 1,024 (2¹⁰) | 1,111 (23.0) | 1,150 (22.3) | 0.97× | **≥ parity** |
| 4,096 (2¹²) | 5,668 (21.7) | 5,792 (21.2) | 0.98× | **≥ parity** |
| 65,536 (2¹⁶) | 137,089 (19.1) | 167,205 (15.7) | 0.82× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,976,152 (17.6) | 4,023,258 (13.0) | 0.74× | **≥ parity** |
| 1,000 (2³·5³) | 1,368 (18.2) | 1,560 (16.0) | 0.88× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,570 (17.3) | 1,568 (17.3) | 1.00× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,486 (21.1) | 2,845 (18.4) | 0.87× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 11,583 (21.2) | 17,392 (14.1) | 10,997 (22.3) | 47,260 (5.2) | 33,285 (7.4) | 1.05× | lags FFTW 1.05× |
| 128x128 | 51,582 (22.2) | 69,100 (16.6) | 59,331 (19.3) | 144,242 (8.0) | 123,089 (9.3) | 0.87× | **≥ parity** |
| 256x256 | 274,497 (19.1) | 333,081 (15.7) | 288,314 (18.2) | 608,664 (8.6) | 455,068 (11.5) | 0.95× | **≥ parity** |
| 512x512 | 1,230,137 (19.2) | 1,706,009 (13.8) | 1,315,956 (17.9) | 2,891,817 (8.2) | 1,949,164 (12.1) | 0.93× | **≥ parity** |
| 1024x1024 | 6,062,685 (17.3) | 8,320,942 (12.6) | 6,964,250 (15.1) | 13,499,010 (7.8) | 9,466,549 (11.1) | 0.87× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 72,266,323 |
| 1,024 | — | — | 133,164,030 |
| 4,096 | — | — | 238,462,109 |
| 65,536 | — | — | 2,142,792,564 |
| 1,048,576 | — | — | 4,046,713,639 |
| 1,000 | — | — | 162,635,384 |
| 1,080 | — | — | 368,323,037 |
| 1,920 | — | — | 577,568,922 |
| 1,009 | — | — | 144,507,186 |
| 1,296 | — | — | 266,975,591 |
| 10,007 | — | — | 1,240,200,623 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 17/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 256 (2⁸): 1.31×
- real 256 (2⁸): 1.25×
- real 1,080 (2³·3³·5): 1.11×
- complex 1,024 (2¹⁰): 1.08×
- complex 1,000 (2³·5³): 1.08×
- complex 1,080 (2³·3³·5): 1.07×
- 2-D 64x64: 1.05×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
