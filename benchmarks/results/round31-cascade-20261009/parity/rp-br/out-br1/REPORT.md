# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon Cascade Lake (cfarm151, KVM, AVX-512).
- **Toolchains**: go1.27.1 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512** (3.3.10 built from source by setup.sh); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 465 (22.0) | 362 (28.3) | 9,038 (1.1) | 6,523 (1.6) | 5,524 (1.9) | 1.29× | lags FFTW 1.29× |
| 1,024 (2¹⁰) | 2,469 (20.7) | 2,100 (24.4) | 16,471 (3.1) | 12,039 (4.3) | 29,261 (1.7) | 1.18× | lags FFTW 1.18× |
| 4,096 (2¹²) | 12,187 (20.2) | 12,886 (19.1) | 57,370 (4.3) | 44,923 (5.5) | 133,460 (1.8) | 0.95× | **≥ parity** |
| 65,536 (2¹⁶) | 485,779 (10.8) | 399,368 (13.1) | 1,879,838 (2.8) | 1,377,911 (3.8) | 3,102,116 (1.7) | 1.22× | lags FFTW 1.22× |
| 1,048,576 (2²⁰) | 20,952,442 (5.0) | 26,890,123 (3.9) | 50,179,724 (2.1) | 40,982,757 (2.6) | 99,038,314 (1.1) | 0.78× | **≥ parity** |
| 1,000 (2³·5³) | 2,996 (16.6) | 2,683 (18.6) | 17,223 (2.9) | 14,140 (3.5) | 30,096 (1.7) | 1.12× | lags FFTW 1.12× |
| 1,080 (2³·3³·5) | 3,753 (14.5) | 2,985 (18.2) | 20,751 (2.6) | 13,394 (4.1) | 34,961 (1.6) | 1.26× | lags FFTW 1.26× |
| 1,920 (2⁷·3·5) | 5,436 (19.3) | 4,937 (21.2) | 31,418 (3.3) | 24,046 (4.4) | 61,789 (1.7) | 1.10× | lags FFTW 1.10× |
| 1,009 (prime) | 18,676 (2.7) | 30,298 (1.7) | 82,826 (0.6) | 57,247 (0.9) | 1,471,552 (0.0) | 0.62× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 4,410 (15.2) | 3,664 (18.3) | 20,939 (3.2) | 16,139 (4.2) | 41,816 (1.6) | 1.20× | lags FFTW 1.20× |
| 10,007 (prime) | 336,258 (2.0) | 363,151 (1.8) | 934,617 (0.7) | 730,341 (0.9) | 143,858,562 (0.0) | 0.93× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 388 (13.2) | 360 (14.2) | 6,571 (0.8) | 6,915 (0.7) | 2,644 (1.9) | 1.08× | lags FFTW 1.08× |
| 1,024 (2¹⁰) | 1,278 (20.0) | 1,502 (17.0) | 13,017 (2.0) | 11,922 (2.1) | 13,350 (1.9) | 0.85× | **≥ parity** |
| 4,096 (2¹²) | 7,835 (15.7) | 7,527 (16.3) | 33,234 (3.7) | 31,761 (3.9) | 60,949 (2.0) | 1.04× | **≥ parity** |
| 65,536 (2¹⁶) | 216,641 (12.1) | 189,003 (13.9) | 597,272 (4.4) | 693,763 (3.8) | 1,431,887 (1.8) | 1.15× | lags FFTW 1.15× |
| 1,048,576 (2²⁰) | 11,176,874 (4.7) | 10,217,340 (5.1) | 20,536,433 (2.6) | 21,245,661 (2.5) | 43,716,122 (1.2) | 1.09× | lags FFTW 1.09× |
| 1,000 (2³·5³) | 1,857 (13.4) | 1,902 (13.1) | 14,373 (1.7) | 12,407 (2.0) | 14,211 (1.8) | 0.98× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,084 (13.1) | 2,239 (12.2) | 14,064 (1.9) | 12,891 (2.1) | 16,623 (1.6) | 0.93× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,385 (15.5) | 3,473 (15.1) | 18,021 (2.9) | 17,494 (3.0) | 29,268 (1.8) | 0.97× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 401 (12.8) | 502 (10.2) | 0.80× | **≥ parity** |
| 1,024 (2¹⁰) | 1,445 (17.7) | 1,864 (13.7) | 0.78× | **≥ parity** |
| 4,096 (2¹²) | 7,987 (15.4) | 8,558 (14.4) | 0.93× | **≥ parity** |
| 65,536 (2¹⁶) | 245,975 (10.7) | 211,379 (12.4) | 1.16× | lags FFTW 1.16× |
| 1,048,576 (2²⁰) | 12,454,907 (4.2) | 9,197,267 (5.7) | 1.35× | lags FFTW 1.35× |
| 1,000 (2³·5³) | 1,920 (13.0) | 2,264 (11.0) | 0.85× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,161 (12.6) | 2,552 (10.7) | 0.85× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,518 (14.9) | 4,215 (12.4) | 0.83× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,676 (13.9) | 31,124 (7.9) | 15,106 (16.3) | 55,454 (4.4) | 46,364 (5.3) | 1.17× | lags FFTW 1.17× |
| 128x128 | 80,373 (14.3) | 136,536 (8.4) | 82,034 (14.0) | 205,519 (5.6) | 174,725 (6.6) | 0.98× | **≥ parity** |
| 256x256 | 453,189 (11.6) | 693,280 (7.6) | 482,312 (10.9) | 872,365 (6.0) | 755,604 (6.9) | 0.94× | **≥ parity** |
| 512x512 | 3,117,626 (7.6) | 5,218,734 (4.5) | 2,587,407 (9.1) | 5,729,935 (4.1) | 3,601,749 (6.6) | 1.20× | lags FFTW 1.20× |
| 1024x1024 | 15,533,508 (6.8) | 19,664,619 (5.3) | 27,340,037 (3.8) | 30,539,076 (3.4) | 26,220,127 (4.0) | 0.57× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 22,366 | 10,057 | 68,346,271 |
| 1,024 | 79,632 | 36,403 | 125,888,404 |
| 4,096 | 308,549 | 139,086 | 279,302,996 |
| 65,536 | 4,065,764 | 2,233,336 | 3,075,494,286 |
| 1,048,576 | 75,279,413 | 35,748,674 | 8,320,670,858 |
| 1,000 | 62,709 | 35,939 | 145,080,132 |
| 1,080 | 70,519 | 38,983 | 344,980,817 |
| 1,920 | 139,876 | 67,860 | 565,386,498 |
| 1,009 | 179,774 | 44,087 | 134,909,120 |
| 1,296 | 83,051 | 46,398 | 258,416,763 |
| 10,007 | 2,194,645 | 425,666 | 1,511,307,914 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 12/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 256 (2⁸): 1.29×
- complex 1,080 (2³·3³·5): 1.26×
- complex 65,536 (2¹⁶): 1.22×
- 2-D 512x512: 1.20×
- complex 1,296 (2⁴·3⁴): 1.20×
- complex 1,024 (2¹⁰): 1.18×
- 2-D 64x64: 1.17×
- real 65,536 (2¹⁶): 1.15×
- complex 1,000 (2³·5³): 1.12×
- complex 1,920 (2⁷·3·5): 1.10×
- real 1,048,576 (2²⁰): 1.09×
- real 256 (2⁸): 1.08×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
