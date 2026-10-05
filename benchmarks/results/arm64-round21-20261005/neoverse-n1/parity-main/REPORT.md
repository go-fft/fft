# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: ARM Neoverse-N1, 64 cores, GCC Compile Farm cfarm424 (main 7264f35).
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
| 256 (2⁸) | 1,440 (7.1) | 1,150 (8.9) | 9,743 (1.1) | 7,638 (1.3) | 6,157 (1.7) | 1.25× | lags FFTW 1.25× |
| 1,024 (2¹⁰) | 7,030 (7.3) | 6,308 (8.1) | 19,106 (2.7) | 14,585 (3.5) | 32,281 (1.6) | 1.11× | lags FFTW 1.11× |
| 4,096 (2¹²) | 33,652 (7.3) | 40,520 (6.1) | 59,962 (4.1) | 47,876 (5.1) | 152,942 (1.6) | 0.83× | **≥ parity** |
| 65,536 (2¹⁶) | 841,539 (6.2) | 1,228,910 (4.3) | 2,158,009 (2.4) | 1,455,875 (3.6) | 3,590,003 (1.5) | 0.68× | **≥ parity** |
| 1,048,576 (2²⁰) | 24,039,013 (4.4) | 44,763,191 (2.3) | 48,687,855 (2.2) | 34,012,190 (3.1) | 80,644,186 (1.3) | 0.54× | **≥ parity** |
| 1,000 (2³·5³) | 7,660 (6.5) | 7,156 (7.0) | 19,495 (2.6) | 15,227 (3.3) | 35,110 (1.4) | 1.07× | lags FFTW 1.07× |
| 1,080 (2³·3³·5) | 8,954 (6.1) | 7,664 (7.1) | 21,886 (2.5) | 16,836 (3.2) | 43,426 (1.3) | 1.17× | lags FFTW 1.17× |
| 1,920 (2⁷·3·5) | 15,695 (6.7) | 13,557 (7.7) | 33,298 (3.1) | 26,468 (4.0) | 73,006 (1.4) | 1.16× | lags FFTW 1.16× |
| 1,009 (prime) | 22,780 (2.2) | 48,288 (1.0) | 97,117 (0.5) | 58,081 (0.9) | 2,339,873 (0.0) | 0.47× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 11,487 (5.8) | 11,349 (5.9) | 25,103 (2.7) | 19,695 (3.4) | 49,793 (1.3) | 1.01× | **≥ parity** |
| 10,007 (prime) | 496,880 (1.3) | 595,704 (1.1) | 1,023,669 (0.6) | 761,720 (0.9) | 228,106,372 (0.0) | 0.83× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 984 (5.2) | 720 (7.1) | 8,938 (0.6) | 7,931 (0.6) | 2,997 (1.7) | 1.37× | lags FFTW 1.37× |
| 1,024 (2¹⁰) | 4,370 (5.9) | 4,168 (6.1) | 14,582 (1.8) | 12,377 (2.1) | 15,418 (1.7) | 1.05× | **≥ parity** |
| 4,096 (2¹²) | 20,364 (6.0) | 19,690 (6.2) | 37,676 (3.3) | 32,867 (3.7) | 72,846 (1.7) | 1.03× | **≥ parity** |
| 65,536 (2¹⁶) | 486,075 (5.4) | 559,804 (4.7) | 735,636 (3.6) | 764,279 (3.4) | 1,849,438 (1.4) | 0.87× | **≥ parity** |
| 1,048,576 (2²⁰) | 13,428,356 (3.9) | 21,340,305 (2.5) | 22,654,315 (2.3) | 19,139,432 (2.7) | 46,908,966 (1.1) | 0.63× | **≥ parity** |
| 1,000 (2³·5³) | 4,919 (5.1) | 3,945 (6.3) | 15,888 (1.6) | 12,581 (2.0) | 15,066 (1.7) | 1.25× | lags FFTW 1.25× |
| 1,080 (2³·3³·5) | 5,730 (4.7) | 4,207 (6.5) | 16,308 (1.7) | 14,290 (1.9) | 18,253 (1.5) | 1.36× | lags FFTW 1.36× |
| 1,920 (2⁷·3·5) | 9,305 (5.6) | 7,358 (7.1) | 22,113 (2.4) | 18,916 (2.8) | 31,964 (1.6) | 1.26× | lags FFTW 1.26× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 1,049 (4.9) | 850 (6.0) | 1.23× | lags FFTW 1.23× |
| 1,024 (2¹⁰) | 4,648 (5.5) | 4,479 (5.7) | 1.04× | **≥ parity** |
| 4,096 (2¹²) | 21,135 (5.8) | 21,127 (5.8) | 1.00× | **≥ parity** |
| 65,536 (2¹⁶) | 506,344 (5.2) | 639,747 (4.1) | 0.79× | **≥ parity** |
| 1,048,576 (2²⁰) | 13,058,879 (4.0) | 26,026,365 (2.0) | 0.50× | **≥ parity** |
| 1,000 (2³·5³) | 5,362 (4.6) | 4,259 (5.8) | 1.26× | lags FFTW 1.26× |
| 1,080 (2³·3³·5) | 6,118 (4.4) | 4,448 (6.1) | 1.38× | lags FFTW 1.38× |
| 1,920 (2⁷·3·5) | 9,972 (5.3) | 7,795 (6.7) | 1.28× | lags FFTW 1.28× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 48,436 (5.1) | 116,818 (2.1) | 30,868 (8.0) | 72,497 (3.4) | 50,061 (4.9) | 1.57× | lags FFTW 1.57× |
| 128x128 | 375,782 (3.1) | 398,581 (2.9) | 193,792 (5.9) | 255,552 (4.5) | 199,157 (5.8) | 1.94× | lags FFTW 1.94× |
| 256x256 | 759,820 (6.9) | 929,103 (5.6) | 1,412,104 (3.7) | 1,085,292 (4.8) | 849,493 (6.2) | 0.54× | **≥ parity** |
| 512x512 | 1,697,403 (13.9) | 2,146,937 (11.0) | 8,346,851 (2.8) | 5,286,951 (4.5) | 4,068,217 (5.8) | 0.20× | **≥ parity** |
| 1024x1024 | 3,891,714 (26.9) | 5,328,155 (19.7) | 48,525,678 (2.2) | 25,182,296 (4.2) | 22,691,534 (4.6) | 0.08× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 39,841 | 26,482 | 7,690,569 |
| 1,024 | 141,001 | 97,361 | 3,325,658 |
| 4,096 | 514,491 | 366,002 | 6,158,825 |
| 65,536 | 6,916,098 | 5,458,445 | 21,450,928 |
| 1,048,576 | 87,688,082 | 63,938,927 | 19,975,182 |
| 1,000 | 138,369 | 100,259 | 4,679,680 |
| 1,080 | 158,493 | 105,536 | 12,580,056 |
| 1,920 | 262,900 | 180,660 | 17,802,864 |
| 1,009 | 314,211 | 110,884 | 6,213,548 |
| 1,296 | 206,810 | 121,585 | 10,776,024 |
| 10,007 | 4,009,200 | 965,977 | 35,885,094 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 13/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 23/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 23/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- 2-D 128x128: 1.94×
- 2-D 64x64: 1.57×
- real 256 (2⁸): 1.37×
- real 1,080 (2³·3³·5): 1.36×
- real 1,920 (2⁷·3·5): 1.26×
- complex 256 (2⁸): 1.25×
- real 1,000 (2³·5³): 1.25×
- complex 1,080 (2³·3³·5): 1.17×
- complex 1,920 (2⁷·3·5): 1.16×
- complex 1,024 (2¹⁰): 1.11×
- complex 1,000 (2³·5³): 1.07×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
