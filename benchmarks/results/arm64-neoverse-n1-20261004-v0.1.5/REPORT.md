# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: ARM Neoverse-N1, 64 cores, GCC Compile Farm cfarm424.
- **Toolchains**: go1.26.4 linux/arm64 (cross-compiled); native **FFTW fftw-3.3.10-neon** (3.3.10 built from source with --enable-neon); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 1,897 (5.4) | 1,156 (8.9) | 9,535 (1.1) | 7,770 (1.3) | 10,786 (0.9) | 1.64× | lags FFTW 1.64× |
| 1,024 (2¹⁰) | 9,009 (5.7) | 6,317 (8.1) | 18,723 (2.7) | 14,597 (3.5) | 53,320 (1.0) | 1.43× | lags FFTW 1.43× |
| 4,096 (2¹²) | 47,150 (5.2) | 40,604 (6.1) | 59,349 (4.1) | 48,159 (5.1) | 281,470 (0.9) | 1.16× | lags FFTW 1.16× |
| 65,536 (2¹⁶) | 1,126,744 (4.7) | 1,324,500 (4.0) | 2,256,821 (2.3) | 1,482,226 (3.5) | 6,469,940 (0.8) | 0.85× | **≥ parity** |
| 1,048,576 (2²⁰) | 26,566,310 (3.9) | 44,987,190 (2.3) | 44,213,908 (2.4) | 32,179,429 (3.3) | 141,620,920 (0.7) | 0.59× | **≥ parity** |
| 1,000 (2³·5³) | 11,075 (4.5) | 7,152 (7.0) | 19,142 (2.6) | 15,145 (3.3) | 55,645 (0.9) | 1.55× | lags FFTW 1.55× |
| 1,080 (2³·3³·5) | 12,755 (4.3) | 7,674 (7.1) | 22,138 (2.5) | 16,604 (3.3) | 63,788 (0.9) | 1.66× | lags FFTW 1.66× |
| 1,920 (2⁷·3·5) | 20,799 (5.0) | 13,620 (7.7) | 33,041 (3.2) | 25,894 (4.0) | 116,176 (0.9) | 1.53× | lags FFTW 1.53× |
| 1,009 (prime) | 27,628 (1.8) | 48,453 (1.0) | 96,470 (0.5) | 57,376 (0.9) | 2,344,866 (0.0) | 0.57× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 15,813 (4.2) | 11,367 (5.9) | 24,794 (2.7) | 19,689 (3.4) | 80,856 (0.8) | 1.39× | lags FFTW 1.39× |
| 10,007 (prime) | 815,537 (0.8) | 609,824 (1.1) | 1,044,754 (0.6) | 768,517 (0.9) | 223,174,406 (0.0) | 1.34× | lags FFTW 1.34× |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 1,239 (4.1) | 723 (7.1) | 8,817 (0.6) | 7,741 (0.7) | 6,029 (0.8) | 1.71× | lags FFTW 1.71× |
| 1,024 (2¹⁰) | 5,439 (4.7) | 4,172 (6.1) | 14,286 (1.8) | 12,407 (2.1) | 30,100 (0.9) | 1.30× | lags FFTW 1.30× |
| 4,096 (2¹²) | 24,895 (4.9) | 19,720 (6.2) | 37,944 (3.2) | 33,753 (3.6) | 133,431 (0.9) | 1.26× | lags FFTW 1.26× |
| 65,536 (2¹⁶) | 646,781 (4.1) | 539,920 (4.9) | 738,289 (3.6) | 745,045 (3.5) | 3,231,981 (0.8) | 1.20× | lags FFTW 1.20× |
| 1,048,576 (2²⁰) | 14,361,911 (3.7) | 19,042,966 (2.8) | 20,251,994 (2.6) | 18,649,949 (2.8) | 75,630,164 (0.7) | 0.75× | **≥ parity** |
| 1,000 (2³·5³) | 6,301 (4.0) | 3,947 (6.3) | 15,441 (1.6) | 12,658 (2.0) | 28,621 (0.9) | 1.60× | lags FFTW 1.60× |
| 1,080 (2³·3³·5) | 7,243 (3.8) | 4,206 (6.5) | 16,152 (1.7) | 14,197 (1.9) | 33,385 (0.8) | 1.72× | lags FFTW 1.72× |
| 1,920 (2⁷·3·5) | 12,154 (4.3) | 7,323 (7.1) | 22,217 (2.4) | 18,757 (2.8) | 60,948 (0.9) | 1.66× | lags FFTW 1.66× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 1,324 (3.9) | 812 (6.3) | 1.63× | lags FFTW 1.63× |
| 1,024 (2¹⁰) | 5,495 (4.7) | 4,486 (5.7) | 1.22× | lags FFTW 1.22× |
| 4,096 (2¹²) | 27,301 (4.5) | 21,175 (5.8) | 1.29× | lags FFTW 1.29× |
| 65,536 (2¹⁶) | 634,073 (4.1) | 651,122 (4.0) | 0.97× | **≥ parity** |
| 1,048,576 (2²⁰) | 14,554,225 (3.6) | 21,944,856 (2.4) | 0.66× | **≥ parity** |
| 1,000 (2³·5³) | 6,774 (3.7) | 4,296 (5.8) | 1.58× | lags FFTW 1.58× |
| 1,080 (2³·3³·5) | 7,564 (3.6) | 4,421 (6.2) | 1.71× | lags FFTW 1.71× |
| 1,920 (2⁷·3·5) | 13,396 (3.9) | 7,824 (6.7) | 1.71× | lags FFTW 1.71× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 63,408 (3.9) | 172,237 (1.4) | 30,873 (8.0) | 70,063 (3.5) | 49,955 (4.9) | 2.05× | lags FFTW 2.05× |
| 128x128 | 478,244 (2.4) | 546,855 (2.1) | 192,657 (6.0) | 250,649 (4.6) | 195,179 (5.9) | 2.48× | lags FFTW 2.48× |
| 256x256 | 924,311 (5.7) | 1,255,620 (4.2) | 1,309,105 (4.0) | 1,054,372 (5.0) | 863,151 (6.1) | 0.71× | **≥ parity** |
| 512x512 | 2,362,428 (10.0) | 2,929,776 (8.1) | 7,032,992 (3.4) | 5,538,421 (4.3) | 3,988,878 (5.9) | 0.34× | **≥ parity** |
| 1024x1024 | 6,309,178 (16.6) | 7,836,390 (13.4) | 43,933,250 (2.4) | 25,754,452 (4.1) | 21,814,100 (4.8) | 0.14× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 30,855 | 28,911 | 4,323,544 |
| 1,024 | 111,458 | 106,319 | 3,357,090 |
| 4,096 | 432,308 | 398,632 | 6,275,255 |
| 65,536 | 6,313,725 | 6,057,752 | 21,681,528 |
| 1,048,576 | 77,492,858 | 76,720,848 | 20,299,267 |
| 1,000 | 112,352 | 104,771 | 4,668,469 |
| 1,080 | 123,571 | 118,181 | 28,097,585 |
| 1,920 | 213,896 | 199,074 | 18,018,793 |
| 1,009 | 339,876 | 125,526 | 6,153,533 |
| 1,296 | 150,054 | 140,384 | 10,455,277 |
| 10,007 | 4,778,650 | 1,140,167 | 35,471,737 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 7/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 23/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 20/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- 2-D 128x128: 2.48×
- 2-D 64x64: 2.05×
- real 1,080 (2³·3³·5): 1.72×
- real 256 (2⁸): 1.71×
- complex 1,080 (2³·3³·5): 1.66×
- real 1,920 (2⁷·3·5): 1.66×
- complex 256 (2⁸): 1.64×
- real 1,000 (2³·5³): 1.60×
- complex 1,000 (2³·5³): 1.55×
- complex 1,920 (2⁷·3·5): 1.53×
- complex 1,024 (2¹⁰): 1.43×
- complex 1,296 (2⁴·3⁴): 1.39×
- complex 10,007 (prime): 1.34×
- real 1,024 (2¹⁰): 1.30×
- real 4,096 (2¹²): 1.26×
- real 65,536 (2¹⁶): 1.20×
- complex 4,096 (2¹²): 1.16×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
