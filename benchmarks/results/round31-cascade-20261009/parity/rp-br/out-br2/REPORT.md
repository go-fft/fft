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
| 256 (2⁸) | 467 (21.9) | 382 (26.8) | 8,662 (1.2) | 6,374 (1.6) | 5,493 (1.9) | 1.22× | lags FFTW 1.22× |
| 1,024 (2¹⁰) | 2,512 (20.4) | 2,134 (24.0) | 15,893 (3.2) | 13,349 (3.8) | 28,848 (1.8) | 1.18× | lags FFTW 1.18× |
| 4,096 (2¹²) | 12,242 (20.1) | 13,522 (18.2) | 58,600 (4.2) | 47,097 (5.2) | 131,694 (1.9) | 0.91× | **≥ parity** |
| 65,536 (2¹⁶) | 456,946 (11.5) | 407,660 (12.9) | 1,849,774 (2.8) | 1,389,505 (3.8) | 3,290,352 (1.6) | 1.12× | lags FFTW 1.12× |
| 1,048,576 (2²⁰) | 20,869,018 (5.0) | 27,383,664 (3.8) | 50,109,738 (2.1) | 42,124,835 (2.5) | 98,422,550 (1.1) | 0.76× | **≥ parity** |
| 1,000 (2³·5³) | 3,030 (16.4) | 2,777 (17.9) | 16,716 (3.0) | 13,825 (3.6) | 30,730 (1.6) | 1.09× | lags FFTW 1.09× |
| 1,080 (2³·3³·5) | 3,757 (14.5) | 2,991 (18.2) | 20,080 (2.7) | 14,001 (3.9) | 35,649 (1.5) | 1.26× | lags FFTW 1.26× |
| 1,920 (2⁷·3·5) | 5,503 (19.0) | 4,965 (21.1) | 31,223 (3.4) | 23,383 (4.5) | 62,473 (1.7) | 1.11× | lags FFTW 1.11× |
| 1,009 (prime) | 18,677 (2.7) | 30,710 (1.6) | 80,618 (0.6) | 56,770 (0.9) | 1,480,060 (0.0) | 0.61× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 4,382 (15.3) | 3,644 (18.4) | 22,794 (2.9) | 18,538 (3.6) | 41,602 (1.6) | 1.20× | lags FFTW 1.20× |
| 10,007 (prime) | 330,354 (2.0) | 364,239 (1.8) | 969,677 (0.7) | 736,503 (0.9) | 144,971,216 (0.0) | 0.91× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 391 (13.1) | 364 (14.1) | 6,582 (0.8) | 6,959 (0.7) | 2,670 (1.9) | 1.08× | lags FFTW 1.08× |
| 1,024 (2¹⁰) | 1,304 (19.6) | 1,523 (16.8) | 12,934 (2.0) | 10,627 (2.4) | 13,260 (1.9) | 0.86× | **≥ parity** |
| 4,096 (2¹²) | 8,068 (15.2) | 7,595 (16.2) | 34,413 (3.6) | 31,554 (3.9) | 62,327 (2.0) | 1.06× | lags FFTW 1.06× |
| 65,536 (2¹⁶) | 213,265 (12.3) | 196,888 (13.3) | 608,322 (4.3) | 694,786 (3.8) | 1,442,163 (1.8) | 1.08× | lags FFTW 1.08× |
| 1,048,576 (2²⁰) | 8,180,681 (6.4) | 7,387,786 (7.1) | 32,346,174 (1.6) | 26,850,869 (2.0) | 43,730,701 (1.2) | 1.11× | lags FFTW 1.11× |
| 1,000 (2³·5³) | 1,861 (13.4) | 1,873 (13.3) | 12,782 (1.9) | 11,199 (2.2) | 14,435 (1.7) | 0.99× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,060 (13.2) | 2,019 (13.5) | 13,459 (2.0) | 12,383 (2.2) | 17,221 (1.6) | 1.02× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,436 (15.2) | 3,418 (15.3) | 18,296 (2.9) | 17,749 (2.9) | 29,275 (1.8) | 1.01× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 401 (12.8) | 540 (9.5) | 0.74× | **≥ parity** |
| 1,024 (2¹⁰) | 1,436 (17.8) | 1,846 (13.9) | 0.78× | **≥ parity** |
| 4,096 (2¹²) | 8,006 (15.3) | 8,330 (14.8) | 0.96× | **≥ parity** |
| 65,536 (2¹⁶) | 237,286 (11.0) | 213,370 (12.3) | 1.11× | lags FFTW 1.11× |
| 1,048,576 (2²⁰) | 11,420,831 (4.6) | 9,117,409 (5.8) | 1.25× | lags FFTW 1.25× |
| 1,000 (2³·5³) | 1,936 (12.9) | 2,311 (10.8) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,163 (12.6) | 2,526 (10.8) | 0.86× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,564 (14.7) | 3,932 (13.3) | 0.91× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,457 (14.1) | 31,072 (7.9) | 17,042 (14.4) | 54,435 (4.5) | 44,707 (5.5) | 1.02× | **≥ parity** |
| 128x128 | 82,215 (13.9) | 142,113 (8.1) | 69,930 (16.4) | 211,820 (5.4) | 173,406 (6.6) | 1.18× | lags FFTW 1.18× |
| 256x256 | 485,696 (10.8) | 672,929 (7.8) | 482,313 (10.9) | 937,168 (5.6) | 762,220 (6.9) | 1.01× | **≥ parity** |
| 512x512 | 2,304,167 (10.2) | 5,012,977 (4.7) | 2,359,481 (10.0) | 4,769,061 (4.9) | 3,501,876 (6.7) | 0.98× | **≥ parity** |
| 1024x1024 | 16,487,325 (6.4) | 19,227,931 (5.5) | 17,649,444 (5.9) | 30,429,946 (3.4) | 26,601,115 (3.9) | 0.93× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 22,469 | 9,884 | 68,478,929 |
| 1,024 | 78,586 | 35,703 | 126,771,938 |
| 4,096 | 304,786 | 136,125 | 277,663,847 |
| 65,536 | 4,007,900 | 2,158,851 | 3,090,051,738 |
| 1,048,576 | 73,106,607 | 35,609,502 | 8,323,094,817 |
| 1,000 | 61,953 | 35,181 | 143,300,239 |
| 1,080 | 76,877 | 37,994 | 350,163,201 |
| 1,920 | 160,607 | 65,483 | 566,216,508 |
| 1,009 | 186,886 | 43,234 | 135,915,579 |
| 1,296 | 81,176 | 44,893 | 258,969,462 |
| 10,007 | 2,118,597 | 415,160 | 1,505,619,565 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 12/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 1,080 (2³·3³·5): 1.26×
- complex 256 (2⁸): 1.22×
- complex 1,296 (2⁴·3⁴): 1.20×
- complex 1,024 (2¹⁰): 1.18×
- 2-D 128x128: 1.18×
- complex 65,536 (2¹⁶): 1.12×
- complex 1,920 (2⁷·3·5): 1.11×
- real 1,048,576 (2²⁰): 1.11×
- complex 1,000 (2³·5³): 1.09×
- real 65,536 (2¹⁶): 1.08×
- real 256 (2⁸): 1.08×
- real 4,096 (2¹²): 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
