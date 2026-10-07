# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon (Cascade Lake), cfarm151 VM, one pinned core.
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
| 256 (2⁸) | 500 (20.5) | 373 (27.5) | 8,818 (1.2) | 6,336 (1.6) | 5,502 (1.9) | 1.34× | lags FFTW 1.34× |
| 1,024 (2¹⁰) | 2,769 (18.5) | 2,121 (24.1) | 16,107 (3.2) | 11,948 (4.3) | 28,551 (1.8) | 1.31× | lags FFTW 1.31× |
| 4,096 (2¹²) | 12,991 (18.9) | 13,073 (18.8) | 64,640 (3.8) | 46,867 (5.2) | 134,730 (1.8) | 0.99× | **≥ parity** |
| 65,536 (2¹⁶) | 456,866 (11.5) | 405,182 (12.9) | 2,051,201 (2.6) | 1,408,951 (3.7) | 3,409,644 (1.5) | 1.13× | lags FFTW 1.13× |
| 1,048,576 (2²⁰) | 19,873,614 (5.3) | 26,182,727 (4.0) | 51,248,281 (2.0) | 42,173,867 (2.5) | 97,162,644 (1.1) | 0.76× | **≥ parity** |
| 1,000 (2³·5³) | 3,359 (14.8) | 2,733 (18.2) | 17,379 (2.9) | 14,207 (3.5) | 30,279 (1.6) | 1.23× | lags FFTW 1.23× |
| 1,080 (2³·3³·5) | 4,215 (12.9) | 2,968 (18.3) | 20,196 (2.7) | 12,945 (4.2) | 34,883 (1.6) | 1.42× | lags FFTW 1.42× |
| 1,920 (2⁷·3·5) | 6,418 (16.3) | 5,066 (20.7) | 31,108 (3.4) | 23,245 (4.5) | 62,080 (1.7) | 1.27× | lags FFTW 1.27× |
| 1,009 (prime) | 18,667 (2.7) | 30,203 (1.7) | 78,723 (0.6) | 56,022 (0.9) | 1,431,728 (0.0) | 0.62× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 4,396 (15.2) | 3,591 (18.7) | 20,780 (3.2) | 16,636 (4.0) | 41,661 (1.6) | 1.22× | lags FFTW 1.22× |
| 10,007 (prime) | 333,692 (2.0) | 370,076 (1.8) | 912,183 (0.7) | 742,660 (0.9) | 140,276,211 (0.0) | 0.90× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 479 (10.7) | 369 (13.9) | 6,573 (0.8) | 6,761 (0.8) | 2,883 (1.8) | 1.30× | lags FFTW 1.30× |
| 1,024 (2¹⁰) | 1,788 (14.3) | 1,491 (17.2) | 12,251 (2.1) | 10,252 (2.5) | 13,156 (1.9) | 1.20× | lags FFTW 1.20× |
| 4,096 (2¹²) | 9,385 (13.1) | 7,350 (16.7) | 32,711 (3.8) | 31,348 (3.9) | 62,244 (2.0) | 1.28× | lags FFTW 1.28× |
| 65,536 (2¹⁶) | 255,816 (10.2) | 192,419 (13.6) | 594,070 (4.4) | 695,695 (3.8) | 1,433,739 (1.8) | 1.33× | lags FFTW 1.33× |
| 1,048,576 (2²⁰) | 10,545,654 (5.0) | 10,495,771 (5.0) | 22,831,281 (2.3) | 22,509,818 (2.3) | 44,999,450 (1.2) | 1.00× | **≥ parity** |
| 1,000 (2³·5³) | 2,201 (11.3) | 1,894 (13.2) | 14,417 (1.7) | 12,039 (2.1) | 14,553 (1.7) | 1.16× | lags FFTW 1.16× |
| 1,080 (2³·3³·5) | 2,438 (11.2) | 2,042 (13.3) | 13,485 (2.0) | 12,506 (2.2) | 16,658 (1.6) | 1.19× | lags FFTW 1.19× |
| 1,920 (2⁷·3·5) | 3,950 (13.3) | 3,437 (15.2) | 18,171 (2.9) | 17,115 (3.1) | 29,464 (1.8) | 1.15× | lags FFTW 1.15× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 504 (10.2) | 540 (9.5) | 0.93× | **≥ parity** |
| 1,024 (2¹⁰) | 1,778 (14.4) | 1,880 (13.6) | 0.95× | **≥ parity** |
| 4,096 (2¹²) | 9,501 (12.9) | 8,367 (14.7) | 1.14× | lags FFTW 1.14× |
| 65,536 (2¹⁶) | 257,435 (10.2) | 216,929 (12.1) | 1.19× | lags FFTW 1.19× |
| 1,048,576 (2²⁰) | 11,104,636 (4.7) | 10,813,778 (4.8) | 1.03× | **≥ parity** |
| 1,000 (2³·5³) | 2,244 (11.1) | 2,290 (10.9) | 0.98× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,430 (11.2) | 2,522 (10.8) | 0.96× | **≥ parity** |
| 1,920 (2⁷·3·5) | 4,062 (12.9) | 3,999 (13.1) | 1.02× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 18,079 (13.6) | 30,472 (8.1) | 17,074 (14.4) | 55,048 (4.5) | 44,810 (5.5) | 1.06× | lags FFTW 1.06× |
| 128x128 | 86,954 (13.2) | 143,472 (8.0) | 74,452 (15.4) | 208,644 (5.5) | 171,425 (6.7) | 1.17× | lags FFTW 1.17× |
| 256x256 | 493,953 (10.6) | 716,480 (7.3) | 477,712 (11.0) | 924,082 (5.7) | 756,374 (6.9) | 1.03× | **≥ parity** |
| 512x512 | 2,505,325 (9.4) | 4,769,350 (4.9) | 2,955,590 (8.0) | 6,600,821 (3.6) | 4,182,236 (5.6) | 0.85× | **≥ parity** |
| 1024x1024 | 15,877,370 (6.6) | 19,668,892 (5.3) | 24,253,126 (4.3) | 30,807,799 (3.4) | 27,590,827 (3.8) | 0.65× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 18,292 | 10,306 | 68,444,008 |
| 1,024 | 64,579 | 36,476 | 125,126,740 |
| 4,096 | 244,967 | 136,483 | 277,954,504 |
| 65,536 | 3,994,558 | 2,135,081 | 3,070,183,529 |
| 1,048,576 | 74,855,606 | 35,372,094 | 8,183,240,343 |
| 1,000 | 65,241 | 35,024 | 142,838,765 |
| 1,080 | 76,432 | 38,337 | 345,000,342 |
| 1,920 | 121,558 | 66,920 | 569,062,553 |
| 1,009 | 183,398 | 43,517 | 135,299,463 |
| 1,296 | 84,474 | 45,032 | 259,554,589 |
| 10,007 | 2,150,355 | 415,064 | 1,518,784,501 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 8/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 1,080 (2³·3³·5): 1.42×
- complex 256 (2⁸): 1.34×
- real 65,536 (2¹⁶): 1.33×
- complex 1,024 (2¹⁰): 1.31×
- real 256 (2⁸): 1.30×
- real 4,096 (2¹²): 1.28×
- complex 1,920 (2⁷·3·5): 1.27×
- complex 1,000 (2³·5³): 1.23×
- complex 1,296 (2⁴·3⁴): 1.22×
- real 1,024 (2¹⁰): 1.20×
- real 1,080 (2³·3³·5): 1.19×
- 2-D 128x128: 1.17×
- real 1,000 (2³·5³): 1.16×
- real 1,920 (2⁷·3·5): 1.15×
- complex 65,536 (2¹⁶): 1.13×
- 2-D 64x64: 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
