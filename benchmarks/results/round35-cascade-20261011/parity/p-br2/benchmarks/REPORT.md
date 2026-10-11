# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon (Cascade Lake), cfarm151, 1 pinned core.
- **Toolchains**: go1.27.2 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512** (3.3.10 built from source with AVX-512 (setup.sh)); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 389 (26.3) | 370 (27.7) | 9,024 (1.1) | 6,342 (1.6) | 5,506 (1.9) | 1.05× | lags FFTW 1.05× |
| 1,024 (2¹⁰) | 2,559 (20.0) | 2,154 (23.8) | 16,526 (3.1) | 11,757 (4.4) | 28,707 (1.8) | 1.19× | lags FFTW 1.19× |
| 4,096 (2¹²) | 12,411 (19.8) | 12,861 (19.1) | 58,809 (4.2) | 43,743 (5.6) | 132,999 (1.8) | 0.97× | **≥ parity** |
| 65,536 (2¹⁶) | 440,382 (11.9) | 437,233 (12.0) | 1,940,162 (2.7) | 1,369,891 (3.8) | 3,472,357 (1.5) | 1.01× | **≥ parity** |
| 1,048,576 (2²⁰) | 17,842,066 (5.9) | 27,125,068 (3.9) | 52,427,724 (2.0) | 41,220,682 (2.5) | 98,239,436 (1.1) | 0.66× | **≥ parity** |
| 1,000 (2³·5³) | 2,996 (16.6) | 2,748 (18.1) | 19,243 (2.6) | 14,359 (3.5) | 30,095 (1.7) | 1.09× | lags FFTW 1.09× |
| 1,080 (2³·3³·5) | 3,423 (15.9) | 3,077 (17.7) | 20,762 (2.6) | 13,314 (4.1) | 35,028 (1.6) | 1.11× | lags FFTW 1.11× |
| 1,920 (2⁷·3·5) | 5,517 (19.0) | 4,930 (21.2) | 31,462 (3.3) | 23,134 (4.5) | 61,815 (1.7) | 1.12× | lags FFTW 1.12× |
| 1,009 (prime) | 18,021 (2.8) | 30,805 (1.6) | 79,901 (0.6) | 56,389 (0.9) | 1,422,763 (0.0) | 0.59× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,966 (16.9) | 3,681 (18.2) | 20,682 (3.2) | 15,449 (4.3) | 41,459 (1.6) | 1.08× | lags FFTW 1.08× |
| 10,007 (prime) | 336,600 (2.0) | 366,570 (1.8) | 959,596 (0.7) | 725,905 (0.9) | 139,922,710 (0.0) | 0.92× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 346 (14.8) | 363 (14.1) | 6,658 (0.8) | 6,497 (0.8) | 2,648 (1.9) | 0.96× | **≥ parity** |
| 1,024 (2¹⁰) | 1,209 (21.2) | 1,514 (16.9) | 13,207 (1.9) | 10,047 (2.5) | 13,159 (1.9) | 0.80× | **≥ parity** |
| 4,096 (2¹²) | 8,101 (15.2) | 7,710 (15.9) | 32,624 (3.8) | 31,562 (3.9) | 60,975 (2.0) | 1.05× | lags FFTW 1.05× |
| 65,536 (2¹⁶) | 211,527 (12.4) | 186,129 (14.1) | 627,858 (4.2) | 703,982 (3.7) | 1,437,309 (1.8) | 1.14× | lags FFTW 1.14× |
| 1,048,576 (2²⁰) | 8,961,396 (5.9) | 9,630,197 (5.4) | 18,154,104 (2.9) | 21,834,688 (2.4) | 44,567,239 (1.2) | 0.93× | **≥ parity** |
| 1,000 (2³·5³) | 1,765 (14.1) | 1,866 (13.4) | 14,227 (1.8) | 12,002 (2.1) | 14,222 (1.8) | 0.95× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,066 (13.2) | 2,186 (12.4) | 13,537 (2.0) | 11,263 (2.4) | 18,737 (1.5) | 0.94× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,394 (15.4) | 3,445 (15.2) | 18,305 (2.9) | 17,197 (3.0) | 33,024 (1.6) | 0.99× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 363 (14.1) | 536 (9.6) | 0.68× | **≥ parity** |
| 1,024 (2¹⁰) | 1,323 (19.3) | 1,868 (13.7) | 0.71× | **≥ parity** |
| 4,096 (2¹²) | 8,141 (15.1) | 8,723 (14.1) | 0.93× | **≥ parity** |
| 65,536 (2¹⁶) | 253,185 (10.4) | 218,033 (12.0) | 1.16× | lags FFTW 1.16× |
| 1,048,576 (2²⁰) | 10,924,506 (4.8) | 10,197,779 (5.1) | 1.07× | lags FFTW 1.07× |
| 1,000 (2³·5³) | 1,891 (13.2) | 2,263 (11.0) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,231 (12.2) | 2,559 (10.6) | 0.87× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,710 (14.1) | 3,936 (13.3) | 0.94× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,942 (13.7) | 31,174 (7.9) | 15,127 (16.2) | 56,171 (4.4) | 45,993 (5.3) | 1.19× | lags FFTW 1.19× |
| 128x128 | 79,385 (14.4) | 145,621 (7.9) | 82,010 (14.0) | 210,087 (5.5) | 173,331 (6.6) | 0.97× | **≥ parity** |
| 256x256 | 460,378 (11.4) | 781,497 (6.7) | 471,219 (11.1) | 910,584 (5.8) | 757,926 (6.9) | 0.98× | **≥ parity** |
| 512x512 | 3,552,983 (6.6) | 5,170,398 (4.6) | 2,596,248 (9.1) | 5,952,878 (4.0) | 3,888,032 (6.1) | 1.37× | lags FFTW 1.37× |
| 1024x1024 | 16,266,557 (6.4) | 20,849,043 (5.0) | 24,133,954 (4.3) | 30,008,724 (3.5) | 25,627,305 (4.1) | 0.67× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 23,155 | 10,567 | 69,722,138 |
| 1,024 | 81,069 | 38,978 | 127,205,381 |
| 4,096 | 327,200 | 146,598 | 277,326,716 |
| 65,536 | 4,353,301 | 2,326,794 | 3,126,718,637 |
| 1,048,576 | 74,662,203 | 35,880,702 | 8,365,775,963 |
| 1,000 | 67,713 | 37,261 | 145,155,873 |
| 1,080 | 70,428 | 39,068 | 342,450,103 |
| 1,920 | 146,463 | 68,618 | 561,990,738 |
| 1,009 | 179,578 | 44,899 | 136,755,571 |
| 1,296 | 83,623 | 47,925 | 262,165,122 |
| 10,007 | 2,202,408 | 423,936 | 1,525,238,277 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 14/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- 2-D 512x512: 1.37×
- complex 1,024 (2¹⁰): 1.19×
- 2-D 64x64: 1.19×
- real 65,536 (2¹⁶): 1.14×
- complex 1,920 (2⁷·3·5): 1.12×
- complex 1,080 (2³·3³·5): 1.11×
- complex 1,000 (2³·5³): 1.09×
- complex 1,296 (2⁴·3⁴): 1.08×
- complex 256 (2⁸): 1.05×
- real 4,096 (2¹²): 1.05×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
