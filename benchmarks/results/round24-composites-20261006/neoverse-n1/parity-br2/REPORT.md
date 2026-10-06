# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Ampere Altra (Neoverse-N1), cfarm424, 64 cores, pinned to one core.
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
| 256 (2⁸) | 1,109 (9.2) | 1,181 (8.7) | 10,589 (1.0) | 8,278 (1.2) | 5,813 (1.8) | 0.94× | **≥ parity** |
| 1,024 (2¹⁰) | 5,230 (9.8) | 6,597 (7.8) | 20,239 (2.5) | 15,242 (3.4) | 30,328 (1.7) | 0.79× | **≥ parity** |
| 4,096 (2¹²) | 27,547 (8.9) | 42,183 (5.8) | 61,460 (4.0) | 48,379 (5.1) | 145,857 (1.7) | 0.65× | **≥ parity** |
| 65,536 (2¹⁶) | 734,890 (7.1) | 1,432,597 (3.7) | 2,419,369 (2.2) | 1,508,256 (3.5) | 3,297,895 (1.6) | 0.51× | **≥ parity** |
| 1,048,576 (2²⁰) | 22,678,198 (4.6) | 48,199,943 (2.2) | 45,490,557 (2.3) | 31,628,638 (3.3) | 76,727,370 (1.4) | 0.47× | **≥ parity** |
| 1,000 (2³·5³) | 5,997 (8.3) | 7,353 (6.8) | 19,105 (2.6) | 15,024 (3.3) | 33,712 (1.5) | 0.82× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,811 (8.0) | 7,800 (7.0) | 21,611 (2.5) | 16,596 (3.3) | 41,427 (1.3) | 0.87× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,867 (8.8) | 13,736 (7.6) | 32,667 (3.2) | 25,799 (4.1) | 69,919 (1.5) | 0.86× | **≥ parity** |
| 1,009 (prime) | 21,361 (2.4) | 48,538 (1.0) | 96,233 (0.5) | 57,954 (0.9) | 2,225,385 (0.0) | 0.44× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,722 (7.7) | 11,408 (5.9) | 25,400 (2.6) | 20,205 (3.3) | 48,493 (1.4) | 0.76× | **≥ parity** |
| 10,007 (prime) | 424,271 (1.6) | 606,176 (1.1) | 1,042,818 (0.6) | 773,434 (0.9) | 216,646,738 (0.0) | 0.70× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 823 (6.2) | 724 (7.1) | 9,186 (0.6) | 8,036 (0.6) | 2,993 (1.7) | 1.14× | lags FFTW 1.14× |
| 1,024 (2¹⁰) | 3,544 (7.2) | 4,248 (6.0) | 14,347 (1.8) | 12,383 (2.1) | 15,231 (1.7) | 0.83× | **≥ parity** |
| 4,096 (2¹²) | 16,460 (7.5) | 19,877 (6.2) | 37,335 (3.3) | 33,013 (3.7) | 68,627 (1.8) | 0.83× | **≥ parity** |
| 65,536 (2¹⁶) | 420,017 (6.2) | 575,112 (4.6) | 757,750 (3.5) | 781,033 (3.4) | 1,804,297 (1.5) | 0.73× | **≥ parity** |
| 1,048,576 (2²⁰) | 11,758,743 (4.5) | 22,331,136 (2.3) | 21,774,284 (2.4) | 19,411,334 (2.7) | 45,496,208 (1.2) | 0.53× | **≥ parity** |
| 1,000 (2³·5³) | 4,115 (6.1) | 3,942 (6.3) | 15,526 (1.6) | 12,550 (2.0) | 15,433 (1.6) | 1.04× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,611 (5.9) | 4,199 (6.5) | 15,913 (1.7) | 14,069 (1.9) | 17,846 (1.5) | 1.10× | lags FFTW 1.10× |
| 1,920 (2⁷·3·5) | 7,588 (6.9) | 7,330 (7.1) | 22,306 (2.3) | 18,932 (2.8) | 31,108 (1.7) | 1.04× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 953 (5.4) | 809 (6.3) | 1.18× | lags FFTW 1.18× |
| 1,024 (2¹⁰) | 3,955 (6.5) | 4,438 (5.8) | 0.89× | **≥ parity** |
| 4,096 (2¹²) | 17,798 (6.9) | 21,160 (5.8) | 0.84× | **≥ parity** |
| 65,536 (2¹⁶) | 447,801 (5.9) | 648,285 (4.0) | 0.69× | **≥ parity** |
| 1,048,576 (2²⁰) | 13,229,668 (4.0) | 27,572,530 (1.9) | 0.48× | **≥ parity** |
| 1,000 (2³·5³) | 4,468 (5.6) | 4,323 (5.8) | 1.03× | **≥ parity** |
| 1,080 (2³·3³·5) | 5,004 (5.4) | 4,504 (6.0) | 1.11× | lags FFTW 1.11× |
| 1,920 (2⁷·3·5) | 8,213 (6.4) | 7,895 (6.6) | 1.04× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 35,015 (7.0) | 42,207 (5.8) | 31,621 (7.8) | 69,870 (3.5) | 50,219 (4.9) | 1.11× | lags FFTW 1.11× |
| 128x128 | 142,043 (8.1) | 174,782 (6.6) | 200,040 (5.7) | 250,180 (4.6) | 200,016 (5.7) | 0.71× | **≥ parity** |
| 256x256 | 760,399 (6.9) | 877,469 (6.0) | 1,505,370 (3.5) | 1,029,281 (5.1) | 846,996 (6.2) | 0.51× | **≥ parity** |
| 512x512 | 3,709,414 (6.4) | 4,613,456 (5.1) | 9,019,644 (2.6) | 6,339,961 (3.7) | 4,249,940 (5.6) | 0.41× | **≥ parity** |
| 1024x1024 | 20,004,028 (5.2) | 20,965,761 (5.0) | 50,986,421 (2.1) | 27,137,605 (3.9) | 23,309,490 (4.5) | 0.39× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 15,919 | 10,898 | 6,241,074 |
| 1,024 | 50,888 | 37,304 | 3,359,640 |
| 4,096 | 183,158 | 137,289 | 7,355,688 |
| 65,536 | 3,145,606 | 2,056,783 | 22,156,943 |
| 1,048,576 | 42,861,923 | 34,593,944 | 21,335,611 |
| 1,000 | 50,222 | 36,634 | 4,796,056 |
| 1,080 | 57,442 | 40,089 | 14,402,250 |
| 1,920 | 94,321 | 67,696 | 17,389,325 |
| 1,009 | 121,192 | 43,777 | 6,285,515 |
| 1,296 | 80,547 | 55,276 | 10,766,609 |
| 10,007 | 1,483,048 | 414,558 | 36,372,269 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 21/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.14×
- 2-D 64x64: 1.11×
- real 1,080 (2³·3³·5): 1.10×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
