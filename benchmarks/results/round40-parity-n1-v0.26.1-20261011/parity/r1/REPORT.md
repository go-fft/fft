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
| 256 (2⁸) | 1,011 (10.1) | 1,149 (8.9) | 9,499 (1.1) | 7,676 (1.3) | 5,854 (1.7) | 0.88× | **≥ parity** |
| 1,024 (2¹⁰) | 4,900 (10.4) | 6,335 (8.1) | 18,833 (2.7) | 14,551 (3.5) | 30,533 (1.7) | 0.77× | **≥ parity** |
| 4,096 (2¹²) | 23,789 (10.3) | 41,033 (6.0) | 60,003 (4.1) | 48,683 (5.0) | 145,843 (1.7) | 0.58× | **≥ parity** |
| 65,536 (2¹⁶) | 731,396 (7.2) | 1,232,651 (4.3) | 2,216,937 (2.4) | 1,477,209 (3.5) | 3,257,534 (1.6) | 0.59× | **≥ parity** |
| 1,048,576 (2²⁰) | 21,646,098 (4.8) | 44,140,574 (2.4) | 45,131,694 (2.3) | 31,515,140 (3.3) | 75,363,070 (1.4) | 0.49× | **≥ parity** |
| 1,000 (2³·5³) | 6,006 (8.3) | 7,180 (6.9) | 19,159 (2.6) | 15,233 (3.3) | 33,757 (1.5) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,859 (7.9) | 7,667 (7.1) | 21,819 (2.5) | 16,768 (3.2) | 41,667 (1.3) | 0.89× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,974 (8.7) | 13,588 (7.7) | 32,987 (3.2) | 26,248 (4.0) | 70,426 (1.5) | 0.88× | **≥ parity** |
| 1,009 (prime) | 21,611 (2.3) | 47,979 (1.0) | 95,926 (0.5) | 57,332 (0.9) | 2,219,780 (0.0) | 0.45× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,744 (7.7) | 11,336 (5.9) | 24,995 (2.7) | 19,642 (3.4) | 49,151 (1.4) | 0.77× | **≥ parity** |
| 10,007 (prime) | 415,927 (1.6) | 586,057 (1.1) | 1,023,746 (0.6) | 769,411 (0.9) | 220,554,999 (0.0) | 0.71× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 713 (7.2) | 722 (7.1) | 8,650 (0.6) | 8,042 (0.6) | 2,987 (1.7) | 0.99× | **≥ parity** |
| 1,024 (2¹⁰) | 2,927 (8.7) | 4,205 (6.1) | 14,412 (1.8) | 12,654 (2.0) | 15,167 (1.7) | 0.70× | **≥ parity** |
| 4,096 (2¹²) | 14,195 (8.7) | 19,845 (6.2) | 37,338 (3.3) | 33,149 (3.7) | 67,988 (1.8) | 0.72× | **≥ parity** |
| 65,536 (2¹⁶) | 390,412 (6.7) | 545,857 (4.8) | 722,322 (3.6) | 750,677 (3.5) | 1,756,153 (1.5) | 0.72× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,536,405 (5.0) | 18,481,928 (2.8) | 19,792,830 (2.6) | 18,671,014 (2.8) | 43,080,151 (1.2) | 0.57× | **≥ parity** |
| 1,000 (2³·5³) | 3,708 (6.7) | 3,957 (6.3) | 15,170 (1.6) | 12,628 (2.0) | 14,933 (1.7) | 0.94× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,150 (6.6) | 4,206 (6.5) | 15,887 (1.7) | 13,908 (2.0) | 17,464 (1.6) | 0.99× | **≥ parity** |
| 1,920 (2⁷·3·5) | 6,841 (7.7) | 7,361 (7.1) | 21,729 (2.4) | 19,089 (2.7) | 30,546 (1.7) | 0.93× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 739 (6.9) | 809 (6.3) | 0.91× | **≥ parity** |
| 1,024 (2¹⁰) | 2,993 (8.6) | 4,459 (5.7) | 0.67× | **≥ parity** |
| 4,096 (2¹²) | 14,079 (8.7) | 21,156 (5.8) | 0.67× | **≥ parity** |
| 65,536 (2¹⁶) | 378,728 (6.9) | 634,626 (4.1) | 0.60× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,946,908 (4.8) | 21,667,693 (2.4) | 0.51× | **≥ parity** |
| 1,000 (2³·5³) | 3,732 (6.7) | 4,241 (5.9) | 0.88× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,180 (6.5) | 4,442 (6.1) | 0.94× | **≥ parity** |
| 1,920 (2⁷·3·5) | 6,839 (7.7) | 7,806 (6.7) | 0.88× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 24,657 (10.0) | 30,044 (8.2) | 30,946 (7.9) | 70,966 (3.5) | 50,416 (4.9) | 0.80× | **≥ parity** |
| 128x128 | 129,316 (8.9) | 149,407 (7.7) | 196,271 (5.8) | 261,087 (4.4) | 205,610 (5.6) | 0.66× | **≥ parity** |
| 256x256 | 699,614 (7.5) | 715,123 (7.3) | 1,273,538 (4.1) | 1,094,361 (4.8) | 876,954 (6.0) | 0.55× | **≥ parity** |
| 512x512 | 3,291,636 (7.2) | 4,017,777 (5.9) | 6,834,860 (3.5) | 6,356,333 (3.7) | 4,435,853 (5.3) | 0.48× | **≥ parity** |
| 1024x1024 | 18,398,055 (5.7) | 20,687,325 (5.1) | 43,784,148 (2.4) | 27,353,023 (3.8) | 23,182,003 (4.5) | 0.42× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 14,880 | 9,955 | 4,588,148 |
| 1,024 | 52,484 | 34,159 | 3,411,170 |
| 4,096 | 194,391 | 133,128 | 6,169,731 |
| 65,536 | 3,188,470 | 2,007,726 | 21,422,640 |
| 1,048,576 | 48,377,317 | 28,833,517 | 20,316,064 |
| 1,000 | 55,345 | 35,376 | 4,781,512 |
| 1,080 | 63,504 | 38,869 | 12,617,510 |
| 1,920 | 132,686 | 65,485 | 17,326,258 |
| 1,009 | 116,646 | 40,452 | 6,082,930 |
| 1,296 | 70,999 | 44,416 | 10,332,914 |
| 10,007 | 1,473,492 | 383,154 | 35,233,565 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 24/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
