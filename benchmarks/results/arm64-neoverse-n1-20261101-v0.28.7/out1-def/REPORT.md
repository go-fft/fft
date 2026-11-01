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
| 256 (2⁸) | 1,004 (10.2) | 1,146 (8.9) | 9,502 (1.1) | 7,599 (1.3) | 5,766 (1.8) | 0.88× | **≥ parity** |
| 1,024 (2¹⁰) | 4,858 (10.5) | 6,313 (8.1) | 18,858 (2.7) | 14,537 (3.5) | 30,207 (1.7) | 0.77× | **≥ parity** |
| 4,096 (2¹²) | 23,678 (10.4) | 40,838 (6.0) | 59,782 (4.1) | 47,368 (5.2) | 146,634 (1.7) | 0.58× | **≥ parity** |
| 65,536 (2¹⁶) | 710,815 (7.4) | 1,216,773 (4.3) | 2,101,347 (2.5) | 1,412,537 (3.7) | 3,379,866 (1.6) | 0.58× | **≥ parity** |
| 1,048,576 (2²⁰) | 19,657,276 (5.3) | 43,312,258 (2.4) | 42,457,389 (2.5) | 31,137,566 (3.4) | 73,488,399 (1.4) | 0.45× | **≥ parity** |
| 1,000 (2³·5³) | 5,983 (8.3) | 7,147 (7.0) | 19,224 (2.6) | 15,080 (3.3) | 33,883 (1.5) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,808 (8.0) | 7,660 (7.1) | 21,570 (2.5) | 16,628 (3.3) | 41,173 (1.3) | 0.89× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,828 (8.9) | 13,488 (7.8) | 33,090 (3.2) | 26,010 (4.0) | 69,646 (1.5) | 0.88× | **≥ parity** |
| 1,009 (prime) | 21,444 (2.3) | 47,993 (1.0) | 95,811 (0.5) | 56,988 (0.9) | 2,210,893 (0.0) | 0.45× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,600 (7.8) | 11,294 (5.9) | 24,767 (2.7) | 19,497 (3.4) | 48,612 (1.4) | 0.76× | **≥ parity** |
| 10,007 (prime) | 406,173 (1.6) | 592,640 (1.1) | 1,020,437 (0.7) | 757,715 (0.9) | 216,196,532 (0.0) | 0.69× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 709 (7.2) | 718 (7.1) | 8,671 (0.6) | 7,633 (0.7) | 2,971 (1.7) | 0.99× | **≥ parity** |
| 1,024 (2¹⁰) | 2,915 (8.8) | 4,200 (6.1) | 14,233 (1.8) | 12,301 (2.1) | 15,102 (1.7) | 0.69× | **≥ parity** |
| 4,096 (2¹²) | 14,055 (8.7) | 19,848 (6.2) | 37,111 (3.3) | 32,705 (3.8) | 67,800 (1.8) | 0.71× | **≥ parity** |
| 65,536 (2¹⁶) | 383,863 (6.8) | 539,256 (4.9) | 725,954 (3.6) | 751,489 (3.5) | 1,722,266 (1.5) | 0.71× | **≥ parity** |
| 1,048,576 (2²⁰) | 9,926,789 (5.3) | 16,781,994 (3.1) | 22,179,475 (2.4) | 19,453,660 (2.7) | 41,473,246 (1.3) | 0.59× | **≥ parity** |
| 1,000 (2³·5³) | 3,664 (6.8) | 3,939 (6.3) | 15,571 (1.6) | 12,838 (1.9) | 14,840 (1.7) | 0.93× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,158 (6.5) | 4,199 (6.5) | 16,011 (1.7) | 13,965 (1.9) | 17,333 (1.6) | 0.99× | **≥ parity** |
| 1,920 (2⁷·3·5) | 6,771 (7.7) | 7,306 (7.2) | 21,817 (2.4) | 18,921 (2.8) | 30,351 (1.7) | 0.93× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 729 (7.0) | 808 (6.3) | 0.90× | **≥ parity** |
| 1,024 (2¹⁰) | 2,940 (8.7) | 4,444 (5.8) | 0.66× | **≥ parity** |
| 4,096 (2¹²) | 13,958 (8.8) | 21,035 (5.8) | 0.66× | **≥ parity** |
| 65,536 (2¹⁶) | 378,405 (6.9) | 619,210 (4.2) | 0.61× | **≥ parity** |
| 1,048,576 (2²⁰) | 9,890,601 (5.3) | 19,675,612 (2.7) | 0.50× | **≥ parity** |
| 1,000 (2³·5³) | 3,664 (6.8) | 4,241 (5.9) | 0.86× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,151 (6.6) | 4,415 (6.2) | 0.94× | **≥ parity** |
| 1,920 (2⁷·3·5) | 6,713 (7.8) | 7,807 (6.7) | 0.86× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 24,268 (10.1) | 29,479 (8.3) | 30,916 (7.9) | 70,063 (3.5) | 49,693 (4.9) | 0.78× | **≥ parity** |
| 128x128 | 125,502 (9.1) | 145,081 (7.9) | 190,682 (6.0) | 258,839 (4.4) | 203,514 (5.6) | 0.66× | **≥ parity** |
| 256x256 | 605,822 (8.7) | 690,864 (7.6) | 1,248,590 (4.2) | 1,193,719 (4.4) | 946,860 (5.5) | 0.49× | **≥ parity** |
| 512x512 | 3,097,910 (7.6) | 3,787,080 (6.2) | 6,604,645 (3.6) | 5,392,053 (4.4) | 4,097,312 (5.8) | 0.47× | **≥ parity** |
| 1024x1024 | 17,408,653 (6.0) | 17,699,636 (5.9) | 43,273,710 (2.4) | 26,768,473 (3.9) | 20,765,919 (5.0) | 0.40× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 13,893 | 9,500 | 4,227,038 |
| 1,024 | 50,352 | 33,557 | 3,257,101 |
| 4,096 | 187,854 | 127,287 | 6,136,194 |
| 65,536 | 3,154,111 | 1,897,124 | 21,681,640 |
| 1,048,576 | 45,723,683 | 30,460,038 | 19,899,446 |
| 1,000 | 50,979 | 33,685 | 4,651,284 |
| 1,080 | 57,674 | 36,441 | 12,415,947 |
| 1,920 | 96,207 | 62,605 | 17,285,757 |
| 1,009 | 111,650 | 39,914 | 6,069,792 |
| 1,296 | 82,940 | 43,467 | 10,362,031 |
| 10,007 | 1,332,130 | 375,600 | 35,267,210 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 24/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
