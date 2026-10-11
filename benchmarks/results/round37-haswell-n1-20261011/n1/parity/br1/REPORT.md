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
| 256 (2⁸) | 1,016 (10.1) | 1,152 (8.9) | 9,559 (1.1) | 7,679 (1.3) | 5,782 (1.8) | 0.88× | **≥ parity** |
| 1,024 (2¹⁰) | 4,872 (10.5) | 6,318 (8.1) | 18,643 (2.7) | 14,495 (3.5) | 30,281 (1.7) | 0.77× | **≥ parity** |
| 4,096 (2¹²) | 23,830 (10.3) | 41,095 (6.0) | 60,057 (4.1) | 47,696 (5.2) | 146,967 (1.7) | 0.58× | **≥ parity** |
| 65,536 (2¹⁶) | 724,235 (7.2) | 1,222,530 (4.3) | 2,114,561 (2.5) | 1,430,422 (3.7) | 3,451,185 (1.5) | 0.59× | **≥ parity** |
| 1,048,576 (2²⁰) | 21,306,734 (4.9) | 43,954,977 (2.4) | 43,439,148 (2.4) | 32,300,691 (3.2) | 75,716,288 (1.4) | 0.48× | **≥ parity** |
| 1,000 (2³·5³) | 6,029 (8.3) | 7,201 (6.9) | 19,825 (2.5) | 15,347 (3.2) | 33,875 (1.5) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,853 (7.9) | 7,758 (7.0) | 21,814 (2.5) | 16,761 (3.2) | 41,750 (1.3) | 0.88× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,954 (8.8) | 13,662 (7.7) | 32,847 (3.2) | 25,962 (4.0) | 69,628 (1.5) | 0.88× | **≥ parity** |
| 1,009 (prime) | 21,555 (2.3) | 48,141 (1.0) | 98,207 (0.5) | 58,564 (0.9) | 2,229,629 (0.0) | 0.45× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,623 (7.8) | 11,344 (5.9) | 24,779 (2.7) | 19,578 (3.4) | 48,887 (1.4) | 0.76× | **≥ parity** |
| 10,007 (prime) | 422,199 (1.6) | 596,047 (1.1) | 1,027,206 (0.6) | 763,022 (0.9) | 220,238,083 (0.0) | 0.71× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 713 (7.2) | 719 (7.1) | 8,615 (0.6) | 7,814 (0.7) | 3,001 (1.7) | 0.99× | **≥ parity** |
| 1,024 (2¹⁰) | 2,902 (8.8) | 4,241 (6.0) | 14,163 (1.8) | 12,470 (2.1) | 15,397 (1.7) | 0.68× | **≥ parity** |
| 4,096 (2¹²) | 14,146 (8.7) | 19,937 (6.2) | 37,140 (3.3) | 33,339 (3.7) | 69,766 (1.8) | 0.71× | **≥ parity** |
| 65,536 (2¹⁶) | 386,253 (6.8) | 547,904 (4.8) | 723,378 (3.6) | 755,349 (3.5) | 1,726,518 (1.5) | 0.70× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,089,798 (5.2) | 17,253,789 (3.0) | 19,790,867 (2.6) | 18,152,784 (2.9) | 41,445,989 (1.3) | 0.58× | **≥ parity** |
| 1,000 (2³·5³) | 3,693 (6.7) | 3,958 (6.3) | 15,088 (1.7) | 12,596 (2.0) | 14,855 (1.7) | 0.93× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,192 (6.5) | 4,206 (6.5) | 15,921 (1.7) | 14,099 (1.9) | 17,357 (1.6) | 1.00× | **≥ parity** |
| 1,920 (2⁷·3·5) | 6,773 (7.7) | 7,345 (7.1) | 22,303 (2.3) | 19,038 (2.7) | 30,783 (1.7) | 0.92× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 728 (7.0) | 811 (6.3) | 0.90× | **≥ parity** |
| 1,024 (2¹⁰) | 3,012 (8.5) | 4,444 (5.8) | 0.68× | **≥ parity** |
| 4,096 (2¹²) | 13,974 (8.8) | 21,093 (5.8) | 0.66× | **≥ parity** |
| 65,536 (2¹⁶) | 377,620 (6.9) | 631,181 (4.2) | 0.60× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,469,549 (5.0) | 20,433,672 (2.6) | 0.51× | **≥ parity** |
| 1,000 (2³·5³) | 3,673 (6.8) | 4,247 (5.9) | 0.86× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,174 (6.5) | 4,428 (6.1) | 0.94× | **≥ parity** |
| 1,920 (2⁷·3·5) | 6,731 (7.8) | 7,811 (6.7) | 0.86× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 24,732 (9.9) | 30,219 (8.1) | 30,905 (8.0) | 71,438 (3.4) | 50,162 (4.9) | 0.80× | **≥ parity** |
| 128x128 | 127,768 (9.0) | 149,997 (7.6) | 192,766 (6.0) | 253,387 (4.5) | 200,397 (5.7) | 0.66× | **≥ parity** |
| 256x256 | 619,681 (8.5) | 775,909 (6.8) | 1,266,098 (4.1) | 1,083,241 (4.8) | 920,037 (5.7) | 0.49× | **≥ parity** |
| 512x512 | 3,238,426 (7.3) | 4,137,216 (5.7) | 6,726,153 (3.5) | 5,712,964 (4.1) | 4,191,557 (5.6) | 0.48× | **≥ parity** |
| 1024x1024 | 18,142,522 (5.8) | 18,690,534 (5.6) | 43,394,569 (2.4) | 26,408,435 (4.0) | 20,866,965 (5.0) | 0.42× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 14,173 | 9,918 | 4,424,365 |
| 1,024 | 51,265 | 34,525 | 3,275,624 |
| 4,096 | 197,721 | 131,543 | 6,194,238 |
| 65,536 | 3,338,620 | 2,002,875 | 21,623,770 |
| 1,048,576 | 50,518,624 | 28,922,083 | 20,017,499 |
| 1,000 | 53,056 | 34,425 | 4,806,651 |
| 1,080 | 59,069 | 37,033 | 12,762,081 |
| 1,920 | 98,685 | 63,490 | 18,080,262 |
| 1,009 | 115,330 | 39,561 | 6,308,518 |
| 1,296 | 85,164 | 42,944 | 10,607,600 |
| 10,007 | 1,370,172 | 372,663 | 36,096,521 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 24/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
