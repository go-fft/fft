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
| 256 (2⁸) | 1,011 (10.1) | 1,149 (8.9) | 9,546 (1.1) | 7,608 (1.3) | 5,870 (1.7) | 0.88× | **≥ parity** |
| 1,024 (2¹⁰) | 4,872 (10.5) | 6,319 (8.1) | 18,784 (2.7) | 14,524 (3.5) | 30,329 (1.7) | 0.77× | **≥ parity** |
| 4,096 (2¹²) | 24,050 (10.2) | 40,867 (6.0) | 60,212 (4.1) | 47,503 (5.2) | 142,982 (1.7) | 0.59× | **≥ parity** |
| 65,536 (2¹⁶) | 720,079 (7.3) | 1,271,714 (4.1) | 2,270,416 (2.3) | 1,542,077 (3.4) | 3,434,139 (1.5) | 0.57× | **≥ parity** |
| 1,048,576 (2²⁰) | 20,280,017 (5.2) | 45,968,755 (2.3) | 47,948,825 (2.2) | 32,231,346 (3.3) | 74,944,384 (1.4) | 0.44× | **≥ parity** |
| 1,000 (2³·5³) | 6,013 (8.3) | 7,156 (7.0) | 19,248 (2.6) | 15,151 (3.3) | 34,007 (1.5) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,831 (8.0) | 7,668 (7.1) | 21,647 (2.5) | 16,762 (3.2) | 42,186 (1.3) | 0.89× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,896 (8.8) | 13,493 (7.8) | 33,067 (3.2) | 26,006 (4.0) | 71,179 (1.5) | 0.88× | **≥ parity** |
| 1,009 (prime) | 21,451 (2.3) | 48,302 (1.0) | 97,400 (0.5) | 57,174 (0.9) | 2,239,021 (0.0) | 0.44× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,713 (7.7) | 11,403 (5.9) | 25,022 (2.7) | 19,577 (3.4) | 48,455 (1.4) | 0.76× | **≥ parity** |
| 10,007 (prime) | 408,275 (1.6) | 620,476 (1.1) | 1,023,025 (0.6) | 751,954 (0.9) | 217,875,329 (0.0) | 0.66× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 714 (7.2) | 719 (7.1) | 8,630 (0.6) | 7,667 (0.7) | 3,006 (1.7) | 0.99× | **≥ parity** |
| 1,024 (2¹⁰) | 2,922 (8.8) | 4,202 (6.1) | 14,253 (1.8) | 12,473 (2.1) | 15,197 (1.7) | 0.70× | **≥ parity** |
| 4,096 (2¹²) | 14,268 (8.6) | 19,883 (6.2) | 37,290 (3.3) | 33,313 (3.7) | 69,073 (1.8) | 0.72× | **≥ parity** |
| 65,536 (2¹⁶) | 392,089 (6.7) | 542,864 (4.8) | 752,839 (3.5) | 769,884 (3.4) | 1,757,927 (1.5) | 0.72× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,209,310 (5.1) | 19,187,894 (2.7) | 19,443,220 (2.7) | 18,320,560 (2.9) | 43,466,057 (1.2) | 0.53× | **≥ parity** |
| 1,000 (2³·5³) | 3,707 (6.7) | 3,977 (6.3) | 15,170 (1.6) | 12,554 (2.0) | 14,945 (1.7) | 0.93× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,188 (6.5) | 4,209 (6.5) | 15,901 (1.7) | 13,942 (2.0) | 17,464 (1.6) | 0.99× | **≥ parity** |
| 1,920 (2⁷·3·5) | 6,738 (7.8) | 7,313 (7.2) | 21,753 (2.4) | 18,844 (2.8) | 30,786 (1.7) | 0.92× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 727 (7.0) | 813 (6.3) | 0.89× | **≥ parity** |
| 1,024 (2¹⁰) | 2,943 (8.7) | 4,451 (5.8) | 0.66× | **≥ parity** |
| 4,096 (2¹²) | 13,968 (8.8) | 21,111 (5.8) | 0.66× | **≥ parity** |
| 65,536 (2¹⁶) | 374,772 (7.0) | 641,325 (4.1) | 0.58× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,405,310 (5.0) | 23,425,424 (2.2) | 0.44× | **≥ parity** |
| 1,000 (2³·5³) | 3,722 (6.7) | 4,242 (5.9) | 0.88× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,222 (6.4) | 4,422 (6.2) | 0.95× | **≥ parity** |
| 1,920 (2⁷·3·5) | 6,741 (7.8) | 7,890 (6.6) | 0.85× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 24,302 (10.1) | 32,092 (7.7) | 31,076 (7.9) | 70,458 (3.5) | 50,403 (4.9) | 0.78× | **≥ parity** |
| 128x128 | 125,822 (9.1) | 155,134 (7.4) | 193,680 (5.9) | 257,740 (4.4) | 196,046 (5.9) | 0.65× | **≥ parity** |
| 256x256 | 665,450 (7.9) | 798,345 (6.6) | 1,271,900 (4.1) | 1,089,916 (4.8) | 913,042 (5.7) | 0.52× | **≥ parity** |
| 512x512 | 3,177,381 (7.4) | 4,187,000 (5.6) | 6,873,959 (3.4) | 6,658,681 (3.5) | 4,204,568 (5.6) | 0.46× | **≥ parity** |
| 1024x1024 | 18,050,259 (5.8) | 18,343,699 (5.7) | 44,445,458 (2.4) | 27,320,842 (3.8) | 20,967,843 (5.0) | 0.41× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 14,418 | 9,679 | 4,591,148 |
| 1,024 | 52,052 | 35,170 | 3,285,250 |
| 4,096 | 194,674 | 129,509 | 6,180,612 |
| 65,536 | 3,140,743 | 1,919,446 | 21,795,966 |
| 1,048,576 | 100,717,782 | 31,751,173 | 20,034,898 |
| 1,000 | 52,790 | 34,283 | 4,667,189 |
| 1,080 | 59,664 | 37,006 | 12,483,105 |
| 1,920 | 97,808 | 63,070 | 17,653,985 |
| 1,009 | 114,879 | 39,972 | 6,098,570 |
| 1,296 | 86,265 | 44,323 | 10,338,874 |
| 10,007 | 1,357,866 | 400,037 | 37,025,312 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 24/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
