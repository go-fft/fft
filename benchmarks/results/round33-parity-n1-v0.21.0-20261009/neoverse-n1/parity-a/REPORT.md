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
| 256 (2⁸) | 1,011 (10.1) | 1,158 (8.8) | 9,398 (1.1) | 7,649 (1.3) | 5,764 (1.8) | 0.87× | **≥ parity** |
| 1,024 (2¹⁰) | 4,890 (10.5) | 6,364 (8.0) | 18,877 (2.7) | 14,508 (3.5) | 30,240 (1.7) | 0.77× | **≥ parity** |
| 4,096 (2¹²) | 23,772 (10.3) | 41,241 (6.0) | 60,278 (4.1) | 47,939 (5.1) | 146,450 (1.7) | 0.58× | **≥ parity** |
| 65,536 (2¹⁶) | 725,198 (7.2) | 1,229,518 (4.3) | 2,143,704 (2.4) | 1,419,418 (3.7) | 3,388,652 (1.5) | 0.59× | **≥ parity** |
| 1,048,576 (2²⁰) | 20,268,708 (5.2) | 44,166,537 (2.4) | 44,825,774 (2.3) | 30,310,153 (3.5) | 75,345,834 (1.4) | 0.46× | **≥ parity** |
| 1,000 (2³·5³) | 6,109 (8.2) | 7,164 (7.0) | 19,281 (2.6) | 15,919 (3.1) | 33,992 (1.5) | 0.85× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,850 (7.9) | 7,745 (7.0) | 22,093 (2.5) | 17,154 (3.2) | 41,628 (1.3) | 0.88× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,927 (8.8) | 13,605 (7.7) | 32,885 (3.2) | 25,837 (4.1) | 70,217 (1.5) | 0.88× | **≥ parity** |
| 1,009 (prime) | 21,796 (2.3) | 48,144 (1.0) | 97,647 (0.5) | 57,321 (0.9) | 2,233,492 (0.0) | 0.45× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,616 (7.8) | 11,418 (5.9) | 24,935 (2.7) | 19,715 (3.4) | 48,613 (1.4) | 0.75× | **≥ parity** |
| 10,007 (prime) | 420,364 (1.6) | 645,158 (1.0) | 1,023,738 (0.6) | 757,778 (0.9) | 217,062,368 (0.0) | 0.65× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 750 (6.8) | 721 (7.1) | 8,654 (0.6) | 7,754 (0.7) | 2,983 (1.7) | 1.04× | **≥ parity** |
| 1,024 (2¹⁰) | 3,095 (8.3) | 4,208 (6.1) | 14,211 (1.8) | 12,496 (2.0) | 15,171 (1.7) | 0.74× | **≥ parity** |
| 4,096 (2¹²) | 14,745 (8.3) | 19,871 (6.2) | 36,812 (3.3) | 32,687 (3.8) | 68,736 (1.8) | 0.74× | **≥ parity** |
| 65,536 (2¹⁶) | 396,473 (6.6) | 545,029 (4.8) | 717,588 (3.7) | 739,826 (3.5) | 1,738,680 (1.5) | 0.73× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,166,813 (5.2) | 18,631,661 (2.8) | 20,134,950 (2.6) | 17,970,094 (2.9) | 43,449,067 (1.2) | 0.55× | **≥ parity** |
| 1,000 (2³·5³) | 3,814 (6.5) | 3,952 (6.3) | 15,176 (1.6) | 12,551 (2.0) | 14,881 (1.7) | 0.97× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,347 (6.3) | 4,219 (6.4) | 15,914 (1.7) | 14,149 (1.9) | 17,368 (1.6) | 1.03× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,080 (7.4) | 7,328 (7.1) | 21,716 (2.4) | 18,828 (2.8) | 30,729 (1.7) | 0.97× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 774 (6.6) | 813 (6.3) | 0.95× | **≥ parity** |
| 1,024 (2¹⁰) | 3,096 (8.3) | 4,441 (5.8) | 0.70× | **≥ parity** |
| 4,096 (2¹²) | 14,585 (8.4) | 21,050 (5.8) | 0.69× | **≥ parity** |
| 65,536 (2¹⁶) | 395,194 (6.6) | 645,565 (4.1) | 0.61× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,463,167 (5.0) | 21,952,347 (2.4) | 0.48× | **≥ parity** |
| 1,000 (2³·5³) | 3,820 (6.5) | 4,273 (5.8) | 0.89× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,374 (6.2) | 4,441 (6.1) | 0.99× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,047 (7.4) | 8,009 (6.5) | 0.88× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 24,495 (10.0) | 29,697 (8.3) | 31,415 (7.8) | 70,071 (3.5) | 50,050 (4.9) | 0.78× | **≥ parity** |
| 128x128 | 129,111 (8.9) | 151,959 (7.5) | 198,863 (5.8) | 252,643 (4.5) | 204,026 (5.6) | 0.65× | **≥ parity** |
| 256x256 | 614,503 (8.5) | 705,002 (7.4) | 1,240,346 (4.2) | 1,194,762 (4.4) | 976,989 (5.4) | 0.50× | **≥ parity** |
| 512x512 | 3,189,134 (7.4) | 3,888,516 (6.1) | 6,578,719 (3.6) | 5,960,154 (4.0) | 4,075,000 (5.8) | 0.48× | **≥ parity** |
| 1024x1024 | 19,093,929 (5.5) | 18,391,010 (5.7) | 43,106,629 (2.4) | 27,360,033 (3.8) | 20,876,845 (5.0) | 0.44× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 14,850 | 9,962 | 4,448,056 |
| 1,024 | 52,661 | 34,419 | 3,927,928 |
| 4,096 | 195,130 | 132,727 | 6,338,179 |
| 65,536 | 3,125,266 | 2,003,724 | 21,717,193 |
| 1,048,576 | 48,854,348 | 29,476,370 | 20,215,440 |
| 1,000 | 55,163 | 35,080 | 4,729,863 |
| 1,080 | 62,373 | 38,025 | 12,479,631 |
| 1,920 | 102,804 | 63,722 | 17,793,668 |
| 1,009 | 117,066 | 39,905 | 6,099,213 |
| 1,296 | 70,851 | 43,530 | 10,391,507 |
| 10,007 | 1,415,622 | 376,506 | 35,402,570 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 24/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
