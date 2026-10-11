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
| 256 (2⁸) | 1,015 (10.1) | 1,160 (8.8) | 9,539 (1.1) | 7,767 (1.3) | 5,859 (1.7) | 0.88× | **≥ parity** |
| 1,024 (2¹⁰) | 4,953 (10.3) | 6,385 (8.0) | 18,837 (2.7) | 14,703 (3.5) | 30,556 (1.7) | 0.78× | **≥ parity** |
| 4,096 (2¹²) | 24,100 (10.2) | 41,411 (5.9) | 59,967 (4.1) | 49,188 (5.0) | 147,725 (1.7) | 0.58× | **≥ parity** |
| 65,536 (2¹⁶) | 726,143 (7.2) | 1,239,516 (4.2) | 2,290,928 (2.3) | 1,544,854 (3.4) | 3,439,932 (1.5) | 0.59× | **≥ parity** |
| 1,048,576 (2²⁰) | 20,220,921 (5.2) | 43,931,365 (2.4) | 44,387,083 (2.4) | 32,741,965 (3.2) | 74,598,799 (1.4) | 0.46× | **≥ parity** |
| 1,000 (2³·5³) | 6,109 (8.2) | 7,296 (6.8) | 19,585 (2.5) | 15,141 (3.3) | 34,121 (1.5) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,871 (7.9) | 7,724 (7.0) | 21,733 (2.5) | 16,786 (3.2) | 41,528 (1.3) | 0.89× | **≥ parity** |
| 1,920 (2⁷·3·5) | 12,047 (8.7) | 13,608 (7.7) | 32,910 (3.2) | 26,076 (4.0) | 69,774 (1.5) | 0.89× | **≥ parity** |
| 1,009 (prime) | 21,712 (2.3) | 48,574 (1.0) | 96,317 (0.5) | 57,708 (0.9) | 2,229,852 (0.0) | 0.45× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,744 (7.7) | 11,323 (5.9) | 24,794 (2.7) | 19,723 (3.4) | 48,691 (1.4) | 0.77× | **≥ parity** |
| 10,007 (prime) | 420,584 (1.6) | 597,804 (1.1) | 1,031,826 (0.6) | 765,576 (0.9) | 216,546,780 (0.0) | 0.70× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 750 (6.8) | 723 (7.1) | 8,778 (0.6) | 7,751 (0.7) | 3,005 (1.7) | 1.04× | **≥ parity** |
| 1,024 (2¹⁰) | 3,150 (8.1) | 4,217 (6.1) | 14,278 (1.8) | 12,862 (2.0) | 15,195 (1.7) | 0.75× | **≥ parity** |
| 4,096 (2¹²) | 14,880 (8.3) | 19,864 (6.2) | 38,065 (3.2) | 33,005 (3.7) | 69,728 (1.8) | 0.75× | **≥ parity** |
| 65,536 (2¹⁶) | 404,996 (6.5) | 542,640 (4.8) | 741,056 (3.5) | 781,127 (3.4) | 1,732,221 (1.5) | 0.75× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,260,836 (5.1) | 18,589,410 (2.8) | 19,781,637 (2.7) | 18,452,277 (2.8) | 42,250,673 (1.2) | 0.55× | **≥ parity** |
| 1,000 (2³·5³) | 3,885 (6.4) | 3,944 (6.3) | 15,463 (1.6) | 12,963 (1.9) | 14,915 (1.7) | 0.99× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,346 (6.3) | 4,217 (6.5) | 16,464 (1.7) | 14,304 (1.9) | 17,299 (1.6) | 1.03× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,145 (7.3) | 7,324 (7.1) | 21,828 (2.4) | 18,898 (2.8) | 30,341 (1.7) | 0.98× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 764 (6.7) | 809 (6.3) | 0.94× | **≥ parity** |
| 1,024 (2¹⁰) | 3,103 (8.3) | 4,455 (5.7) | 0.70× | **≥ parity** |
| 4,096 (2¹²) | 14,691 (8.4) | 21,100 (5.8) | 0.70× | **≥ parity** |
| 65,536 (2¹⁶) | 384,605 (6.8) | 646,271 (4.1) | 0.60× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,143,550 (5.2) | 21,861,797 (2.4) | 0.46× | **≥ parity** |
| 1,000 (2³·5³) | 3,830 (6.5) | 4,326 (5.8) | 0.89× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,344 (6.3) | 4,439 (6.1) | 0.98× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,109 (7.4) | 7,881 (6.6) | 0.90× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 24,624 (10.0) | 32,285 (7.6) | 31,250 (7.9) | 70,078 (3.5) | 50,153 (4.9) | 0.79× | **≥ parity** |
| 128x128 | 127,430 (9.0) | 155,759 (7.4) | 194,238 (5.9) | 254,909 (4.5) | 202,651 (5.7) | 0.66× | **≥ parity** |
| 256x256 | 632,558 (8.3) | 793,206 (6.6) | 1,245,788 (4.2) | 1,087,144 (4.8) | 858,650 (6.1) | 0.51× | **≥ parity** |
| 512x512 | 3,218,732 (7.3) | 4,158,252 (5.7) | 7,201,799 (3.3) | 5,490,442 (4.3) | 4,104,454 (5.7) | 0.45× | **≥ parity** |
| 1024x1024 | 18,018,482 (5.8) | 19,181,750 (5.5) | 44,975,786 (2.3) | 27,205,990 (3.9) | 22,741,916 (4.6) | 0.40× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 14,909 | 10,014 | 4,651,949 |
| 1,024 | 52,902 | 34,278 | 3,293,753 |
| 4,096 | 197,028 | 129,147 | 6,259,346 |
| 65,536 | 3,218,177 | 1,999,033 | 21,733,474 |
| 1,048,576 | 48,891,225 | 28,536,056 | 20,718,757 |
| 1,000 | 53,626 | 35,805 | 4,836,872 |
| 1,080 | 61,692 | 37,774 | 12,657,415 |
| 1,920 | 103,075 | 66,223 | 17,630,648 |
| 1,009 | 152,555 | 39,993 | 6,268,630 |
| 1,296 | 71,104 | 45,559 | 10,431,141 |
| 10,007 | 1,405,526 | 389,256 | 35,447,702 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 24/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
