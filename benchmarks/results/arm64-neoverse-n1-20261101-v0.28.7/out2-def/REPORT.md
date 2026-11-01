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
| 256 (2⁸) | 1,004 (10.2) | 1,150 (8.9) | 9,472 (1.1) | 7,601 (1.3) | 5,767 (1.8) | 0.87× | **≥ parity** |
| 1,024 (2¹⁰) | 4,861 (10.5) | 6,325 (8.1) | 18,745 (2.7) | 14,503 (3.5) | 30,191 (1.7) | 0.77× | **≥ parity** |
| 4,096 (2¹²) | 23,731 (10.4) | 40,863 (6.0) | 59,177 (4.2) | 48,013 (5.1) | 145,153 (1.7) | 0.58× | **≥ parity** |
| 65,536 (2¹⁶) | 719,790 (7.3) | 1,224,343 (4.3) | 2,181,447 (2.4) | 1,471,102 (3.6) | 3,375,183 (1.6) | 0.59× | **≥ parity** |
| 1,048,576 (2²⁰) | 20,047,084 (5.2) | 43,212,813 (2.4) | 45,376,913 (2.3) | 30,794,065 (3.4) | 73,817,120 (1.4) | 0.46× | **≥ parity** |
| 1,000 (2³·5³) | 5,992 (8.3) | 7,157 (7.0) | 19,243 (2.6) | 15,117 (3.3) | 33,755 (1.5) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,812 (8.0) | 7,670 (7.1) | 21,592 (2.5) | 16,609 (3.3) | 41,332 (1.3) | 0.89× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,828 (8.9) | 13,503 (7.8) | 32,708 (3.2) | 25,807 (4.1) | 69,660 (1.5) | 0.88× | **≥ parity** |
| 1,009 (prime) | 21,429 (2.3) | 48,036 (1.0) | 96,480 (0.5) | 56,934 (0.9) | 2,210,436 (0.0) | 0.45× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,613 (7.8) | 11,328 (5.9) | 24,769 (2.7) | 19,470 (3.4) | 48,535 (1.4) | 0.76× | **≥ parity** |
| 10,007 (prime) | 399,789 (1.7) | 599,019 (1.1) | 1,028,763 (0.6) | 755,236 (0.9) | 216,298,604 (0.0) | 0.67× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 708 (7.2) | 718 (7.1) | 8,641 (0.6) | 7,525 (0.7) | 2,972 (1.7) | 0.99× | **≥ parity** |
| 1,024 (2¹⁰) | 2,957 (8.7) | 4,200 (6.1) | 14,224 (1.8) | 12,233 (2.1) | 15,051 (1.7) | 0.70× | **≥ parity** |
| 4,096 (2¹²) | 14,039 (8.8) | 19,896 (6.2) | 37,066 (3.3) | 32,336 (3.8) | 69,814 (1.8) | 0.71× | **≥ parity** |
| 65,536 (2¹⁶) | 379,950 (6.9) | 549,363 (4.8) | 732,281 (3.6) | 729,830 (3.6) | 1,746,609 (1.5) | 0.69× | **≥ parity** |
| 1,048,576 (2²⁰) | 9,831,143 (5.3) | 17,803,248 (2.9) | 19,277,077 (2.7) | 17,625,431 (3.0) | 40,661,260 (1.3) | 0.55× | **≥ parity** |
| 1,000 (2³·5³) | 3,654 (6.8) | 3,940 (6.3) | 15,207 (1.6) | 12,540 (2.0) | 14,857 (1.7) | 0.93× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,132 (6.6) | 4,201 (6.5) | 15,911 (1.7) | 13,992 (1.9) | 17,173 (1.6) | 0.98× | **≥ parity** |
| 1,920 (2⁷·3·5) | 6,763 (7.7) | 7,311 (7.2) | 21,740 (2.4) | 18,844 (2.8) | 30,421 (1.7) | 0.93× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 728 (7.0) | 809 (6.3) | 0.90× | **≥ parity** |
| 1,024 (2¹⁰) | 2,933 (8.7) | 4,445 (5.8) | 0.66× | **≥ parity** |
| 4,096 (2¹²) | 13,971 (8.8) | 21,030 (5.8) | 0.66× | **≥ parity** |
| 65,536 (2¹⁶) | 374,295 (7.0) | 675,857 (3.9) | 0.55× | **≥ parity** |
| 1,048,576 (2²⁰) | 9,699,194 (5.4) | 21,136,971 (2.5) | 0.46× | **≥ parity** |
| 1,000 (2³·5³) | 3,663 (6.8) | 4,241 (5.9) | 0.86× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,153 (6.6) | 4,416 (6.2) | 0.94× | **≥ parity** |
| 1,920 (2⁷·3·5) | 6,762 (7.7) | 7,802 (6.7) | 0.87× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 24,288 (10.1) | 29,652 (8.3) | 30,879 (8.0) | 71,071 (3.5) | 50,109 (4.9) | 0.79× | **≥ parity** |
| 128x128 | 124,800 (9.2) | 145,520 (7.9) | 192,958 (5.9) | 247,728 (4.6) | 194,879 (5.9) | 0.65× | **≥ parity** |
| 256x256 | 608,777 (8.6) | 708,348 (7.4) | 1,357,947 (3.9) | 1,077,789 (4.9) | 905,083 (5.8) | 0.45× | **≥ parity** |
| 512x512 | 3,134,498 (7.5) | 3,953,684 (6.0) | 7,366,402 (3.2) | 5,941,094 (4.0) | 4,086,259 (5.8) | 0.43× | **≥ parity** |
| 1024x1024 | 17,540,135 (6.0) | 18,149,524 (5.8) | 45,776,516 (2.3) | 25,571,840 (4.1) | 21,017,791 (5.0) | 0.38× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 13,966 | 9,559 | 4,031,002 |
| 1,024 | 50,227 | 33,775 | 3,268,005 |
| 4,096 | 187,155 | 127,728 | 6,146,967 |
| 65,536 | 3,146,196 | 1,908,495 | 21,386,506 |
| 1,048,576 | 56,692,890 | 30,212,746 | 19,944,066 |
| 1,000 | 51,770 | 35,090 | 4,664,566 |
| 1,080 | 57,843 | 36,959 | 12,431,057 |
| 1,920 | 96,925 | 62,858 | 17,330,065 |
| 1,009 | 112,619 | 39,925 | 6,088,490 |
| 1,296 | 80,455 | 44,124 | 10,732,777 |
| 10,007 | 1,337,553 | 381,329 | 35,240,818 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 24/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
