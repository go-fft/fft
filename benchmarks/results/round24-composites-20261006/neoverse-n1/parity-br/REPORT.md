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
| 256 (2⁸) | 1,111 (9.2) | 1,151 (8.9) | 9,732 (1.1) | 7,638 (1.3) | 5,823 (1.8) | 0.97× | **≥ parity** |
| 1,024 (2¹⁰) | 5,230 (9.8) | 6,318 (8.1) | 18,854 (2.7) | 15,388 (3.3) | 30,246 (1.7) | 0.83× | **≥ parity** |
| 4,096 (2¹²) | 27,584 (8.9) | 40,635 (6.0) | 59,331 (4.1) | 48,901 (5.0) | 145,403 (1.7) | 0.68× | **≥ parity** |
| 65,536 (2¹⁶) | 724,123 (7.2) | 1,233,911 (4.2) | 2,170,957 (2.4) | 1,435,590 (3.7) | 3,309,755 (1.6) | 0.59× | **≥ parity** |
| 1,048,576 (2²⁰) | 20,448,227 (5.1) | 44,183,917 (2.4) | 46,583,661 (2.3) | 33,034,362 (3.2) | 76,593,573 (1.4) | 0.46× | **≥ parity** |
| 1,000 (2³·5³) | 5,978 (8.3) | 7,173 (6.9) | 19,220 (2.6) | 15,272 (3.3) | 33,610 (1.5) | 0.83× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,834 (8.0) | 7,688 (7.1) | 21,693 (2.5) | 16,667 (3.3) | 41,361 (1.3) | 0.89× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,987 (8.7) | 13,562 (7.7) | 33,007 (3.2) | 26,061 (4.0) | 69,695 (1.5) | 0.88× | **≥ parity** |
| 1,009 (prime) | 21,577 (2.3) | 48,290 (1.0) | 97,299 (0.5) | 57,722 (0.9) | 2,208,803 (0.0) | 0.45× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,630 (7.8) | 11,434 (5.9) | 24,818 (2.7) | 19,667 (3.4) | 48,564 (1.4) | 0.75× | **≥ parity** |
| 10,007 (prime) | 433,067 (1.5) | 590,604 (1.1) | 1,014,295 (0.7) | 763,991 (0.9) | 217,031,493 (0.0) | 0.73× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 824 (6.2) | 722 (7.1) | 8,814 (0.6) | 7,725 (0.7) | 3,008 (1.7) | 1.14× | lags FFTW 1.14× |
| 1,024 (2¹⁰) | 3,550 (7.2) | 4,189 (6.1) | 14,197 (1.8) | 12,487 (2.1) | 15,269 (1.7) | 0.85× | **≥ parity** |
| 4,096 (2¹²) | 16,369 (7.5) | 19,685 (6.2) | 37,197 (3.3) | 32,934 (3.7) | 69,377 (1.8) | 0.83× | **≥ parity** |
| 65,536 (2¹⁶) | 421,812 (6.2) | 555,956 (4.7) | 731,403 (3.6) | 743,003 (3.5) | 1,766,347 (1.5) | 0.76× | **≥ parity** |
| 1,048,576 (2²⁰) | 13,444,708 (3.9) | 22,339,447 (2.3) | 23,958,699 (2.2) | 19,565,035 (2.7) | 44,282,858 (1.2) | 0.60× | **≥ parity** |
| 1,000 (2³·5³) | 4,131 (6.0) | 3,962 (6.3) | 15,284 (1.6) | 12,687 (2.0) | 14,673 (1.7) | 1.04× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,601 (5.9) | 4,202 (6.5) | 15,985 (1.7) | 14,106 (1.9) | 17,141 (1.6) | 1.09× | lags FFTW 1.09× |
| 1,920 (2⁷·3·5) | 7,606 (6.9) | 7,328 (7.1) | 21,923 (2.4) | 18,899 (2.8) | 30,332 (1.7) | 1.04× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 942 (5.4) | 813 (6.3) | 1.16× | lags FFTW 1.16× |
| 1,024 (2¹⁰) | 3,863 (6.6) | 4,468 (5.7) | 0.86× | **≥ parity** |
| 4,096 (2¹²) | 17,829 (6.9) | 21,244 (5.8) | 0.84× | **≥ parity** |
| 65,536 (2¹⁶) | 439,314 (6.0) | 626,335 (4.2) | 0.70× | **≥ parity** |
| 1,048,576 (2²⁰) | 12,165,831 (4.3) | 23,294,928 (2.3) | 0.52× | **≥ parity** |
| 1,000 (2³·5³) | 4,447 (5.6) | 4,258 (5.9) | 1.04× | **≥ parity** |
| 1,080 (2³·3³·5) | 5,011 (5.4) | 4,428 (6.1) | 1.13× | lags FFTW 1.13× |
| 1,920 (2⁷·3·5) | 8,226 (6.4) | 7,808 (6.7) | 1.05× | lags FFTW 1.05× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 34,427 (7.1) | 40,504 (6.1) | 30,937 (7.9) | 69,375 (3.5) | 50,723 (4.8) | 1.11× | lags FFTW 1.11× |
| 128x128 | 141,009 (8.1) | 164,823 (7.0) | 195,613 (5.9) | 254,353 (4.5) | 201,279 (5.7) | 0.72× | **≥ parity** |
| 256x256 | 751,869 (7.0) | 830,836 (6.3) | 1,310,868 (4.0) | 1,089,495 (4.8) | 920,835 (5.7) | 0.57× | **≥ parity** |
| 512x512 | 3,580,687 (6.6) | 4,268,708 (5.5) | 7,527,904 (3.1) | 6,791,966 (3.5) | 4,571,875 (5.2) | 0.48× | **≥ parity** |
| 1024x1024 | 17,993,261 (5.8) | 19,211,329 (5.5) | 46,190,041 (2.3) | 29,275,094 (3.6) | 24,865,057 (4.2) | 0.39× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 14,410 | 11,200 | 4,562,054 |
| 1,024 | 48,851 | 35,821 | 3,276,758 |
| 4,096 | 169,937 | 133,708 | 6,343,795 |
| 65,536 | 3,657,128 | 1,943,123 | 21,417,091 |
| 1,048,576 | 39,246,089 | 28,811,787 | 20,345,477 |
| 1,000 | 46,996 | 33,812 | 4,682,496 |
| 1,080 | 52,872 | 36,770 | 12,511,866 |
| 1,920 | 89,783 | 63,526 | 17,584,166 |
| 1,009 | 120,721 | 40,447 | 6,090,112 |
| 1,296 | 65,458 | 44,009 | 10,615,043 |
| 10,007 | 1,445,170 | 409,258 | 36,838,433 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 21/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.14×
- 2-D 64x64: 1.11×
- real 1,080 (2³·3³·5): 1.09×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
