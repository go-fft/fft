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
| 256 (2⁸) | 1,014 (10.1) | 1,152 (8.9) | 9,470 (1.1) | 7,716 (1.3) | 5,838 (1.8) | 0.88× | **≥ parity** |
| 1,024 (2¹⁰) | 4,888 (10.5) | 6,322 (8.1) | 18,777 (2.7) | 14,500 (3.5) | 30,552 (1.7) | 0.77× | **≥ parity** |
| 4,096 (2¹²) | 23,913 (10.3) | 41,110 (6.0) | 59,092 (4.2) | 47,243 (5.2) | 145,758 (1.7) | 0.58× | **≥ parity** |
| 65,536 (2¹⁶) | 712,370 (7.4) | 1,225,504 (4.3) | 2,287,804 (2.3) | 1,415,487 (3.7) | 3,386,032 (1.5) | 0.58× | **≥ parity** |
| 1,048,576 (2²⁰) | 20,088,309 (5.2) | 45,056,854 (2.3) | 43,897,203 (2.4) | 31,774,493 (3.3) | 74,624,315 (1.4) | 0.45× | **≥ parity** |
| 1,000 (2³·5³) | 6,008 (8.3) | 7,195 (6.9) | 19,317 (2.6) | 15,300 (3.3) | 34,088 (1.5) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,867 (7.9) | 7,757 (7.0) | 21,658 (2.5) | 16,686 (3.3) | 41,559 (1.3) | 0.89× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,973 (8.7) | 13,624 (7.7) | 32,750 (3.2) | 25,864 (4.0) | 70,245 (1.5) | 0.88× | **≥ parity** |
| 1,009 (prime) | 21,711 (2.3) | 48,512 (1.0) | 95,892 (0.5) | 57,070 (0.9) | 2,225,371 (0.0) | 0.45× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,719 (7.7) | 11,441 (5.9) | 25,029 (2.7) | 19,798 (3.4) | 48,704 (1.4) | 0.76× | **≥ parity** |
| 10,007 (prime) | 425,426 (1.6) | 603,698 (1.1) | 1,018,777 (0.7) | 774,097 (0.9) | 216,885,437 (0.0) | 0.70× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 713 (7.2) | 724 (7.1) | 8,757 (0.6) | 7,790 (0.7) | 2,997 (1.7) | 0.98× | **≥ parity** |
| 1,024 (2¹⁰) | 2,931 (8.7) | 4,233 (6.0) | 14,445 (1.8) | 12,494 (2.0) | 15,153 (1.7) | 0.69× | **≥ parity** |
| 4,096 (2¹²) | 14,196 (8.7) | 20,077 (6.1) | 37,131 (3.3) | 32,605 (3.8) | 69,727 (1.8) | 0.71× | **≥ parity** |
| 65,536 (2¹⁶) | 388,146 (6.8) | 557,025 (4.7) | 720,975 (3.6) | 749,527 (3.5) | 1,745,242 (1.5) | 0.70× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,388,787 (5.0) | 19,246,399 (2.7) | 19,472,511 (2.7) | 18,441,418 (2.8) | 43,994,012 (1.2) | 0.54× | **≥ parity** |
| 1,000 (2³·5³) | 3,705 (6.7) | 3,968 (6.3) | 15,306 (1.6) | 13,021 (1.9) | 14,962 (1.7) | 0.93× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,172 (6.5) | 4,244 (6.4) | 16,045 (1.7) | 14,179 (1.9) | 17,415 (1.6) | 0.98× | **≥ parity** |
| 1,920 (2⁷·3·5) | 6,792 (7.7) | 7,397 (7.1) | 21,838 (2.4) | 18,991 (2.8) | 30,559 (1.7) | 0.92× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 728 (7.0) | 821 (6.2) | 0.89× | **≥ parity** |
| 1,024 (2¹⁰) | 2,954 (8.7) | 4,464 (5.7) | 0.66× | **≥ parity** |
| 4,096 (2¹²) | 14,146 (8.7) | 21,190 (5.8) | 0.67× | **≥ parity** |
| 65,536 (2¹⁶) | 375,070 (7.0) | 629,759 (4.2) | 0.60× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,236,983 (5.1) | 22,065,553 (2.4) | 0.46× | **≥ parity** |
| 1,000 (2³·5³) | 3,702 (6.7) | 4,256 (5.9) | 0.87× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,177 (6.5) | 4,446 (6.1) | 0.94× | **≥ parity** |
| 1,920 (2⁷·3·5) | 6,812 (7.7) | 7,871 (6.7) | 0.87× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 24,811 (9.9) | 30,331 (8.1) | 31,447 (7.8) | 70,194 (3.5) | 50,575 (4.9) | 0.79× | **≥ parity** |
| 128x128 | 128,070 (9.0) | 149,355 (7.7) | 195,891 (5.9) | 267,403 (4.3) | 198,390 (5.8) | 0.65× | **≥ parity** |
| 256x256 | 620,520 (8.4) | 732,379 (7.2) | 1,240,354 (4.2) | 1,091,661 (4.8) | 899,239 (5.8) | 0.50× | **≥ parity** |
| 512x512 | 3,170,272 (7.4) | 4,092,415 (5.8) | 6,931,355 (3.4) | 5,614,662 (4.2) | 4,018,651 (5.9) | 0.46× | **≥ parity** |
| 1024x1024 | 17,903,920 (5.9) | 18,690,590 (5.6) | 43,617,180 (2.4) | 26,254,598 (4.0) | 20,211,770 (5.2) | 0.41× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 14,149 | 10,073 | 4,498,959 |
| 1,024 | 53,357 | 35,726 | 3,274,657 |
| 4,096 | 189,556 | 132,697 | 6,276,909 |
| 65,536 | 3,343,276 | 1,971,098 | 21,933,462 |
| 1,048,576 | 47,742,593 | 30,945,372 | 20,390,676 |
| 1,000 | 52,038 | 34,719 | 4,796,643 |
| 1,080 | 59,116 | 37,358 | 12,805,583 |
| 1,920 | 98,585 | 65,093 | 17,532,066 |
| 1,009 | 127,708 | 40,172 | 6,226,547 |
| 1,296 | 82,735 | 44,065 | 10,625,385 |
| 10,007 | 1,388,612 | 402,438 | 35,858,503 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 24/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
