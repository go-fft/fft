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
| 256 (2⁸) | 1,119 (9.2) | 1,154 (8.9) | 9,528 (1.1) | 7,742 (1.3) | 5,877 (1.7) | 0.97× | **≥ parity** |
| 1,024 (2¹⁰) | 5,286 (9.7) | 6,331 (8.1) | 19,044 (2.7) | 14,568 (3.5) | 30,545 (1.7) | 0.83× | **≥ parity** |
| 4,096 (2¹²) | 27,980 (8.8) | 41,078 (6.0) | 59,887 (4.1) | 47,750 (5.1) | 147,302 (1.7) | 0.68× | **≥ parity** |
| 65,536 (2¹⁶) | 718,855 (7.3) | 1,248,012 (4.2) | 2,123,611 (2.5) | 1,407,148 (3.7) | 3,422,430 (1.5) | 0.58× | **≥ parity** |
| 1,048,576 (2²⁰) | 19,979,411 (5.2) | 44,741,628 (2.3) | 43,731,941 (2.4) | 32,495,348 (3.2) | 75,262,706 (1.4) | 0.45× | **≥ parity** |
| 1,000 (2³·5³) | 6,054 (8.2) | 7,172 (6.9) | 19,351 (2.6) | 15,360 (3.2) | 33,708 (1.5) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,900 (7.9) | 7,717 (7.1) | 21,833 (2.5) | 16,763 (3.2) | 41,611 (1.3) | 0.89× | **≥ parity** |
| 1,920 (2⁷·3·5) | 12,094 (8.7) | 13,552 (7.7) | 33,154 (3.2) | 25,966 (4.0) | 69,771 (1.5) | 0.89× | **≥ parity** |
| 1,009 (prime) | 21,758 (2.3) | 48,006 (1.0) | 96,423 (0.5) | 57,057 (0.9) | 2,227,719 (0.0) | 0.45× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,656 (7.7) | 11,351 (5.9) | 24,838 (2.7) | 19,532 (3.4) | 49,172 (1.4) | 0.76× | **≥ parity** |
| 10,007 (prime) | 423,606 (1.6) | 608,088 (1.1) | 1,026,270 (0.6) | 778,987 (0.9) | 218,235,021 (0.0) | 0.70× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 768 (6.7) | 721 (7.1) | 8,794 (0.6) | 8,135 (0.6) | 2,997 (1.7) | 1.07× | lags FFTW 1.07× |
| 1,024 (2¹⁰) | 3,326 (7.7) | 4,215 (6.1) | 14,302 (1.8) | 12,540 (2.0) | 15,133 (1.7) | 0.79× | **≥ parity** |
| 4,096 (2¹²) | 15,627 (7.9) | 20,079 (6.1) | 37,255 (3.3) | 32,744 (3.8) | 70,075 (1.8) | 0.78× | **≥ parity** |
| 65,536 (2¹⁶) | 399,726 (6.6) | 550,984 (4.8) | 714,202 (3.7) | 735,846 (3.6) | 1,746,813 (1.5) | 0.73× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,365,029 (5.1) | 18,604,085 (2.8) | 19,596,998 (2.7) | 18,229,459 (2.9) | 43,034,484 (1.2) | 0.56× | **≥ parity** |
| 1,000 (2³·5³) | 3,884 (6.4) | 3,941 (6.3) | 15,283 (1.6) | 12,956 (1.9) | 14,944 (1.7) | 0.99× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,369 (6.2) | 4,204 (6.5) | 15,938 (1.7) | 14,075 (1.9) | 17,305 (1.6) | 1.04× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,169 (7.3) | 7,353 (7.1) | 22,103 (2.4) | 18,961 (2.8) | 30,855 (1.7) | 0.97× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 773 (6.6) | 811 (6.3) | 0.95× | **≥ parity** |
| 1,024 (2¹⁰) | 3,286 (7.8) | 4,486 (5.7) | 0.73× | **≥ parity** |
| 4,096 (2¹²) | 15,372 (8.0) | 21,073 (5.8) | 0.73× | **≥ parity** |
| 65,536 (2¹⁶) | 392,194 (6.7) | 668,733 (3.9) | 0.59× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,172,465 (5.2) | 23,882,302 (2.2) | 0.43× | **≥ parity** |
| 1,000 (2³·5³) | 3,891 (6.4) | 4,254 (5.9) | 0.91× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,370 (6.2) | 4,419 (6.2) | 0.99× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,167 (7.3) | 7,827 (6.7) | 0.92× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 27,685 (8.9) | 32,775 (7.5) | 30,890 (8.0) | 71,373 (3.4) | 50,869 (4.8) | 0.90× | **≥ parity** |
| 128x128 | 126,578 (9.1) | 151,113 (7.6) | 196,407 (5.8) | 254,431 (4.5) | 201,357 (5.7) | 0.64× | **≥ parity** |
| 256x256 | 677,358 (7.7) | 759,619 (6.9) | 1,422,188 (3.7) | 1,068,015 (4.9) | 938,159 (5.6) | 0.48× | **≥ parity** |
| 512x512 | 3,502,749 (6.7) | 4,191,030 (5.6) | 7,837,988 (3.0) | 5,999,256 (3.9) | 4,320,504 (5.5) | 0.45× | **≥ parity** |
| 1024x1024 | 18,482,459 (5.7) | 18,970,605 (5.5) | 47,216,461 (2.2) | 26,662,964 (3.9) | 20,395,469 (5.1) | 0.39× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 15,151 | 9,746 | 4,483,623 |
| 1,024 | 52,380 | 34,078 | 3,280,926 |
| 4,096 | 191,402 | 130,339 | 6,236,007 |
| 65,536 | 3,130,838 | 2,023,418 | 22,007,151 |
| 1,048,576 | 78,731,647 | 31,771,365 | 20,894,214 |
| 1,000 | 53,766 | 34,414 | 4,890,950 |
| 1,080 | 59,434 | 36,731 | 12,892,824 |
| 1,920 | 99,810 | 64,087 | 17,843,371 |
| 1,009 | 131,267 | 40,104 | 6,477,851 |
| 1,296 | 79,427 | 44,942 | 10,303,665 |
| 10,007 | 1,430,738 | 400,547 | 35,473,222 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 23/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.07×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
