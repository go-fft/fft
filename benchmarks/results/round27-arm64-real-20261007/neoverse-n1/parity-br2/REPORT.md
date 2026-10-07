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
| 256 (2⁸) | 1,107 (9.3) | 1,156 (8.9) | 9,673 (1.1) | 7,767 (1.3) | 5,857 (1.7) | 0.96× | **≥ parity** |
| 1,024 (2¹⁰) | 5,290 (9.7) | 6,327 (8.1) | 19,107 (2.7) | 14,683 (3.5) | 30,391 (1.7) | 0.84× | **≥ parity** |
| 4,096 (2¹²) | 27,755 (8.9) | 41,074 (6.0) | 60,620 (4.1) | 48,881 (5.0) | 146,572 (1.7) | 0.68× | **≥ parity** |
| 65,536 (2¹⁶) | 719,022 (7.3) | 1,258,770 (4.2) | 2,217,306 (2.4) | 1,454,544 (3.6) | 3,403,682 (1.5) | 0.57× | **≥ parity** |
| 1,048,576 (2²⁰) | 20,407,904 (5.1) | 44,492,925 (2.4) | 46,317,743 (2.3) | 33,048,267 (3.2) | 74,161,852 (1.4) | 0.46× | **≥ parity** |
| 1,000 (2³·5³) | 6,029 (8.3) | 7,239 (6.9) | 19,354 (2.6) | 15,371 (3.2) | 34,016 (1.5) | 0.83× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,866 (7.9) | 7,771 (7.0) | 21,968 (2.5) | 16,854 (3.2) | 41,586 (1.3) | 0.88× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,958 (8.8) | 13,787 (7.6) | 33,658 (3.1) | 27,104 (3.9) | 69,794 (1.5) | 0.87× | **≥ parity** |
| 1,009 (prime) | 21,461 (2.3) | 48,543 (1.0) | 98,182 (0.5) | 57,760 (0.9) | 2,241,627 (0.0) | 0.44× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,655 (7.7) | 11,552 (5.8) | 25,602 (2.6) | 19,768 (3.4) | 49,528 (1.4) | 0.75× | **≥ parity** |
| 10,007 (prime) | 409,418 (1.6) | 598,675 (1.1) | 1,046,121 (0.6) | 774,515 (0.9) | 220,242,338 (0.0) | 0.68× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 771 (6.6) | 731 (7.0) | 8,681 (0.6) | 7,833 (0.7) | 3,063 (1.7) | 1.05× | lags FFTW 1.05× |
| 1,024 (2¹⁰) | 3,272 (7.8) | 4,220 (6.1) | 14,301 (1.8) | 12,433 (2.1) | 15,534 (1.6) | 0.78× | **≥ parity** |
| 4,096 (2¹²) | 15,492 (7.9) | 20,231 (6.1) | 37,528 (3.3) | 33,173 (3.7) | 68,498 (1.8) | 0.77× | **≥ parity** |
| 65,536 (2¹⁶) | 405,417 (6.5) | 558,973 (4.7) | 718,094 (3.7) | 765,605 (3.4) | 1,787,719 (1.5) | 0.73× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,414,911 (5.0) | 18,816,464 (2.8) | 19,269,423 (2.7) | 18,096,695 (2.9) | 42,235,436 (1.2) | 0.55× | **≥ parity** |
| 1,000 (2³·5³) | 3,856 (6.5) | 4,019 (6.2) | 15,204 (1.6) | 12,663 (2.0) | 14,873 (1.7) | 0.96× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,368 (6.2) | 4,269 (6.4) | 15,962 (1.7) | 14,088 (1.9) | 17,422 (1.6) | 1.02× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,224 (7.2) | 7,447 (7.0) | 22,100 (2.4) | 19,018 (2.8) | 30,907 (1.7) | 0.97× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 773 (6.6) | 810 (6.3) | 0.95× | **≥ parity** |
| 1,024 (2¹⁰) | 3,375 (7.6) | 4,458 (5.7) | 0.76× | **≥ parity** |
| 4,096 (2¹²) | 15,429 (8.0) | 20,991 (5.9) | 0.74× | **≥ parity** |
| 65,536 (2¹⁶) | 388,297 (6.8) | 657,208 (4.0) | 0.59× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,358,317 (5.1) | 21,323,533 (2.5) | 0.49× | **≥ parity** |
| 1,000 (2³·5³) | 3,903 (6.4) | 4,248 (5.9) | 0.92× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,452 (6.1) | 4,454 (6.1) | 1.00× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,269 (7.2) | 7,816 (6.7) | 0.93× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 28,060 (8.8) | 33,318 (7.4) | 30,842 (8.0) | 72,575 (3.4) | 50,470 (4.9) | 0.91× | **≥ parity** |
| 128x128 | 127,604 (9.0) | 149,245 (7.7) | 193,020 (5.9) | 258,448 (4.4) | 198,764 (5.8) | 0.66× | **≥ parity** |
| 256x256 | 680,878 (7.7) | 787,979 (6.7) | 1,251,540 (4.2) | 1,187,653 (4.4) | 947,338 (5.5) | 0.54× | **≥ parity** |
| 512x512 | 3,486,009 (6.8) | 4,220,728 (5.6) | 6,747,951 (3.5) | 5,195,350 (4.5) | 4,060,175 (5.8) | 0.52× | **≥ parity** |
| 1024x1024 | 18,091,751 (5.8) | 19,060,544 (5.5) | 44,134,627 (2.4) | 26,344,551 (4.0) | 19,851,258 (5.3) | 0.41× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 14,955 | 9,730 | 4,340,505 |
| 1,024 | 55,082 | 34,027 | 3,273,409 |
| 4,096 | 199,787 | 128,956 | 6,213,371 |
| 65,536 | 3,244,564 | 2,008,966 | 22,110,168 |
| 1,048,576 | 53,163,607 | 30,455,424 | 20,157,060 |
| 1,000 | 52,752 | 34,705 | 4,938,032 |
| 1,080 | 59,028 | 37,075 | 13,079,513 |
| 1,920 | 98,054 | 62,493 | 18,018,909 |
| 1,009 | 127,978 | 39,201 | 6,334,575 |
| 1,296 | 77,052 | 43,058 | 10,835,921 |
| 10,007 | 1,443,448 | 380,379 | 36,073,575 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 23/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.05×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
