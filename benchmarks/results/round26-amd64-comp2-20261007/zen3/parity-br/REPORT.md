# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3), cfarm420, one pinned core.
- **Toolchains**: go1.27.1 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128** (3.3.10 built from source by setup.sh); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 350 (29.3) | 275 (37.3) | 6,693 (1.5) | 5,464 (1.9) | 5,499 (1.9) | 1.27× | lags FFTW 1.27× |
| 1,024 (2¹⁰) | 1,677 (30.5) | 1,502 (34.1) | 13,080 (3.9) | 10,063 (5.1) | 28,815 (1.8) | 1.12× | lags FFTW 1.12× |
| 4,096 (2¹²) | 9,177 (26.8) | 11,116 (22.1) | 44,109 (5.6) | 33,026 (7.4) | 131,241 (1.9) | 0.83× | **≥ parity** |
| 65,536 (2¹⁶) | 257,335 (20.4) | 289,666 (18.1) | 1,948,173 (2.7) | 912,308 (5.7) | 2,926,895 (1.8) | 0.89× | **≥ parity** |
| 1,048,576 (2²⁰) | 6,039,244 (17.4) | 11,915,055 (8.8) | 43,839,631 (2.4) | 15,074,569 (7.0) | 58,535,695 (1.8) | 0.51× | **≥ parity** |
| 1,000 (2³·5³) | 2,202 (22.6) | 2,062 (24.2) | 13,602 (3.7) | 10,073 (4.9) | 31,136 (1.6) | 1.07× | lags FFTW 1.07× |
| 1,080 (2³·3³·5) | 2,581 (21.1) | 2,397 (22.7) | 14,247 (3.8) | 10,847 (5.0) | 35,539 (1.5) | 1.08× | lags FFTW 1.08× |
| 1,920 (2⁷·3·5) | 3,868 (27.1) | 4,771 (21.9) | 21,506 (4.9) | 16,169 (6.5) | 61,433 (1.7) | 0.81× | **≥ parity** |
| 1,009 (prime) | 13,364 (3.8) | 20,265 (2.5) | 61,020 (0.8) | 38,307 (1.3) | 1,152,030 (0.0) | 0.66× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,002 (22.3) | 2,951 (22.7) | 16,663 (4.0) | 12,660 (5.3) | 41,719 (1.6) | 1.02× | **≥ parity** |
| 10,007 (prime) | 206,887 (3.2) | 204,097 (3.3) | 638,934 (1.0) | 465,202 (1.4) | 119,052,159 (0.0) | 1.01× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 298 (17.2) | 268 (19.1) | 5,991 (0.9) | 5,424 (0.9) | 2,513 (2.0) | 1.11× | lags FFTW 1.11× |
| 1,024 (2¹⁰) | 1,115 (23.0) | 1,224 (20.9) | 9,479 (2.7) | 8,562 (3.0) | 12,573 (2.0) | 0.91× | **≥ parity** |
| 4,096 (2¹²) | 5,883 (20.9) | 5,078 (24.2) | 25,422 (4.8) | 22,071 (5.6) | 60,379 (2.0) | 1.16× | lags FFTW 1.16× |
| 65,536 (2¹⁶) | 141,437 (18.5) | 157,870 (16.6) | 459,115 (5.7) | 497,963 (5.3) | 1,322,137 (2.0) | 0.90× | **≥ parity** |
| 1,048,576 (2²⁰) | 3,080,403 (17.0) | 3,775,358 (13.9) | 8,785,754 (6.0) | 10,412,562 (5.0) | 27,104,467 (1.9) | 0.82× | **≥ parity** |
| 1,000 (2³·5³) | 1,433 (17.4) | 1,358 (18.3) | 10,403 (2.4) | 8,618 (2.9) | 13,395 (1.9) | 1.06× | lags FFTW 1.06× |
| 1,080 (2³·3³·5) | 1,631 (16.7) | 1,415 (19.2) | 10,967 (2.5) | 9,249 (2.9) | 15,739 (1.7) | 1.15× | lags FFTW 1.15× |
| 1,920 (2⁷·3·5) | 2,587 (20.2) | 2,297 (22.8) | 14,596 (3.6) | 12,276 (4.3) | 28,378 (1.8) | 1.13× | lags FFTW 1.13× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 309 (16.6) | 386 (13.3) | 0.80× | **≥ parity** |
| 1,024 (2¹⁰) | 1,158 (22.1) | 1,387 (18.5) | 0.83× | **≥ parity** |
| 4,096 (2¹²) | 5,958 (20.6) | 5,819 (21.1) | 1.02× | **≥ parity** |
| 65,536 (2¹⁶) | 141,040 (18.6) | 164,417 (15.9) | 0.86× | **≥ parity** |
| 1,048,576 (2²⁰) | 3,079,089 (17.0) | 4,114,804 (12.7) | 0.75× | **≥ parity** |
| 1,000 (2³·5³) | 1,438 (17.3) | 1,468 (17.0) | 0.98× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,652 (16.5) | 1,536 (17.7) | 1.08× | lags FFTW 1.08× |
| 1,920 (2⁷·3·5) | 2,671 (19.6) | 2,822 (18.6) | 0.95× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 11,046 (22.2) | 15,019 (16.4) | 11,789 (20.8) | 49,444 (5.0) | 33,422 (7.4) | 0.94× | **≥ parity** |
| 128x128 | 52,790 (21.7) | 64,441 (17.8) | 65,448 (17.5) | 150,578 (7.6) | 123,730 (9.3) | 0.81× | **≥ parity** |
| 256x256 | 279,324 (18.8) | 331,213 (15.8) | 280,463 (18.7) | 682,655 (7.7) | 453,217 (11.6) | 1.00× | **≥ parity** |
| 512x512 | 1,359,178 (17.4) | 1,498,658 (15.7) | 1,321,722 (17.9) | 2,907,282 (8.1) | 1,900,464 (12.4) | 1.03× | **≥ parity** |
| 1024x1024 | 6,605,082 (15.9) | 8,197,183 (12.8) | 7,028,828 (14.9) | 13,284,659 (7.9) | 9,424,822 (11.1) | 0.94× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 11,219 | 7,305 | 71,284,532 |
| 1,024 | 41,068 | 27,160 | 130,430,099 |
| 4,096 | 163,180 | 112,677 | 243,779,741 |
| 65,536 | 2,564,664 | 1,586,923 | 2,305,613,498 |
| 1,048,576 | 55,433,414 | 25,301,221 | 4,245,805,619 |
| 1,000 | 35,629 | 27,554 | 152,223,509 |
| 1,080 | 39,591 | 29,945 | 363,027,125 |
| 1,920 | 75,232 | 50,910 | 592,046,049 |
| 1,009 | 74,849 | 31,642 | 143,173,225 |
| 1,296 | 44,060 | 35,451 | 264,409,241 |
| 10,007 | 1,019,933 | 299,358 | 1,293,352,147 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 15/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 256 (2⁸): 1.27×
- real 4,096 (2¹²): 1.16×
- real 1,080 (2³·3³·5): 1.15×
- real 1,920 (2⁷·3·5): 1.13×
- complex 1,024 (2¹⁰): 1.12×
- real 256 (2⁸): 1.11×
- complex 1,080 (2³·3³·5): 1.08×
- complex 1,000 (2³·5³): 1.07×
- real 1,000 (2³·5³): 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
