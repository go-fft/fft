# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3), cfarm420.
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
| 256 (2⁸) | 304 (33.7) | 280 (36.5) | 7,092 (1.4) | 5,711 (1.8) | 5,635 (1.8) | 1.08× | lags FFTW 1.08× |
| 1,024 (2¹⁰) | 1,726 (29.7) | 1,950 (26.3) | 13,786 (3.7) | 10,141 (5.0) | 29,404 (1.7) | 0.89× | **≥ parity** |
| 4,096 (2¹²) | 19,657 (12.5) | 10,257 (24.0) | 44,827 (5.5) | 33,999 (7.2) | 134,789 (1.8) | 1.92× | lags FFTW 1.92× |
| 65,536 (2¹⁶) | 265,538 (19.7) | 296,794 (17.7) | 1,919,202 (2.7) | 973,979 (5.4) | 2,989,912 (1.8) | 0.89× | **≥ parity** |
| 1,048,576 (2²⁰) | 6,314,715 (16.6) | 11,435,642 (9.2) | 23,865,873 (4.4) | 14,839,522 (7.1) | 59,891,317 (1.8) | 0.55× | **≥ parity** |
| 1,000 (2³·5³) | 2,180 (22.9) | 2,112 (23.6) | 13,745 (3.6) | 10,158 (4.9) | 30,081 (1.7) | 1.03× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,606 (20.9) | 2,433 (22.4) | 14,608 (3.7) | 10,698 (5.1) | 37,619 (1.4) | 1.07× | lags FFTW 1.07× |
| 1,920 (2⁷·3·5) | 3,840 (27.3) | 3,773 (27.8) | 22,485 (4.7) | 16,573 (6.3) | 64,104 (1.6) | 1.02× | **≥ parity** |
| 1,009 (prime) | 13,345 (3.8) | 20,286 (2.5) | 62,310 (0.8) | 39,844 (1.3) | 1,210,428 (0.0) | 0.66× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,134 (21.4) | 2,911 (23.0) | 16,842 (4.0) | 12,798 (5.2) | 42,967 (1.6) | 1.08× | lags FFTW 1.08× |
| 10,007 (prime) | 213,531 (3.1) | 219,747 (3.0) | 674,300 (1.0) | 469,075 (1.4) | 121,278,051 (0.0) | 0.97× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 233 (22.0) | 226 (22.6) | 6,165 (0.8) | 5,457 (0.9) | 2,535 (2.0) | 1.03× | **≥ parity** |
| 1,024 (2¹⁰) | 1,027 (24.9) | 1,005 (25.5) | 9,655 (2.7) | 8,552 (3.0) | 12,773 (2.0) | 1.02× | **≥ parity** |
| 4,096 (2¹²) | 5,772 (21.3) | 5,351 (23.0) | 25,767 (4.8) | 22,427 (5.5) | 60,945 (2.0) | 1.08× | lags FFTW 1.08× |
| 65,536 (2¹⁶) | 151,632 (17.3) | 170,366 (15.4) | 444,539 (5.9) | 497,587 (5.3) | 1,346,906 (1.9) | 0.89× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,788,297 (18.8) | 3,906,236 (13.4) | 8,511,844 (6.2) | 10,076,797 (5.2) | 28,870,105 (1.8) | 0.71× | **≥ parity** |
| 1,000 (2³·5³) | 1,246 (20.0) | 1,235 (20.2) | 11,109 (2.2) | 9,205 (2.7) | 13,332 (1.9) | 1.01× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,505 (18.1) | 1,419 (19.2) | 11,535 (2.4) | 9,248 (2.9) | 17,444 (1.6) | 1.06× | lags FFTW 1.06× |
| 1,920 (2⁷·3·5) | 2,459 (21.3) | 2,311 (22.7) | 14,287 (3.7) | 12,781 (4.1) | 29,639 (1.8) | 1.06× | lags FFTW 1.06× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 255 (20.1) | 406 (12.6) | 0.63× | **≥ parity** |
| 1,024 (2¹⁰) | 1,029 (24.9) | 1,197 (21.4) | 0.86× | **≥ parity** |
| 4,096 (2¹²) | 5,805 (21.2) | 5,814 (21.1) | 1.00× | **≥ parity** |
| 65,536 (2¹⁶) | 135,684 (19.3) | 166,564 (15.7) | 0.81× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,954,649 (17.7) | 4,332,296 (12.1) | 0.68× | **≥ parity** |
| 1,000 (2³·5³) | 1,308 (19.0) | 1,553 (16.0) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,588 (17.1) | 1,594 (17.1) | 1.00× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,582 (20.3) | 2,654 (19.7) | 0.97× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 12,060 (20.4) | 17,574 (14.0) | 11,370 (21.6) | 51,081 (4.8) | 33,786 (7.3) | 1.06× | lags FFTW 1.06× |
| 128x128 | 52,539 (21.8) | 68,747 (16.7) | 58,828 (19.5) | 155,743 (7.4) | 129,501 (8.9) | 0.89× | **≥ parity** |
| 256x256 | 277,604 (18.9) | 346,051 (15.2) | 286,047 (18.3) | 659,618 (7.9) | 477,206 (11.0) | 0.97× | **≥ parity** |
| 512x512 | 1,242,977 (19.0) | 1,528,307 (15.4) | 1,352,195 (17.4) | 3,036,429 (7.8) | 1,945,477 (12.1) | 0.92× | **≥ parity** |
| 1024x1024 | 5,988,554 (17.5) | 7,489,721 (14.0) | 6,930,148 (15.1) | 14,953,917 (7.0) | 10,391,461 (10.1) | 0.86× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 72,218,113 |
| 1,024 | — | — | 136,741,062 |
| 4,096 | — | — | 242,533,030 |
| 65,536 | — | — | 2,169,186,856 |
| 1,048,576 | — | — | 4,264,748,958 |
| 1,000 | — | — | 160,645,053 |
| 1,080 | — | — | 381,961,474 |
| 1,920 | — | — | 575,483,522 |
| 1,009 | — | — | 141,148,704 |
| 1,296 | — | — | 323,019,376 |
| 10,007 | — | — | 1,232,741,300 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 16/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 4,096 (2¹²): 1.92×
- complex 256 (2⁸): 1.08×
- real 4,096 (2¹²): 1.08×
- complex 1,296 (2⁴·3⁴): 1.08×
- complex 1,080 (2³·3³·5): 1.07×
- real 1,920 (2⁷·3·5): 1.06×
- real 1,080 (2³·3³·5): 1.06×
- 2-D 64x64: 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
