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
| 256 (2⁸) | 1,129 (9.1) | 1,158 (8.8) | 9,585 (1.1) | 7,735 (1.3) | 5,858 (1.7) | 0.97× | **≥ parity** |
| 1,024 (2¹⁰) | 5,284 (9.7) | 6,320 (8.1) | 18,908 (2.7) | 14,672 (3.5) | 30,627 (1.7) | 0.84× | **≥ parity** |
| 4,096 (2¹²) | 28,362 (8.7) | 41,146 (6.0) | 61,998 (4.0) | 49,341 (5.0) | 145,320 (1.7) | 0.69× | **≥ parity** |
| 65,536 (2¹⁶) | 728,694 (7.2) | 1,332,228 (3.9) | 2,355,966 (2.2) | 1,467,898 (3.6) | 3,374,199 (1.6) | 0.55× | **≥ parity** |
| 1,048,576 (2²⁰) | 23,032,794 (4.6) | 47,892,705 (2.2) | 45,763,525 (2.3) | 30,347,268 (3.5) | 74,767,322 (1.4) | 0.48× | **≥ parity** |
| 1,000 (2³·5³) | 7,301 (6.8) | 7,191 (6.9) | 19,372 (2.6) | 15,262 (3.3) | 33,717 (1.5) | 1.02× | **≥ parity** |
| 1,080 (2³·3³·5) | 8,713 (6.2) | 7,690 (7.1) | 21,907 (2.5) | 16,747 (3.2) | 41,614 (1.3) | 1.13× | lags FFTW 1.13× |
| 1,920 (2⁷·3·5) | 14,990 (7.0) | 13,539 (7.7) | 32,830 (3.2) | 25,953 (4.0) | 70,061 (1.5) | 1.11× | lags FFTW 1.11× |
| 1,009 (prime) | 21,679 (2.3) | 48,284 (1.0) | 96,620 (0.5) | 57,184 (0.9) | 2,219,754 (0.0) | 0.45× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 11,579 (5.8) | 11,367 (5.9) | 25,061 (2.7) | 19,868 (3.4) | 47,303 (1.4) | 1.02× | **≥ parity** |
| 10,007 (prime) | 473,085 (1.4) | 608,748 (1.1) | 1,064,511 (0.6) | 803,237 (0.8) | 216,802,565 (0.0) | 0.78× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 836 (6.1) | 718 (7.1) | 8,813 (0.6) | 7,953 (0.6) | 2,974 (1.7) | 1.16× | lags FFTW 1.16× |
| 1,024 (2¹⁰) | 3,572 (7.2) | 4,214 (6.1) | 14,584 (1.8) | 12,605 (2.0) | 15,180 (1.7) | 0.85× | **≥ parity** |
| 4,096 (2¹²) | 16,653 (7.4) | 19,908 (6.2) | 37,227 (3.3) | 33,317 (3.7) | 69,829 (1.8) | 0.84× | **≥ parity** |
| 65,536 (2¹⁶) | 411,706 (6.4) | 570,927 (4.6) | 740,295 (3.5) | 833,979 (3.1) | 1,746,280 (1.5) | 0.72× | **≥ parity** |
| 1,048,576 (2²⁰) | 12,014,452 (4.4) | 19,380,606 (2.7) | 27,765,697 (1.9) | 21,616,222 (2.4) | 42,906,707 (1.2) | 0.62× | **≥ parity** |
| 1,000 (2³·5³) | 4,710 (5.3) | 3,949 (6.3) | 15,499 (1.6) | 13,089 (1.9) | 14,871 (1.7) | 1.19× | lags FFTW 1.19× |
| 1,080 (2³·3³·5) | 5,462 (5.0) | 4,209 (6.5) | 16,347 (1.7) | 14,631 (1.9) | 17,391 (1.6) | 1.30× | lags FFTW 1.30× |
| 1,920 (2⁷·3·5) | 8,885 (5.9) | 7,326 (7.1) | 22,555 (2.3) | 19,402 (2.7) | 30,469 (1.7) | 1.21× | lags FFTW 1.21× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 959 (5.3) | 810 (6.3) | 1.18× | lags FFTW 1.18× |
| 1,024 (2¹⁰) | 3,918 (6.5) | 4,440 (5.8) | 0.88× | **≥ parity** |
| 4,096 (2¹²) | 17,893 (6.9) | 21,196 (5.8) | 0.84× | **≥ parity** |
| 65,536 (2¹⁶) | 432,263 (6.1) | 646,930 (4.1) | 0.67× | **≥ parity** |
| 1,048,576 (2²⁰) | 11,379,131 (4.6) | 22,999,846 (2.3) | 0.49× | **≥ parity** |
| 1,000 (2³·5³) | 5,074 (4.9) | 4,274 (5.8) | 1.19× | lags FFTW 1.19× |
| 1,080 (2³·3³·5) | 5,859 (4.6) | 4,475 (6.1) | 1.31× | lags FFTW 1.31× |
| 1,920 (2⁷·3·5) | 9,491 (5.5) | 7,896 (6.6) | 1.20× | lags FFTW 1.20× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 34,331 (7.2) | 41,253 (6.0) | 31,036 (7.9) | 72,449 (3.4) | 50,932 (4.8) | 1.11× | lags FFTW 1.11× |
| 128x128 | 147,074 (7.8) | 174,762 (6.6) | 194,193 (5.9) | 263,307 (4.4) | 206,233 (5.6) | 0.76× | **≥ parity** |
| 256x256 | 765,977 (6.8) | 853,520 (6.1) | 1,266,843 (4.1) | 1,420,308 (3.7) | 1,036,691 (5.1) | 0.60× | **≥ parity** |
| 512x512 | 3,639,307 (6.5) | 3,806,146 (6.2) | 6,906,432 (3.4) | 7,239,404 (3.3) | 5,357,729 (4.4) | 0.53× | **≥ parity** |
| 1024x1024 | 18,923,916 (5.5) | 19,003,016 (5.5) | 44,097,065 (2.4) | 31,077,672 (3.4) | 24,404,993 (4.3) | 0.43× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 13,051 | 10,329 | 4,569,974 |
| 1,024 | 47,204 | 34,735 | 3,274,320 |
| 4,096 | 171,818 | 129,199 | 6,183,954 |
| 65,536 | 2,826,316 | 1,948,342 | 21,864,418 |
| 1,048,576 | 40,684,020 | 33,372,211 | 20,091,196 |
| 1,000 | 44,889 | 33,638 | 4,741,656 |
| 1,080 | 52,164 | 37,358 | 12,655,070 |
| 1,920 | 82,841 | 62,521 | 17,361,086 |
| 1,009 | 117,624 | 39,764 | 6,104,913 |
| 1,296 | 76,555 | 43,385 | 10,406,082 |
| 10,007 | 1,464,308 | 380,472 | 35,547,061 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 17/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 1,080 (2³·3³·5): 1.30×
- real 1,920 (2⁷·3·5): 1.21×
- real 1,000 (2³·5³): 1.19×
- real 256 (2⁸): 1.16×
- complex 1,080 (2³·3³·5): 1.13×
- complex 1,920 (2⁷·3·5): 1.11×
- 2-D 64x64: 1.11×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
