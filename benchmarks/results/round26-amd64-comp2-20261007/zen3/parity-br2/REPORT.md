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
| 256 (2⁸) | 347 (29.5) | 287 (35.7) | 7,055 (1.5) | 5,473 (1.9) | 5,470 (1.9) | 1.21× | lags FFTW 1.21× |
| 1,024 (2¹⁰) | 1,727 (29.6) | 1,509 (33.9) | 13,542 (3.8) | 9,899 (5.2) | 28,251 (1.8) | 1.14× | lags FFTW 1.14× |
| 4,096 (2¹²) | 9,599 (25.6) | 11,295 (21.8) | 47,171 (5.2) | 34,540 (7.1) | 132,154 (1.9) | 0.85× | **≥ parity** |
| 65,536 (2¹⁶) | 260,897 (20.1) | 298,621 (17.6) | 1,891,459 (2.8) | 914,418 (5.7) | 2,842,938 (1.8) | 0.87× | **≥ parity** |
| 1,048,576 (2²⁰) | 5,970,248 (17.6) | 11,505,252 (9.1) | 27,424,859 (3.8) | 14,850,006 (7.1) | 59,965,099 (1.7) | 0.52× | **≥ parity** |
| 1,000 (2³·5³) | 2,182 (22.8) | 2,060 (24.2) | 14,817 (3.4) | 9,899 (5.0) | 29,897 (1.7) | 1.06× | lags FFTW 1.06× |
| 1,080 (2³·3³·5) | 2,603 (20.9) | 2,322 (23.4) | 14,157 (3.8) | 10,659 (5.1) | 35,262 (1.5) | 1.12× | lags FFTW 1.12× |
| 1,920 (2⁷·3·5) | 3,852 (27.2) | 3,947 (26.5) | 21,865 (4.8) | 16,346 (6.4) | 61,434 (1.7) | 0.98× | **≥ parity** |
| 1,009 (prime) | 13,032 (3.9) | 20,086 (2.5) | 60,741 (0.8) | 38,166 (1.3) | 1,171,675 (0.0) | 0.65× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 2,955 (22.7) | 2,912 (23.0) | 16,206 (4.1) | 12,415 (5.4) | 42,387 (1.6) | 1.01× | **≥ parity** |
| 10,007 (prime) | 210,266 (3.2) | 206,839 (3.2) | 641,946 (1.0) | 463,639 (1.4) | 128,548,175 (0.0) | 1.02× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 296 (17.3) | 227 (22.6) | 5,943 (0.9) | 5,542 (0.9) | 2,551 (2.0) | 1.30× | lags FFTW 1.30× |
| 1,024 (2¹⁰) | 1,124 (22.8) | 979 (26.1) | 9,443 (2.7) | 9,026 (2.8) | 12,648 (2.0) | 1.15× | lags FFTW 1.15× |
| 4,096 (2¹²) | 5,831 (21.1) | 5,248 (23.4) | 25,405 (4.8) | 21,905 (5.6) | 59,229 (2.1) | 1.11× | lags FFTW 1.11× |
| 65,536 (2¹⁶) | 137,990 (19.0) | 154,529 (17.0) | 402,780 (6.5) | 500,832 (5.2) | 1,307,470 (2.0) | 0.89× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,893,913 (18.1) | 3,748,442 (14.0) | 8,287,442 (6.3) | 9,956,538 (5.3) | 27,908,205 (1.9) | 0.77× | **≥ parity** |
| 1,000 (2³·5³) | 1,372 (18.2) | 1,222 (20.4) | 10,186 (2.4) | 8,506 (2.9) | 14,448 (1.7) | 1.12× | lags FFTW 1.12× |
| 1,080 (2³·3³·5) | 1,596 (17.0) | 1,418 (19.2) | 10,742 (2.5) | 9,371 (2.9) | 15,744 (1.7) | 1.13× | lags FFTW 1.13× |
| 1,920 (2⁷·3·5) | 2,564 (20.4) | 2,360 (22.2) | 14,149 (3.7) | 12,189 (4.3) | 28,990 (1.8) | 1.09× | lags FFTW 1.09× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 307 (16.7) | 347 (14.8) | 0.88× | **≥ parity** |
| 1,024 (2¹⁰) | 1,148 (22.3) | 1,191 (21.5) | 0.96× | **≥ parity** |
| 4,096 (2¹²) | 5,930 (20.7) | 5,788 (21.2) | 1.02× | **≥ parity** |
| 65,536 (2¹⁶) | 140,312 (18.7) | 171,179 (15.3) | 0.82× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,882,704 (18.2) | 4,126,716 (12.7) | 0.70× | **≥ parity** |
| 1,000 (2³·5³) | 1,418 (17.6) | 1,426 (17.5) | 0.99× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,654 (16.4) | 1,585 (17.2) | 1.04× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,675 (19.6) | 2,662 (19.7) | 1.00× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 10,990 (22.4) | 15,450 (15.9) | 10,742 (22.9) | 50,400 (4.9) | 32,844 (7.5) | 1.02× | **≥ parity** |
| 128x128 | 52,771 (21.7) | 71,355 (16.1) | 62,073 (18.5) | 147,509 (7.8) | 127,001 (9.0) | 0.85× | **≥ parity** |
| 256x256 | 275,994 (19.0) | 316,445 (16.6) | 287,810 (18.2) | 594,783 (8.8) | 454,994 (11.5) | 0.96× | **≥ parity** |
| 512x512 | 1,254,306 (18.8) | 1,511,382 (15.6) | 1,373,084 (17.2) | 2,953,141 (8.0) | 1,916,868 (12.3) | 0.91× | **≥ parity** |
| 1024x1024 | 6,846,428 (15.3) | 8,436,663 (12.4) | 6,886,025 (15.2) | 13,316,283 (7.9) | 9,361,665 (11.2) | 0.99× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 12,236 | 7,953 | 71,501,223 |
| 1,024 | 42,680 | 31,694 | 134,634,561 |
| 4,096 | 199,299 | 112,733 | 242,767,311 |
| 65,536 | 2,594,775 | 1,767,675 | 2,272,988,053 |
| 1,048,576 | 51,832,705 | 28,962,263 | 4,256,926,304 |
| 1,000 | 41,473 | 27,798 | 152,839,659 |
| 1,080 | 38,814 | 31,591 | 369,955,408 |
| 1,920 | 81,038 | 51,825 | 586,159,177 |
| 1,009 | 82,607 | 32,621 | 151,009,439 |
| 1,296 | 44,509 | 35,145 | 266,554,061 |
| 10,007 | 1,091,802 | 308,185 | 1,272,705,968 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 14/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.30×
- complex 256 (2⁸): 1.21×
- real 1,024 (2¹⁰): 1.15×
- complex 1,024 (2¹⁰): 1.14×
- real 1,080 (2³·3³·5): 1.13×
- real 1,000 (2³·5³): 1.12×
- complex 1,080 (2³·3³·5): 1.12×
- real 4,096 (2¹²): 1.11×
- real 1,920 (2⁷·3·5): 1.09×
- complex 1,000 (2³·5³): 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
