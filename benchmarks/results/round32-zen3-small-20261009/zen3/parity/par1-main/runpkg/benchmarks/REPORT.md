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
| 256 (2⁸) | 365 (28.1) | 278 (36.9) | 6,757 (1.5) | 5,429 (1.9) | 5,941 (1.7) | 1.31× | lags FFTW 1.31× |
| 1,024 (2¹⁰) | 1,690 (30.3) | 1,523 (33.6) | 13,859 (3.7) | 9,966 (5.1) | 28,669 (1.8) | 1.11× | lags FFTW 1.11× |
| 4,096 (2¹²) | 9,316 (26.4) | 10,300 (23.9) | 45,178 (5.4) | 33,838 (7.3) | 136,775 (1.8) | 0.90× | **≥ parity** |
| 65,536 (2¹⁶) | 256,666 (20.4) | 298,456 (17.6) | 1,986,440 (2.6) | 916,341 (5.7) | 2,915,070 (1.8) | 0.86× | **≥ parity** |
| 1,048,576 (2²⁰) | 5,221,819 (20.1) | 11,370,727 (9.2) | 23,282,257 (4.5) | 14,547,807 (7.2) | 66,139,088 (1.6) | 0.46× | **≥ parity** |
| 1,000 (2³·5³) | 2,178 (22.9) | 2,163 (23.0) | 13,796 (3.6) | 10,123 (4.9) | 30,018 (1.7) | 1.01× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,595 (21.0) | 2,297 (23.7) | 14,369 (3.8) | 10,844 (5.0) | 36,620 (1.5) | 1.13× | lags FFTW 1.13× |
| 1,920 (2⁷·3·5) | 3,816 (27.4) | 3,987 (26.3) | 21,929 (4.8) | 16,819 (6.2) | 63,786 (1.6) | 0.96× | **≥ parity** |
| 1,009 (prime) | 12,967 (3.9) | 20,207 (2.5) | 60,495 (0.8) | 38,636 (1.3) | 1,146,553 (0.0) | 0.64× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,009 (22.3) | 2,921 (22.9) | 16,531 (4.1) | 12,826 (5.2) | 42,582 (1.6) | 1.03× | **≥ parity** |
| 10,007 (prime) | 216,342 (3.1) | 201,873 (3.3) | 641,424 (1.0) | 470,510 (1.4) | 124,200,105 (0.0) | 1.07× | lags FFTW 1.07× |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 298 (17.2) | 232 (22.1) | 6,215 (0.8) | 5,243 (1.0) | 2,611 (2.0) | 1.28× | lags FFTW 1.28× |
| 1,024 (2¹⁰) | 1,045 (24.5) | 964 (26.6) | 9,538 (2.7) | 8,275 (3.1) | 12,642 (2.0) | 1.08× | lags FFTW 1.08× |
| 4,096 (2¹²) | 5,650 (21.7) | 5,169 (23.8) | 25,035 (4.9) | 22,133 (5.6) | 59,912 (2.1) | 1.09× | lags FFTW 1.09× |
| 65,536 (2¹⁶) | 137,809 (19.0) | 155,281 (16.9) | 449,275 (5.8) | 499,924 (5.2) | 1,290,333 (2.0) | 0.89× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,719,040 (19.3) | 3,832,094 (13.7) | 8,384,483 (6.3) | 10,136,058 (5.2) | 26,339,614 (2.0) | 0.71× | **≥ parity** |
| 1,000 (2³·5³) | 1,330 (18.7) | 1,254 (19.9) | 10,174 (2.4) | 8,544 (2.9) | 13,224 (1.9) | 1.06× | lags FFTW 1.06× |
| 1,080 (2³·3³·5) | 1,519 (17.9) | 1,330 (20.5) | 10,933 (2.5) | 8,988 (3.0) | 15,909 (1.7) | 1.14× | lags FFTW 1.14× |
| 1,920 (2⁷·3·5) | 2,385 (22.0) | 2,297 (22.8) | 14,394 (3.6) | 12,126 (4.3) | 28,194 (1.9) | 1.04× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 284 (18.0) | 291 (17.6) | 0.98× | **≥ parity** |
| 1,024 (2¹⁰) | 1,087 (23.6) | 1,220 (21.0) | 0.89× | **≥ parity** |
| 4,096 (2¹²) | 5,730 (21.4) | 5,824 (21.1) | 0.98× | **≥ parity** |
| 65,536 (2¹⁶) | 133,647 (19.6) | 171,093 (15.3) | 0.78× | **≥ parity** |
| 1,048,576 (2²⁰) | 3,034,654 (17.3) | 4,068,843 (12.9) | 0.75× | **≥ parity** |
| 1,000 (2³·5³) | 1,424 (17.5) | 1,434 (17.4) | 0.99× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,589 (17.1) | 1,620 (16.8) | 0.98× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,501 (20.9) | 2,775 (18.9) | 0.90× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 11,347 (21.7) | 15,457 (15.9) | 10,762 (22.8) | 46,984 (5.2) | 34,385 (7.1) | 1.05× | lags FFTW 1.05× |
| 128x128 | 55,106 (20.8) | 70,825 (16.2) | 60,826 (18.9) | 147,202 (7.8) | 127,133 (9.0) | 0.91× | **≥ parity** |
| 256x256 | 279,972 (18.7) | 372,859 (14.1) | 282,275 (18.6) | 693,742 (7.6) | 456,880 (11.5) | 0.99× | **≥ parity** |
| 512x512 | 1,288,607 (18.3) | 1,558,167 (15.1) | 1,323,816 (17.8) | 3,054,710 (7.7) | 1,992,589 (11.8) | 0.97× | **≥ parity** |
| 1024x1024 | 6,494,312 (16.1) | 8,525,955 (12.3) | 6,393,202 (16.4) | 14,813,130 (7.1) | 9,594,180 (10.9) | 1.02× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,771,443 |
| 1,024 | — | — | 139,281,153 |
| 4,096 | — | — | 242,156,420 |
| 65,536 | — | — | 2,155,927,260 |
| 1,048,576 | — | — | 4,120,178,943 |
| 1,000 | — | — | 151,736,649 |
| 1,080 | — | — | 364,622,836 |
| 1,920 | — | — | 569,736,759 |
| 1,009 | — | — | 141,041,434 |
| 1,296 | — | — | 265,826,711 |
| 10,007 | — | — | 1,261,279,783 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 14/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 256 (2⁸): 1.31×
- real 256 (2⁸): 1.28×
- real 1,080 (2³·3³·5): 1.14×
- complex 1,080 (2³·3³·5): 1.13×
- complex 1,024 (2¹⁰): 1.11×
- real 4,096 (2¹²): 1.09×
- real 1,024 (2¹⁰): 1.08×
- complex 10,007 (prime): 1.07×
- real 1,000 (2³·5³): 1.06×
- 2-D 64x64: 1.05×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
