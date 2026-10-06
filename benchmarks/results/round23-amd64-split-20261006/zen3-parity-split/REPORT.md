# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3), cfarm420, one pinned core.
- **Toolchains**: go1.27.1 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128** (3.3.10 built from source with --enable-sse2 --enable-avx --enable-avx2 --enable-fma); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 351 (29.2) | 274 (37.4) | 6,798 (1.5) | 5,573 (1.8) | 5,598 (1.8) | 1.28× | lags FFTW 1.28× |
| 1,024 (2¹⁰) | 1,715 (29.9) | 1,550 (33.0) | 13,337 (3.8) | 9,832 (5.2) | 29,314 (1.7) | 1.11× | lags FFTW 1.11× |
| 4,096 (2¹²) | 9,327 (26.3) | 10,312 (23.8) | 44,564 (5.5) | 34,791 (7.1) | 137,085 (1.8) | 0.90× | **≥ parity** |
| 65,536 (2¹⁶) | 254,383 (20.6) | 292,489 (17.9) | 1,966,543 (2.7) | 924,424 (5.7) | 2,907,079 (1.8) | 0.87× | **≥ parity** |
| 1,048,576 (2²⁰) | 5,467,427 (19.2) | 11,255,953 (9.3) | 28,335,980 (3.7) | 14,391,151 (7.3) | 61,447,493 (1.7) | 0.49× | **≥ parity** |
| 1,000 (2³·5³) | 2,392 (20.8) | 2,026 (24.6) | 14,151 (3.5) | 10,346 (4.8) | 30,323 (1.6) | 1.18× | lags FFTW 1.18× |
| 1,080 (2³·3³·5) | 2,824 (19.3) | 2,227 (24.4) | 14,238 (3.8) | 10,869 (5.0) | 35,426 (1.5) | 1.27× | lags FFTW 1.27× |
| 1,920 (2⁷·3·5) | 4,461 (23.5) | 3,863 (27.1) | 22,105 (4.7) | 16,094 (6.5) | 61,995 (1.7) | 1.15× | lags FFTW 1.15× |
| 1,009 (prime) | 12,938 (3.9) | 20,119 (2.5) | 60,505 (0.8) | 39,112 (1.3) | 1,130,438 (0.0) | 0.64× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,588 (18.7) | 2,907 (23.0) | 16,513 (4.1) | 12,282 (5.5) | 42,402 (1.6) | 1.23× | lags FFTW 1.23× |
| 10,007 (prime) | 212,798 (3.1) | 203,640 (3.3) | 631,061 (1.1) | 467,700 (1.4) | 118,186,667 (0.0) | 1.04× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 317 (16.2) | 226 (22.6) | 6,066 (0.8) | 5,286 (1.0) | 2,551 (2.0) | 1.40× | lags FFTW 1.40× |
| 1,024 (2¹⁰) | 1,115 (23.0) | 958 (26.7) | 9,365 (2.7) | 8,180 (3.1) | 13,173 (1.9) | 1.16× | lags FFTW 1.16× |
| 4,096 (2¹²) | 5,834 (21.1) | 5,698 (21.6) | 25,308 (4.9) | 22,060 (5.6) | 60,797 (2.0) | 1.02× | **≥ parity** |
| 65,536 (2¹⁶) | 139,285 (18.8) | 148,963 (17.6) | 404,328 (6.5) | 492,580 (5.3) | 1,330,379 (2.0) | 0.94× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,982,926 (17.6) | 3,711,035 (14.1) | 8,424,702 (6.2) | 10,133,481 (5.2) | 27,002,137 (1.9) | 0.80× | **≥ parity** |
| 1,000 (2³·5³) | 1,484 (16.8) | 1,260 (19.8) | 10,421 (2.4) | 8,769 (2.8) | 13,456 (1.9) | 1.18× | lags FFTW 1.18× |
| 1,080 (2³·3³·5) | 1,708 (15.9) | 1,308 (20.8) | 11,045 (2.5) | 8,962 (3.0) | 16,049 (1.7) | 1.31× | lags FFTW 1.31× |
| 1,920 (2⁷·3·5) | 2,746 (19.1) | 2,308 (22.7) | 14,106 (3.7) | 11,945 (4.4) | 27,518 (1.9) | 1.19× | lags FFTW 1.19× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 310 (16.5) | 282 (18.2) | 1.10× | lags FFTW 1.10× |
| 1,024 (2¹⁰) | 1,189 (21.5) | 1,185 (21.6) | 1.00× | **≥ parity** |
| 4,096 (2¹²) | 6,033 (20.4) | 6,081 (20.2) | 0.99× | **≥ parity** |
| 65,536 (2¹⁶) | 139,832 (18.7) | 168,007 (15.6) | 0.83× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,911,271 (18.0) | 4,204,482 (12.5) | 0.69× | **≥ parity** |
| 1,000 (2³·5³) | 1,569 (15.9) | 1,444 (17.2) | 1.09× | lags FFTW 1.09× |
| 1,080 (2³·3³·5) | 1,723 (15.8) | 1,574 (17.3) | 1.09× | lags FFTW 1.09× |
| 1,920 (2⁷·3·5) | 2,837 (18.5) | 2,743 (19.1) | 1.03× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 10,921 (22.5) | 14,672 (16.8) | 11,596 (21.2) | 47,923 (5.1) | 33,681 (7.3) | 0.94× | **≥ parity** |
| 128x128 | 52,146 (22.0) | 64,307 (17.8) | 61,560 (18.6) | 145,907 (7.9) | 131,501 (8.7) | 0.85× | **≥ parity** |
| 256x256 | 274,582 (19.1) | 321,293 (16.3) | 283,960 (18.5) | 607,283 (8.6) | 458,078 (11.4) | 0.97× | **≥ parity** |
| 512x512 | 1,236,579 (19.1) | 1,425,699 (16.5) | 1,373,258 (17.2) | 2,924,952 (8.1) | 1,903,332 (12.4) | 0.90× | **≥ parity** |
| 1024x1024 | 5,811,776 (18.0) | 7,042,133 (14.9) | 6,681,744 (15.7) | 13,445,512 (7.8) | 9,391,654 (11.2) | 0.87× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 72,925,383 |
| 1,024 | — | — | 130,806,979 |
| 4,096 | — | — | 241,435,770 |
| 65,536 | — | — | 2,157,558,120 |
| 1,048,576 | — | — | 3,980,377,079 |
| 1,000 | — | — | 151,728,239 |
| 1,080 | — | — | 363,669,965 |
| 1,920 | — | — | 568,808,278 |
| 1,009 | — | — | 141,358,215 |
| 1,296 | — | — | 264,671,090 |
| 10,007 | — | — | 1,243,818,806 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 13/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.40×
- real 1,080 (2³·3³·5): 1.31×
- complex 256 (2⁸): 1.28×
- complex 1,080 (2³·3³·5): 1.27×
- complex 1,296 (2⁴·3⁴): 1.23×
- real 1,920 (2⁷·3·5): 1.19×
- complex 1,000 (2³·5³): 1.18×
- real 1,000 (2³·5³): 1.18×
- real 1,024 (2¹⁰): 1.16×
- complex 1,920 (2⁷·3·5): 1.15×
- complex 1,024 (2¹⁰): 1.11×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
