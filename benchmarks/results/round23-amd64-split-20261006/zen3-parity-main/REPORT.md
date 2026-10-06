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
| 256 (2⁸) | 386 (26.5) | 286 (35.8) | 6,701 (1.5) | 5,377 (1.9) | 5,616 (1.8) | 1.35× | lags FFTW 1.35× |
| 1,024 (2¹⁰) | 1,924 (26.6) | 1,521 (33.7) | 12,957 (4.0) | 9,680 (5.3) | 29,261 (1.7) | 1.26× | lags FFTW 1.26× |
| 4,096 (2¹²) | 12,767 (19.2) | 10,913 (22.5) | 45,589 (5.4) | 33,427 (7.4) | 133,644 (1.8) | 1.17× | lags FFTW 1.17× |
| 65,536 (2¹⁶) | 284,686 (18.4) | 291,838 (18.0) | 1,870,873 (2.8) | 906,844 (5.8) | 2,867,751 (1.8) | 0.98× | **≥ parity** |
| 1,048,576 (2²⁰) | 6,515,523 (16.1) | 11,308,693 (9.3) | 23,265,391 (4.5) | 13,902,579 (7.5) | 58,620,556 (1.8) | 0.58× | **≥ parity** |
| 1,000 (2³·5³) | 2,351 (21.2) | 1,965 (25.4) | 13,737 (3.6) | 10,197 (4.9) | 29,900 (1.7) | 1.20× | lags FFTW 1.20× |
| 1,080 (2³·3³·5) | 2,770 (19.6) | 2,315 (23.5) | 14,278 (3.8) | 10,781 (5.0) | 35,533 (1.5) | 1.20× | lags FFTW 1.20× |
| 1,920 (2⁷·3·5) | 4,755 (22.0) | 3,986 (26.3) | 22,180 (4.7) | 16,572 (6.3) | 63,406 (1.7) | 1.19× | lags FFTW 1.19× |
| 1,009 (prime) | 12,895 (3.9) | 20,421 (2.5) | 61,220 (0.8) | 38,514 (1.3) | 1,141,255 (0.0) | 0.63× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,739 (17.9) | 2,960 (22.6) | 16,665 (4.0) | 12,468 (5.4) | 42,550 (1.6) | 1.26× | lags FFTW 1.26× |
| 10,007 (prime) | 249,488 (2.7) | 209,829 (3.2) | 634,214 (1.0) | 466,332 (1.4) | 118,578,939 (0.0) | 1.19× | lags FFTW 1.19× |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 324 (15.8) | 328 (15.6) | 6,033 (0.8) | 5,442 (0.9) | 2,915 (1.8) | 0.99× | **≥ parity** |
| 1,024 (2¹⁰) | 1,251 (20.5) | 990 (25.9) | 9,484 (2.7) | 8,402 (3.0) | 12,633 (2.0) | 1.26× | lags FFTW 1.26× |
| 4,096 (2¹²) | 6,805 (18.1) | 5,414 (22.7) | 25,343 (4.8) | 22,211 (5.5) | 59,595 (2.1) | 1.26× | lags FFTW 1.26× |
| 65,536 (2¹⁶) | 161,963 (16.2) | 158,920 (16.5) | 399,020 (6.6) | 490,954 (5.3) | 1,315,583 (2.0) | 1.02× | **≥ parity** |
| 1,048,576 (2²⁰) | 3,362,933 (15.6) | 3,587,563 (14.6) | 8,271,844 (6.3) | 9,958,202 (5.3) | 26,700,008 (2.0) | 0.94× | **≥ parity** |
| 1,000 (2³·5³) | 1,479 (16.8) | 1,271 (19.6) | 10,325 (2.4) | 8,673 (2.9) | 13,391 (1.9) | 1.16× | lags FFTW 1.16× |
| 1,080 (2³·3³·5) | 1,699 (16.0) | 1,326 (20.5) | 10,889 (2.5) | 9,249 (2.9) | 16,066 (1.7) | 1.28× | lags FFTW 1.28× |
| 1,920 (2⁷·3·5) | 2,802 (18.7) | 2,301 (22.7) | 14,264 (3.7) | 12,270 (4.3) | 28,128 (1.9) | 1.22× | lags FFTW 1.22× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 344 (14.9) | 313 (16.4) | 1.10× | lags FFTW 1.10× |
| 1,024 (2¹⁰) | 1,293 (19.8) | 1,193 (21.5) | 1.08× | lags FFTW 1.08× |
| 4,096 (2¹²) | 6,913 (17.8) | 5,837 (21.1) | 1.18× | lags FFTW 1.18× |
| 65,536 (2¹⁶) | 161,341 (16.2) | 170,150 (15.4) | 0.95× | **≥ parity** |
| 1,048,576 (2²⁰) | 3,353,720 (15.6) | 4,129,543 (12.7) | 0.81× | **≥ parity** |
| 1,000 (2³·5³) | 1,522 (16.4) | 1,416 (17.6) | 1.07× | lags FFTW 1.07× |
| 1,080 (2³·3³·5) | 1,739 (15.6) | 1,539 (17.7) | 1.13× | lags FFTW 1.13× |
| 1,920 (2⁷·3·5) | 2,851 (18.4) | 2,696 (19.4) | 1.06× | lags FFTW 1.06× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 11,329 (21.7) | 15,090 (16.3) | 11,165 (22.0) | 47,832 (5.1) | 34,143 (7.2) | 1.01× | **≥ parity** |
| 128x128 | 55,915 (20.5) | 65,814 (17.4) | 61,596 (18.6) | 143,687 (8.0) | 128,468 (8.9) | 0.91× | **≥ parity** |
| 256x256 | 280,102 (18.7) | 319,743 (16.4) | 282,641 (18.6) | 676,789 (7.7) | 458,424 (11.4) | 0.99× | **≥ parity** |
| 512x512 | 1,307,556 (18.0) | 1,443,988 (16.3) | 1,299,680 (18.2) | 3,046,434 (7.7) | 1,880,936 (12.5) | 1.01× | **≥ parity** |
| 1024x1024 | 6,166,283 (17.0) | 7,582,652 (13.8) | 6,205,268 (16.9) | 15,017,569 (7.0) | 9,458,572 (11.1) | 0.99× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 72,870,703 |
| 1,024 | — | — | 130,957,450 |
| 4,096 | — | — | 239,501,309 |
| 65,536 | — | — | 2,147,486,046 |
| 1,048,576 | — | — | 3,966,977,233 |
| 1,000 | — | — | 152,218,269 |
| 1,080 | — | — | 363,905,585 |
| 1,920 | — | — | 570,758,920 |
| 1,009 | — | — | 143,153,645 |
| 1,296 | — | — | 264,615,930 |
| 10,007 | — | — | 1,224,151,016 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 11/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 256 (2⁸): 1.35×
- real 1,080 (2³·3³·5): 1.28×
- complex 1,024 (2¹⁰): 1.26×
- real 1,024 (2¹⁰): 1.26×
- complex 1,296 (2⁴·3⁴): 1.26×
- real 4,096 (2¹²): 1.26×
- real 1,920 (2⁷·3·5): 1.22×
- complex 1,080 (2³·3³·5): 1.20×
- complex 1,000 (2³·5³): 1.20×
- complex 1,920 (2⁷·3·5): 1.19×
- complex 10,007 (prime): 1.19×
- complex 4,096 (2¹²): 1.17×
- real 1,000 (2³·5³): 1.16×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
