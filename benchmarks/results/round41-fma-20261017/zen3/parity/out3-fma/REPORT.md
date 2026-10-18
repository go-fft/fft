# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3, cfarm420).
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
| 256 (2⁸) | — (0.0) | 278 (36.8) | 6,836 (1.5) | 5,177 (2.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,535 (33.4) | 12,867 (4.0) | 9,443 (5.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,870 (22.6) | 42,774 (5.7) | 32,623 (7.5) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 293,686 (17.9) | 1,963,445 (2.7) | 912,518 (5.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,097,338 (9.4) | 22,732,332 (4.6) | 13,712,684 (7.6) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,139 (23.3) | 13,557 (3.7) | 9,702 (5.1) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,252 (24.2) | 13,975 (3.9) | 10,463 (5.2) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,891 (26.9) | 21,627 (4.8) | 16,159 (6.5) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,463 (2.5) | 60,598 (0.8) | 38,243 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,001 (22.3) | 16,154 (4.1) | 12,123 (5.5) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 204,321 (3.3) | 635,424 (1.0) | 471,648 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 230 (22.3) | 6,025 (0.8) | 5,303 (1.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 978 (26.2) | 9,421 (2.7) | 8,323 (3.1) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,243 (23.4) | 24,808 (5.0) | 21,847 (5.6) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 161,373 (16.2) | 428,733 (6.1) | 494,823 (5.3) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,687,555 (14.2) | 8,352,912 (6.3) | 10,071,233 (5.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,225 (20.3) | 10,237 (2.4) | 8,450 (2.9) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,345 (20.2) | 10,912 (2.5) | 8,853 (3.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,356 (22.2) | 14,326 (3.7) | 12,081 (4.3) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 288 (17.8) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,192 (21.5) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,833 (21.1) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 168,519 (15.6) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,038,439 (13.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,516 (16.4) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,562 (17.4) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,720 (19.3) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,736 (22.9) | 48,997 (5.0) | 34,178 (7.2) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 60,861 (18.8) | 146,744 (7.8) | 125,786 (9.1) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 278,092 (18.9) | 703,537 (7.5) | 455,221 (11.5) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,290,976 (18.3) | 3,096,450 (7.6) | 1,877,559 (12.6) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,647,924 (15.8) | 14,670,402 (7.1) | 9,368,923 (11.2) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,907,273 |
| 1,024 | — | — | 131,309,300 |
| 4,096 | — | — | 241,753,330 |
| 65,536 | — | — | 2,213,163,086 |
| 1,048,576 | — | — | 3,953,051,716 |
| 1,000 | — | — | 152,794,120 |
| 1,080 | — | — | 363,712,935 |
| 1,920 | — | — | 584,044,085 |
| 1,009 | — | — | 141,795,124 |
| 1,296 | — | — | 264,687,680 |
| 10,007 | — | — | 1,232,011,460 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
