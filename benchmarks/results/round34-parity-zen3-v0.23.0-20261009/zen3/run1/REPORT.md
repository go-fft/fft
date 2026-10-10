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
| 256 (2⁸) | — (0.0) | 282 (36.3) | 6,855 (1.5) | 5,451 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,541 (33.2) | 13,492 (3.8) | 10,345 (4.9) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,269 (23.9) | 45,990 (5.3) | 34,903 (7.0) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 345,370 (15.2) | 1,961,583 (2.7) | 923,347 (5.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 12,620,879 (8.3) | 42,500,377 (2.5) | 14,715,193 (7.1) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,069 (24.1) | 14,233 (3.5) | 10,530 (4.7) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,331 (23.3) | 14,445 (3.8) | 11,143 (4.9) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,976 (26.3) | 21,821 (4.8) | 16,567 (6.3) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,608 (2.4) | 60,270 (0.8) | 39,849 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,092 (21.7) | 16,188 (4.1) | 12,520 (5.4) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 212,137 (3.1) | 651,407 (1.0) | 498,499 (1.3) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 230 (22.3) | 6,069 (0.8) | 5,348 (1.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 993 (25.8) | 9,426 (2.7) | 8,460 (3.0) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,332 (23.0) | 24,842 (4.9) | 22,283 (5.5) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 156,600 (16.7) | 434,202 (6.0) | 512,704 (5.1) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,815,924 (13.7) | 8,604,308 (6.1) | 10,035,274 (5.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,238 (20.1) | 10,184 (2.4) | 9,067 (2.7) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,692 (16.1) | 10,789 (2.5) | 9,046 (3.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,350 (22.3) | 14,037 (3.7) | 12,217 (4.3) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 304 (16.8) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,278 (20.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 6,011 (20.4) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 173,131 (15.1) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,174,592 (12.6) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,512 (16.5) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,597 (17.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,720 (19.2) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,670 (23.0) | 47,910 (5.1) | 34,049 (7.2) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 61,377 (18.7) | 144,898 (7.9) | 124,882 (9.2) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 283,849 (18.5) | 707,461 (7.4) | 463,060 (11.3) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,372,599 (17.2) | 3,020,898 (7.8) | 1,917,006 (12.3) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,901,955 (15.2) | 13,495,368 (7.8) | 9,450,979 (11.1) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 72,384,493 |
| 1,024 | — | — | 133,301,621 |
| 4,096 | — | — | 239,926,939 |
| 65,536 | — | — | 2,178,539,580 |
| 1,048,576 | — | — | 4,773,155,859 |
| 1,000 | — | — | 159,402,212 |
| 1,080 | — | — | 365,477,606 |
| 1,920 | — | — | 581,807,454 |
| 1,009 | — | — | 145,967,326 |
| 1,296 | — | — | 271,986,304 |
| 10,007 | — | — | 1,294,222,508 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
