# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Xeon Cascade Lake (cfarm151).
- **Toolchains**: go1.27.1 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512** (3.3.10 built from source by setup.sh); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | — (0.0) | 363 (28.2) | 8,578 (1.2) | 6,549 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,162 (23.7) | 16,236 (3.2) | 12,010 (4.3) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 13,142 (18.7) | 63,746 (3.9) | 49,948 (4.9) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 391,216 (13.4) | 1,966,781 (2.7) | 1,379,347 (3.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 26,892,027 (3.9) | 54,694,486 (1.9) | 42,500,494 (2.5) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,748 (18.1) | 16,992 (2.9) | 14,142 (3.5) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,982 (18.2) | 20,466 (2.7) | 13,044 (4.2) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,980 (21.0) | 31,550 (3.3) | 23,165 (4.5) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 30,425 (1.7) | 88,462 (0.6) | 55,171 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,611 (18.6) | 20,893 (3.2) | 16,106 (4.2) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 357,511 (1.9) | 917,592 (0.7) | 721,166 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 364 (14.1) | 6,721 (0.8) | 7,379 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,498 (17.1) | 13,284 (1.9) | 10,873 (2.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,736 (15.9) | 34,048 (3.6) | 32,707 (3.8) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 192,251 (13.6) | 600,389 (4.4) | 701,987 (3.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 9,732,880 (5.4) | 18,513,285 (2.8) | 22,983,283 (2.3) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,868 (13.3) | 13,292 (1.9) | 12,069 (2.1) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,984 (13.7) | 13,631 (2.0) | 12,700 (2.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,463 (15.1) | 18,520 (2.8) | 17,268 (3.0) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 488 (10.5) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,932 (13.2) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,996 (13.7) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 213,484 (12.3) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 10,747,084 (4.9) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,306 (10.8) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,591 (10.5) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,049 (12.9) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 17,205 (14.3) | 57,777 (4.3) | 47,119 (5.2) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 73,322 (15.6) | 209,740 (5.5) | 184,212 (6.2) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 469,932 (11.2) | 924,472 (5.7) | 772,547 (6.8) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 2,863,964 (8.2) | 5,397,399 (4.4) | 3,611,087 (6.5) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 22,882,089 (4.6) | 35,625,740 (2.9) | 29,008,535 (3.6) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,498,774 |
| 1,024 | — | — | 128,115,909 |
| 4,096 | — | — | 282,625,284 |
| 65,536 | — | — | 3,066,719,806 |
| 1,048,576 | — | — | 8,668,794,321 |
| 1,000 | — | — | 143,446,807 |
| 1,080 | — | — | 344,919,836 |
| 1,920 | — | — | 569,068,143 |
| 1,009 | — | — | 136,074,930 |
| 1,296 | — | — | 260,086,885 |
| 10,007 | — | — | 1,512,728,136 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
