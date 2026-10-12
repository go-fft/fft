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
| 256 (2⁸) | — (0.0) | 277 (37.0) | 6,930 (1.5) | 5,354 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,540 (33.2) | 13,485 (3.8) | 9,779 (5.2) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,854 (22.6) | 44,502 (5.5) | 37,094 (6.6) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 334,787 (15.7) | 2,048,132 (2.6) | 928,007 (5.6) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 12,556,722 (8.4) | 25,531,064 (4.1) | 15,143,396 (6.9) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,140 (23.3) | 13,622 (3.7) | 10,349 (4.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,401 (22.7) | 14,930 (3.6) | 11,109 (4.9) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,995 (26.2) | 21,953 (4.8) | 16,470 (6.4) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,760 (2.4) | 64,527 (0.8) | 38,880 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,081 (21.7) | 16,539 (4.1) | 12,826 (5.2) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 206,319 (3.2) | 681,872 (1.0) | 478,417 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 227 (22.6) | 6,023 (0.9) | 5,468 (0.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 995 (25.7) | 9,838 (2.6) | 8,433 (3.0) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,336 (23.0) | 26,261 (4.7) | 22,676 (5.4) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 157,046 (16.7) | 418,773 (6.3) | 537,728 (4.9) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,197,702 (12.5) | 8,964,059 (5.8) | 10,656,154 (4.9) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,520 (16.4) | 10,629 (2.3) | 9,447 (2.6) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,389 (19.6) | 11,938 (2.3) | 9,634 (2.8) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,388 (21.9) | 15,642 (3.3) | 13,926 (3.8) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 350 (14.6) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,244 (20.6) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,988 (20.5) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 169,217 (15.5) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,180,768 (12.5) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,425 (17.5) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,552 (17.5) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,808 (18.6) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,844 (22.7) | 52,381 (4.7) | 36,253 (6.8) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 58,912 (19.5) | 163,429 (7.0) | 136,294 (8.4) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 334,362 (15.7) | 683,632 (7.7) | 477,063 (11.0) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,354,047 (17.4) | 2,667,071 (8.8) | 1,967,073 (12.0) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 7,161,628 (14.6) | 14,336,677 (7.3) | 9,589,642 (10.9) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,807,032 |
| 1,024 | — | — | 130,851,350 |
| 4,096 | — | — | 239,937,919 |
| 65,536 | — | — | 2,327,101,157 |
| 1,048,576 | — | — | 4,413,379,875 |
| 1,000 | — | — | 183,127,743 |
| 1,080 | — | — | 384,524,855 |
| 1,920 | — | — | 624,335,234 |
| 1,009 | — | — | 143,844,095 |
| 1,296 | — | — | 280,151,477 |
| 10,007 | — | — | 1,269,503,727 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
