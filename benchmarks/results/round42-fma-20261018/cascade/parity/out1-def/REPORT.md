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
| 256 (2⁸) | — (0.0) | 376 (27.2) | 9,104 (1.1) | 6,538 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,126 (24.1) | 16,454 (3.1) | 11,912 (4.3) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 12,958 (19.0) | 60,698 (4.0) | 50,006 (4.9) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 406,476 (12.9) | 1,944,932 (2.7) | 1,364,078 (3.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 26,496,718 (4.0) | 48,891,926 (2.1) | 42,575,889 (2.5) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,697 (18.5) | 18,353 (2.7) | 14,242 (3.5) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,986 (18.2) | 21,379 (2.5) | 13,526 (4.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,907 (21.3) | 32,321 (3.2) | 23,361 (4.5) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 30,238 (1.7) | 81,031 (0.6) | 56,147 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,614 (18.5) | 21,391 (3.1) | 16,353 (4.1) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 360,121 (1.8) | 922,077 (0.7) | 731,756 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 363 (14.1) | 6,748 (0.8) | 7,008 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,526 (16.8) | 13,166 (1.9) | 10,357 (2.5) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,684 (16.0) | 32,404 (3.8) | 31,356 (3.9) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 192,407 (13.6) | 603,258 (4.3) | 682,829 (3.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 9,549,889 (5.5) | 16,723,305 (3.1) | 21,782,850 (2.4) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,888 (13.2) | 14,638 (1.7) | 12,365 (2.0) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,018 (13.5) | 13,841 (2.0) | 12,991 (2.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,452 (15.2) | 18,528 (2.8) | 17,416 (3.0) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 537 (9.5) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,857 (13.8) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,709 (14.1) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 224,788 (11.7) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 10,214,517 (5.1) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,246 (11.1) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,228 (12.2) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,812 (13.7) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 14,868 (16.5) | 56,591 (4.3) | 45,246 (5.4) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 81,243 (14.1) | 207,832 (5.5) | 183,302 (6.3) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 486,167 (10.8) | 935,397 (5.6) | 765,912 (6.8) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 2,467,353 (9.6) | 4,386,449 (5.4) | 3,392,631 (7.0) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 24,118,215 (4.3) | 30,255,410 (3.5) | 25,166,693 (4.2) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 69,237,309 |
| 1,024 | — | — | 127,010,008 |
| 4,096 | — | — | 280,055,590 |
| 65,536 | — | — | 3,121,172,277 |
| 1,048,576 | — | — | 8,422,077,226 |
| 1,000 | — | — | 142,572,295 |
| 1,080 | — | — | 343,295,107 |
| 1,920 | — | — | 567,923,142 |
| 1,009 | — | — | 137,003,582 |
| 1,296 | — | — | 258,116,905 |
| 10,007 | — | — | 1,509,243,283 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
