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
| 256 (2⁸) | — (0.0) | 280 (36.5) | 6,821 (1.5) | 5,458 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,534 (33.4) | 13,005 (3.9) | 9,885 (5.2) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 11,005 (22.3) | 46,039 (5.3) | 33,566 (7.3) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 300,972 (17.4) | 1,990,189 (2.6) | 920,950 (5.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,616,958 (9.0) | 23,109,309 (4.5) | 14,070,800 (7.5) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,989 (25.1) | 13,898 (3.6) | 10,806 (4.6) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,359 (23.1) | 14,856 (3.7) | 11,113 (4.9) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,110 (25.5) | 23,527 (4.5) | 17,495 (6.0) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,904 (2.4) | 61,747 (0.8) | 39,376 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 2,942 (22.8) | 16,531 (4.1) | 12,716 (5.3) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 208,130 (3.2) | 660,762 (1.0) | 480,154 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 229 (22.4) | 6,024 (0.8) | 5,445 (0.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 994 (25.7) | 9,420 (2.7) | 8,501 (3.0) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,711 (21.5) | 25,238 (4.9) | 22,136 (5.6) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 158,753 (16.5) | 415,153 (6.3) | 503,564 (5.2) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,820,464 (13.7) | 8,692,032 (6.0) | 10,366,444 (5.1) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,267 (19.7) | 10,664 (2.3) | 8,622 (2.9) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,579 (17.2) | 11,195 (2.4) | 9,304 (2.9) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,354 (22.2) | 14,357 (3.6) | 12,263 (4.3) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 302 (16.9) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,183 (21.6) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,894 (20.8) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 168,525 (15.6) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,033,992 (13.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,558 (16.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,550 (17.6) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,771 (18.9) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,819 (22.7) | 48,612 (5.1) | 33,319 (7.4) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 61,827 (18.6) | 149,297 (7.7) | 127,233 (9.0) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 295,259 (17.8) | 616,174 (8.5) | 462,622 (11.3) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,313,353 (18.0) | 2,986,043 (7.9) | 1,980,990 (11.9) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,355,214 (16.5) | 13,220,834 (7.9) | 9,473,959 (11.1) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,852,073 |
| 1,024 | — | — | 133,442,491 |
| 4,096 | — | — | 243,679,190 |
| 65,536 | — | — | 2,219,111,158 |
| 1,048,576 | — | — | 4,173,524,187 |
| 1,000 | — | — | 154,575,310 |
| 1,080 | — | — | 372,163,769 |
| 1,920 | — | — | 576,143,302 |
| 1,009 | — | — | 144,173,275 |
| 1,296 | — | — | 270,768,983 |
| 10,007 | — | — | 1,231,583,790 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
