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
| 256 (2⁸) | — (0.0) | 276 (37.1) | 6,813 (1.5) | 5,162 (2.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,516 (33.8) | 13,271 (3.9) | 9,596 (5.3) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 12,118 (20.3) | 43,888 (5.6) | 33,439 (7.3) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 294,599 (17.8) | 1,954,272 (2.7) | 906,332 (5.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,175,245 (9.4) | 22,261,636 (4.7) | 13,731,286 (7.6) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,994 (25.0) | 13,816 (3.6) | 9,816 (5.1) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,447 (22.2) | 14,588 (3.7) | 10,475 (5.2) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,909 (26.8) | 22,434 (4.7) | 15,908 (6.6) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,521 (2.5) | 60,881 (0.8) | 38,337 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,066 (21.9) | 16,232 (4.1) | 12,352 (5.4) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 205,973 (3.2) | 638,533 (1.0) | 467,325 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 224 (22.9) | 6,180 (0.8) | 5,282 (1.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 983 (26.0) | 9,564 (2.7) | 8,256 (3.1) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,291 (23.2) | 25,415 (4.8) | 21,850 (5.6) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 155,811 (16.8) | 422,915 (6.2) | 498,802 (5.3) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,695,453 (14.2) | 8,388,628 (6.2) | 10,160,711 (5.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,237 (20.1) | 10,412 (2.4) | 8,706 (2.9) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,310 (20.8) | 11,022 (2.5) | 8,925 (3.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,259 (23.2) | 14,202 (3.7) | 11,965 (4.4) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 352 (14.6) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,181 (21.7) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 6,060 (20.3) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 168,161 (15.6) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,025,249 (13.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,416 (17.6) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,573 (17.3) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,639 (19.8) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,773 (22.8) | 47,931 (5.1) | 32,587 (7.5) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 61,468 (18.7) | 147,845 (7.8) | 126,651 (9.1) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 301,418 (17.4) | 596,559 (8.8) | 449,097 (11.7) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,588,080 (14.9) | 2,895,427 (8.1) | 1,931,403 (12.2) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,638,990 (15.8) | 13,354,089 (7.9) | 9,329,422 (11.2) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,342,692 |
| 1,024 | — | — | 135,354,222 |
| 4,096 | — | — | 249,390,843 |
| 65,536 | — | — | 2,125,755,336 |
| 1,048,576 | — | — | 3,983,036,361 |
| 1,000 | — | — | 151,666,729 |
| 1,080 | — | — | 363,691,596 |
| 1,920 | — | — | 588,909,828 |
| 1,009 | — | — | 141,125,824 |
| 1,296 | — | — | 269,920,712 |
| 10,007 | — | — | 1,212,069,981 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
