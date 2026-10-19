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
| 256 (2⁸) | — (0.0) | 282 (36.3) | 6,940 (1.5) | 5,351 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,535 (33.3) | 14,403 (3.6) | 12,811 (4.0) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,450 (23.5) | 46,158 (5.3) | 35,135 (7.0) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 313,315 (16.7) | 2,225,525 (2.4) | 1,019,779 (5.1) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,558,534 (9.1) | 27,136,864 (3.9) | 20,550,706 (5.1) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,128 (23.4) | 14,481 (3.4) | 10,339 (4.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,356 (23.1) | 16,711 (3.3) | 10,910 (5.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,919 (26.7) | 24,992 (4.2) | 18,048 (5.8) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,700 (2.4) | 62,527 (0.8) | 39,593 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 2,954 (22.7) | 16,862 (4.0) | 12,420 (5.4) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 217,478 (3.1) | 672,477 (1.0) | 498,409 (1.3) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 226 (22.7) | 6,453 (0.8) | 5,369 (1.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 979 (26.2) | 10,673 (2.4) | 9,597 (2.7) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,225 (23.5) | 25,679 (4.8) | 23,121 (5.3) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 157,846 (16.6) | 436,823 (6.0) | 515,377 (5.1) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,219,564 (12.4) | 9,067,967 (5.8) | 10,577,032 (5.0) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,253 (19.9) | 10,581 (2.4) | 8,817 (2.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,335 (20.4) | 11,937 (2.3) | 9,280 (2.9) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,500 (20.9) | 15,072 (3.5) | 12,979 (4.0) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 303 (16.9) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,408 (18.2) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,170 (15.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 184,678 (14.2) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,336,660 (12.1) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,547 (16.1) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,760 (15.5) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,811 (18.6) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 11,418 (21.5) | 52,801 (4.7) | 34,115 (7.2) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 61,175 (18.7) | 149,546 (7.7) | 132,114 (8.7) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 294,493 (17.8) | 693,206 (7.6) | 462,808 (11.3) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,433,289 (16.5) | 3,311,426 (7.1) | 3,162,599 (7.5) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 7,166,470 (14.6) | 17,240,408 (6.1) | 11,603,734 (9.0) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 72,075,262 |
| 1,024 | — | — | 134,231,921 |
| 4,096 | — | — | 266,581,311 |
| 65,536 | — | — | 2,201,548,290 |
| 1,048,576 | — | — | 4,115,217,030 |
| 1,000 | — | — | 154,862,171 |
| 1,080 | — | — | 366,910,937 |
| 1,920 | — | — | 574,971,422 |
| 1,009 | — | — | 143,590,596 |
| 1,296 | — | — | 267,396,301 |
| 10,007 | — | — | 1,319,816,500 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
