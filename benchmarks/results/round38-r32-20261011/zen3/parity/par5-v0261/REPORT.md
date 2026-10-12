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
| 256 (2⁸) | — (0.0) | 296 (34.6) | 6,959 (1.5) | 5,272 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,523 (33.6) | 13,351 (3.8) | 9,706 (5.3) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,798 (22.8) | 44,576 (5.5) | 33,450 (7.3) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 296,684 (17.7) | 1,971,171 (2.7) | 920,748 (5.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,188,041 (9.4) | 27,495,421 (3.8) | 14,463,193 (7.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,073 (24.0) | 14,635 (3.4) | 10,415 (4.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,315 (23.5) | 14,793 (3.7) | 11,138 (4.9) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,935 (26.6) | 22,400 (4.7) | 16,588 (6.3) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 19,925 (2.5) | 62,520 (0.8) | 41,444 (1.2) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 2,939 (22.8) | 16,737 (4.0) | 12,661 (5.3) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 206,123 (3.2) | 642,099 (1.0) | 465,274 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 223 (23.0) | 6,313 (0.8) | 5,391 (0.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 983 (26.0) | 9,761 (2.6) | 8,531 (3.0) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,226 (23.5) | 25,084 (4.9) | 22,108 (5.6) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 152,568 (17.2) | 427,141 (6.1) | 491,419 (5.3) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,753,168 (14.0) | 8,533,059 (6.1) | 10,116,900 (5.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,242 (20.1) | 10,406 (2.4) | 8,543 (2.9) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,306 (20.8) | 10,982 (2.5) | 9,227 (2.9) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,453 (21.3) | 14,410 (3.6) | 12,312 (4.3) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 349 (14.7) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,155 (22.2) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,878 (20.9) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 166,999 (15.7) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,294,406 (12.2) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,540 (16.2) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,608 (16.9) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,809 (18.6) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,696 (23.0) | 48,469 (5.1) | 33,161 (7.4) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 61,308 (18.7) | 147,210 (7.8) | 122,554 (9.4) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 290,785 (18.0) | 671,950 (7.8) | 472,131 (11.1) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,287,423 (18.3) | 2,826,631 (8.3) | 1,939,512 (12.2) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,599,940 (15.9) | 14,434,877 (7.3) | 9,702,352 (10.8) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 73,862,493 |
| 1,024 | — | — | 132,192,500 |
| 4,096 | — | — | 240,181,649 |
| 65,536 | — | — | 2,194,153,067 |
| 1,048,576 | — | — | 3,960,899,430 |
| 1,000 | — | — | 154,559,270 |
| 1,080 | — | — | 369,097,558 |
| 1,920 | — | — | 574,948,831 |
| 1,009 | — | — | 141,797,714 |
| 1,296 | — | — | 265,339,471 |
| 10,007 | — | — | 1,223,705,996 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
