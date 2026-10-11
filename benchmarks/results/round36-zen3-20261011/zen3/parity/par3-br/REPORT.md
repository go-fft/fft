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
| 256 (2⁸) | — (0.0) | 273 (37.5) | 6,720 (1.5) | 5,429 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,590 (32.2) | 13,023 (3.9) | 9,948 (5.1) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,660 (23.1) | 43,994 (5.6) | 33,279 (7.4) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 290,960 (18.0) | 1,898,139 (2.8) | 938,016 (5.6) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,806,330 (8.9) | 40,689,991 (2.6) | 13,953,676 (7.5) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,099 (23.7) | 13,480 (3.7) | 9,991 (5.0) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,218 (24.5) | 14,385 (3.8) | 10,879 (5.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,137 (25.3) | 21,625 (4.8) | 16,263 (6.4) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,336 (2.5) | 60,713 (0.8) | 38,873 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,013 (22.2) | 16,213 (4.1) | 12,501 (5.4) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 207,853 (3.2) | 650,574 (1.0) | 473,638 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 226 (22.7) | 6,133 (0.8) | 5,310 (1.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 964 (26.6) | 9,594 (2.7) | 8,357 (3.1) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,373 (22.9) | 24,951 (4.9) | 22,364 (5.5) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 152,242 (17.2) | 414,849 (6.3) | 498,585 (5.3) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,752,087 (14.0) | 8,259,828 (6.3) | 9,936,322 (5.3) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,275 (19.5) | 10,213 (2.4) | 8,551 (2.9) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,326 (20.5) | 10,936 (2.5) | 9,247 (2.9) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,351 (22.3) | 14,294 (3.7) | 12,073 (4.3) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 290 (17.6) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,168 (21.9) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,820 (21.1) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 167,372 (15.7) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,080,534 (12.8) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,479 (16.8) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,580 (17.2) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,758 (19.0) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 11,314 (21.7) | 49,474 (5.0) | 35,329 (7.0) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 64,538 (17.8) | 142,043 (8.1) | 123,347 (9.3) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 300,891 (17.4) | 712,016 (7.4) | 455,858 (11.5) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,323,664 (17.8) | 2,743,511 (8.6) | 1,850,699 (12.7) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,695,705 (15.7) | 13,778,887 (7.6) | 9,599,475 (10.9) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,261,172 |
| 1,024 | — | — | 136,825,812 |
| 4,096 | — | — | 248,686,013 |
| 65,536 | — | — | 2,156,639,671 |
| 1,048,576 | — | — | 4,190,495,385 |
| 1,000 | — | — | 151,689,099 |
| 1,080 | — | — | 376,504,851 |
| 1,920 | — | — | 574,955,851 |
| 1,009 | — | — | 146,763,127 |
| 1,296 | — | — | 268,455,282 |
| 10,007 | — | — | 1,221,735,115 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
