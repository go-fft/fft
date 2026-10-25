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
| 256 (2⁸) | — (0.0) | 287 (35.7) | 6,814 (1.5) | 5,361 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,522 (33.6) | 13,071 (3.9) | 9,883 (5.2) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 11,043 (22.3) | 42,786 (5.7) | 34,067 (7.2) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 299,640 (17.5) | 1,957,365 (2.7) | 918,951 (5.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,455,031 (9.2) | 23,743,017 (4.4) | 13,966,052 (7.5) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,186 (22.8) | 13,824 (3.6) | 10,506 (4.7) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,300 (23.7) | 14,584 (3.7) | 10,883 (5.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,821 (27.4) | 22,171 (4.7) | 16,537 (6.3) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,902 (2.4) | 61,104 (0.8) | 38,754 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 2,927 (22.9) | 16,455 (4.1) | 12,949 (5.2) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 208,287 (3.2) | 643,948 (1.0) | 467,542 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 291 (17.6) | 6,018 (0.9) | 5,412 (0.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 995 (25.7) | 9,583 (2.7) | 8,444 (3.0) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,229 (23.5) | 24,905 (4.9) | 22,015 (5.6) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 156,992 (16.7) | 417,908 (6.3) | 509,714 (5.1) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,899,030 (13.4) | 8,189,860 (6.4) | 10,467,949 (5.0) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,266 (19.7) | 10,377 (2.4) | 8,725 (2.9) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,346 (20.2) | 10,982 (2.5) | 9,101 (3.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,490 (21.0) | 14,277 (3.7) | 12,152 (4.3) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 354 (14.5) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,176 (21.8) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,856 (21.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 170,965 (15.3) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,444,072 (11.8) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,436 (17.4) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,539 (17.7) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,801 (18.7) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,620 (23.1) | 48,340 (5.1) | 33,886 (7.3) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 64,385 (17.8) | 143,516 (8.0) | 127,875 (9.0) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 279,243 (18.8) | 709,967 (7.4) | 456,113 (11.5) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,332,117 (17.7) | 3,011,268 (7.8) | 1,931,616 (12.2) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,583,565 (15.9) | 14,693,688 (7.1) | 9,425,179 (11.1) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 72,595,153 |
| 1,024 | — | — | 130,716,499 |
| 4,096 | — | — | 242,042,540 |
| 65,536 | — | — | 2,250,694,373 |
| 1,048,576 | — | — | 4,235,618,065 |
| 1,000 | — | — | 151,750,159 |
| 1,080 | — | — | 366,106,257 |
| 1,920 | — | — | 569,403,669 |
| 1,009 | — | — | 141,377,714 |
| 1,296 | — | — | 266,328,061 |
| 10,007 | — | — | 1,238,508,123 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
