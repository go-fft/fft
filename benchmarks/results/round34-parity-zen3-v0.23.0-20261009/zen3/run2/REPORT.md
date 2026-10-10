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
| 256 (2⁸) | — (0.0) | 279 (36.7) | 6,792 (1.5) | 5,483 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,513 (33.8) | 12,747 (4.0) | 10,277 (5.0) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,919 (22.5) | 42,511 (5.8) | 34,579 (7.1) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 297,204 (17.6) | 1,969,864 (2.7) | 930,430 (5.6) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 12,097,183 (8.7) | 24,423,784 (4.3) | 14,422,182 (7.3) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,031 (24.5) | 13,777 (3.6) | 10,158 (4.9) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,329 (23.4) | 14,447 (3.8) | 11,067 (4.9) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,945 (26.5) | 22,255 (4.7) | 16,155 (6.5) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,925 (2.4) | 60,221 (0.8) | 38,660 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,088 (21.7) | 16,114 (4.2) | 12,207 (5.5) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 207,431 (3.2) | 629,621 (1.1) | 478,140 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 226 (22.6) | 6,114 (0.8) | 5,353 (1.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 964 (26.6) | 9,511 (2.7) | 8,262 (3.1) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,250 (23.4) | 24,965 (4.9) | 22,255 (5.5) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 153,467 (17.1) | 407,382 (6.4) | 507,178 (5.2) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,837,273 (13.7) | 8,444,410 (6.2) | 10,105,396 (5.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,252 (19.9) | 10,404 (2.4) | 8,841 (2.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,351 (20.1) | 11,070 (2.5) | 9,174 (3.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,414 (21.7) | 14,848 (3.5) | 12,535 (4.2) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 355 (14.4) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,326 (19.3) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 6,286 (19.5) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 164,231 (16.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,139,092 (12.7) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,479 (16.8) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,555 (17.5) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,666 (19.6) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,856 (22.6) | 47,735 (5.1) | 34,455 (7.1) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 61,103 (18.8) | 143,088 (8.0) | 125,130 (9.2) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 294,318 (17.8) | 712,743 (7.4) | 458,090 (11.4) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,323,957 (17.8) | 2,747,832 (8.6) | 1,859,972 (12.7) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,708,787 (15.6) | 13,274,919 (7.9) | 9,385,614 (11.2) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,391,262 |
| 1,024 | — | — | 131,252,470 |
| 4,096 | — | — | 239,461,109 |
| 65,536 | — | — | 2,162,539,373 |
| 1,048,576 | — | — | 4,476,112,994 |
| 1,000 | — | — | 154,272,650 |
| 1,080 | — | — | 371,238,158 |
| 1,920 | — | — | 575,656,562 |
| 1,009 | — | — | 142,681,635 |
| 1,296 | — | — | 264,638,211 |
| 10,007 | — | — | 1,257,736,621 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
