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
| 256 (2⁸) | — (0.0) | 276 (37.1) | 6,722 (1.5) | 5,256 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,597 (32.1) | 12,887 (4.0) | 9,466 (5.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 11,601 (21.2) | 43,520 (5.6) | 33,343 (7.4) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 309,482 (16.9) | 1,953,878 (2.7) | 908,740 (5.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,526,041 (9.1) | 22,738,202 (4.6) | 13,882,418 (7.6) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,086 (23.9) | 13,752 (3.6) | 10,051 (5.0) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,306 (23.6) | 14,264 (3.8) | 10,569 (5.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,207 (24.9) | 21,809 (4.8) | 16,173 (6.5) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,217 (2.5) | 61,348 (0.8) | 38,141 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,028 (22.1) | 16,366 (4.1) | 12,284 (5.5) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 205,255 (3.2) | 638,586 (1.0) | 466,144 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 228 (22.4) | 6,043 (0.8) | 5,408 (0.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 982 (26.1) | 9,638 (2.7) | 8,454 (3.0) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,335 (23.0) | 25,204 (4.9) | 22,104 (5.6) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 153,564 (17.1) | 421,988 (6.2) | 491,986 (5.3) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,582,889 (14.6) | 8,197,031 (6.4) | 10,070,322 (5.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,231 (20.2) | 10,405 (2.4) | 8,919 (2.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,308 (20.8) | 11,300 (2.4) | 9,393 (2.9) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,376 (22.0) | 14,422 (3.6) | 12,328 (4.2) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 303 (16.9) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,181 (21.7) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,800 (21.2) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 167,261 (15.7) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,021,582 (13.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,527 (16.3) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,702 (16.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,761 (19.0) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,932 (22.5) | 48,129 (5.1) | 34,545 (7.1) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 61,860 (18.5) | 146,963 (7.8) | 127,315 (9.0) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 284,511 (18.4) | 680,275 (7.7) | 454,521 (11.5) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,305,653 (18.1) | 2,749,167 (8.6) | 1,933,453 (12.2) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,500,775 (16.1) | 12,817,955 (8.2) | 9,397,295 (11.2) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,240,892 |
| 1,024 | — | — | 131,558,760 |
| 4,096 | — | — | 241,463,270 |
| 65,536 | — | — | 2,273,593,803 |
| 1,048,576 | — | — | 4,071,595,401 |
| 1,000 | — | — | 153,354,190 |
| 1,080 | — | — | 366,825,707 |
| 1,920 | — | — | 571,006,930 |
| 1,009 | — | — | 141,573,684 |
| 1,296 | — | — | 267,840,181 |
| 10,007 | — | — | 1,216,193,533 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
