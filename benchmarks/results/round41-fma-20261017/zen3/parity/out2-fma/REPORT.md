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
| 256 (2⁸) | — (0.0) | 282 (36.3) | 6,591 (1.6) | 5,275 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,739 (29.4) | 13,305 (3.8) | 9,509 (5.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,736 (22.9) | 43,222 (5.7) | 33,428 (7.4) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 296,334 (17.7) | 1,978,917 (2.6) | 916,333 (5.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,212,493 (9.4) | 23,287,059 (4.5) | 14,036,151 (7.5) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,113 (23.6) | 13,853 (3.6) | 9,903 (5.0) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,282 (23.8) | 14,358 (3.8) | 10,564 (5.2) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,805 (27.5) | 22,028 (4.8) | 15,979 (6.6) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,550 (2.5) | 61,422 (0.8) | 38,268 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,796 (17.7) | 16,279 (4.1) | 12,164 (5.5) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 205,661 (3.2) | 642,021 (1.0) | 468,591 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 223 (23.0) | 5,958 (0.9) | 5,240 (1.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 985 (26.0) | 9,364 (2.7) | 8,468 (3.0) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,232 (23.5) | 24,932 (4.9) | 21,676 (5.7) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 159,222 (16.5) | 428,004 (6.1) | 493,046 (5.3) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,561,913 (14.7) | 8,344,188 (6.3) | 10,019,533 (5.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,366 (18.2) | 10,258 (2.4) | 8,562 (2.9) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,358 (20.0) | 10,876 (2.5) | 8,931 (3.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,292 (22.8) | 14,207 (3.7) | 11,951 (4.4) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 297 (17.2) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,191 (21.5) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,914 (20.8) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 168,430 (15.6) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,091,919 (12.8) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,492 (16.7) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,520 (17.9) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,620 (20.0) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,834 (22.7) | 51,017 (4.8) | 33,036 (7.4) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 61,892 (18.5) | 150,045 (7.6) | 129,429 (8.9) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 280,084 (18.7) | 645,796 (8.1) | 462,699 (11.3) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,278,784 (18.4) | 2,913,897 (8.1) | 1,938,794 (12.2) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,398,439 (16.4) | 13,473,527 (7.8) | 9,503,279 (11.0) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,744,802 |
| 1,024 | — | — | 134,347,461 |
| 4,096 | — | — | 243,897,020 |
| 65,536 | — | — | 2,158,229,301 |
| 1,048,576 | — | — | 3,952,196,017 |
| 1,000 | — | — | 152,852,749 |
| 1,080 | — | — | 364,622,296 |
| 1,920 | — | — | 592,273,719 |
| 1,009 | — | — | 142,700,745 |
| 1,296 | — | — | 282,968,759 |
| 10,007 | — | — | 1,236,053,192 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
