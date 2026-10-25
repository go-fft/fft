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
| 256 (2⁸) | — (0.0) | 377 (27.1) | 8,746 (1.2) | 6,386 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,134 (24.0) | 18,390 (2.8) | 11,790 (4.3) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 13,395 (18.3) | 59,181 (4.2) | 44,730 (5.5) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 394,252 (13.3) | 1,915,380 (2.7) | 1,253,143 (4.2) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 25,273,643 (4.1) | 45,991,180 (2.3) | 38,698,139 (2.7) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,809 (17.7) | 16,975 (2.9) | 13,996 (3.6) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,969 (18.3) | 20,393 (2.7) | 13,412 (4.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,990 (21.0) | 31,388 (3.3) | 22,841 (4.6) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 30,763 (1.6) | 81,232 (0.6) | 56,813 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 4,070 (16.5) | 22,815 (2.9) | 16,836 (4.0) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 363,457 (1.8) | 942,728 (0.7) | 718,901 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 366 (14.0) | 6,623 (0.8) | 7,399 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,537 (16.7) | 13,225 (1.9) | 10,351 (2.5) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,484 (16.4) | 33,171 (3.7) | 31,489 (3.9) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 193,199 (13.6) | 612,777 (4.3) | 675,563 (3.9) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 6,190,222 (8.5) | 17,193,427 (3.0) | 18,735,784 (2.8) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,875 (13.3) | 14,356 (1.7) | 10,754 (2.3) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,158 (12.6) | 13,543 (2.0) | 12,621 (2.2) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,406 (15.4) | 18,018 (2.9) | 15,360 (3.4) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 581 (8.8) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,803 (14.2) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,432 (14.6) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 215,908 (12.1) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 9,341,329 (5.6) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,277 (10.9) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,534 (10.7) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,039 (13.0) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 17,095 (14.4) | 54,979 (4.5) | 45,748 (5.4) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 75,065 (15.3) | 212,495 (5.4) | 173,543 (6.6) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 474,374 (11.1) | 868,654 (6.0) | 746,263 (7.0) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 2,216,047 (10.6) | 4,241,541 (5.6) | 3,303,776 (7.1) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 13,129,853 (8.0) | 28,606,101 (3.7) | 25,943,110 (4.0) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 69,111,820 |
| 1,024 | — | — | 125,964,338 |
| 4,096 | — | — | 275,823,571 |
| 65,536 | — | — | 3,175,957,419 |
| 1,048,576 | — | — | 7,519,502,079 |
| 1,000 | — | — | 142,329,680 |
| 1,080 | — | — | 344,057,120 |
| 1,920 | — | — | 566,611,894 |
| 1,009 | — | — | 135,995,058 |
| 1,296 | — | — | 258,152,517 |
| 10,007 | — | — | 1,505,630,435 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
