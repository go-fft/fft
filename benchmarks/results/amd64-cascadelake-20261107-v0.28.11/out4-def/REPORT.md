# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon Cascade Lake, 8 cores (GCC Compile Farm cfarm151).
- **Toolchains**: go1.27.3 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512** (3.3.10 built from source by setup.sh); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | — (0.0) | 376 (27.2) | 8,983 (1.1) | 6,773 (1.5) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,136 (24.0) | 16,812 (3.0) | 12,599 (4.1) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 13,144 (18.7) | 60,761 (4.0) | 48,352 (5.1) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 401,350 (13.1) | 2,201,198 (2.4) | 1,517,737 (3.5) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 27,061,768 (3.9) | 55,314,450 (1.9) | 42,909,279 (2.4) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,745 (18.2) | 16,932 (2.9) | 14,524 (3.4) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 3,082 (17.7) | 20,734 (2.6) | 13,843 (3.9) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 5,069 (20.7) | 31,473 (3.3) | 23,790 (4.4) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 31,004 (1.6) | 79,647 (0.6) | 55,883 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 4,172 (16.1) | 21,791 (3.1) | 16,717 (4.0) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 371,773 (1.8) | 888,624 (0.7) | 742,571 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 374 (13.7) | 6,833 (0.7) | 7,456 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,533 (16.7) | 11,821 (2.2) | 10,646 (2.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,707 (15.9) | 33,718 (3.6) | 28,822 (4.3) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 201,878 (13.0) | 600,306 (4.4) | 707,657 (3.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 10,148,956 (5.2) | 32,266,138 (1.6) | 22,559,763 (2.3) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,889 (13.2) | 13,017 (1.9) | 12,221 (2.0) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,045 (13.3) | 13,778 (2.0) | 11,481 (2.4) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,485 (15.0) | 18,407 (2.8) | 16,319 (3.2) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 502 (10.2) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,921 (13.3) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 9,018 (13.6) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 220,191 (11.9) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,241,831 (4.7) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,462 (10.1) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,707 (10.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,067 (12.9) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 15,569 (15.8) | 58,945 (4.2) | 48,127 (5.1) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 85,650 (13.4) | 220,706 (5.2) | 185,890 (6.2) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 504,173 (10.4) | 964,990 (5.4) | 784,485 (6.7) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 3,446,699 (6.8) | 6,332,932 (3.7) | 4,932,328 (4.8) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 24,299,057 (4.3) | 29,759,045 (3.5) | 28,074,916 (3.7) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,106,998 |
| 1,024 | — | — | 128,723,755 |
| 4,096 | — | — | 284,786,180 |
| 65,536 | — | — | 3,124,356,806 |
| 1,048,576 | — | — | 8,571,764,767 |
| 1,000 | — | — | 145,882,317 |
| 1,080 | — | — | 352,021,745 |
| 1,920 | — | — | 579,141,987 |
| 1,009 | — | — | 137,692,523 |
| 1,296 | — | — | 262,768,274 |
| 10,007 | — | — | 1,541,822,989 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
