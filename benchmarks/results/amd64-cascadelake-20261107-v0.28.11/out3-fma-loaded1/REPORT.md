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
| 256 (2⁸) | — (0.0) | 370 (27.7) | 9,638 (1.1) | 6,546 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,127 (24.1) | 16,844 (3.0) | 12,477 (4.1) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 14,105 (17.4) | 63,908 (3.8) | 47,782 (5.1) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 414,295 (12.7) | 2,302,712 (2.3) | 1,552,254 (3.4) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 25,829,251 (4.1) | 57,818,895 (1.8) | 45,483,719 (2.3) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,778 (17.9) | 18,799 (2.7) | 14,194 (3.5) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 3,188 (17.1) | 21,242 (2.6) | 13,923 (3.9) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 5,324 (19.7) | 32,899 (3.2) | 23,562 (4.4) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 30,902 (1.6) | 84,489 (0.6) | 56,450 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,747 (17.9) | 23,754 (2.8) | 17,914 (3.7) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 371,905 (1.8) | 951,975 (0.7) | 763,076 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 366 (14.0) | 7,193 (0.7) | 7,574 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,557 (16.4) | 13,624 (1.9) | 10,603 (2.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,758 (15.8) | 35,339 (3.5) | 32,576 (3.8) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 204,690 (12.8) | 654,417 (4.0) | 755,672 (3.5) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 9,735,750 (5.4) | 36,066,944 (1.5) | 25,016,267 (2.1) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,888 (13.2) | 15,000 (1.7) | 12,469 (2.0) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,247 (12.1) | 14,702 (1.9) | 11,847 (2.3) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,483 (15.0) | 19,528 (2.7) | 17,548 (3.0) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 500 (10.2) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,981 (12.9) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,344 (14.7) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 222,690 (11.8) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 10,938,582 (4.8) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,345 (10.6) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,947 (9.2) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,158 (12.6) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 15,691 (15.7) | 60,635 (4.1) | 47,906 (5.1) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 86,627 (13.2) | 224,291 (5.1) | 194,305 (5.9) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 521,803 (10.0) | 1,120,793 (4.7) | 816,658 (6.4) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 3,474,588 (6.8) | 7,396,853 (3.2) | 5,445,531 (4.3) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 22,471,226 (4.7) | 34,016,001 (3.1) | 28,850,101 (3.6) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,532,388 |
| 1,024 | — | — | 128,265,016 |
| 4,096 | — | — | 283,374,124 |
| 65,536 | — | — | 3,202,117,453 |
| 1,048,576 | — | — | 8,624,150,039 |
| 1,000 | — | — | 144,769,536 |
| 1,080 | — | — | 349,350,538 |
| 1,920 | — | — | 577,812,465 |
| 1,009 | — | — | 138,725,881 |
| 1,296 | — | — | 264,492,862 |
| 10,007 | — | — | 1,548,188,520 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
