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
| 256 (2⁸) | — (0.0) | 375 (27.3) | 8,642 (1.2) | 6,477 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,122 (24.1) | 16,202 (3.2) | 11,893 (4.3) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 13,137 (18.7) | 65,689 (3.7) | 47,052 (5.2) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 403,216 (13.0) | 2,043,093 (2.6) | 1,432,533 (3.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 26,520,330 (4.0) | 51,854,917 (2.0) | 42,303,510 (2.5) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,741 (18.2) | 17,495 (2.8) | 13,962 (3.6) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 3,022 (18.0) | 20,261 (2.7) | 13,472 (4.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 5,033 (20.8) | 31,696 (3.3) | 23,760 (4.4) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 30,512 (1.6) | 84,810 (0.6) | 56,484 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,743 (17.9) | 21,430 (3.1) | 16,670 (4.0) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 368,688 (1.8) | 956,968 (0.7) | 741,258 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 366 (14.0) | 6,628 (0.8) | 7,271 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,529 (16.7) | 13,106 (2.0) | 10,523 (2.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,881 (15.6) | 33,136 (3.7) | 31,991 (3.8) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 189,434 (13.8) | 596,678 (4.4) | 710,351 (3.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 10,006,129 (5.2) | 24,054,320 (2.2) | 21,434,138 (2.4) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,903 (13.1) | 13,472 (1.8) | 12,152 (2.1) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,996 (13.6) | 13,529 (2.0) | 12,671 (2.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,610 (14.5) | 18,315 (2.9) | 17,103 (3.1) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 524 (9.8) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,935 (13.2) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,467 (14.5) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 217,208 (12.1) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 10,038,791 (5.2) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,226 (11.2) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,502 (10.9) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,082 (12.8) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 15,204 (16.2) | 54,522 (4.5) | 45,704 (5.4) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 75,760 (15.1) | 207,164 (5.5) | 185,241 (6.2) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 471,852 (11.1) | 893,732 (5.9) | 768,625 (6.8) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 2,638,640 (8.9) | 5,874,601 (4.0) | 3,880,716 (6.1) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 22,508,452 (4.7) | 32,002,693 (3.3) | 26,211,808 (4.0) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 69,706,927 |
| 1,024 | — | — | 127,701,432 |
| 4,096 | — | — | 285,640,533 |
| 65,536 | — | — | 3,106,908,081 |
| 1,048,576 | — | — | 8,470,460,161 |
| 1,000 | — | — | 145,783,279 |
| 1,080 | — | — | 348,908,798 |
| 1,920 | — | — | 578,883,375 |
| 1,009 | — | — | 137,573,783 |
| 1,296 | — | — | 262,295,844 |
| 10,007 | — | — | 1,525,697,304 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
