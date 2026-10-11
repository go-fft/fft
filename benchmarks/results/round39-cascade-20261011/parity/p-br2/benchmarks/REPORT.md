# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon (Cascade Lake), cfarm151, 1 pinned core.
- **Toolchains**: go1.27.1 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512** (3.3.10 built from source with AVX-512 (setup.sh)); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | — (0.0) | 368 (27.8) | 8,988 (1.1) | 6,604 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,149 (23.8) | 16,449 (3.1) | 12,020 (4.3) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 13,187 (18.6) | 62,170 (4.0) | 45,619 (5.4) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 399,743 (13.1) | 1,933,195 (2.7) | 1,331,653 (3.9) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 25,832,232 (4.1) | 47,904,643 (2.2) | 40,963,983 (2.6) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,790 (17.9) | 17,066 (2.9) | 14,092 (3.5) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 3,024 (18.0) | 20,466 (2.7) | 13,258 (4.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,949 (21.2) | 31,842 (3.3) | 22,881 (4.6) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 30,268 (1.7) | 80,510 (0.6) | 56,576 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,587 (18.7) | 20,912 (3.2) | 16,615 (4.0) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 364,640 (1.8) | 853,232 (0.8) | 729,883 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 365 (14.0) | 6,611 (0.8) | 7,079 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,520 (16.8) | 12,943 (2.0) | 10,331 (2.5) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,423 (16.6) | 33,385 (3.7) | 31,729 (3.9) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 190,381 (13.8) | 605,496 (4.3) | 698,940 (3.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 7,950,502 (6.6) | 20,561,194 (2.5) | 21,002,844 (2.5) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,875 (13.3) | 13,835 (1.8) | 11,979 (2.1) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,981 (13.7) | 13,625 (2.0) | 12,452 (2.2) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,502 (14.9) | 18,315 (2.9) | 17,090 (3.1) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 496 (10.3) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,866 (13.7) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,755 (14.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 209,786 (12.5) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 9,170,403 (5.7) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,257 (11.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,307 (11.8) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,940 (13.3) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 16,935 (14.5) | 59,151 (4.2) | 45,184 (5.4) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 80,597 (14.2) | 202,249 (5.7) | 182,654 (6.3) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 478,062 (11.0) | 864,101 (6.1) | 755,425 (6.9) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 2,302,006 (10.2) | 4,644,383 (5.1) | 3,477,537 (6.8) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 17,288,938 (6.1) | 26,211,110 (4.0) | 23,393,134 (4.5) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 68,922,704 |
| 1,024 | — | — | 127,117,289 |
| 4,096 | — | — | 280,791,057 |
| 65,536 | — | — | 3,102,212,517 |
| 1,048,576 | — | — | 8,229,704,609 |
| 1,000 | — | — | 143,052,593 |
| 1,080 | — | — | 342,802,715 |
| 1,920 | — | — | 567,031,957 |
| 1,009 | — | — | 135,251,652 |
| 1,296 | — | — | 260,682,421 |
| 10,007 | — | — | 1,519,151,291 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
