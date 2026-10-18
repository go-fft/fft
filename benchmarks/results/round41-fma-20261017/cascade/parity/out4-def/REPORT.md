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
| 256 (2⁸) | — (0.0) | 376 (27.2) | 8,754 (1.2) | 6,306 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,123 (24.1) | 16,820 (3.0) | 11,826 (4.3) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 12,931 (19.0) | 59,826 (4.1) | 45,463 (5.4) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 413,399 (12.7) | 1,923,405 (2.7) | 1,332,884 (3.9) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 29,389,982 (3.6) | 46,830,371 (2.2) | 41,421,820 (2.5) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,771 (18.0) | 17,220 (2.9) | 14,114 (3.5) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,960 (18.4) | 20,695 (2.6) | 13,722 (4.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,833 (21.7) | 32,718 (3.2) | 23,766 (4.4) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 30,261 (1.7) | 79,530 (0.6) | 56,659 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,586 (18.7) | 20,491 (3.3) | 16,071 (4.2) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 360,644 (1.8) | 855,346 (0.8) | 717,737 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 366 (14.0) | 6,905 (0.7) | 7,120 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,506 (17.0) | 13,190 (1.9) | 10,319 (2.5) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,437 (16.5) | 33,717 (3.6) | 31,593 (3.9) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 194,533 (13.5) | 601,706 (4.4) | 700,177 (3.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 8,906,894 (5.9) | 17,440,092 (3.0) | 26,018,057 (2.0) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,862 (13.4) | 14,362 (1.7) | 11,965 (2.1) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,153 (12.6) | 13,366 (2.0) | 12,373 (2.2) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,454 (15.2) | 17,949 (2.9) | 16,497 (3.2) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 491 (10.4) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,856 (13.8) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,675 (14.2) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 217,070 (12.1) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 10,688,500 (4.9) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,324 (10.7) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,350 (11.6) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,891 (13.5) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 14,910 (16.5) | 55,147 (4.5) | 44,716 (5.5) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 72,915 (15.7) | 212,270 (5.4) | 180,406 (6.4) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 481,862 (10.9) | 852,302 (6.2) | 737,769 (7.1) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 2,428,526 (9.7) | 4,174,017 (5.7) | 3,375,238 (7.0) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 17,703,819 (5.9) | 28,441,797 (3.7) | 26,121,214 (4.0) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 69,196,330 |
| 1,024 | — | — | 126,726,560 |
| 4,096 | — | — | 294,795,457 |
| 65,536 | — | — | 3,146,867,087 |
| 1,048,576 | — | — | 9,100,996,163 |
| 1,000 | — | — | 142,455,420 |
| 1,080 | — | — | 345,697,321 |
| 1,920 | — | — | 564,566,903 |
| 1,009 | — | — | 135,279,620 |
| 1,296 | — | — | 259,343,706 |
| 10,007 | — | — | 1,530,621,437 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
