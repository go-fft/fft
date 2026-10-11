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
| 256 (2⁸) | — (0.0) | 282 (36.2) | 6,915 (1.5) | 5,262 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,593 (32.1) | 14,380 (3.6) | 10,142 (5.0) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,942 (22.5) | 45,621 (5.4) | 34,592 (7.1) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 295,621 (17.7) | 2,018,870 (2.6) | 915,679 (5.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,570,688 (9.1) | 24,210,792 (4.3) | 14,495,820 (7.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,206 (22.6) | 15,286 (3.3) | 10,319 (4.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,470 (22.0) | 14,564 (3.7) | 10,930 (5.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,008 (26.1) | 21,793 (4.8) | 16,117 (6.5) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 21,379 (2.4) | 60,319 (0.8) | 40,291 (1.2) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,095 (21.6) | 16,152 (4.1) | 12,455 (5.4) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 223,238 (3.0) | 640,158 (1.0) | 467,673 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 223 (23.0) | 6,057 (0.8) | 5,527 (0.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 985 (26.0) | 9,971 (2.6) | 8,463 (3.0) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,372 (22.9) | 25,069 (4.9) | 24,620 (5.0) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 156,471 (16.8) | 410,891 (6.4) | 503,486 (5.2) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,747,749 (14.0) | 8,699,295 (6.0) | 9,979,881 (5.3) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,236 (20.1) | 10,503 (2.4) | 8,762 (2.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,375 (19.8) | 11,119 (2.4) | 9,696 (2.8) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,407 (21.8) | 14,756 (3.5) | 12,731 (4.1) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 296 (17.3) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,208 (21.2) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,896 (20.8) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 169,730 (15.4) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,220,268 (12.4) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,595 (15.6) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,703 (16.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,933 (17.9) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 11,459 (21.4) | 49,929 (4.9) | 33,863 (7.3) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 65,696 (17.5) | 148,002 (7.7) | 128,446 (8.9) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 300,830 (17.4) | 722,196 (7.3) | 470,398 (11.1) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,338,403 (17.6) | 2,985,674 (7.9) | 2,053,989 (11.5) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,556,521 (16.0) | 14,336,635 (7.3) | 9,864,575 (10.6) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 73,921,513 |
| 1,024 | — | — | 135,820,462 |
| 4,096 | — | — | 242,112,060 |
| 65,536 | — | — | 2,509,269,731 |
| 1,048,576 | — | — | 4,236,303,395 |
| 1,000 | — | — | 154,093,260 |
| 1,080 | — | — | 369,505,108 |
| 1,920 | — | — | 584,199,236 |
| 1,009 | — | — | 146,014,986 |
| 1,296 | — | — | 280,208,698 |
| 10,007 | — | — | 1,270,582,728 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
