# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: control r42 binary.
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
| 256 (2⁸) | — (0.0) | 371 (27.6) | 9,001 (1.1) | 6,437 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,351 (21.8) | 16,473 (3.1) | 13,097 (3.9) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 13,106 (18.8) | 61,545 (4.0) | 47,295 (5.2) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 402,333 (13.0) | 2,085,856 (2.5) | 1,412,949 (3.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 27,101,201 (3.9) | 64,016,271 (1.6) | 49,767,646 (2.1) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,841 (17.5) | 17,126 (2.9) | 13,783 (3.6) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 3,022 (18.0) | 20,211 (2.7) | 12,972 (4.2) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,981 (21.0) | 31,769 (3.3) | 22,868 (4.6) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 30,522 (1.6) | 81,027 (0.6) | 54,881 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,694 (18.1) | 21,292 (3.1) | 16,585 (4.0) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 369,916 (1.8) | 915,822 (0.7) | 732,270 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 364 (14.1) | 6,777 (0.8) | 6,923 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,526 (16.8) | 13,101 (2.0) | 10,133 (2.5) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,558 (16.3) | 32,733 (3.8) | 31,526 (3.9) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 191,708 (13.7) | 601,289 (4.4) | 699,675 (3.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 9,313,715 (5.6) | 20,556,558 (2.6) | 25,177,295 (2.1) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,877 (13.3) | 14,527 (1.7) | 12,117 (2.1) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,055 (13.2) | 15,792 (1.7) | 11,100 (2.5) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,478 (15.1) | 20,627 (2.5) | 17,122 (3.1) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 532 (9.6) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,900 (13.5) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,599 (14.3) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 215,092 (12.2) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 10,728,181 (4.9) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,360 (10.6) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,571 (10.6) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,074 (12.8) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 17,213 (14.3) | 58,301 (4.2) | 46,383 (5.3) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 85,114 (13.5) | 205,940 (5.6) | 174,891 (6.6) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 492,033 (10.7) | 943,985 (5.6) | 787,212 (6.7) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 2,843,124 (8.3) | 6,977,013 (3.4) | 4,941,127 (4.8) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 24,138,369 (4.3) | 31,593,707 (3.3) | 29,060,707 (3.6) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 70,366,906 |
| 1,024 | — | — | 128,393,235 |
| 4,096 | — | — | 280,983,863 |
| 65,536 | — | — | 3,130,544,525 |
| 1,048,576 | — | — | 8,635,765,414 |
| 1,000 | — | — | 144,322,763 |
| 1,080 | — | — | 348,553,957 |
| 1,920 | — | — | 577,782,878 |
| 1,009 | — | — | 136,229,758 |
| 1,296 | — | — | 260,526,533 |
| 10,007 | — | — | 1,525,981,997 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
