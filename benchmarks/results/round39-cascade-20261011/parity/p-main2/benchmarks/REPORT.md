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
| 256 (2⁸) | — (0.0) | 414 (24.8) | 8,669 (1.2) | 6,220 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,191 (23.4) | 16,784 (3.1) | 11,981 (4.3) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 13,058 (18.8) | 60,789 (4.0) | 44,595 (5.5) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 430,169 (12.2) | 1,869,479 (2.8) | 1,329,237 (3.9) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 26,190,387 (4.0) | 50,310,135 (2.1) | 41,418,817 (2.5) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,710 (18.4) | 17,433 (2.9) | 13,929 (3.6) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 3,039 (17.9) | 20,375 (2.7) | 14,203 (3.8) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 4,915 (21.3) | 31,230 (3.4) | 23,615 (4.4) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 30,578 (1.6) | 85,478 (0.6) | 56,864 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 4,042 (16.6) | 20,916 (3.2) | 16,120 (4.2) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 357,326 (1.9) | 919,820 (0.7) | 726,069 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 344 (14.9) | 6,652 (0.8) | 6,947 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,498 (17.1) | 13,063 (2.0) | 10,150 (2.5) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,590 (16.2) | 32,423 (3.8) | 31,280 (3.9) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 182,545 (14.4) | 598,321 (4.4) | 679,785 (3.9) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 8,955,166 (5.9) | 19,145,393 (2.7) | 19,805,586 (2.6) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,888 (13.2) | 14,070 (1.8) | 12,012 (2.1) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,200 (12.4) | 13,360 (2.0) | 12,695 (2.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,495 (15.0) | 18,279 (2.9) | 17,144 (3.1) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 492 (10.4) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,842 (13.9) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,648 (14.2) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 220,486 (11.9) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 8,927,508 (5.9) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,334 (10.7) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,317 (11.7) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,974 (13.2) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 17,019 (14.4) | 57,538 (4.3) | 44,863 (5.5) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 70,711 (16.2) | 207,046 (5.5) | 173,580 (6.6) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 461,652 (11.4) | 863,004 (6.1) | 746,290 (7.0) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 2,397,192 (9.8) | 4,632,040 (5.1) | 3,615,222 (6.5) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 15,852,957 (6.6) | 26,177,921 (4.0) | 24,333,467 (4.3) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 70,210,716 |
| 1,024 | — | — | 128,449,226 |
| 4,096 | — | — | 282,873,314 |
| 65,536 | — | — | 3,144,693,067 |
| 1,048,576 | — | — | 8,303,428,939 |
| 1,000 | — | — | 143,950,282 |
| 1,080 | — | — | 344,751,405 |
| 1,920 | — | — | 572,265,390 |
| 1,009 | — | — | 135,829,324 |
| 1,296 | — | — | 261,640,944 |
| 10,007 | — | — | 1,504,457,215 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
