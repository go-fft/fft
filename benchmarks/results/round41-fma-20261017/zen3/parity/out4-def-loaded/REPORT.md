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
| 256 (2⁸) | — (0.0) | 274 (37.3) | 6,767 (1.5) | 5,159 (2.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,521 (33.7) | 13,049 (3.9) | 9,554 (5.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,236 (24.0) | 43,917 (5.6) | 33,262 (7.4) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 298,117 (17.6) | 1,951,241 (2.7) | 928,810 (5.6) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,317,213 (9.3) | 22,618,642 (4.6) | 13,770,045 (7.6) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,141 (23.3) | 13,615 (3.7) | 9,884 (5.0) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,398 (22.7) | 14,223 (3.8) | 10,550 (5.2) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,958 (26.5) | 21,478 (4.9) | 16,180 (6.5) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,728 (2.4) | 61,201 (0.8) | 38,466 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 2,961 (22.6) | 15,995 (4.2) | 12,190 (5.5) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 207,836 (3.2) | 645,066 (1.0) | 475,189 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 224 (22.8) | 6,030 (0.8) | 5,400 (0.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 993 (25.8) | 9,398 (2.7) | 8,200 (3.1) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,131 (23.9) | 25,344 (4.8) | 21,914 (5.6) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 151,728 (17.3) | 404,600 (6.5) | 492,524 (5.3) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,595,143 (14.6) | 8,327,238 (6.3) | 10,043,719 (5.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,285 (19.4) | 10,254 (2.4) | 8,532 (2.9) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,390 (19.6) | 10,875 (2.5) | 8,893 (3.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,357 (22.2) | 14,226 (3.7) | 12,134 (4.3) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 353 (14.5) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,157 (22.1) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,901 (20.8) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 164,348 (16.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,099,378 (12.8) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,441 (17.3) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,542 (17.6) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,603 (20.1) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,509 (23.4) | 48,696 (5.0) | 33,900 (7.2) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 64,834 (17.7) | 143,871 (8.0) | 122,085 (9.4) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 285,639 (18.4) | 716,561 (7.3) | 460,628 (11.4) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,321,005 (17.9) | 2,786,620 (8.5) | 1,928,588 (12.2) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,349,844 (16.5) | 13,146,054 (8.0) | 9,322,382 (11.2) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,481,503 |
| 1,024 | — | — | 131,793,020 |
| 4,096 | — | — | 239,840,219 |
| 65,536 | — | — | 2,171,648,757 |
| 1,048,576 | — | — | 4,016,539,845 |
| 1,000 | — | — | 154,519,381 |
| 1,080 | — | — | 373,236,360 |
| 1,920 | — | — | 640,251,711 |
| 1,009 | — | — | 145,515,426 |
| 1,296 | — | — | 267,435,052 |
| 10,007 | — | — | 1,216,399,323 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
