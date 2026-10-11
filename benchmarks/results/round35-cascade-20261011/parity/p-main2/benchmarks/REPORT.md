# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon (Cascade Lake), cfarm151, 1 pinned core.
- **Toolchains**: go1.27.2 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512** (3.3.10 built from source with AVX-512 (setup.sh)); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 469 (21.8) | 378 (27.1) | 8,782 (1.2) | 6,590 (1.6) | 5,535 (1.9) | 1.24× | lags FFTW 1.24× |
| 1,024 (2¹⁰) | 2,527 (20.3) | 2,104 (24.3) | 16,339 (3.1) | 12,149 (4.2) | 28,670 (1.8) | 1.20× | lags FFTW 1.20× |
| 4,096 (2¹²) | 11,973 (20.5) | 12,969 (18.9) | 62,337 (3.9) | 45,784 (5.4) | 132,759 (1.9) | 0.92× | **≥ parity** |
| 65,536 (2¹⁶) | 465,318 (11.3) | 408,654 (12.8) | 2,058,408 (2.5) | 1,268,178 (4.1) | 3,659,090 (1.4) | 1.14× | lags FFTW 1.14× |
| 1,048,576 (2²⁰) | 21,204,056 (4.9) | 25,803,593 (4.1) | 49,632,198 (2.1) | 41,995,222 (2.5) | 98,328,165 (1.1) | 0.82× | **≥ parity** |
| 1,000 (2³·5³) | 2,995 (16.6) | 2,816 (17.7) | 16,649 (3.0) | 14,115 (3.5) | 30,198 (1.7) | 1.06× | lags FFTW 1.06× |
| 1,080 (2³·3³·5) | 3,789 (14.4) | 3,016 (18.0) | 20,251 (2.7) | 13,519 (4.0) | 35,136 (1.5) | 1.26× | lags FFTW 1.26× |
| 1,920 (2⁷·3·5) | 5,483 (19.1) | 4,892 (21.4) | 31,430 (3.3) | 22,536 (4.6) | 61,865 (1.7) | 1.12× | lags FFTW 1.12× |
| 1,009 (prime) | 18,703 (2.7) | 30,209 (1.7) | 80,995 (0.6) | 56,582 (0.9) | 1,437,349 (0.0) | 0.62× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 4,324 (15.5) | 3,720 (18.0) | 20,663 (3.2) | 17,420 (3.8) | 41,538 (1.6) | 1.16× | lags FFTW 1.16× |
| 10,007 (prime) | 330,835 (2.0) | 361,944 (1.8) | 887,772 (0.7) | 716,873 (0.9) | 140,713,895 (0.0) | 0.91× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 397 (12.9) | 359 (14.2) | 6,417 (0.8) | 7,279 (0.7) | 2,683 (1.9) | 1.11× | lags FFTW 1.11× |
| 1,024 (2¹⁰) | 1,286 (19.9) | 1,516 (16.9) | 12,967 (2.0) | 10,397 (2.5) | 13,300 (1.9) | 0.85× | **≥ parity** |
| 4,096 (2¹²) | 7,863 (15.6) | 7,754 (15.8) | 34,039 (3.6) | 32,051 (3.8) | 62,002 (2.0) | 1.01× | **≥ parity** |
| 65,536 (2¹⁶) | 215,086 (12.2) | 190,472 (13.8) | 590,348 (4.4) | 694,263 (3.8) | 1,435,655 (1.8) | 1.13× | lags FFTW 1.13× |
| 1,048,576 (2²⁰) | 9,432,776 (5.6) | 9,072,879 (5.8) | 18,031,026 (2.9) | 20,939,051 (2.5) | 42,801,375 (1.2) | 1.04× | **≥ parity** |
| 1,000 (2³·5³) | 1,866 (13.4) | 1,900 (13.1) | 13,026 (1.9) | 12,095 (2.1) | 14,515 (1.7) | 0.98× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,105 (12.9) | 2,047 (13.3) | 13,438 (2.0) | 12,587 (2.2) | 16,613 (1.6) | 1.03× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,422 (15.3) | 3,514 (14.9) | 18,268 (2.9) | 17,281 (3.0) | 29,671 (1.8) | 0.97× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 410 (12.5) | 523 (9.8) | 0.78× | **≥ parity** |
| 1,024 (2¹⁰) | 1,443 (17.7) | 1,936 (13.2) | 0.75× | **≥ parity** |
| 4,096 (2¹²) | 8,064 (15.2) | 8,702 (14.1) | 0.93× | **≥ parity** |
| 65,536 (2¹⁶) | 249,395 (10.5) | 214,607 (12.2) | 1.16× | lags FFTW 1.16× |
| 1,048,576 (2²⁰) | 11,329,438 (4.6) | 9,571,502 (5.5) | 1.18× | lags FFTW 1.18× |
| 1,000 (2³·5³) | 1,958 (12.7) | 2,321 (10.7) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,184 (12.5) | 2,476 (11.0) | 0.88× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,575 (14.6) | 3,997 (13.1) | 0.89× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,603 (14.0) | 31,232 (7.9) | 15,042 (16.3) | 55,788 (4.4) | 45,733 (5.4) | 1.17× | lags FFTW 1.17× |
| 128x128 | 83,145 (13.8) | 139,074 (8.2) | 84,584 (13.6) | 210,270 (5.5) | 174,715 (6.6) | 0.98× | **≥ parity** |
| 256x256 | 488,357 (10.7) | 699,317 (7.5) | 485,358 (10.8) | 863,438 (6.1) | 768,512 (6.8) | 1.01× | **≥ parity** |
| 512x512 | 2,594,554 (9.1) | 5,165,217 (4.6) | 2,477,342 (9.5) | 5,481,955 (4.3) | 3,542,425 (6.7) | 1.05× | **≥ parity** |
| 1024x1024 | 19,746,131 (5.3) | 19,258,057 (5.4) | 23,914,272 (4.4) | 30,392,621 (3.5) | 26,987,991 (3.9) | 0.83× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 22,990 | 10,331 | 69,568,040 |
| 1,024 | 80,187 | 37,159 | 126,596,034 |
| 4,096 | 311,980 | 145,154 | 280,898,472 |
| 65,536 | 4,247,250 | 2,293,541 | 3,088,881,731 |
| 1,048,576 | 75,611,896 | 36,104,566 | 8,194,584,774 |
| 1,000 | 64,111 | 36,122 | 143,481,173 |
| 1,080 | 71,835 | 40,024 | 344,929,529 |
| 1,920 | 141,439 | 69,715 | 569,912,646 |
| 1,009 | 182,907 | 46,725 | 136,744,963 |
| 1,296 | 83,949 | 46,960 | 257,650,295 |
| 10,007 | 2,160,179 | 426,498 | 1,528,918,922 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 14/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 1,080 (2³·3³·5): 1.26×
- complex 256 (2⁸): 1.24×
- complex 1,024 (2¹⁰): 1.20×
- 2-D 64x64: 1.17×
- complex 1,296 (2⁴·3⁴): 1.16×
- complex 65,536 (2¹⁶): 1.14×
- real 65,536 (2¹⁶): 1.13×
- complex 1,920 (2⁷·3·5): 1.12×
- real 256 (2⁸): 1.11×
- complex 1,000 (2³·5³): 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
