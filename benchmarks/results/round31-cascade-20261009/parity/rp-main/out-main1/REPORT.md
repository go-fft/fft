# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon Cascade Lake (cfarm151, KVM, AVX-512).
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
| 256 (2⁸) | 464 (22.0) | 371 (27.6) | 8,656 (1.2) | 6,607 (1.5) | 5,508 (1.9) | 1.25× | lags FFTW 1.25× |
| 1,024 (2¹⁰) | 2,570 (19.9) | 2,163 (23.7) | 17,072 (3.0) | 12,050 (4.2) | 28,384 (1.8) | 1.19× | lags FFTW 1.19× |
| 4,096 (2¹²) | 12,319 (19.9) | 12,812 (19.2) | 58,455 (4.2) | 45,083 (5.5) | 133,039 (1.8) | 0.96× | **≥ parity** |
| 65,536 (2¹⁶) | 466,718 (11.2) | 410,535 (12.8) | 1,935,782 (2.7) | 1,297,542 (4.0) | 3,755,846 (1.4) | 1.14× | lags FFTW 1.14× |
| 1,048,576 (2²⁰) | 18,169,784 (5.8) | 26,854,012 (3.9) | 48,722,343 (2.2) | 41,998,993 (2.5) | 101,122,235 (1.0) | 0.68× | **≥ parity** |
| 1,000 (2³·5³) | 3,354 (14.9) | 2,743 (18.2) | 16,809 (3.0) | 14,296 (3.5) | 30,074 (1.7) | 1.22× | lags FFTW 1.22× |
| 1,080 (2³·3³·5) | 4,201 (13.0) | 3,006 (18.1) | 20,104 (2.7) | 13,546 (4.0) | 34,922 (1.6) | 1.40× | lags FFTW 1.40× |
| 1,920 (2⁷·3·5) | 6,516 (16.1) | 4,910 (21.3) | 31,191 (3.4) | 23,252 (4.5) | 61,711 (1.7) | 1.33× | lags FFTW 1.33× |
| 1,009 (prime) | 18,899 (2.7) | 31,027 (1.6) | 80,171 (0.6) | 55,208 (0.9) | 1,435,308 (0.0) | 0.61× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 4,453 (15.0) | 3,666 (18.3) | 20,491 (3.3) | 16,137 (4.2) | 41,507 (1.6) | 1.21× | lags FFTW 1.21× |
| 10,007 (prime) | 343,987 (1.9) | 364,136 (1.8) | 887,674 (0.7) | 727,091 (0.9) | 140,848,492 (0.0) | 0.94× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 392 (13.1) | 361 (14.2) | 6,569 (0.8) | 7,440 (0.7) | 2,650 (1.9) | 1.08× | lags FFTW 1.08× |
| 1,024 (2¹⁰) | 1,461 (17.5) | 1,514 (16.9) | 12,583 (2.0) | 10,471 (2.4) | 13,523 (1.9) | 0.96× | **≥ parity** |
| 4,096 (2¹²) | 8,807 (14.0) | 7,562 (16.2) | 32,651 (3.8) | 32,255 (3.8) | 61,218 (2.0) | 1.16× | lags FFTW 1.16× |
| 65,536 (2¹⁶) | 216,006 (12.1) | 187,690 (14.0) | 612,738 (4.3) | 690,161 (3.8) | 1,458,306 (1.8) | 1.15× | lags FFTW 1.15× |
| 1,048,576 (2²⁰) | 9,672,713 (5.4) | 10,146,152 (5.2) | 29,585,862 (1.8) | 22,989,628 (2.3) | 47,247,399 (1.1) | 0.95× | **≥ parity** |
| 1,000 (2³·5³) | 2,055 (12.1) | 1,860 (13.4) | 14,099 (1.8) | 12,361 (2.0) | 14,173 (1.8) | 1.10× | lags FFTW 1.10× |
| 1,080 (2³·3³·5) | 2,233 (12.2) | 2,272 (12.0) | 13,577 (2.0) | 12,706 (2.1) | 16,475 (1.7) | 0.98× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,715 (14.1) | 3,470 (15.1) | 18,010 (2.9) | 17,495 (3.0) | 29,390 (1.8) | 1.07× | lags FFTW 1.07× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 409 (12.5) | 528 (9.7) | 0.77× | **≥ parity** |
| 1,024 (2¹⁰) | 1,616 (15.8) | 1,921 (13.3) | 0.84× | **≥ parity** |
| 4,096 (2¹²) | 8,775 (14.0) | 8,354 (14.7) | 1.05× | lags FFTW 1.05× |
| 65,536 (2¹⁶) | 258,432 (10.1) | 209,082 (12.5) | 1.24× | lags FFTW 1.24× |
| 1,048,576 (2²⁰) | 13,288,029 (3.9) | 9,065,703 (5.8) | 1.47× | lags FFTW 1.47× |
| 1,000 (2³·5³) | 2,112 (11.8) | 2,249 (11.1) | 0.94× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,258 (12.0) | 2,302 (11.8) | 0.98× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,816 (13.7) | 3,964 (13.2) | 0.96× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,651 (13.9) | 30,763 (8.0) | 16,711 (14.7) | 55,500 (4.4) | 47,003 (5.2) | 1.06× | lags FFTW 1.06× |
| 128x128 | 80,580 (14.2) | 137,853 (8.3) | 72,018 (15.9) | 202,559 (5.7) | 172,497 (6.6) | 1.12× | lags FFTW 1.12× |
| 256x256 | 492,464 (10.6) | 677,386 (7.7) | 469,889 (11.2) | 912,681 (5.7) | 757,490 (6.9) | 1.05× | **≥ parity** |
| 512x512 | 2,377,519 (9.9) | 5,100,525 (4.6) | 2,404,719 (9.8) | 6,508,045 (3.6) | 4,567,777 (5.2) | 0.99× | **≥ parity** |
| 1024x1024 | 17,676,140 (5.9) | 19,987,913 (5.2) | 25,332,221 (4.1) | 31,252,721 (3.4) | 27,605,362 (3.8) | 0.70× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 22,350 | 10,131 | 68,019,932 |
| 1,024 | 78,673 | 38,776 | 128,196,232 |
| 4,096 | 307,094 | 150,679 | 278,158,121 |
| 65,536 | 4,154,629 | 2,396,314 | 3,126,714,643 |
| 1,048,576 | 77,690,886 | 36,612,245 | 8,498,581,467 |
| 1,000 | 65,732 | 37,886 | 142,189,592 |
| 1,080 | 79,573 | 39,858 | 340,762,354 |
| 1,920 | 118,746 | 66,595 | 564,089,129 |
| 1,009 | 188,191 | 43,305 | 136,443,678 |
| 1,296 | 89,220 | 45,483 | 258,347,696 |
| 10,007 | 2,153,631 | 415,416 | 1,502,228,655 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 10/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 1,080 (2³·3³·5): 1.40×
- complex 1,920 (2⁷·3·5): 1.33×
- complex 256 (2⁸): 1.25×
- complex 1,000 (2³·5³): 1.22×
- complex 1,296 (2⁴·3⁴): 1.21×
- complex 1,024 (2¹⁰): 1.19×
- real 4,096 (2¹²): 1.16×
- real 65,536 (2¹⁶): 1.15×
- complex 65,536 (2¹⁶): 1.14×
- 2-D 128x128: 1.12×
- real 1,000 (2³·5³): 1.10×
- real 256 (2⁸): 1.08×
- real 1,920 (2⁷·3·5): 1.07×
- 2-D 64x64: 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
