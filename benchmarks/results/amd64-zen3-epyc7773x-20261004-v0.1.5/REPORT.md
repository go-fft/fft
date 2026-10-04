# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3), GCC Compile Farm cfarm420.
- **Toolchains**: go1.26.4 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128** (3.3.10 built from source with --enable-sse2 --enable-avx --enable-avx2 --enable-fma); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 403 (25.4) | 278 (36.9) | 6,784 (1.5) | 5,439 (1.9) | 12,300 (0.8) | 1.45× | lags FFTW 1.45× |
| 1,024 (2¹⁰) | 2,053 (24.9) | 1,525 (33.6) | 13,203 (3.9) | 9,709 (5.3) | 61,064 (0.8) | 1.35× | lags FFTW 1.35× |
| 4,096 (2¹²) | 12,967 (19.0) | 10,519 (23.4) | 48,063 (5.1) | 32,864 (7.5) | 286,577 (0.9) | 1.23× | lags FFTW 1.23× |
| 65,536 (2¹⁶) | 287,755 (18.2) | 295,993 (17.7) | 1,950,255 (2.7) | 907,537 (5.8) | 6,261,047 (0.8) | 0.97× | **≥ parity** |
| 1,048,576 (2²⁰) | 7,015,491 (14.9) | 11,150,846 (9.4) | 23,327,232 (4.5) | 13,742,973 (7.6) | 131,640,385 (0.8) | 0.63× | **≥ parity** |
| 1,000 (2³·5³) | 2,606 (19.1) | 2,048 (24.3) | 13,867 (3.6) | 10,114 (4.9) | 68,944 (0.7) | 1.27× | lags FFTW 1.27× |
| 1,080 (2³·3³·5) | 3,044 (17.9) | 2,305 (23.6) | 14,331 (3.8) | 11,853 (4.6) | 69,808 (0.8) | 1.32× | lags FFTW 1.32× |
| 1,920 (2⁷·3·5) | 5,093 (20.6) | 3,934 (26.6) | 21,847 (4.8) | 16,313 (6.4) | 129,547 (0.8) | 1.29× | lags FFTW 1.29× |
| 1,009 (prime) | 13,257 (3.8) | 20,771 (2.4) | 60,288 (0.8) | 38,431 (1.3) | 1,549,023 (0.0) | 0.64× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 4,069 (16.5) | 2,997 (22.4) | 16,314 (4.1) | 12,375 (5.4) | 83,992 (0.8) | 1.36× | lags FFTW 1.36× |
| 10,007 (prime) | 253,076 (2.6) | 203,981 (3.3) | 649,788 (1.0) | 465,714 (1.4) | 155,005,073 (0.0) | 1.24× | lags FFTW 1.24× |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 322 (15.9) | 227 (22.6) | 6,264 (0.8) | 5,461 (0.9) | 5,211 (1.0) | 1.42× | lags FFTW 1.42× |
| 1,024 (2¹⁰) | 1,276 (20.1) | 1,004 (25.5) | 9,596 (2.7) | 8,831 (2.9) | 24,897 (1.0) | 1.27× | lags FFTW 1.27× |
| 4,096 (2¹²) | 6,965 (17.6) | 5,396 (22.8) | 25,097 (4.9) | 22,256 (5.5) | 109,442 (1.1) | 1.29× | lags FFTW 1.29× |
| 65,536 (2¹⁶) | 163,344 (16.0) | 154,110 (17.0) | 408,729 (6.4) | 497,040 (5.3) | 2,308,755 (1.1) | 1.06× | lags FFTW 1.06× |
| 1,048,576 (2²⁰) | 3,651,718 (14.4) | 3,679,195 (14.2) | 8,396,119 (6.2) | 10,128,341 (5.2) | 45,926,821 (1.1) | 0.99× | **≥ parity** |
| 1,000 (2³·5³) | 1,594 (15.6) | 1,253 (19.9) | 10,347 (2.4) | 8,598 (2.9) | 27,577 (0.9) | 1.27× | lags FFTW 1.27× |
| 1,080 (2³·3³·5) | 1,827 (14.9) | 1,326 (20.5) | 10,865 (2.5) | 9,005 (3.0) | 28,149 (1.0) | 1.38× | lags FFTW 1.38× |
| 1,920 (2⁷·3·5) | 2,992 (17.5) | 2,398 (21.8) | 14,273 (3.7) | 12,149 (4.3) | 51,694 (1.0) | 1.25× | lags FFTW 1.25× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 362 (14.2) | 348 (14.7) | 1.04× | **≥ parity** |
| 1,024 (2¹⁰) | 1,281 (20.0) | 1,176 (21.8) | 1.09× | lags FFTW 1.09× |
| 4,096 (2¹²) | 7,191 (17.1) | 5,810 (21.1) | 1.24× | lags FFTW 1.24× |
| 65,536 (2¹⁶) | 167,713 (15.6) | 166,455 (15.7) | 1.01× | **≥ parity** |
| 1,048,576 (2²⁰) | 3,417,614 (15.3) | 4,105,172 (12.8) | 0.83× | **≥ parity** |
| 1,000 (2³·5³) | 1,618 (15.4) | 1,491 (16.7) | 1.09× | lags FFTW 1.09× |
| 1,080 (2³·3³·5) | 1,828 (14.9) | 1,530 (17.8) | 1.19× | lags FFTW 1.19× |
| 1,920 (2⁷·3·5) | 3,058 (17.1) | 2,712 (19.3) | 1.13× | lags FFTW 1.13× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 23,520 (10.4) | 39,639 (6.2) | 10,702 (23.0) | 50,854 (4.8) | 34,042 (7.2) | 2.20× | lags FFTW 2.20× |
| 128x128 | 129,283 (8.9) | 207,589 (5.5) | 66,104 (17.4) | 143,128 (8.0) | 127,998 (9.0) | 1.96× | lags FFTW 1.96× |
| 256x256 | 328,888 (15.9) | 571,409 (9.2) | 283,132 (18.5) | 649,841 (8.1) | 450,754 (11.6) | 1.16× | lags FFTW 1.16× |
| 512x512 | 1,094,156 (21.6) | 1,591,244 (14.8) | 1,296,141 (18.2) | 2,899,721 (8.1) | 1,934,815 (12.2) | 0.84× | **≥ parity** |
| 1024x1024 | 2,347,602 (44.7) | 5,534,835 (18.9) | 6,554,636 (16.0) | 13,416,170 (7.8) | 9,461,339 (11.1) | 0.36× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 26,793 | 13,302 | 71,996,623 |
| 1,024 | 125,191 | 47,838 | 131,383,450 |
| 4,096 | 327,830 | 183,225 | 240,525,890 |
| 65,536 | 4,168,248 | 2,716,996 | 2,342,020,155 |
| 1,048,576 | 63,207,135 | 30,360,723 | 3,957,114,848 |
| 1,000 | 91,892 | 48,564 | 152,092,449 |
| 1,080 | 110,543 | 54,866 | 364,103,336 |
| 1,920 | 176,183 | 102,372 | 568,628,779 |
| 1,009 | 136,051 | 54,723 | 141,853,434 |
| 1,296 | 131,634 | 62,158 | 265,098,061 |
| 10,007 | 1,886,034 | 477,487 | 1,217,974,414 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 6/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- 2-D 64x64: 2.20×
- 2-D 128x128: 1.96×
- complex 256 (2⁸): 1.45×
- real 256 (2⁸): 1.42×
- real 1,080 (2³·3³·5): 1.38×
- complex 1,296 (2⁴·3⁴): 1.36×
- complex 1,024 (2¹⁰): 1.35×
- complex 1,080 (2³·3³·5): 1.32×
- complex 1,920 (2⁷·3·5): 1.29×
- real 4,096 (2¹²): 1.29×
- complex 1,000 (2³·5³): 1.27×
- real 1,000 (2³·5³): 1.27×
- real 1,024 (2¹⁰): 1.27×
- real 1,920 (2⁷·3·5): 1.25×
- complex 10,007 (prime): 1.24×
- complex 4,096 (2¹²): 1.23×
- 2-D 256x256: 1.16×
- real 65,536 (2¹⁶): 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
