# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon (Cascade Lake), cfarm151, 8 vCPUs, pinned to one core.
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
| 256 (2⁸) | 500 (20.5) | 370 (27.7) | 9,071 (1.1) | 6,328 (1.6) | 5,510 (1.9) | 1.35× | lags FFTW 1.35× |
| 1,024 (2¹⁰) | 2,798 (18.3) | 2,121 (24.1) | 16,269 (3.1) | 11,585 (4.4) | 28,513 (1.8) | 1.32× | lags FFTW 1.32× |
| 4,096 (2¹²) | 13,129 (18.7) | 13,018 (18.9) | 58,795 (4.2) | 44,889 (5.5) | 131,239 (1.9) | 1.01× | **≥ parity** |
| 65,536 (2¹⁶) | 459,309 (11.4) | 408,581 (12.8) | 2,064,741 (2.5) | 1,488,579 (3.5) | 3,403,604 (1.5) | 1.12× | lags FFTW 1.12× |
| 1,048,576 (2²⁰) | 22,583,959 (4.6) | 26,376,708 (4.0) | 53,046,375 (2.0) | 42,167,493 (2.5) | 100,543,892 (1.0) | 0.86× | **≥ parity** |
| 1,000 (2³·5³) | 3,331 (15.0) | 2,711 (18.4) | 17,501 (2.8) | 13,892 (3.6) | 30,191 (1.7) | 1.23× | lags FFTW 1.23× |
| 1,080 (2³·3³·5) | 4,155 (13.1) | 2,956 (18.4) | 20,762 (2.6) | 13,653 (4.0) | 35,064 (1.6) | 1.41× | lags FFTW 1.41× |
| 1,920 (2⁷·3·5) | 7,292 (14.4) | 4,893 (21.4) | 31,813 (3.3) | 23,287 (4.5) | 61,533 (1.7) | 1.49× | lags FFTW 1.49× |
| 1,009 (prime) | 18,680 (2.7) | 30,309 (1.7) | 83,014 (0.6) | 56,623 (0.9) | 1,437,736 (0.0) | 0.62× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 5,535 (12.1) | 3,650 (18.4) | 21,573 (3.1) | 15,755 (4.3) | 41,554 (1.6) | 1.52× | lags FFTW 1.52× |
| 10,007 (prime) | 334,833 (2.0) | 363,801 (1.8) | 974,857 (0.7) | 751,955 (0.9) | 140,618,973 (0.0) | 0.92× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 470 (10.9) | 362 (14.1) | 6,663 (0.8) | 6,205 (0.8) | 2,705 (1.9) | 1.30× | lags FFTW 1.30× |
| 1,024 (2¹⁰) | 1,771 (14.5) | 1,503 (17.0) | 13,382 (1.9) | 10,541 (2.4) | 13,424 (1.9) | 1.18× | lags FFTW 1.18× |
| 4,096 (2¹²) | 9,247 (13.3) | 7,520 (16.3) | 33,896 (3.6) | 31,171 (3.9) | 61,634 (2.0) | 1.23× | lags FFTW 1.23× |
| 65,536 (2¹⁶) | 247,343 (10.6) | 188,027 (13.9) | 609,509 (4.3) | 700,037 (3.7) | 1,446,655 (1.8) | 1.32× | lags FFTW 1.32× |
| 1,048,576 (2²⁰) | 12,987,357 (4.0) | 10,747,993 (4.9) | 36,068,648 (1.5) | 23,907,400 (2.2) | 46,134,556 (1.1) | 1.21× | lags FFTW 1.21× |
| 1,000 (2³·5³) | 2,232 (11.2) | 1,884 (13.2) | 14,407 (1.7) | 11,186 (2.2) | 14,248 (1.7) | 1.18× | lags FFTW 1.18× |
| 1,080 (2³·3³·5) | 2,566 (10.6) | 1,990 (13.7) | 14,168 (1.9) | 11,073 (2.5) | 17,045 (1.6) | 1.29× | lags FFTW 1.29× |
| 1,920 (2⁷·3·5) | 4,160 (12.6) | 3,491 (15.0) | 18,316 (2.9) | 16,942 (3.1) | 29,547 (1.8) | 1.19× | lags FFTW 1.19× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 501 (10.2) | 498 (10.3) | 1.00× | **≥ parity** |
| 1,024 (2¹⁰) | 1,755 (14.6) | 1,859 (13.8) | 0.94× | **≥ parity** |
| 4,096 (2¹²) | 9,289 (13.2) | 8,670 (14.2) | 1.07× | lags FFTW 1.07× |
| 65,536 (2¹⁶) | 269,819 (9.7) | 216,088 (12.1) | 1.25× | lags FFTW 1.25× |
| 1,048,576 (2²⁰) | 13,788,400 (3.8) | 10,209,951 (5.1) | 1.35× | lags FFTW 1.35× |
| 1,000 (2³·5³) | 2,253 (11.1) | 2,328 (10.7) | 0.97× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,564 (10.6) | 2,536 (10.7) | 1.01× | **≥ parity** |
| 1,920 (2⁷·3·5) | 4,191 (12.5) | 3,930 (13.3) | 1.07× | lags FFTW 1.07× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,372 (14.1) | 30,394 (8.1) | 17,077 (14.4) | 59,892 (4.1) | 45,640 (5.4) | 1.02× | **≥ parity** |
| 128x128 | 86,237 (13.3) | 142,994 (8.0) | 83,114 (13.8) | 211,884 (5.4) | 175,481 (6.5) | 1.04× | **≥ parity** |
| 256x256 | 462,759 (11.3) | 728,918 (7.2) | 472,066 (11.1) | 921,605 (5.7) | 773,556 (6.8) | 0.98× | **≥ parity** |
| 512x512 | 4,061,447 (5.8) | 5,452,224 (4.3) | 2,658,883 (8.9) | 6,321,692 (3.7) | 5,221,605 (4.5) | 1.53× | lags FFTW 1.53× |
| 1024x1024 | 15,414,192 (6.8) | 19,689,540 (5.3) | 24,793,450 (4.2) | 29,383,197 (3.6) | 28,362,277 (3.7) | 0.62× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 18,510 | 10,516 | 68,128,573 |
| 1,024 | 65,578 | 38,311 | 126,105,920 |
| 4,096 | 246,784 | 145,668 | 282,483,241 |
| 65,536 | 4,228,659 | 2,360,818 | 3,131,031,487 |
| 1,048,576 | 76,777,528 | 35,794,109 | 8,388,243,131 |
| 1,000 | 68,593 | 37,077 | 144,648,586 |
| 1,080 | 76,978 | 39,600 | 343,572,326 |
| 1,920 | 127,561 | 68,433 | 564,600,537 |
| 1,009 | 186,726 | 45,178 | 135,245,912 |
| 1,296 | 92,021 | 46,735 | 257,310,526 |
| 10,007 | 2,214,091 | 438,727 | 1,510,073,632 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 8/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- 2-D 512x512: 1.53×
- complex 1,296 (2⁴·3⁴): 1.52×
- complex 1,920 (2⁷·3·5): 1.49×
- complex 1,080 (2³·3³·5): 1.41×
- complex 256 (2⁸): 1.35×
- complex 1,024 (2¹⁰): 1.32×
- real 65,536 (2¹⁶): 1.32×
- real 256 (2⁸): 1.30×
- real 1,080 (2³·3³·5): 1.29×
- real 4,096 (2¹²): 1.23×
- complex 1,000 (2³·5³): 1.23×
- real 1,048,576 (2²⁰): 1.21×
- real 1,920 (2⁷·3·5): 1.19×
- real 1,000 (2³·5³): 1.18×
- real 1,024 (2¹⁰): 1.18×
- complex 65,536 (2¹⁶): 1.12×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
