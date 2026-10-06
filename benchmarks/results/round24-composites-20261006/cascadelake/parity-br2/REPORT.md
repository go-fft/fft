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
| 256 (2⁸) | 499 (20.5) | 372 (27.5) | 8,693 (1.2) | 6,511 (1.6) | 5,524 (1.9) | 1.34× | lags FFTW 1.34× |
| 1,024 (2¹⁰) | 2,762 (18.5) | 2,167 (23.6) | 16,203 (3.2) | 12,103 (4.2) | 28,566 (1.8) | 1.27× | lags FFTW 1.27× |
| 4,096 (2¹²) | 13,196 (18.6) | 13,275 (18.5) | 64,818 (3.8) | 50,567 (4.9) | 133,808 (1.8) | 0.99× | **≥ parity** |
| 65,536 (2¹⁶) | 463,761 (11.3) | 406,334 (12.9) | 2,034,918 (2.6) | 1,407,701 (3.7) | 3,697,282 (1.4) | 1.14× | lags FFTW 1.14× |
| 1,048,576 (2²⁰) | 22,406,480 (4.7) | 26,949,749 (3.9) | 52,935,115 (2.0) | 42,810,735 (2.4) | 100,291,249 (1.0) | 0.83× | **≥ parity** |
| 1,000 (2³·5³) | 3,299 (15.1) | 2,780 (17.9) | 16,854 (3.0) | 14,259 (3.5) | 30,088 (1.7) | 1.19× | lags FFTW 1.19× |
| 1,080 (2³·3³·5) | 4,195 (13.0) | 2,949 (18.5) | 20,177 (2.7) | 13,432 (4.1) | 35,073 (1.6) | 1.42× | lags FFTW 1.42× |
| 1,920 (2⁷·3·5) | 6,394 (16.4) | 4,937 (21.2) | 31,386 (3.3) | 22,368 (4.7) | 62,258 (1.7) | 1.30× | lags FFTW 1.30× |
| 1,009 (prime) | 18,766 (2.7) | 30,211 (1.7) | 81,157 (0.6) | 57,164 (0.9) | 1,473,616 (0.0) | 0.62× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 4,447 (15.1) | 3,638 (18.4) | 20,819 (3.2) | 15,833 (4.2) | 41,942 (1.6) | 1.22× | lags FFTW 1.22× |
| 10,007 (prime) | 336,074 (2.0) | 362,651 (1.8) | 960,391 (0.7) | 729,061 (0.9) | 145,645,925 (0.0) | 0.93× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 481 (10.6) | 341 (15.0) | 6,541 (0.8) | 6,896 (0.7) | 2,665 (1.9) | 1.41× | lags FFTW 1.41× |
| 1,024 (2¹⁰) | 1,776 (14.4) | 1,524 (16.8) | 12,217 (2.1) | 10,850 (2.4) | 13,471 (1.9) | 1.17× | lags FFTW 1.17× |
| 4,096 (2¹²) | 9,163 (13.4) | 7,434 (16.5) | 32,595 (3.8) | 30,861 (4.0) | 61,024 (2.0) | 1.23× | lags FFTW 1.23× |
| 65,536 (2¹⁶) | 248,412 (10.6) | 203,216 (12.9) | 628,835 (4.2) | 690,553 (3.8) | 1,456,763 (1.8) | 1.22× | lags FFTW 1.22× |
| 1,048,576 (2²⁰) | 13,202,533 (4.0) | 9,377,926 (5.6) | 27,994,073 (1.9) | 22,042,025 (2.4) | 47,707,965 (1.1) | 1.41× | lags FFTW 1.41× |
| 1,000 (2³·5³) | 2,224 (11.2) | 1,864 (13.4) | 14,046 (1.8) | 12,074 (2.1) | 16,457 (1.5) | 1.19× | lags FFTW 1.19× |
| 1,080 (2³·3³·5) | 2,437 (11.2) | 2,165 (12.6) | 13,392 (2.0) | 12,422 (2.2) | 19,633 (1.4) | 1.13× | lags FFTW 1.13× |
| 1,920 (2⁷·3·5) | 3,987 (13.1) | 3,354 (15.6) | 18,182 (2.9) | 17,265 (3.0) | 33,097 (1.6) | 1.19× | lags FFTW 1.19× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 495 (10.3) | 499 (10.3) | 0.99× | **≥ parity** |
| 1,024 (2¹⁰) | 1,767 (14.5) | 1,874 (13.7) | 0.94× | **≥ parity** |
| 4,096 (2¹²) | 9,201 (13.4) | 8,240 (14.9) | 1.12× | lags FFTW 1.12× |
| 65,536 (2¹⁶) | 254,929 (10.3) | 217,122 (12.1) | 1.17× | lags FFTW 1.17× |
| 1,048,576 (2²⁰) | 12,685,175 (4.1) | 10,078,087 (5.2) | 1.26× | lags FFTW 1.26× |
| 1,000 (2³·5³) | 2,273 (11.0) | 2,204 (11.3) | 1.03× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,359 (11.5) | 2,262 (12.0) | 1.04× | **≥ parity** |
| 1,920 (2⁷·3·5) | 4,043 (12.9) | 3,961 (13.2) | 1.02× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,747 (13.8) | 30,662 (8.0) | 14,876 (16.5) | 54,454 (4.5) | 46,046 (5.3) | 1.19× | lags FFTW 1.19× |
| 128x128 | 88,788 (12.9) | 145,390 (7.9) | 75,492 (15.2) | 209,632 (5.5) | 171,678 (6.7) | 1.18× | lags FFTW 1.18× |
| 256x256 | 458,918 (11.4) | 728,443 (7.2) | 474,552 (11.0) | 874,517 (6.0) | 757,019 (6.9) | 0.97× | **≥ parity** |
| 512x512 | 3,127,451 (7.5) | 5,366,330 (4.4) | 2,945,041 (8.0) | 6,600,314 (3.6) | 4,332,727 (5.4) | 1.06× | lags FFTW 1.06× |
| 1024x1024 | 20,187,824 (5.2) | 21,233,777 (4.9) | 26,051,359 (4.0) | 30,900,113 (3.4) | 27,850,079 (3.8) | 0.77× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 18,776 | 11,108 | 70,136,335 |
| 1,024 | 68,447 | 40,057 | 129,814,932 |
| 4,096 | 251,699 | 149,098 | 284,479,367 |
| 65,536 | 4,287,596 | 2,433,788 | 3,147,441,555 |
| 1,048,576 | 79,226,000 | 36,337,969 | 8,494,121,080 |
| 1,000 | 70,230 | 37,986 | 145,170,469 |
| 1,080 | 76,384 | 41,319 | 349,759,005 |
| 1,920 | 125,304 | 72,044 | 564,686,490 |
| 1,009 | 185,884 | 46,813 | 134,900,913 |
| 1,296 | 88,820 | 51,638 | 258,045,819 |
| 10,007 | 2,220,624 | 445,454 | 1,510,852,745 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 6/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 1,080 (2³·3³·5): 1.42×
- real 256 (2⁸): 1.41×
- real 1,048,576 (2²⁰): 1.41×
- complex 256 (2⁸): 1.34×
- complex 1,920 (2⁷·3·5): 1.30×
- complex 1,024 (2¹⁰): 1.27×
- real 4,096 (2¹²): 1.23×
- real 65,536 (2¹⁶): 1.22×
- complex 1,296 (2⁴·3⁴): 1.22×
- real 1,000 (2³·5³): 1.19×
- 2-D 64x64: 1.19×
- real 1,920 (2⁷·3·5): 1.19×
- complex 1,000 (2³·5³): 1.19×
- 2-D 128x128: 1.18×
- real 1,024 (2¹⁰): 1.17×
- complex 65,536 (2¹⁶): 1.14×
- real 1,080 (2³·3³·5): 1.13×
- 2-D 512x512: 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
