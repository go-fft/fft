# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon (Cascade Lake, AVX-512), 8 vCPUs, GCC Compile Farm cfarm151.
- **Toolchains**: go1.26.4 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512** (3.3.10 built from source with --enable-sse2 --enable-avx --enable-avx2 --enable-avx512 --enable-fma); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 518 (19.8) | 364 (28.1) | 8,971 (1.1) | 6,527 (1.6) | 9,095 (1.1) | 1.42× | lags FFTW 1.42× |
| 1,024 (2¹⁰) | 2,777 (18.4) | 2,078 (24.6) | 16,468 (3.1) | 11,988 (4.3) | 46,636 (1.1) | 1.34× | lags FFTW 1.34× |
| 4,096 (2¹²) | 13,195 (18.6) | 11,925 (20.6) | 55,881 (4.4) | 43,591 (5.6) | 223,296 (1.1) | 1.11× | lags FFTW 1.11× |
| 65,536 (2¹⁶) | 630,588 (8.3) | 402,862 (13.0) | 2,010,010 (2.6) | 1,379,240 (3.8) | 5,532,786 (0.9) | 1.57× | lags FFTW 1.57× |
| 1,048,576 (2²⁰) | 30,219,146 (3.5) | 26,034,898 (4.0) | 52,323,462 (2.0) | 40,861,867 (2.6) | 137,193,076 (0.8) | 1.16× | lags FFTW 1.16× |
| 1,000 (2³·5³) | 3,940 (12.6) | 2,767 (18.0) | 17,142 (2.9) | 14,140 (3.5) | 46,444 (1.1) | 1.42× | lags FFTW 1.42× |
| 1,080 (2³·3³·5) | 4,681 (11.6) | 3,001 (18.1) | 20,430 (2.7) | 13,235 (4.1) | 56,431 (1.0) | 1.56× | lags FFTW 1.56× |
| 1,920 (2⁷·3·5) | 7,900 (13.3) | 4,918 (21.3) | 30,208 (3.5) | 21,510 (4.9) | 97,921 (1.1) | 1.61× | lags FFTW 1.61× |
| 1,009 (prime) | 19,932 (2.5) | 30,736 (1.6) | 79,351 (0.6) | 55,867 (0.9) | 1,955,698 (0.0) | 0.65× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 6,369 (10.5) | 3,628 (18.5) | 20,564 (3.3) | 15,806 (4.2) | 66,231 (1.0) | 1.76× | lags FFTW 1.76× |
| 10,007 (prime) | 346,752 (1.9) | 359,673 (1.8) | 852,248 (0.8) | 728,135 (0.9) | 192,346,904 (0.0) | 0.96× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 494 (10.4) | 370 (13.9) | 6,517 (0.8) | 6,643 (0.8) | 4,607 (1.1) | 1.34× | lags FFTW 1.34× |
| 1,024 (2¹⁰) | 1,753 (14.6) | 1,506 (17.0) | 12,139 (2.1) | 10,123 (2.5) | 22,355 (1.1) | 1.16× | lags FFTW 1.16× |
| 4,096 (2¹²) | 9,144 (13.4) | 7,354 (16.7) | 32,419 (3.8) | 31,497 (3.9) | 97,673 (1.3) | 1.24× | lags FFTW 1.24× |
| 65,536 (2¹⁶) | 245,318 (10.7) | 194,778 (13.5) | 616,113 (4.3) | 688,198 (3.8) | 2,192,859 (1.2) | 1.26× | lags FFTW 1.26× |
| 1,048,576 (2²⁰) | 16,068,287 (3.3) | 10,403,525 (5.0) | 20,155,236 (2.6) | 21,152,562 (2.5) | 66,732,806 (0.8) | 1.54× | lags FFTW 1.54× |
| 1,000 (2³·5³) | 2,464 (10.1) | 2,016 (12.4) | 13,082 (1.9) | 11,991 (2.1) | 22,643 (1.1) | 1.22× | lags FFTW 1.22× |
| 1,080 (2³·3³·5) | 2,846 (9.6) | 2,156 (12.6) | 13,432 (2.0) | 11,119 (2.4) | 24,921 (1.1) | 1.32× | lags FFTW 1.32× |
| 1,920 (2⁷·3·5) | 4,477 (11.7) | 3,456 (15.1) | 18,370 (2.8) | 15,260 (3.4) | 45,519 (1.2) | 1.30× | lags FFTW 1.30× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 520 (9.8) | 529 (9.7) | 0.98× | **≥ parity** |
| 1,024 (2¹⁰) | 1,813 (14.1) | 1,976 (13.0) | 0.92× | **≥ parity** |
| 4,096 (2¹²) | 9,263 (13.3) | 8,378 (14.7) | 1.11× | lags FFTW 1.11× |
| 65,536 (2¹⁶) | 252,618 (10.4) | 215,436 (12.2) | 1.17× | lags FFTW 1.17× |
| 1,048,576 (2²⁰) | 15,050,916 (3.5) | 9,680,486 (5.4) | 1.55× | lags FFTW 1.55× |
| 1,000 (2³·5³) | 2,421 (10.3) | 2,279 (10.9) | 1.06× | lags FFTW 1.06× |
| 1,080 (2³·3³·5) | 2,801 (9.7) | 2,329 (11.7) | 1.20× | lags FFTW 1.20× |
| 1,920 (2⁷·3·5) | 4,535 (11.5) | 3,960 (13.2) | 1.15× | lags FFTW 1.15× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 34,023 (7.2) | 48,515 (5.1) | 15,083 (16.3) | 57,151 (4.3) | 44,842 (5.5) | 2.26× | lags FFTW 2.26× |
| 128x128 | 140,357 (8.2) | 191,900 (6.0) | 84,074 (13.6) | 211,819 (5.4) | 180,727 (6.3) | 1.67× | lags FFTW 1.67× |
| 256x256 | 498,210 (10.5) | 734,341 (7.1) | 480,842 (10.9) | 945,551 (5.5) | 770,487 (6.8) | 1.04× | **≥ parity** |
| 512x512 | 1,855,475 (12.7) | 2,943,242 (8.0) | 2,627,125 (9.0) | 6,672,469 (3.5) | 4,669,821 (5.1) | 0.71× | **≥ parity** |
| 1024x1024 | 8,430,820 (12.4) | 11,388,587 (9.2) | 25,825,030 (4.1) | 30,855,494 (3.4) | 27,295,744 (3.8) | 0.33× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 20,655 | 11,465 | 74,310,483 |
| 1,024 | 71,010 | 38,857 | 128,371,341 |
| 4,096 | 269,004 | 146,386 | 279,635,927 |
| 65,536 | 4,267,035 | 2,406,238 | 3,113,902,356 |
| 1,048,576 | 77,685,766 | 35,946,593 | 8,281,462,066 |
| 1,000 | 66,542 | 37,361 | 142,381,813 |
| 1,080 | 77,290 | 43,454 | 343,181,150 |
| 1,920 | 125,693 | 76,320 | 564,288,274 |
| 1,009 | 184,291 | 48,499 | 135,736,069 |
| 1,296 | 95,584 | 51,662 | 257,637,938 |
| 10,007 | 2,293,679 | 466,077 | 1,505,562,410 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 5/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- 2-D 64x64: 2.26×
- complex 1,296 (2⁴·3⁴): 1.76×
- 2-D 128x128: 1.67×
- complex 1,920 (2⁷·3·5): 1.61×
- complex 65,536 (2¹⁶): 1.57×
- complex 1,080 (2³·3³·5): 1.56×
- real 1,048,576 (2²⁰): 1.54×
- complex 1,000 (2³·5³): 1.42×
- complex 256 (2⁸): 1.42×
- complex 1,024 (2¹⁰): 1.34×
- real 256 (2⁸): 1.34×
- real 1,080 (2³·3³·5): 1.32×
- real 1,920 (2⁷·3·5): 1.30×
- real 65,536 (2¹⁶): 1.26×
- real 4,096 (2¹²): 1.24×
- real 1,000 (2³·5³): 1.22×
- real 1,024 (2¹⁰): 1.16×
- complex 1,048,576 (2²⁰): 1.16×
- complex 4,096 (2¹²): 1.11×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
