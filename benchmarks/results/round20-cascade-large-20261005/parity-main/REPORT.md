# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon (Cascade Lake), 8 vCPUs, GCC Compile Farm cfarm151.
- **Toolchains**: go1.27.1 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512** (Homebrew arm64 bottle, NEON, linked from C); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 503 (20.4) | 368 (27.9) | 8,802 (1.2) | 6,401 (1.6) | 6,640 (1.5) | 1.37× | lags FFTW 1.37× |
| 1,024 (2¹⁰) | 2,778 (18.4) | 2,089 (24.5) | 16,289 (3.1) | 11,903 (4.3) | 29,072 (1.8) | 1.33× | lags FFTW 1.33× |
| 4,096 (2¹²) | 13,056 (18.8) | 12,998 (18.9) | 57,181 (4.3) | 47,699 (5.2) | 131,861 (1.9) | 1.00× | **≥ parity** |
| 65,536 (2¹⁶) | 592,581 (8.8) | 411,888 (12.7) | 1,848,350 (2.8) | 1,337,142 (3.9) | 3,625,367 (1.4) | 1.44× | lags FFTW 1.44× |
| 1,048,576 (2²⁰) | 27,265,483 (3.8) | 25,769,625 (4.1) | 51,165,232 (2.0) | 41,457,565 (2.5) | 98,662,647 (1.1) | 1.06× | lags FFTW 1.06× |
| 1,000 (2³·5³) | 3,340 (14.9) | 2,824 (17.6) | 16,964 (2.9) | 14,017 (3.6) | 30,359 (1.6) | 1.18× | lags FFTW 1.18× |
| 1,080 (2³·3³·5) | 4,166 (13.1) | 2,995 (18.2) | 20,379 (2.7) | 13,123 (4.1) | 34,812 (1.6) | 1.39× | lags FFTW 1.39× |
| 1,920 (2⁷·3·5) | 7,353 (14.2) | 4,865 (21.5) | 31,320 (3.3) | 23,258 (4.5) | 61,650 (1.7) | 1.51× | lags FFTW 1.51× |
| 1,009 (prime) | 18,809 (2.7) | 30,331 (1.7) | 80,552 (0.6) | 56,756 (0.9) | 1,431,074 (0.0) | 0.62× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 5,537 (12.1) | 3,621 (18.5) | 20,468 (3.3) | 17,320 (3.9) | 41,976 (1.6) | 1.53× | lags FFTW 1.53× |
| 10,007 (prime) | 328,457 (2.0) | 364,867 (1.8) | 944,236 (0.7) | 733,065 (0.9) | 140,015,519 (0.0) | 0.90× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 482 (10.6) | 354 (14.5) | 6,733 (0.8) | 7,145 (0.7) | 2,654 (1.9) | 1.36× | lags FFTW 1.36× |
| 1,024 (2¹⁰) | 1,781 (14.4) | 1,475 (17.4) | 13,217 (1.9) | 10,382 (2.5) | 13,185 (1.9) | 1.21× | lags FFTW 1.21× |
| 4,096 (2¹²) | 9,078 (13.5) | 7,379 (16.7) | 33,002 (3.7) | 31,387 (3.9) | 61,284 (2.0) | 1.23× | lags FFTW 1.23× |
| 65,536 (2¹⁶) | 245,608 (10.7) | 195,123 (13.4) | 585,631 (4.5) | 694,945 (3.8) | 1,435,223 (1.8) | 1.26× | lags FFTW 1.26× |
| 1,048,576 (2²⁰) | 11,738,254 (4.5) | 9,406,781 (5.6) | 23,261,745 (2.3) | 21,624,592 (2.4) | 44,167,895 (1.2) | 1.25× | lags FFTW 1.25× |
| 1,000 (2³·5³) | 2,205 (11.3) | 1,862 (13.4) | 14,274 (1.7) | 12,090 (2.1) | 14,239 (1.7) | 1.18× | lags FFTW 1.18× |
| 1,080 (2³·3³·5) | 2,550 (10.7) | 2,204 (12.3) | 13,563 (2.0) | 10,791 (2.5) | 16,483 (1.7) | 1.16× | lags FFTW 1.16× |
| 1,920 (2⁷·3·5) | 4,136 (12.7) | 3,420 (15.3) | 18,065 (2.9) | 16,744 (3.1) | 29,488 (1.8) | 1.21× | lags FFTW 1.21× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 495 (10.3) | 528 (9.7) | 0.94× | **≥ parity** |
| 1,024 (2¹⁰) | 1,795 (14.3) | 1,831 (14.0) | 0.98× | **≥ parity** |
| 4,096 (2¹²) | 9,080 (13.5) | 8,315 (14.8) | 1.09× | lags FFTW 1.09× |
| 65,536 (2¹⁶) | 263,274 (10.0) | 215,841 (12.1) | 1.22× | lags FFTW 1.22× |
| 1,048,576 (2²⁰) | 11,210,646 (4.7) | 8,910,373 (5.9) | 1.26× | lags FFTW 1.26× |
| 1,000 (2³·5³) | 2,244 (11.1) | 2,237 (11.1) | 1.00× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,597 (10.5) | 2,298 (11.8) | 1.13× | lags FFTW 1.13× |
| 1,920 (2⁷·3·5) | 4,166 (12.6) | 4,118 (12.7) | 1.01× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,705 (13.9) | 30,446 (8.1) | 14,733 (16.7) | 55,599 (4.4) | 45,466 (5.4) | 1.20× | lags FFTW 1.20× |
| 128x128 | 93,238 (12.3) | 144,847 (7.9) | 80,068 (14.3) | 209,343 (5.5) | 170,541 (6.7) | 1.16× | lags FFTW 1.16× |
| 256x256 | 513,093 (10.2) | 720,835 (7.3) | 517,261 (10.1) | 915,083 (5.7) | 755,740 (6.9) | 0.99× | **≥ parity** |
| 512x512 | 2,912,650 (8.1) | 5,069,590 (4.7) | 2,555,369 (9.2) | 6,066,013 (3.9) | 3,983,835 (5.9) | 1.14× | lags FFTW 1.14× |
| 1024x1024 | 15,119,261 (6.9) | 19,677,059 (5.3) | 22,603,987 (4.6) | 31,999,032 (3.3) | 26,936,646 (3.9) | 0.67× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 68,080,565 |
| 1,024 | — | — | 125,903,115 |
| 4,096 | — | — | 278,448,127 |
| 65,536 | — | — | 3,068,184,178 |
| 1,048,576 | — | — | 8,262,890,279 |
| 1,000 | — | — | 147,531,141 |
| 1,080 | — | — | 349,448,439 |
| 1,920 | — | — | 561,753,599 |
| 1,009 | — | — | 135,019,340 |
| 1,296 | — | — | 258,345,712 |
| 10,007 | — | — | 1,502,960,308 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 5/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 1,296 (2⁴·3⁴): 1.53×
- complex 1,920 (2⁷·3·5): 1.51×
- complex 65,536 (2¹⁶): 1.44×
- complex 1,080 (2³·3³·5): 1.39×
- complex 256 (2⁸): 1.37×
- real 256 (2⁸): 1.36×
- complex 1,024 (2¹⁰): 1.33×
- real 65,536 (2¹⁶): 1.26×
- real 1,048,576 (2²⁰): 1.25×
- real 4,096 (2¹²): 1.23×
- real 1,920 (2⁷·3·5): 1.21×
- real 1,024 (2¹⁰): 1.21×
- 2-D 64x64: 1.20×
- real 1,000 (2³·5³): 1.18×
- complex 1,000 (2³·5³): 1.18×
- 2-D 128x128: 1.16×
- real 1,080 (2³·3³·5): 1.16×
- 2-D 512x512: 1.14×
- complex 1,048,576 (2²⁰): 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
