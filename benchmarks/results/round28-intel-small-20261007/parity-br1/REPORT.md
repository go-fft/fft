# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon (Cascade Lake), cfarm151 VM, one pinned core.
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
| 256 (2⁸) | 456 (22.5) | 378 (27.1) | 8,410 (1.2) | 6,343 (1.6) | 5,539 (1.8) | 1.21× | lags FFTW 1.21× |
| 1,024 (2¹⁰) | 2,483 (20.6) | 2,137 (24.0) | 16,443 (3.1) | 11,745 (4.4) | 28,763 (1.8) | 1.16× | lags FFTW 1.16× |
| 4,096 (2¹²) | 12,198 (20.1) | 12,675 (19.4) | 60,383 (4.1) | 46,364 (5.3) | 132,987 (1.8) | 0.96× | **≥ parity** |
| 65,536 (2¹⁶) | 461,043 (11.4) | 401,617 (13.1) | 1,921,658 (2.7) | 1,341,121 (3.9) | 3,575,859 (1.5) | 1.15× | lags FFTW 1.15× |
| 1,048,576 (2²⁰) | 22,285,662 (4.7) | 26,719,241 (3.9) | 49,615,545 (2.1) | 41,813,976 (2.5) | 100,575,336 (1.0) | 0.83× | **≥ parity** |
| 1,000 (2³·5³) | 3,337 (14.9) | 2,698 (18.5) | 17,281 (2.9) | 13,841 (3.6) | 30,434 (1.6) | 1.24× | lags FFTW 1.24× |
| 1,080 (2³·3³·5) | 4,159 (13.1) | 3,035 (17.9) | 20,474 (2.7) | 13,203 (4.1) | 35,711 (1.5) | 1.37× | lags FFTW 1.37× |
| 1,920 (2⁷·3·5) | 6,415 (16.3) | 5,114 (20.5) | 31,558 (3.3) | 23,065 (4.5) | 62,549 (1.7) | 1.25× | lags FFTW 1.25× |
| 1,009 (prime) | 18,392 (2.7) | 30,297 (1.7) | 84,037 (0.6) | 55,784 (0.9) | 1,450,090 (0.0) | 0.61× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 4,304 (15.6) | 3,584 (18.7) | 20,884 (3.2) | 16,857 (4.0) | 41,702 (1.6) | 1.20× | lags FFTW 1.20× |
| 10,007 (prime) | 327,725 (2.0) | 361,767 (1.8) | 890,403 (0.7) | 723,707 (0.9) | 140,233,070 (0.0) | 0.91× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 436 (11.8) | 359 (14.2) | 6,502 (0.8) | 6,996 (0.7) | 2,670 (1.9) | 1.21× | lags FFTW 1.21× |
| 1,024 (2¹⁰) | 1,669 (15.3) | 1,467 (17.5) | 13,091 (2.0) | 10,158 (2.5) | 13,271 (1.9) | 1.14× | lags FFTW 1.14× |
| 4,096 (2¹²) | 9,244 (13.3) | 7,421 (16.6) | 32,420 (3.8) | 31,246 (3.9) | 61,470 (2.0) | 1.25× | lags FFTW 1.25× |
| 65,536 (2¹⁶) | 239,121 (11.0) | 181,732 (14.4) | 635,439 (4.1) | 693,091 (3.8) | 1,441,216 (1.8) | 1.32× | lags FFTW 1.32× |
| 1,048,576 (2²⁰) | 12,257,694 (4.3) | 10,239,803 (5.1) | 19,069,636 (2.7) | 20,506,673 (2.6) | 44,853,069 (1.2) | 1.20× | lags FFTW 1.20× |
| 1,000 (2³·5³) | 2,206 (11.3) | 1,944 (12.8) | 13,522 (1.8) | 11,889 (2.1) | 14,144 (1.8) | 1.13× | lags FFTW 1.13× |
| 1,080 (2³·3³·5) | 2,415 (11.3) | 2,137 (12.7) | 13,541 (2.0) | 11,002 (2.5) | 16,472 (1.7) | 1.13× | lags FFTW 1.13× |
| 1,920 (2⁷·3·5) | 3,990 (13.1) | 3,405 (15.4) | 18,085 (2.9) | 17,061 (3.1) | 28,964 (1.8) | 1.17× | lags FFTW 1.17× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 452 (11.3) | 518 (9.9) | 0.87× | **≥ parity** |
| 1,024 (2¹⁰) | 1,733 (14.8) | 1,816 (14.1) | 0.95× | **≥ parity** |
| 4,096 (2¹²) | 8,991 (13.7) | 8,884 (13.8) | 1.01× | **≥ parity** |
| 65,536 (2¹⁶) | 250,269 (10.5) | 217,149 (12.1) | 1.15× | lags FFTW 1.15× |
| 1,048,576 (2²⁰) | 12,528,171 (4.2) | 9,211,820 (5.7) | 1.36× | lags FFTW 1.36× |
| 1,000 (2³·5³) | 2,251 (11.1) | 2,309 (10.8) | 0.97× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,390 (11.4) | 3,076 (8.8) | 0.78× | **≥ parity** |
| 1,920 (2⁷·3·5) | 4,028 (13.0) | 3,934 (13.3) | 1.02× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 18,194 (13.5) | 30,672 (8.0) | 16,984 (14.5) | 53,891 (4.6) | 44,623 (5.5) | 1.07× | lags FFTW 1.07× |
| 128x128 | 79,844 (14.4) | 138,178 (8.3) | 71,085 (16.1) | 213,609 (5.4) | 173,224 (6.6) | 1.12× | lags FFTW 1.12× |
| 256x256 | 472,045 (11.1) | 693,174 (7.6) | 476,516 (11.0) | 927,324 (5.7) | 755,400 (6.9) | 0.99× | **≥ parity** |
| 512x512 | 3,328,023 (7.1) | 5,013,472 (4.7) | 2,384,709 (9.9) | 5,940,593 (4.0) | 3,732,272 (6.3) | 1.40× | lags FFTW 1.40× |
| 1024x1024 | 16,977,023 (6.2) | 21,042,877 (5.0) | 24,154,386 (4.3) | 29,699,207 (3.5) | 25,133,298 (4.2) | 0.70× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 23,700 | 10,568 | 68,352,750 |
| 1,024 | 83,759 | 37,818 | 127,434,423 |
| 4,096 | 318,495 | 145,308 | 280,565,200 |
| 65,536 | 4,437,745 | 2,273,714 | 3,142,708,512 |
| 1,048,576 | 76,752,946 | 35,796,027 | 8,423,444,828 |
| 1,000 | 66,736 | 36,466 | 143,644,184 |
| 1,080 | 74,528 | 39,473 | 344,436,647 |
| 1,920 | 120,716 | 68,499 | 567,105,219 |
| 1,009 | 181,627 | 44,632 | 135,500,098 |
| 1,296 | 87,703 | 47,173 | 260,742,046 |
| 10,007 | 2,232,682 | 431,016 | 1,530,651,193 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 6/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- 2-D 512x512: 1.40×
- complex 1,080 (2³·3³·5): 1.37×
- real 65,536 (2¹⁶): 1.32×
- complex 1,920 (2⁷·3·5): 1.25×
- real 4,096 (2¹²): 1.25×
- complex 1,000 (2³·5³): 1.24×
- real 256 (2⁸): 1.21×
- complex 256 (2⁸): 1.21×
- complex 1,296 (2⁴·3⁴): 1.20×
- real 1,048,576 (2²⁰): 1.20×
- real 1,920 (2⁷·3·5): 1.17×
- complex 1,024 (2¹⁰): 1.16×
- complex 65,536 (2¹⁶): 1.15×
- real 1,024 (2¹⁰): 1.14×
- real 1,000 (2³·5³): 1.13×
- real 1,080 (2³·3³·5): 1.13×
- 2-D 128x128: 1.12×
- 2-D 64x64: 1.07×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
