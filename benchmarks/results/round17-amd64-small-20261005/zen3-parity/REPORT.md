# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3), GCC Compile Farm cfarm420.
- **Toolchains**: go1.27.1 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128** (3.3.10 built from source with --enable-sse2 --enable-avx --enable-avx2 --enable-fma); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 401 (25.5) | 286 (35.8) | 6,744 (1.5) | 5,335 (1.9) | 5,570 (1.8) | 1.41× | lags FFTW 1.41× |
| 1,024 (2¹⁰) | 2,057 (24.9) | 1,734 (29.5) | 12,888 (4.0) | 9,439 (5.4) | 28,806 (1.8) | 1.19× | lags FFTW 1.19× |
| 4,096 (2¹²) | 12,823 (19.2) | 10,109 (24.3) | 46,469 (5.3) | 32,964 (7.5) | 132,503 (1.9) | 1.27× | lags FFTW 1.27× |
| 65,536 (2¹⁶) | 297,569 (17.6) | 291,379 (18.0) | 1,882,640 (2.8) | 938,437 (5.6) | 2,895,850 (1.8) | 1.02× | **≥ parity** |
| 1,048,576 (2²⁰) | 6,436,742 (16.3) | 11,157,857 (9.4) | 22,740,294 (4.6) | 14,094,420 (7.4) | 59,842,748 (1.8) | 0.58× | **≥ parity** |
| 1,000 (2³·5³) | 2,391 (20.8) | 2,188 (22.8) | 13,648 (3.7) | 10,145 (4.9) | 29,777 (1.7) | 1.09× | lags FFTW 1.09× |
| 1,080 (2³·3³·5) | 2,876 (18.9) | 2,321 (23.4) | 14,100 (3.9) | 10,704 (5.1) | 35,560 (1.5) | 1.24× | lags FFTW 1.24× |
| 1,920 (2⁷·3·5) | 4,649 (22.5) | 3,818 (27.4) | 21,689 (4.8) | 16,306 (6.4) | 63,637 (1.6) | 1.22× | lags FFTW 1.22× |
| 1,009 (prime) | 12,988 (3.9) | 20,103 (2.5) | 59,440 (0.8) | 37,217 (1.4) | 1,153,476 (0.0) | 0.65× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,666 (18.3) | 3,080 (21.8) | 16,392 (4.1) | 12,240 (5.5) | 41,888 (1.6) | 1.19× | lags FFTW 1.19× |
| 10,007 (prime) | 252,134 (2.6) | 209,411 (3.2) | 653,129 (1.0) | 466,779 (1.4) | 121,112,274 (0.0) | 1.20× | lags FFTW 1.20× |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 316 (16.2) | 222 (23.0) | 6,055 (0.8) | 5,218 (1.0) | 2,613 (2.0) | 1.42× | lags FFTW 1.42× |
| 1,024 (2¹⁰) | 1,249 (20.5) | 1,038 (24.7) | 9,451 (2.7) | 8,293 (3.1) | 12,751 (2.0) | 1.20× | lags FFTW 1.20× |
| 4,096 (2¹²) | 6,696 (18.4) | 5,192 (23.7) | 25,466 (4.8) | 22,383 (5.5) | 61,206 (2.0) | 1.29× | lags FFTW 1.29× |
| 65,536 (2¹⁶) | 163,443 (16.0) | 168,232 (15.6) | 397,263 (6.6) | 491,324 (5.3) | 1,351,758 (1.9) | 0.97× | **≥ parity** |
| 1,048,576 (2²⁰) | 3,368,360 (15.6) | 3,621,895 (14.5) | 8,300,800 (6.3) | 10,219,383 (5.1) | 28,513,298 (1.8) | 0.93× | **≥ parity** |
| 1,000 (2³·5³) | 1,515 (16.4) | 1,232 (20.2) | 10,365 (2.4) | 8,895 (2.8) | 13,699 (1.8) | 1.23× | lags FFTW 1.23× |
| 1,080 (2³·3³·5) | 1,733 (15.7) | 1,323 (20.6) | 11,037 (2.5) | 9,282 (2.9) | 15,959 (1.7) | 1.31× | lags FFTW 1.31× |
| 1,920 (2⁷·3·5) | 2,821 (18.6) | 2,351 (22.3) | 14,697 (3.6) | 12,696 (4.1) | 28,023 (1.9) | 1.20× | lags FFTW 1.20× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 335 (15.3) | 353 (14.5) | 0.95× | **≥ parity** |
| 1,024 (2¹⁰) | 1,278 (20.0) | 1,164 (22.0) | 1.10× | lags FFTW 1.10× |
| 4,096 (2¹²) | 6,833 (18.0) | 6,146 (20.0) | 1.11× | lags FFTW 1.11× |
| 65,536 (2¹⁶) | 164,218 (16.0) | 167,106 (15.7) | 0.98× | **≥ parity** |
| 1,048,576 (2²⁰) | 3,604,579 (14.5) | 4,075,471 (12.9) | 0.88× | **≥ parity** |
| 1,000 (2³·5³) | 1,577 (15.8) | 1,423 (17.5) | 1.11× | lags FFTW 1.11× |
| 1,080 (2³·3³·5) | 1,742 (15.6) | 1,708 (15.9) | 1.02× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,932 (17.9) | 3,084 (17.0) | 0.95× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 10,818 (22.7) | 15,047 (16.3) | 12,001 (20.5) | 49,681 (4.9) | 33,642 (7.3) | 0.90× | **≥ parity** |
| 128x128 | 55,124 (20.8) | 69,689 (16.5) | 61,732 (18.6) | 142,413 (8.1) | 129,620 (8.8) | 0.89× | **≥ parity** |
| 256x256 | 285,933 (18.3) | 345,785 (15.2) | 286,703 (18.3) | 639,109 (8.2) | 454,439 (11.5) | 1.00× | **≥ parity** |
| 512x512 | 1,333,251 (17.7) | 1,493,579 (15.8) | 1,256,457 (18.8) | 2,903,627 (8.1) | 1,869,173 (12.6) | 1.06× | lags FFTW 1.06× |
| 1024x1024 | 5,869,450 (17.9) | 7,428,613 (14.1) | 6,448,914 (16.3) | 13,542,561 (7.7) | 9,748,028 (10.8) | 0.91× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 9,871 | 7,263 | 71,738,203 |
| 1,024 | 35,930 | 27,056 | 132,036,990 |
| 4,096 | 133,132 | 104,313 | 243,839,321 |
| 65,536 | 2,102,426 | 1,600,307 | 2,131,104,849 |
| 1,048,576 | 39,176,074 | 24,345,882 | 4,042,579,927 |
| 1,000 | 34,414 | 26,568 | 153,454,979 |
| 1,080 | 40,177 | 28,823 | 379,822,332 |
| 1,920 | 68,684 | 50,553 | 568,591,408 |
| 1,009 | 83,386 | 30,761 | 141,432,044 |
| 1,296 | 45,030 | 34,444 | 263,815,050 |
| 10,007 | 1,094,257 | 297,454 | 1,229,420,979 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 9/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.42×
- complex 256 (2⁸): 1.41×
- real 1,080 (2³·3³·5): 1.31×
- real 4,096 (2¹²): 1.29×
- complex 4,096 (2¹²): 1.27×
- complex 1,080 (2³·3³·5): 1.24×
- real 1,000 (2³·5³): 1.23×
- complex 1,920 (2⁷·3·5): 1.22×
- complex 10,007 (prime): 1.20×
- real 1,024 (2¹⁰): 1.20×
- real 1,920 (2⁷·3·5): 1.20×
- complex 1,296 (2⁴·3⁴): 1.19×
- complex 1,024 (2¹⁰): 1.19×
- complex 1,000 (2³·5³): 1.09×
- 2-D 512x512: 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
