# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3), cfarm420.
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
| 256 (2⁸) | 307 (33.4) | 280 (36.6) | 6,959 (1.5) | 5,488 (1.9) | 5,694 (1.8) | 1.10× | lags FFTW 1.10× |
| 1,024 (2¹⁰) | 1,734 (29.5) | 1,624 (31.5) | 13,111 (3.9) | 9,812 (5.2) | 30,027 (1.7) | 1.07× | lags FFTW 1.07× |
| 4,096 (2¹²) | 10,101 (24.3) | 10,362 (23.7) | 43,309 (5.7) | 34,323 (7.2) | 137,454 (1.8) | 0.97× | **≥ parity** |
| 65,536 (2¹⁶) | 251,830 (20.8) | 295,238 (17.8) | 1,914,789 (2.7) | 909,080 (5.8) | 2,937,567 (1.8) | 0.85× | **≥ parity** |
| 1,048,576 (2²⁰) | 5,929,743 (17.7) | 11,363,933 (9.2) | 40,194,491 (2.6) | 14,534,897 (7.2) | 62,225,158 (1.7) | 0.52× | **≥ parity** |
| 1,000 (2³·5³) | 2,118 (23.5) | 2,029 (24.6) | 13,731 (3.6) | 10,955 (4.5) | 30,760 (1.6) | 1.04× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,558 (21.3) | 2,350 (23.2) | 14,559 (3.7) | 10,948 (5.0) | 36,564 (1.5) | 1.09× | lags FFTW 1.09× |
| 1,920 (2⁷·3·5) | 3,860 (27.1) | 4,271 (24.5) | 23,713 (4.4) | 16,366 (6.4) | 62,605 (1.7) | 0.90× | **≥ parity** |
| 1,009 (prime) | 12,994 (3.9) | 20,933 (2.4) | 60,667 (0.8) | 38,808 (1.3) | 1,136,720 (0.0) | 0.62× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 2,992 (22.4) | 3,206 (20.9) | 16,356 (4.1) | 12,585 (5.3) | 42,197 (1.6) | 0.93× | **≥ parity** |
| 10,007 (prime) | 213,419 (3.1) | 208,930 (3.2) | 645,967 (1.0) | 468,146 (1.4) | 126,551,425 (0.0) | 1.02× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 244 (21.0) | 221 (23.1) | 6,096 (0.8) | 5,209 (1.0) | 2,520 (2.0) | 1.10× | lags FFTW 1.10× |
| 1,024 (2¹⁰) | 1,012 (25.3) | 951 (26.9) | 9,508 (2.7) | 8,288 (3.1) | 13,239 (1.9) | 1.06× | lags FFTW 1.06× |
| 4,096 (2¹²) | 5,754 (21.4) | 5,257 (23.4) | 24,978 (4.9) | 21,870 (5.6) | 59,305 (2.1) | 1.09× | lags FFTW 1.09× |
| 65,536 (2¹⁶) | 136,320 (19.2) | 153,051 (17.1) | 404,934 (6.5) | 494,993 (5.3) | 1,337,320 (2.0) | 0.89× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,790,872 (18.8) | 3,730,711 (14.1) | 8,289,902 (6.3) | 10,264,072 (5.1) | 27,509,503 (1.9) | 0.75× | **≥ parity** |
| 1,000 (2³·5³) | 1,246 (20.0) | 1,350 (18.5) | 10,240 (2.4) | 8,617 (2.9) | 13,709 (1.8) | 0.92× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,525 (17.8) | 1,327 (20.5) | 10,838 (2.5) | 8,964 (3.0) | 15,787 (1.7) | 1.15× | lags FFTW 1.15× |
| 1,920 (2⁷·3·5) | 2,453 (21.3) | 2,292 (22.8) | 14,271 (3.7) | 11,990 (4.4) | 27,946 (1.9) | 1.07× | lags FFTW 1.07× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 251 (20.4) | 356 (14.4) | 0.71× | **≥ parity** |
| 1,024 (2¹⁰) | 1,013 (25.3) | 1,180 (21.7) | 0.86× | **≥ parity** |
| 4,096 (2¹²) | 5,603 (21.9) | 5,903 (20.8) | 0.95× | **≥ parity** |
| 65,536 (2¹⁶) | 143,555 (18.3) | 166,422 (15.8) | 0.86× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,845,784 (18.4) | 4,142,441 (12.7) | 0.69× | **≥ parity** |
| 1,000 (2³·5³) | 1,305 (19.1) | 1,480 (16.8) | 0.88× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,571 (17.3) | 1,598 (17.0) | 0.98× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,524 (20.7) | 2,636 (19.9) | 0.96× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 12,372 (19.9) | 16,372 (15.0) | 10,974 (22.4) | 46,543 (5.3) | 32,886 (7.5) | 1.13× | lags FFTW 1.13× |
| 128x128 | 56,239 (20.4) | 72,599 (15.8) | 62,628 (18.3) | 147,314 (7.8) | 128,327 (8.9) | 0.90× | **≥ parity** |
| 256x256 | 287,190 (18.3) | 342,390 (15.3) | 282,813 (18.5) | 611,748 (8.6) | 458,430 (11.4) | 1.02× | **≥ parity** |
| 512x512 | 1,256,880 (18.8) | 1,460,868 (16.1) | 1,342,129 (17.6) | 2,984,404 (7.9) | 1,932,963 (12.2) | 0.94× | **≥ parity** |
| 1024x1024 | 6,002,581 (17.5) | 8,099,058 (12.9) | 6,622,667 (15.8) | 13,051,902 (8.0) | 9,445,091 (11.1) | 0.91× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,244,342 |
| 1,024 | — | — | 132,018,960 |
| 4,096 | — | — | 243,743,011 |
| 65,536 | — | — | 2,266,589,410 |
| 1,048,576 | — | — | 4,141,135,502 |
| 1,000 | — | — | 153,708,660 |
| 1,080 | — | — | 367,252,267 |
| 1,920 | — | — | 570,956,150 |
| 1,009 | — | — | 143,455,625 |
| 1,296 | — | — | 272,676,834 |
| 10,007 | — | — | 1,241,165,234 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 15/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 1,080 (2³·3³·5): 1.15×
- 2-D 64x64: 1.13×
- real 256 (2⁸): 1.10×
- complex 256 (2⁸): 1.10×
- real 4,096 (2¹²): 1.09×
- complex 1,080 (2³·3³·5): 1.09×
- real 1,920 (2⁷·3·5): 1.07×
- complex 1,024 (2¹⁰): 1.07×
- real 1,024 (2¹⁰): 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
