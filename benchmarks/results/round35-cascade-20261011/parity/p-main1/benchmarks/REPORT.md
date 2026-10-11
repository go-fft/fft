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
| 256 (2⁸) | 472 (21.7) | 356 (28.7) | 8,924 (1.1) | 6,406 (1.6) | 5,508 (1.9) | 1.32× | lags FFTW 1.32× |
| 1,024 (2¹⁰) | 2,542 (20.1) | 2,085 (24.6) | 16,496 (3.1) | 11,916 (4.3) | 28,350 (1.8) | 1.22× | lags FFTW 1.22× |
| 4,096 (2¹²) | 12,288 (20.0) | 13,734 (17.9) | 59,738 (4.1) | 46,302 (5.3) | 131,905 (1.9) | 0.89× | **≥ parity** |
| 65,536 (2¹⁶) | 453,240 (11.6) | 390,128 (13.4) | 1,942,466 (2.7) | 1,325,087 (4.0) | 3,524,488 (1.5) | 1.16× | lags FFTW 1.16× |
| 1,048,576 (2²⁰) | 20,509,162 (5.1) | 26,085,969 (4.0) | 46,909,421 (2.2) | 38,236,009 (2.7) | 97,186,755 (1.1) | 0.79× | **≥ parity** |
| 1,000 (2³·5³) | 3,007 (16.6) | 2,776 (17.9) | 17,293 (2.9) | 14,109 (3.5) | 30,573 (1.6) | 1.08× | lags FFTW 1.08× |
| 1,080 (2³·3³·5) | 3,761 (14.5) | 3,034 (17.9) | 20,671 (2.6) | 13,503 (4.0) | 34,969 (1.6) | 1.24× | lags FFTW 1.24× |
| 1,920 (2⁷·3·5) | 5,493 (19.1) | 4,851 (21.6) | 31,378 (3.3) | 22,836 (4.6) | 61,763 (1.7) | 1.13× | lags FFTW 1.13× |
| 1,009 (prime) | 18,642 (2.7) | 30,404 (1.7) | 81,688 (0.6) | 55,620 (0.9) | 1,427,931 (0.0) | 0.61× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 4,386 (15.3) | 3,667 (18.3) | 21,182 (3.2) | 15,605 (4.3) | 41,360 (1.6) | 1.20× | lags FFTW 1.20× |
| 10,007 (prime) | 335,319 (2.0) | 357,078 (1.9) | 863,583 (0.8) | 740,333 (0.9) | 140,705,713 (0.0) | 0.94× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 394 (13.0) | 336 (15.2) | 6,742 (0.8) | 6,894 (0.7) | 2,656 (1.9) | 1.17× | lags FFTW 1.17× |
| 1,024 (2¹⁰) | 1,285 (19.9) | 1,543 (16.6) | 13,047 (2.0) | 10,271 (2.5) | 13,243 (1.9) | 0.83× | **≥ parity** |
| 4,096 (2¹²) | 7,749 (15.9) | 7,664 (16.0) | 33,531 (3.7) | 31,371 (3.9) | 61,627 (2.0) | 1.01× | **≥ parity** |
| 65,536 (2¹⁶) | 214,672 (12.2) | 186,056 (14.1) | 605,980 (4.3) | 685,924 (3.8) | 1,418,362 (1.8) | 1.15× | lags FFTW 1.15× |
| 1,048,576 (2²⁰) | 8,162,070 (6.4) | 6,905,738 (7.6) | 18,852,583 (2.8) | 20,522,864 (2.6) | 39,473,459 (1.3) | 1.18× | lags FFTW 1.18× |
| 1,000 (2³·5³) | 1,900 (13.1) | 1,867 (13.3) | 14,315 (1.7) | 11,972 (2.1) | 14,159 (1.8) | 1.02× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,099 (13.0) | 1,940 (14.0) | 13,757 (2.0) | 11,142 (2.4) | 16,960 (1.6) | 1.08× | lags FFTW 1.08× |
| 1,920 (2⁷·3·5) | 3,415 (15.3) | 3,393 (15.4) | 18,956 (2.8) | 17,260 (3.0) | 29,734 (1.8) | 1.01× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 408 (12.5) | 535 (9.6) | 0.76× | **≥ parity** |
| 1,024 (2¹⁰) | 1,434 (17.9) | 1,888 (13.6) | 0.76× | **≥ parity** |
| 4,096 (2¹²) | 7,961 (15.4) | 8,550 (14.4) | 0.93× | **≥ parity** |
| 65,536 (2¹⁶) | 241,639 (10.8) | 214,865 (12.2) | 1.12× | lags FFTW 1.12× |
| 1,048,576 (2²⁰) | 9,761,902 (5.4) | 9,515,854 (5.5) | 1.03× | **≥ parity** |
| 1,000 (2³·5³) | 1,936 (12.9) | 2,300 (10.8) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,190 (12.4) | 2,382 (11.4) | 0.92× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,544 (14.8) | 3,950 (13.3) | 0.90× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,874 (13.7) | 30,944 (7.9) | 14,780 (16.6) | 54,993 (4.5) | 45,172 (5.4) | 1.21× | lags FFTW 1.21× |
| 128x128 | 79,621 (14.4) | 136,651 (8.4) | 74,288 (15.4) | 205,282 (5.6) | 181,791 (6.3) | 1.07× | lags FFTW 1.07× |
| 256x256 | 466,432 (11.2) | 671,298 (7.8) | 483,403 (10.8) | 869,027 (6.0) | 763,978 (6.9) | 0.96× | **≥ parity** |
| 512x512 | 1,960,336 (12.0) | 4,475,443 (5.3) | 2,312,245 (10.2) | 4,105,587 (5.7) | 3,286,783 (7.2) | 0.85× | **≥ parity** |
| 1024x1024 | 17,279,668 (6.1) | 19,920,550 (5.3) | 16,312,336 (6.4) | 31,557,805 (3.3) | 23,944,947 (4.4) | 1.06× | lags FFTW 1.06× |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 21,660 | 10,108 | 69,600,359 |
| 1,024 | 79,120 | 36,890 | 126,501,091 |
| 4,096 | 308,310 | 140,688 | 278,283,568 |
| 65,536 | 4,160,738 | 2,255,101 | 3,035,122,544 |
| 1,048,576 | 75,173,462 | 35,825,358 | 7,960,810,425 |
| 1,000 | 62,566 | 35,004 | 144,207,574 |
| 1,080 | 70,216 | 37,934 | 343,759,202 |
| 1,920 | 139,765 | 66,742 | 565,325,800 |
| 1,009 | 179,144 | 43,489 | 135,500,314 |
| 1,296 | 81,556 | 45,245 | 260,129,679 |
| 10,007 | 2,150,647 | 412,321 | 1,526,880,873 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 10/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 256 (2⁸): 1.32×
- complex 1,080 (2³·3³·5): 1.24×
- complex 1,024 (2¹⁰): 1.22×
- 2-D 64x64: 1.21×
- complex 1,296 (2⁴·3⁴): 1.20×
- real 1,048,576 (2²⁰): 1.18×
- real 256 (2⁸): 1.17×
- complex 65,536 (2¹⁶): 1.16×
- real 65,536 (2¹⁶): 1.15×
- complex 1,920 (2⁷·3·5): 1.13×
- complex 1,000 (2³·5³): 1.08×
- real 1,080 (2³·3³·5): 1.08×
- 2-D 128x128: 1.07×
- 2-D 1024x1024: 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
