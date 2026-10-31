# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3, cfarm420), one pinned core.
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
| 256 (2⁸) | 306 (33.4) | 277 (37.0) | 6,627 (1.5) | 5,171 (2.0) | 5,542 (1.8) | 1.11× | lags FFTW 1.11× |
| 1,024 (2¹⁰) | 1,711 (29.9) | 1,552 (33.0) | 12,703 (4.0) | 9,500 (5.4) | 28,532 (1.8) | 1.10× | lags FFTW 1.10× |
| 4,096 (2¹²) | 9,123 (26.9) | 10,070 (24.4) | 43,657 (5.6) | 33,672 (7.3) | 130,710 (1.9) | 0.91× | **≥ parity** |
| 65,536 (2¹⁶) | 245,927 (21.3) | 294,205 (17.8) | 1,906,030 (2.8) | 910,086 (5.8) | 2,820,632 (1.9) | 0.84× | **≥ parity** |
| 1,048,576 (2²⁰) | 5,382,511 (19.5) | 11,149,960 (9.4) | 23,092,242 (4.5) | 13,705,836 (7.7) | 58,126,414 (1.8) | 0.48× | **≥ parity** |
| 1,000 (2³·5³) | 1,961 (25.4) | 2,075 (24.0) | 13,423 (3.7) | 10,144 (4.9) | 30,053 (1.7) | 0.95× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,471 (22.0) | 2,363 (23.0) | 14,226 (3.8) | 10,802 (5.0) | 35,937 (1.5) | 1.05× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,801 (27.5) | 3,931 (26.6) | 21,678 (4.8) | 16,280 (6.4) | 61,017 (1.7) | 0.97× | **≥ parity** |
| 1,009 (prime) | 12,687 (4.0) | 20,843 (2.4) | 60,987 (0.8) | 38,792 (1.3) | 1,131,646 (0.0) | 0.61× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 2,737 (24.5) | 2,996 (22.4) | 16,186 (4.1) | 12,750 (5.3) | 41,482 (1.6) | 0.91× | **≥ parity** |
| 10,007 (prime) | 165,988 (4.0) | 201,933 (3.3) | 635,511 (1.0) | 466,738 (1.4) | 119,113,602 (0.0) | 0.82× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 236 (21.7) | 222 (23.1) | 5,937 (0.9) | 5,251 (1.0) | 2,514 (2.0) | 1.06× | lags FFTW 1.06× |
| 1,024 (2¹⁰) | 1,002 (25.5) | 1,018 (25.1) | 9,397 (2.7) | 8,307 (3.1) | 12,763 (2.0) | 0.98× | **≥ parity** |
| 4,096 (2¹²) | 5,487 (22.4) | 5,282 (23.3) | 24,529 (5.0) | 22,348 (5.5) | 59,295 (2.1) | 1.04× | **≥ parity** |
| 65,536 (2¹⁶) | 134,036 (19.6) | 153,498 (17.1) | 399,453 (6.6) | 489,389 (5.4) | 1,297,862 (2.0) | 0.87× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,727,313 (19.2) | 3,629,360 (14.4) | 8,563,772 (6.1) | 9,879,158 (5.3) | 26,522,593 (2.0) | 0.75× | **≥ parity** |
| 1,000 (2³·5³) | 1,140 (21.9) | 1,220 (20.4) | 10,054 (2.5) | 8,764 (2.8) | 13,804 (1.8) | 0.93× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,366 (19.9) | 1,408 (19.3) | 10,699 (2.5) | 8,896 (3.1) | 15,817 (1.7) | 0.97× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,440 (21.5) | 2,412 (21.7) | 14,084 (3.7) | 11,914 (4.4) | 27,739 (1.9) | 1.01× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 258 (19.9) | 346 (14.8) | 0.75× | **≥ parity** |
| 1,024 (2¹⁰) | 1,030 (24.9) | 1,152 (22.2) | 0.89× | **≥ parity** |
| 4,096 (2¹²) | 5,629 (21.8) | 5,714 (21.5) | 0.99× | **≥ parity** |
| 65,536 (2¹⁶) | 134,296 (19.5) | 164,753 (15.9) | 0.82× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,825,675 (18.6) | 4,091,724 (12.8) | 0.69× | **≥ parity** |
| 1,000 (2³·5³) | 1,196 (20.8) | 1,631 (15.3) | 0.73× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,384 (19.7) | 1,624 (16.8) | 0.85× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,524 (20.7) | 2,861 (18.3) | 0.88× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 9,550 (25.7) | 12,853 (19.1) | 10,989 (22.4) | 45,830 (5.4) | 32,743 (7.5) | 0.87× | **≥ parity** |
| 128x128 | 50,255 (22.8) | 59,795 (19.2) | 64,796 (17.7) | 140,590 (8.2) | 126,549 (9.1) | 0.78× | **≥ parity** |
| 256x256 | 263,102 (19.9) | 309,010 (17.0) | 282,426 (18.6) | 688,330 (7.6) | 456,064 (11.5) | 0.93× | **≥ parity** |
| 512x512 | 1,205,445 (19.6) | 1,392,395 (16.9) | 1,329,046 (17.8) | 3,027,687 (7.8) | 1,905,403 (12.4) | 0.91× | **≥ parity** |
| 1024x1024 | 5,888,338 (17.8) | 7,022,978 (14.9) | 6,307,910 (16.6) | 14,841,542 (7.1) | 9,505,706 (11.0) | 0.93× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 11,630 | 7,411 | 72,254,104 |
| 1,024 | 41,748 | 27,381 | 135,629,592 |
| 4,096 | 160,263 | 116,871 | 241,442,190 |
| 65,536 | 2,388,869 | 1,657,726 | 2,170,510,966 |
| 1,048,576 | 42,628,344 | 25,224,906 | 3,923,768,944 |
| 1,000 | 41,709 | 27,792 | 151,727,308 |
| 1,080 | 44,206 | 29,980 | 366,835,916 |
| 1,920 | 79,071 | 52,398 | 566,926,918 |
| 1,009 | 74,750 | 33,671 | 143,868,395 |
| 1,296 | 54,955 | 36,122 | 264,805,101 |
| 10,007 | 990,410 | 312,053 | 1,220,043,895 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 21/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 256 (2⁸): 1.11×
- complex 1,024 (2¹⁰): 1.10×
- real 256 (2⁸): 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
