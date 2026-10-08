# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3, cfarm420), core 40.
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
| 256 (2⁸) | 350 (29.2) | 289 (35.4) | 6,831 (1.5) | 5,260 (1.9) | 6,689 (1.5) | 1.21× | lags FFTW 1.21× |
| 1,024 (2¹⁰) | 1,720 (29.8) | 2,130 (24.0) | 13,023 (3.9) | 9,454 (5.4) | 29,166 (1.8) | 0.81× | **≥ parity** |
| 4,096 (2¹²) | 9,466 (26.0) | 10,761 (22.8) | 45,329 (5.4) | 32,744 (7.5) | 133,165 (1.8) | 0.88× | **≥ parity** |
| 65,536 (2¹⁶) | 269,903 (19.4) | 301,138 (17.4) | 1,934,129 (2.7) | 963,944 (5.4) | 2,921,350 (1.8) | 0.90× | **≥ parity** |
| 1,048,576 (2²⁰) | 5,635,022 (18.6) | 11,694,796 (9.0) | 25,459,376 (4.1) | 14,086,760 (7.4) | 60,879,220 (1.7) | 0.48× | **≥ parity** |
| 1,000 (2³·5³) | 2,135 (23.3) | 2,032 (24.5) | 13,855 (3.6) | 10,042 (5.0) | 30,144 (1.7) | 1.05× | lags FFTW 1.05× |
| 1,080 (2³·3³·5) | 2,633 (20.7) | 2,227 (24.4) | 14,407 (3.8) | 10,624 (5.1) | 35,793 (1.5) | 1.18× | lags FFTW 1.18× |
| 1,920 (2⁷·3·5) | 3,943 (26.6) | 3,885 (27.0) | 22,069 (4.7) | 16,343 (6.4) | 62,645 (1.7) | 1.01× | **≥ parity** |
| 1,009 (prime) | 12,692 (4.0) | 20,389 (2.5) | 60,661 (0.8) | 38,400 (1.3) | 1,175,343 (0.0) | 0.62× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,013 (22.2) | 3,003 (22.3) | 16,400 (4.1) | 12,708 (5.3) | 42,760 (1.6) | 1.00× | **≥ parity** |
| 10,007 (prime) | 226,795 (2.9) | 200,762 (3.3) | 671,948 (1.0) | 467,808 (1.4) | 117,883,249 (0.0) | 1.13× | lags FFTW 1.13× |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 314 (16.3) | 229 (22.3) | 6,028 (0.8) | 5,206 (1.0) | 2,540 (2.0) | 1.37× | lags FFTW 1.37× |
| 1,024 (2¹⁰) | 1,146 (22.3) | 961 (26.6) | 9,532 (2.7) | 7,981 (3.2) | 12,999 (2.0) | 1.19× | lags FFTW 1.19× |
| 4,096 (2¹²) | 5,907 (20.8) | 5,126 (24.0) | 25,114 (4.9) | 24,379 (5.0) | 61,466 (2.0) | 1.15× | lags FFTW 1.15× |
| 65,536 (2¹⁶) | 144,285 (18.2) | 151,709 (17.3) | 461,914 (5.7) | 515,202 (5.1) | 1,355,068 (1.9) | 0.95× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,977,234 (17.6) | 3,466,754 (15.1) | 8,446,447 (6.2) | 10,118,892 (5.2) | 26,321,692 (2.0) | 0.86× | **≥ parity** |
| 1,000 (2³·5³) | 1,411 (17.7) | 1,325 (18.8) | 10,259 (2.4) | 8,678 (2.9) | 13,346 (1.9) | 1.07× | lags FFTW 1.07× |
| 1,080 (2³·3³·5) | 1,596 (17.0) | 1,376 (19.8) | 10,813 (2.5) | 8,864 (3.1) | 15,772 (1.7) | 1.16× | lags FFTW 1.16× |
| 1,920 (2⁷·3·5) | 2,577 (20.3) | 2,378 (22.0) | 14,900 (3.5) | 12,157 (4.3) | 29,164 (1.8) | 1.08× | lags FFTW 1.08× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 336 (15.3) | 297 (17.2) | 1.13× | lags FFTW 1.13× |
| 1,024 (2¹⁰) | 1,212 (21.1) | 1,202 (21.3) | 1.01× | **≥ parity** |
| 4,096 (2¹²) | 6,079 (20.2) | 5,788 (21.2) | 1.05× | lags FFTW 1.05× |
| 65,536 (2¹⁶) | 141,578 (18.5) | 167,611 (15.6) | 0.84× | **≥ parity** |
| 1,048,576 (2²⁰) | 3,003,139 (17.5) | 4,012,523 (13.1) | 0.75× | **≥ parity** |
| 1,000 (2³·5³) | 1,456 (17.1) | 1,560 (16.0) | 0.93× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,639 (16.6) | 1,617 (16.8) | 1.01× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,637 (19.9) | 2,619 (20.0) | 1.01× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 10,878 (22.6) | 15,698 (15.7) | 10,966 (22.4) | 48,702 (5.0) | 35,343 (7.0) | 0.99× | **≥ parity** |
| 128x128 | 53,465 (21.5) | 67,344 (17.0) | 61,755 (18.6) | 159,145 (7.2) | 132,289 (8.7) | 0.87× | **≥ parity** |
| 256x256 | 275,909 (19.0) | 324,240 (16.2) | 282,825 (18.5) | 635,506 (8.2) | 454,100 (11.5) | 0.98× | **≥ parity** |
| 512x512 | 1,254,572 (18.8) | 1,544,031 (15.3) | 1,448,905 (16.3) | 2,978,305 (7.9) | 1,992,350 (11.8) | 0.87× | **≥ parity** |
| 1024x1024 | 6,117,812 (17.1) | 7,118,219 (14.7) | 6,909,531 (15.2) | 13,518,235 (7.8) | 9,254,347 (11.3) | 0.89× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 74,483,524 |
| 1,024 | — | — | 140,603,024 |
| 4,096 | — | — | 242,749,820 |
| 65,536 | — | — | 2,335,603,952 |
| 1,048,576 | — | — | 4,032,990,053 |
| 1,000 | — | — | 156,544,521 |
| 1,080 | — | — | 363,793,875 |
| 1,920 | — | — | 574,103,311 |
| 1,009 | — | — | 147,959,257 |
| 1,296 | — | — | 268,487,882 |
| 10,007 | — | — | 1,213,959,602 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 14/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.37×
- complex 256 (2⁸): 1.21×
- real 1,024 (2¹⁰): 1.19×
- complex 1,080 (2³·3³·5): 1.18×
- real 1,080 (2³·3³·5): 1.16×
- real 4,096 (2¹²): 1.15×
- complex 10,007 (prime): 1.13×
- real 1,920 (2⁷·3·5): 1.08×
- real 1,000 (2³·5³): 1.07×
- complex 1,000 (2³·5³): 1.05×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
