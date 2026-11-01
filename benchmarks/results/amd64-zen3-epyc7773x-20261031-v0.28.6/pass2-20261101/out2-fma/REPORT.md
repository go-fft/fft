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
| 256 (2⁸) | 295 (34.7) | 274 (37.4) | 6,690 (1.5) | 5,180 (2.0) | 5,463 (1.9) | 1.08× | lags FFTW 1.08× |
| 1,024 (2¹⁰) | 1,686 (30.4) | 1,521 (33.7) | 13,246 (3.9) | 9,537 (5.4) | 28,359 (1.8) | 1.11× | lags FFTW 1.11× |
| 4,096 (2¹²) | 9,219 (26.7) | 10,159 (24.2) | 42,622 (5.8) | 32,487 (7.6) | 129,997 (1.9) | 0.91× | **≥ parity** |
| 65,536 (2¹⁶) | 253,473 (20.7) | 291,528 (18.0) | 1,885,373 (2.8) | 948,381 (5.5) | 2,835,444 (1.8) | 0.87× | **≥ parity** |
| 1,048,576 (2²⁰) | 5,718,705 (18.3) | 11,263,504 (9.3) | 21,665,179 (4.8) | 14,380,158 (7.3) | 57,691,751 (1.8) | 0.51× | **≥ parity** |
| 1,000 (2³·5³) | 1,912 (26.1) | 2,049 (24.3) | 13,590 (3.7) | 9,968 (5.0) | 30,254 (1.6) | 0.93× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,496 (21.8) | 2,280 (23.9) | 14,150 (3.8) | 10,478 (5.2) | 36,023 (1.5) | 1.09× | lags FFTW 1.09× |
| 1,920 (2⁷·3·5) | 3,838 (27.3) | 3,959 (26.4) | 21,857 (4.8) | 16,318 (6.4) | 62,371 (1.7) | 0.97× | **≥ parity** |
| 1,009 (prime) | 12,314 (4.1) | 20,616 (2.4) | 60,868 (0.8) | 38,222 (1.3) | 1,142,992 (0.0) | 0.60× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 2,711 (24.7) | 2,930 (22.9) | 16,119 (4.2) | 12,185 (5.5) | 41,424 (1.6) | 0.93× | **≥ parity** |
| 10,007 (prime) | 162,991 (4.1) | 203,866 (3.3) | 641,382 (1.0) | 468,054 (1.4) | 117,844,232 (0.0) | 0.80× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 235 (21.8) | 226 (22.6) | 5,961 (0.9) | 5,245 (1.0) | 2,552 (2.0) | 1.04× | **≥ parity** |
| 1,024 (2¹⁰) | 966 (26.5) | 980 (26.1) | 9,401 (2.7) | 8,157 (3.1) | 12,891 (2.0) | 0.99× | **≥ parity** |
| 4,096 (2¹²) | 5,399 (22.8) | 5,154 (23.8) | 25,285 (4.9) | 21,823 (5.6) | 59,487 (2.1) | 1.05× | **≥ parity** |
| 65,536 (2¹⁶) | 130,855 (20.0) | 151,568 (17.3) | 399,680 (6.6) | 490,611 (5.3) | 1,323,708 (2.0) | 0.86× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,698,300 (19.4) | 3,623,583 (14.5) | 8,246,231 (6.4) | 10,161,069 (5.2) | 26,930,242 (1.9) | 0.74× | **≥ parity** |
| 1,000 (2³·5³) | 1,126 (22.1) | 1,230 (20.2) | 10,280 (2.4) | 8,706 (2.9) | 13,655 (1.8) | 0.92× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,348 (20.2) | 1,341 (20.3) | 10,938 (2.5) | 8,994 (3.0) | 15,858 (1.7) | 1.01× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,407 (21.8) | 2,392 (21.9) | 14,437 (3.6) | 11,978 (4.4) | 28,439 (1.8) | 1.01× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 253 (20.2) | 295 (17.4) | 0.86× | **≥ parity** |
| 1,024 (2¹⁰) | 1,006 (25.4) | 1,206 (21.2) | 0.83× | **≥ parity** |
| 4,096 (2¹²) | 5,553 (22.1) | 5,867 (20.9) | 0.95× | **≥ parity** |
| 65,536 (2¹⁶) | 133,953 (19.6) | 165,880 (15.8) | 0.81× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,812,814 (18.6) | 4,024,075 (13.0) | 0.70× | **≥ parity** |
| 1,000 (2³·5³) | 1,187 (21.0) | 1,469 (17.0) | 0.81× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,415 (19.2) | 1,538 (17.7) | 0.92× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,498 (21.0) | 2,659 (19.7) | 0.94× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 9,136 (26.9) | 12,663 (19.4) | 10,457 (23.5) | 47,630 (5.2) | 33,186 (7.4) | 0.87× | **≥ parity** |
| 128x128 | 50,081 (22.9) | 59,837 (19.2) | 61,649 (18.6) | 146,648 (7.8) | 126,034 (9.1) | 0.81× | **≥ parity** |
| 256x256 | 259,458 (20.2) | 317,283 (16.5) | 279,135 (18.8) | 670,715 (7.8) | 451,860 (11.6) | 0.93× | **≥ parity** |
| 512x512 | 1,188,548 (19.9) | 1,380,711 (17.1) | 1,341,147 (17.6) | 3,163,403 (7.5) | 1,941,841 (12.1) | 0.89× | **≥ parity** |
| 1024x1024 | 5,698,829 (18.4) | 6,784,013 (15.5) | 5,973,242 (17.6) | 14,616,905 (7.2) | 9,136,885 (11.5) | 0.95× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 11,679 | 7,235 | 71,475,203 |
| 1,024 | 42,399 | 27,116 | 131,571,190 |
| 4,096 | 186,160 | 109,095 | 239,546,119 |
| 65,536 | 2,331,058 | 1,649,223 | 2,154,208,189 |
| 1,048,576 | 48,537,242 | 28,764,020 | 3,994,363,586 |
| 1,000 | 41,931 | 26,296 | 152,203,209 |
| 1,080 | 44,512 | 28,885 | 362,768,215 |
| 1,920 | 74,503 | 49,800 | 576,453,063 |
| 1,009 | 74,099 | 32,390 | 141,563,594 |
| 1,296 | 54,522 | 34,088 | 264,854,270 |
| 10,007 | 981,465 | 302,400 | 1,218,493,464 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 21/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 1,024 (2¹⁰): 1.11×
- complex 1,080 (2³·3³·5): 1.09×
- complex 256 (2⁸): 1.08×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
