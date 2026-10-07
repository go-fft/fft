# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3), cfarm420, one pinned core.
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
| 256 (2⁸) | 353 (29.0) | 274 (37.4) | 6,767 (1.5) | 5,259 (1.9) | 5,466 (1.9) | 1.29× | lags FFTW 1.29× |
| 1,024 (2¹⁰) | 1,668 (30.7) | 1,532 (33.4) | 12,969 (3.9) | 9,492 (5.4) | 28,438 (1.8) | 1.09× | lags FFTW 1.09× |
| 4,096 (2¹²) | 9,058 (27.1) | 10,579 (23.2) | 45,616 (5.4) | 32,912 (7.5) | 132,143 (1.9) | 0.86× | **≥ parity** |
| 65,536 (2¹⁶) | 246,100 (21.3) | 297,124 (17.6) | 1,878,848 (2.8) | 907,072 (5.8) | 2,866,461 (1.8) | 0.83× | **≥ parity** |
| 1,048,576 (2²⁰) | 5,326,835 (19.7) | 11,608,220 (9.0) | 39,233,979 (2.7) | 13,879,531 (7.6) | 58,230,494 (1.8) | 0.46× | **≥ parity** |
| 1,000 (2³·5³) | 2,354 (21.2) | 2,023 (24.6) | 13,642 (3.7) | 10,071 (4.9) | 30,181 (1.7) | 1.16× | lags FFTW 1.16× |
| 1,080 (2³·3³·5) | 2,788 (19.5) | 2,278 (23.9) | 14,157 (3.8) | 10,607 (5.1) | 35,265 (1.5) | 1.22× | lags FFTW 1.22× |
| 1,920 (2⁷·3·5) | 4,437 (23.6) | 3,893 (26.9) | 21,728 (4.8) | 16,219 (6.5) | 61,438 (1.7) | 1.14× | lags FFTW 1.14× |
| 1,009 (prime) | 13,072 (3.9) | 20,118 (2.5) | 60,615 (0.8) | 38,477 (1.3) | 1,132,939 (0.0) | 0.65× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,560 (18.8) | 3,176 (21.1) | 16,403 (4.1) | 12,451 (5.4) | 42,637 (1.6) | 1.12× | lags FFTW 1.12× |
| 10,007 (prime) | 207,632 (3.2) | 203,902 (3.3) | 641,607 (1.0) | 464,785 (1.4) | 118,136,580 (0.0) | 1.02× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 309 (16.6) | 224 (22.9) | 6,015 (0.9) | 5,272 (1.0) | 2,554 (2.0) | 1.38× | lags FFTW 1.38× |
| 1,024 (2¹⁰) | 1,127 (22.7) | 986 (26.0) | 9,535 (2.7) | 8,281 (3.1) | 12,643 (2.0) | 1.14× | lags FFTW 1.14× |
| 4,096 (2¹²) | 5,904 (20.8) | 5,162 (23.8) | 24,959 (4.9) | 22,022 (5.6) | 59,265 (2.1) | 1.14× | lags FFTW 1.14× |
| 65,536 (2¹⁶) | 140,754 (18.6) | 158,248 (16.6) | 394,616 (6.6) | 488,461 (5.4) | 1,312,333 (2.0) | 0.89× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,913,737 (18.0) | 3,902,944 (13.4) | 8,261,850 (6.3) | 10,457,187 (5.0) | 26,835,183 (2.0) | 0.75× | **≥ parity** |
| 1,000 (2³·5³) | 1,502 (16.6) | 1,244 (20.0) | 10,283 (2.4) | 8,643 (2.9) | 13,302 (1.9) | 1.21× | lags FFTW 1.21× |
| 1,080 (2³·3³·5) | 1,737 (15.7) | 1,360 (20.0) | 10,851 (2.5) | 9,009 (3.0) | 15,742 (1.7) | 1.28× | lags FFTW 1.28× |
| 1,920 (2⁷·3·5) | 2,807 (18.7) | 2,589 (20.2) | 14,356 (3.6) | 12,263 (4.3) | 28,858 (1.8) | 1.08× | lags FFTW 1.08× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 332 (15.4) | 302 (17.0) | 1.10× | lags FFTW 1.10× |
| 1,024 (2¹⁰) | 1,146 (22.3) | 1,153 (22.2) | 0.99× | **≥ parity** |
| 4,096 (2¹²) | 5,947 (20.7) | 5,712 (21.5) | 1.04× | **≥ parity** |
| 65,536 (2¹⁶) | 138,045 (19.0) | 167,961 (15.6) | 0.82× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,868,843 (18.3) | 4,136,626 (12.7) | 0.69× | **≥ parity** |
| 1,000 (2³·5³) | 1,527 (16.3) | 1,416 (17.6) | 1.08× | lags FFTW 1.08× |
| 1,080 (2³·3³·5) | 1,731 (15.7) | 1,635 (16.6) | 1.06× | lags FFTW 1.06× |
| 1,920 (2⁷·3·5) | 2,839 (18.4) | 2,813 (18.6) | 1.01× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 10,956 (22.4) | 15,720 (15.6) | 10,695 (23.0) | 47,897 (5.1) | 32,868 (7.5) | 1.02× | **≥ parity** |
| 128x128 | 53,130 (21.6) | 72,139 (15.9) | 64,500 (17.8) | 148,655 (7.7) | 127,042 (9.0) | 0.82× | **≥ parity** |
| 256x256 | 280,204 (18.7) | 345,241 (15.2) | 281,311 (18.6) | 671,329 (7.8) | 451,635 (11.6) | 1.00× | **≥ parity** |
| 512x512 | 1,250,639 (18.9) | 1,501,034 (15.7) | 1,340,569 (17.6) | 2,753,044 (8.6) | 1,896,124 (12.4) | 0.93× | **≥ parity** |
| 1024x1024 | 6,181,986 (17.0) | 7,448,111 (14.1) | 6,554,184 (16.0) | 13,568,382 (7.7) | 9,529,820 (11.0) | 0.94× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 11,056 | 7,358 | 70,669,543 |
| 1,024 | 40,601 | 28,117 | 130,396,769 |
| 4,096 | 159,799 | 104,624 | 240,063,309 |
| 65,536 | 2,526,645 | 1,606,783 | 2,187,000,164 |
| 1,048,576 | 55,398,385 | 25,798,168 | 4,256,367,135 |
| 1,000 | 34,387 | 27,132 | 151,107,319 |
| 1,080 | 38,202 | 29,771 | 362,881,115 |
| 1,920 | 67,997 | 68,828 | 566,441,337 |
| 1,009 | 74,475 | 31,226 | 141,127,804 |
| 1,296 | 45,368 | 34,600 | 265,174,250 |
| 10,007 | 1,001,339 | 308,332 | 1,368,157,052 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 12/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.38×
- complex 256 (2⁸): 1.29×
- real 1,080 (2³·3³·5): 1.28×
- complex 1,080 (2³·3³·5): 1.22×
- real 1,000 (2³·5³): 1.21×
- complex 1,000 (2³·5³): 1.16×
- real 4,096 (2¹²): 1.14×
- real 1,024 (2¹⁰): 1.14×
- complex 1,920 (2⁷·3·5): 1.14×
- complex 1,296 (2⁴·3⁴): 1.12×
- complex 1,024 (2¹⁰): 1.09×
- real 1,920 (2⁷·3·5): 1.08×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
