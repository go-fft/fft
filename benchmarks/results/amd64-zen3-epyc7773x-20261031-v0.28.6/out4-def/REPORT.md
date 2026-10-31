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
| 256 (2⁸) | 303 (33.8) | 272 (37.6) | 6,801 (1.5) | 5,234 (2.0) | 5,411 (1.9) | 1.11× | lags FFTW 1.11× |
| 1,024 (2¹⁰) | 1,718 (29.8) | 1,580 (32.4) | 12,889 (4.0) | 9,423 (5.4) | 28,410 (1.8) | 1.09× | lags FFTW 1.09× |
| 4,096 (2¹²) | 9,582 (25.6) | 9,998 (24.6) | 45,866 (5.4) | 32,914 (7.5) | 130,981 (1.9) | 0.96× | **≥ parity** |
| 65,536 (2¹⁶) | 261,038 (20.1) | 292,444 (17.9) | 2,015,085 (2.6) | 914,855 (5.7) | 2,851,000 (1.8) | 0.89× | **≥ parity** |
| 1,048,576 (2²⁰) | 5,329,186 (19.7) | 11,160,720 (9.4) | 21,525,191 (4.9) | 13,795,848 (7.6) | 58,837,469 (1.8) | 0.48× | **≥ parity** |
| 1,000 (2³·5³) | 1,983 (25.1) | 2,080 (24.0) | 13,737 (3.6) | 9,800 (5.1) | 29,976 (1.7) | 0.95× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,454 (22.2) | 2,247 (24.2) | 14,391 (3.8) | 11,159 (4.9) | 35,870 (1.5) | 1.09× | lags FFTW 1.09× |
| 1,920 (2⁷·3·5) | 3,812 (27.5) | 3,745 (28.0) | 21,683 (4.8) | 15,968 (6.6) | 62,254 (1.7) | 1.02× | **≥ parity** |
| 1,009 (prime) | 13,107 (3.8) | 20,345 (2.5) | 60,562 (0.8) | 37,932 (1.3) | 1,140,467 (0.0) | 0.64× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 2,759 (24.3) | 2,936 (22.8) | 16,112 (4.2) | 12,291 (5.5) | 42,312 (1.6) | 0.94× | **≥ parity** |
| 10,007 (prime) | 170,533 (3.9) | 216,730 (3.1) | 636,310 (1.0) | 464,734 (1.4) | 120,028,351 (0.0) | 0.79× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 243 (21.1) | 224 (22.8) | 5,989 (0.9) | 5,254 (1.0) | 2,608 (2.0) | 1.08× | lags FFTW 1.08× |
| 1,024 (2¹⁰) | 1,027 (24.9) | 959 (26.7) | 9,522 (2.7) | 8,165 (3.1) | 14,065 (1.8) | 1.07× | lags FFTW 1.07× |
| 4,096 (2¹²) | 5,530 (22.2) | 5,144 (23.9) | 25,304 (4.9) | 22,135 (5.6) | 60,698 (2.0) | 1.08× | lags FFTW 1.08× |
| 65,536 (2¹⁶) | 134,202 (19.5) | 159,074 (16.5) | 399,269 (6.6) | 488,702 (5.4) | 1,303,182 (2.0) | 0.84× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,775,841 (18.9) | 3,724,372 (14.1) | 8,412,885 (6.2) | 10,088,845 (5.2) | 26,780,645 (2.0) | 0.75× | **≥ parity** |
| 1,000 (2³·5³) | 1,242 (20.1) | 1,232 (20.2) | 10,265 (2.4) | 8,543 (2.9) | 13,385 (1.9) | 1.01× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,368 (19.9) | 1,386 (19.6) | 10,838 (2.5) | 8,946 (3.0) | 15,876 (1.7) | 0.99× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,448 (21.4) | 2,297 (22.8) | 14,332 (3.7) | 12,030 (4.4) | 27,985 (1.9) | 1.07× | lags FFTW 1.07× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 258 (19.8) | 348 (14.7) | 0.74× | **≥ parity** |
| 1,024 (2¹⁰) | 1,048 (24.4) | 1,128 (22.7) | 0.93× | **≥ parity** |
| 4,096 (2¹²) | 5,811 (21.1) | 5,773 (21.3) | 1.01× | **≥ parity** |
| 65,536 (2¹⁶) | 134,508 (19.5) | 176,059 (14.9) | 0.76× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,898,234 (18.1) | 4,077,218 (12.9) | 0.71× | **≥ parity** |
| 1,000 (2³·5³) | 1,206 (20.7) | 1,456 (17.1) | 0.83× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,440 (18.9) | 1,536 (17.7) | 0.94× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,492 (21.0) | 2,663 (19.7) | 0.94× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 9,686 (25.4) | 13,535 (18.2) | 10,873 (22.6) | 47,733 (5.1) | 32,842 (7.5) | 0.89× | **≥ parity** |
| 128x128 | 49,946 (23.0) | 60,430 (19.0) | 61,152 (18.8) | 147,953 (7.8) | 127,868 (9.0) | 0.82× | **≥ parity** |
| 256x256 | 266,540 (19.7) | 312,223 (16.8) | 278,051 (18.9) | 698,328 (7.5) | 451,421 (11.6) | 0.96× | **≥ parity** |
| 512x512 | 1,238,591 (19.0) | 1,464,171 (16.1) | 1,260,328 (18.7) | 2,937,426 (8.0) | 1,883,522 (12.5) | 0.98× | **≥ parity** |
| 1024x1024 | 5,943,233 (17.6) | 7,042,078 (14.9) | 6,468,020 (16.2) | 13,277,651 (7.9) | 9,230,939 (11.4) | 0.92× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 11,305 | 7,250 | 71,381,933 |
| 1,024 | 42,592 | 27,666 | 133,076,240 |
| 4,096 | 164,735 | 104,082 | 238,687,359 |
| 65,536 | 2,334,970 | 1,601,108 | 2,136,990,671 |
| 1,048,576 | 41,999,085 | 24,215,933 | 3,942,552,031 |
| 1,000 | 40,875 | 29,730 | 157,366,071 |
| 1,080 | 43,637 | 29,661 | 365,631,927 |
| 1,920 | 72,796 | 51,549 | 568,252,538 |
| 1,009 | 79,390 | 31,721 | 140,509,344 |
| 1,296 | 53,290 | 34,912 | 263,933,580 |
| 10,007 | 978,460 | 303,642 | 1,215,481,403 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 17/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- complex 256 (2⁸): 1.11×
- complex 1,080 (2³·3³·5): 1.09×
- complex 1,024 (2¹⁰): 1.09×
- real 256 (2⁸): 1.08×
- real 4,096 (2¹²): 1.08×
- real 1,024 (2¹⁰): 1.07×
- real 1,920 (2⁷·3·5): 1.07×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
