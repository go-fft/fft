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
| 256 (2⁸) | 288 (35.6) | 279 (36.7) | 6,614 (1.5) | 5,299 (1.9) | 5,455 (1.9) | 1.03× | **≥ parity** |
| 1,024 (2¹⁰) | 1,659 (30.9) | 1,561 (32.8) | 12,789 (4.0) | 9,460 (5.4) | 28,692 (1.8) | 1.06× | lags FFTW 1.06× |
| 4,096 (2¹²) | 9,149 (26.9) | 10,846 (22.7) | 42,959 (5.7) | 33,631 (7.3) | 130,545 (1.9) | 0.84× | **≥ parity** |
| 65,536 (2¹⁶) | 252,507 (20.8) | 301,143 (17.4) | 1,973,531 (2.7) | 904,237 (5.8) | 2,896,146 (1.8) | 0.84× | **≥ parity** |
| 1,048,576 (2²⁰) | 5,460,497 (19.2) | 11,174,924 (9.4) | 23,418,148 (4.5) | 13,900,009 (7.5) | 58,945,659 (1.8) | 0.49× | **≥ parity** |
| 1,000 (2³·5³) | 1,945 (25.6) | 2,027 (24.6) | 13,667 (3.6) | 9,864 (5.1) | 30,244 (1.6) | 0.96× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,499 (21.8) | 2,508 (21.7) | 14,683 (3.7) | 10,769 (5.1) | 35,566 (1.5) | 1.00× | **≥ parity** |
| 1,920 (2⁷·3·5) | 3,810 (27.5) | 3,986 (26.3) | 21,772 (4.8) | 16,233 (6.5) | 61,718 (1.7) | 0.96× | **≥ parity** |
| 1,009 (prime) | 12,505 (4.0) | 20,264 (2.5) | 63,142 (0.8) | 39,802 (1.3) | 1,133,326 (0.0) | 0.62× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 2,735 (24.5) | 3,298 (20.3) | 16,381 (4.1) | 12,489 (5.4) | 41,927 (1.6) | 0.83× | **≥ parity** |
| 10,007 (prime) | 162,346 (4.1) | 210,023 (3.2) | 646,713 (1.0) | 482,641 (1.4) | 117,790,916 (0.0) | 0.77× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 233 (22.0) | 226 (22.7) | 6,080 (0.8) | 5,470 (0.9) | 2,541 (2.0) | 1.03× | **≥ parity** |
| 1,024 (2¹⁰) | 953 (26.9) | 978 (26.2) | 9,700 (2.6) | 9,993 (2.6) | 12,770 (2.0) | 0.97× | **≥ parity** |
| 4,096 (2¹²) | 5,533 (22.2) | 5,146 (23.9) | 25,059 (4.9) | 21,996 (5.6) | 62,401 (2.0) | 1.08× | lags FFTW 1.08× |
| 65,536 (2¹⁶) | 132,168 (19.8) | 151,105 (17.3) | 427,315 (6.1) | 493,697 (5.3) | 1,336,150 (2.0) | 0.87× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,795,704 (18.8) | 3,741,691 (14.0) | 8,226,613 (6.4) | 10,008,217 (5.2) | 26,432,032 (2.0) | 0.75× | **≥ parity** |
| 1,000 (2³·5³) | 1,130 (22.0) | 1,231 (20.2) | 10,324 (2.4) | 8,964 (2.8) | 13,292 (1.9) | 0.92× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,331 (20.4) | 1,683 (16.2) | 10,970 (2.5) | 9,283 (2.9) | 15,741 (1.7) | 0.79× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,413 (21.7) | 2,362 (22.2) | 14,304 (3.7) | 12,053 (4.3) | 28,098 (1.9) | 1.02× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 248 (20.6) | 296 (17.3) | 0.84× | **≥ parity** |
| 1,024 (2¹⁰) | 1,004 (25.5) | 1,225 (20.9) | 0.82× | **≥ parity** |
| 4,096 (2¹²) | 5,522 (22.3) | 5,650 (21.7) | 0.98× | **≥ parity** |
| 65,536 (2¹⁶) | 133,783 (19.6) | 165,018 (15.9) | 0.81× | **≥ parity** |
| 1,048,576 (2²⁰) | 2,796,924 (18.7) | 4,092,560 (12.8) | 0.68× | **≥ parity** |
| 1,000 (2³·5³) | 1,217 (20.5) | 1,437 (17.3) | 0.85× | **≥ parity** |
| 1,080 (2³·3³·5) | 1,377 (19.8) | 1,581 (17.2) | 0.87× | **≥ parity** |
| 1,920 (2⁷·3·5) | 2,567 (20.4) | 2,598 (20.1) | 0.99× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 9,214 (26.7) | 12,646 (19.4) | 10,837 (22.7) | 47,659 (5.2) | 33,270 (7.4) | 0.85× | **≥ parity** |
| 128x128 | 50,503 (22.7) | 65,625 (17.5) | 62,030 (18.5) | 147,236 (7.8) | 126,260 (9.1) | 0.81× | **≥ parity** |
| 256x256 | 260,383 (20.1) | 324,333 (16.2) | 286,638 (18.3) | 709,204 (7.4) | 455,881 (11.5) | 0.91× | **≥ parity** |
| 512x512 | 1,214,730 (19.4) | 1,501,561 (15.7) | 1,318,840 (17.9) | 2,729,149 (8.6) | 1,859,954 (12.7) | 0.92× | **≥ parity** |
| 1024x1024 | 6,035,738 (17.4) | 8,045,411 (13.0) | 6,496,390 (16.1) | 12,856,061 (8.2) | 9,353,522 (11.2) | 0.93× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 11,572 | 7,599 | 71,252,292 |
| 1,024 | 40,890 | 27,283 | 132,933,891 |
| 4,096 | 165,775 | 107,520 | 239,122,499 |
| 65,536 | 2,394,800 | 1,617,194 | 2,163,842,374 |
| 1,048,576 | 45,509,592 | 25,039,001 | 4,047,495,468 |
| 1,000 | 43,054 | 27,876 | 152,199,729 |
| 1,080 | 44,482 | 28,746 | 364,891,735 |
| 1,920 | 75,669 | 53,922 | 584,503,116 |
| 1,009 | 74,164 | 31,500 | 141,996,614 |
| 1,296 | 61,483 | 34,667 | 263,464,999 |
| 10,007 | 1,021,825 | 296,812 | 1,249,072,648 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 22/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 4,096 (2¹²): 1.08×
- complex 1,024 (2¹⁰): 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
