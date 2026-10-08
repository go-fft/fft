# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Neoverse-N1 (cfarm424), core 40.
- **Toolchains**: go1.27.1 linux/arm64 (cross-compiled); native **FFTW fftw-3.3.10-neon** (3.3.10 built from source by setup.sh); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 1,009 (10.1) | 1,157 (8.8) | 9,924 (1.0) | 7,803 (1.3) | 5,802 (1.8) | 0.87× | **≥ parity** |
| 1,024 (2¹⁰) | 4,885 (10.5) | 6,300 (8.1) | 19,041 (2.7) | 14,626 (3.5) | 30,642 (1.7) | 0.78× | **≥ parity** |
| 4,096 (2¹²) | 24,085 (10.2) | 41,083 (6.0) | 60,335 (4.1) | 48,073 (5.1) | 146,648 (1.7) | 0.59× | **≥ parity** |
| 65,536 (2¹⁶) | 721,553 (7.3) | 1,265,585 (4.1) | 2,262,059 (2.3) | 1,475,942 (3.6) | 3,224,362 (1.6) | 0.57× | **≥ parity** |
| 1,048,576 (2²⁰) | 20,680,093 (5.1) | 45,830,125 (2.3) | 44,352,443 (2.4) | 31,902,085 (3.3) | 74,491,033 (1.4) | 0.45× | **≥ parity** |
| 1,000 (2³·5³) | 6,015 (8.3) | 7,208 (6.9) | 19,432 (2.6) | 15,415 (3.2) | 34,039 (1.5) | 0.83× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,900 (7.9) | 7,704 (7.1) | 22,125 (2.5) | 16,788 (3.2) | 41,243 (1.3) | 0.90× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,860 (8.8) | 13,554 (7.7) | 33,019 (3.2) | 25,942 (4.0) | 70,503 (1.5) | 0.88× | **≥ parity** |
| 1,009 (prime) | 21,373 (2.4) | 48,253 (1.0) | 96,178 (0.5) | 57,933 (0.9) | 2,228,333 (0.0) | 0.44× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,724 (7.7) | 11,403 (5.9) | 24,934 (2.7) | 19,801 (3.4) | 49,354 (1.4) | 0.77× | **≥ parity** |
| 10,007 (prime) | 419,571 (1.6) | 615,217 (1.1) | 1,037,037 (0.6) | 758,904 (0.9) | 218,522,019 (0.0) | 0.68× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 754 (6.8) | 721 (7.1) | 8,907 (0.6) | 7,947 (0.6) | 3,017 (1.7) | 1.05× | **≥ parity** |
| 1,024 (2¹⁰) | 3,104 (8.2) | 4,219 (6.1) | 14,471 (1.8) | 12,723 (2.0) | 15,352 (1.7) | 0.74× | **≥ parity** |
| 4,096 (2¹²) | 15,017 (8.2) | 19,818 (6.2) | 37,582 (3.3) | 32,992 (3.7) | 70,829 (1.7) | 0.76× | **≥ parity** |
| 65,536 (2¹⁶) | 401,210 (6.5) | 565,543 (4.6) | 739,238 (3.5) | 771,994 (3.4) | 1,785,665 (1.5) | 0.71× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,189,612 (5.1) | 19,195,971 (2.7) | 20,943,512 (2.5) | 18,788,209 (2.8) | 43,107,018 (1.2) | 0.53× | **≥ parity** |
| 1,000 (2³·5³) | 3,891 (6.4) | 3,974 (6.3) | 15,432 (1.6) | 13,258 (1.9) | 14,896 (1.7) | 0.98× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,311 (6.3) | 4,209 (6.5) | 16,429 (1.7) | 14,220 (1.9) | 17,523 (1.6) | 1.02× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,139 (7.3) | 7,332 (7.1) | 22,090 (2.4) | 19,037 (2.8) | 30,449 (1.7) | 0.97× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 768 (6.7) | 809 (6.3) | 0.95× | **≥ parity** |
| 1,024 (2¹⁰) | 3,160 (8.1) | 4,443 (5.8) | 0.71× | **≥ parity** |
| 4,096 (2¹²) | 14,618 (8.4) | 21,160 (5.8) | 0.69× | **≥ parity** |
| 65,536 (2¹⁶) | 390,106 (6.7) | 665,169 (3.9) | 0.59× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,754,393 (4.9) | 22,975,735 (2.3) | 0.47× | **≥ parity** |
| 1,000 (2³·5³) | 3,837 (6.5) | 4,278 (5.8) | 0.90× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,400 (6.2) | 4,421 (6.2) | 1.00× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,221 (7.3) | 7,825 (6.7) | 0.92× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 24,464 (10.0) | 30,813 (8.0) | 30,899 (8.0) | 73,104 (3.4) | 50,551 (4.9) | 0.79× | **≥ parity** |
| 128x128 | 125,450 (9.1) | 153,345 (7.5) | 193,415 (5.9) | 258,046 (4.4) | 203,020 (5.6) | 0.65× | **≥ parity** |
| 256x256 | 659,428 (8.0) | 781,212 (6.7) | 1,444,895 (3.6) | 1,115,125 (4.7) | 968,471 (5.4) | 0.46× | **≥ parity** |
| 512x512 | 3,219,332 (7.3) | 4,088,444 (5.8) | 7,735,338 (3.0) | 5,921,723 (4.0) | 4,327,970 (5.5) | 0.42× | **≥ parity** |
| 1024x1024 | 18,374,804 (5.7) | 18,754,167 (5.6) | 46,947,400 (2.2) | 26,171,978 (4.0) | 20,093,761 (5.2) | 0.39× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 4,426,774 |
| 1,024 | — | — | 3,330,441 |
| 4,096 | — | — | 6,276,516 |
| 65,536 | — | — | 21,643,067 |
| 1,048,576 | — | — | 20,373,531 |
| 1,000 | — | — | 4,680,416 |
| 1,080 | — | — | 12,739,237 |
| 1,920 | — | — | 17,492,136 |
| 1,009 | — | — | 6,099,194 |
| 1,296 | — | — | 10,555,891 |
| 10,007 | — | — | 35,916,084 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 24/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
