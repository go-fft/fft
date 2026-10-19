# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3, cfarm420).
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
| 256 (2⁸) | — (0.0) | 274 (37.4) | 6,947 (1.5) | 5,421 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,502 (34.1) | 13,322 (3.8) | 9,787 (5.2) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,728 (22.9) | 43,357 (5.7) | 32,873 (7.5) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 306,641 (17.1) | 1,955,525 (2.7) | 924,992 (5.7) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,591,362 (9.0) | 23,094,860 (4.5) | 13,779,633 (7.6) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,181 (22.8) | 13,857 (3.6) | 10,009 (5.0) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,414 (22.5) | 14,332 (3.8) | 10,622 (5.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,888 (26.9) | 22,134 (4.7) | 16,055 (6.5) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 19,859 (2.5) | 61,135 (0.8) | 39,012 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 2,997 (22.4) | 16,563 (4.0) | 12,357 (5.4) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 212,222 (3.1) | 643,301 (1.0) | 478,858 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 223 (22.9) | 5,998 (0.9) | 5,517 (0.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 978 (26.2) | 9,773 (2.6) | 8,724 (2.9) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,220 (23.5) | 26,137 (4.7) | 22,375 (5.5) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 164,540 (15.9) | 431,365 (6.1) | 496,632 (5.3) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,685,023 (14.2) | 8,263,343 (6.3) | 10,239,040 (5.1) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,260 (19.8) | 10,355 (2.4) | 8,827 (2.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,331 (20.4) | 11,060 (2.5) | 9,118 (3.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,373 (22.1) | 14,370 (3.6) | 12,656 (4.1) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 358 (14.3) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,171 (21.9) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,821 (21.1) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 166,633 (15.7) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,000,375 (13.1) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,428 (17.4) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,669 (16.3) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,688 (19.5) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,971 (22.4) | 52,067 (4.7) | 34,399 (7.1) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 60,705 (18.9) | 150,059 (7.6) | 129,122 (8.9) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 278,248 (18.8) | 641,850 (8.2) | 474,619 (11.0) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,267,214 (18.6) | 2,939,278 (8.0) | 1,935,346 (12.2) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,332,405 (16.6) | 13,235,973 (7.9) | 9,442,041 (11.1) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 72,224,492 |
| 1,024 | — | — | 131,802,140 |
| 4,096 | — | — | 239,983,809 |
| 65,536 | — | — | 2,170,130,426 |
| 1,048,576 | — | — | 4,135,169,509 |
| 1,000 | — | — | 152,561,050 |
| 1,080 | — | — | 364,929,396 |
| 1,920 | — | — | 573,113,800 |
| 1,009 | — | — | 142,265,365 |
| 1,296 | — | — | 281,765,848 |
| 10,007 | — | — | 1,219,082,014 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
