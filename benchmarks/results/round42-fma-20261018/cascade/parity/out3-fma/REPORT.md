# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Xeon Cascade Lake (cfarm151).
- **Toolchains**: go1.27.1 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512** (3.3.10 built from source by setup.sh); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | — (0.0) | 370 (27.7) | 8,971 (1.1) | 6,426 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,113 (24.2) | 18,384 (2.8) | 12,034 (4.3) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 12,682 (19.4) | 59,557 (4.1) | 49,047 (5.0) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 363,257 (14.4) | 1,963,504 (2.7) | 1,345,035 (3.9) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 25,268,992 (4.2) | 46,444,374 (2.3) | 41,260,098 (2.5) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,700 (18.5) | 17,925 (2.8) | 14,124 (3.5) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,990 (18.2) | 20,510 (2.7) | 13,546 (4.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 5,185 (20.2) | 31,625 (3.3) | 22,885 (4.6) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 30,712 (1.6) | 80,936 (0.6) | 57,235 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,646 (18.4) | 21,357 (3.1) | 15,805 (4.2) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 360,844 (1.8) | 937,747 (0.7) | 731,232 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 359 (14.2) | 6,596 (0.8) | 6,591 (0.8) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,501 (17.1) | 13,175 (1.9) | 10,520 (2.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,397 (16.6) | 33,204 (3.7) | 31,842 (3.9) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 189,389 (13.8) | 605,281 (4.3) | 694,033 (3.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 7,550,452 (6.9) | 20,053,473 (2.6) | 19,648,929 (2.7) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,870 (13.3) | 14,168 (1.8) | 10,853 (2.3) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,996 (13.6) | 13,546 (2.0) | 12,739 (2.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,430 (15.3) | 18,053 (2.9) | 17,485 (3.0) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 533 (9.6) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,814 (14.1) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,639 (14.2) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 214,066 (12.2) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 8,906,091 (5.9) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,274 (11.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,312 (11.8) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,979 (13.2) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 17,345 (14.2) | 56,050 (4.4) | 46,142 (5.3) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 72,788 (15.8) | 206,362 (5.6) | 172,472 (6.6) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 478,878 (10.9) | 922,358 (5.7) | 773,549 (6.8) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 2,343,332 (10.1) | 4,511,920 (5.2) | 3,334,065 (7.1) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 14,781,686 (7.1) | 29,858,754 (3.5) | 24,009,788 (4.4) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 69,007,332 |
| 1,024 | — | — | 127,301,667 |
| 4,096 | — | — | 277,555,324 |
| 65,536 | — | — | 3,069,936,043 |
| 1,048,576 | — | — | 7,893,764,976 |
| 1,000 | — | — | 143,361,913 |
| 1,080 | — | — | 351,312,744 |
| 1,920 | — | — | 564,900,177 |
| 1,009 | — | — | 135,891,883 |
| 1,296 | — | — | 257,461,568 |
| 10,007 | — | — | 1,515,009,491 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
