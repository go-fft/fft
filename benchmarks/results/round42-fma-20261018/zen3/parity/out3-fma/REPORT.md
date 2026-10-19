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
| 256 (2⁸) | — (0.0) | 287 (35.6) | 6,809 (1.5) | 5,121 (2.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,526 (33.6) | 13,089 (3.9) | 9,425 (5.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,625 (23.1) | 45,005 (5.5) | 33,471 (7.3) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 295,913 (17.7) | 1,905,914 (2.8) | 945,101 (5.5) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,321,336 (9.3) | 23,899,507 (4.4) | 14,524,872 (7.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,030 (24.6) | 13,914 (3.6) | 9,780 (5.1) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,233 (24.4) | 15,023 (3.6) | 10,315 (5.3) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,866 (27.1) | 22,174 (4.7) | 16,097 (6.5) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,130 (2.5) | 61,048 (0.8) | 38,412 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,027 (22.1) | 16,491 (4.1) | 12,496 (5.4) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 206,275 (3.2) | 650,178 (1.0) | 477,636 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 225 (22.8) | 6,072 (0.8) | 5,226 (1.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 985 (26.0) | 9,556 (2.7) | 8,154 (3.1) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,542 (22.2) | 25,393 (4.8) | 21,862 (5.6) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 155,993 (16.8) | 422,063 (6.2) | 496,059 (5.3) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,738,304 (14.0) | 8,466,564 (6.2) | 10,290,751 (5.1) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,275 (19.5) | 11,704 (2.1) | 8,675 (2.9) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,406 (19.4) | 11,193 (2.4) | 9,009 (3.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,411 (21.7) | 14,606 (3.6) | 12,215 (4.3) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 367 (13.9) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,155 (22.2) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,777 (21.3) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 167,312 (15.7) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,098,519 (12.8) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,502 (16.6) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,585 (17.2) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,694 (19.4) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,816 (22.7) | 50,891 (4.8) | 32,685 (7.5) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 63,744 (18.0) | 144,117 (8.0) | 128,322 (8.9) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 366,172 (14.3) | 713,481 (7.3) | 451,701 (11.6) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,367,215 (17.3) | 3,010,818 (7.8) | 2,052,417 (11.5) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 7,700,864 (13.6) | 13,485,055 (7.8) | 9,507,931 (11.0) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,562,172 |
| 1,024 | — | — | 132,105,791 |
| 4,096 | — | — | 240,495,809 |
| 65,536 | — | — | 2,159,966,612 |
| 1,048,576 | — | — | 3,955,493,178 |
| 1,000 | — | — | 152,228,309 |
| 1,080 | — | — | 365,294,676 |
| 1,920 | — | — | 574,853,771 |
| 1,009 | — | — | 142,053,665 |
| 1,296 | — | — | 265,194,980 |
| 10,007 | — | — | 1,237,727,083 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
