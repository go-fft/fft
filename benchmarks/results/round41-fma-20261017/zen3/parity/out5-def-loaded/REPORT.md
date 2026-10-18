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
| 256 (2⁸) | — (0.0) | 272 (37.6) | 7,012 (1.5) | 5,345 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,570 (32.6) | 13,404 (3.8) | 9,747 (5.3) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,669 (23.0) | 44,563 (5.5) | 33,800 (7.3) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 310,844 (16.9) | 2,002,504 (2.6) | 948,241 (5.5) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,230,216 (9.3) | 23,642,680 (4.4) | 13,796,508 (7.6) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,037 (24.5) | 14,188 (3.5) | 10,037 (5.0) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,407 (22.6) | 14,790 (3.7) | 10,747 (5.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,793 (27.6) | 22,307 (4.7) | 16,455 (6.4) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,127 (2.5) | 60,612 (0.8) | 37,990 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 2,940 (22.8) | 16,355 (4.1) | 12,187 (5.5) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 213,454 (3.1) | 644,226 (1.0) | 476,950 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 222 (23.1) | 6,079 (0.8) | 5,145 (1.0) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 990 (25.9) | 9,565 (2.7) | 8,040 (3.2) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,162 (23.8) | 25,416 (4.8) | 21,737 (5.7) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 154,936 (16.9) | 401,572 (6.5) | 492,445 (5.3) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,772,468 (13.9) | 8,322,244 (6.3) | 10,257,194 (5.1) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,234 (20.2) | 10,635 (2.3) | 8,577 (2.9) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,353 (20.1) | 11,302 (2.4) | 8,873 (3.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,314 (22.6) | 14,534 (3.6) | 11,846 (4.4) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 296 (17.3) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,181 (21.7) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,721 (21.5) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 164,630 (15.9) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,113,562 (12.7) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,479 (16.8) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,626 (16.7) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,793 (18.7) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,776 (22.8) | 51,044 (4.8) | 32,627 (7.5) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 59,747 (19.2) | 150,145 (7.6) | 128,159 (8.9) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 289,777 (18.1) | 600,863 (8.7) | 450,018 (11.7) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,356,890 (17.4) | 3,110,698 (7.6) | 1,951,535 (12.1) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,480,502 (16.2) | 14,484,086 (7.2) | 9,389,102 (11.2) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,538,513 |
| 1,024 | — | — | 131,403,509 |
| 4,096 | — | — | 239,483,629 |
| 65,536 | — | — | 2,210,634,584 |
| 1,048,576 | — | — | 3,973,901,036 |
| 1,000 | — | — | 152,712,320 |
| 1,080 | — | — | 365,347,086 |
| 1,920 | — | — | 587,727,267 |
| 1,009 | — | — | 141,407,774 |
| 1,296 | — | — | 268,966,012 |
| 10,007 | — | — | 1,232,219,680 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
