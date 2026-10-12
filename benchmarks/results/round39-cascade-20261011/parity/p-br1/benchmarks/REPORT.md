# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon (Cascade Lake), cfarm151, 1 pinned core.
- **Toolchains**: go1.27.1 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512** (3.3.10 built from source with AVX-512 (setup.sh)); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | — (0.0) | 371 (27.6) | 8,620 (1.2) | 6,406 (1.6) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 2,120 (24.1) | 16,260 (3.1) | 12,286 (4.2) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 13,378 (18.4) | 59,696 (4.1) | 43,324 (5.7) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 396,995 (13.2) | 1,840,928 (2.8) | 1,251,992 (4.2) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 25,392,363 (4.1) | 47,862,044 (2.2) | 41,290,840 (2.5) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,709 (18.4) | 16,925 (2.9) | 13,837 (3.6) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,943 (18.5) | 20,456 (2.7) | 13,212 (4.1) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 5,084 (20.6) | 31,168 (3.4) | 23,040 (4.5) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 30,455 (1.7) | 80,483 (0.6) | 56,586 (0.9) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 3,604 (18.6) | 20,667 (3.2) | 16,443 (4.1) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 365,485 (1.8) | 940,472 (0.7) | 719,881 (0.9) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 364 (14.1) | 6,552 (0.8) | 7,071 (0.7) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,511 (16.9) | 13,024 (2.0) | 10,438 (2.5) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 7,589 (16.2) | 32,405 (3.8) | 31,408 (3.9) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 188,641 (13.9) | 584,085 (4.5) | 680,925 (3.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 6,615,664 (7.9) | 18,978,119 (2.8) | 20,280,753 (2.6) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,855 (13.4) | 13,314 (1.9) | 11,875 (2.1) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,018 (13.5) | 13,484 (2.0) | 10,894 (2.5) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,480 (15.0) | 18,000 (2.9) | 16,174 (3.2) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 499 (10.3) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,887 (13.6) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 8,687 (14.1) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 214,482 (12.2) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 9,516,476 (5.5) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,304 (10.8) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,499 (10.9) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,863 (13.6) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 17,153 (14.3) | 56,093 (4.4) | 45,175 (5.4) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 81,422 (14.1) | 205,936 (5.6) | 182,336 (6.3) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 476,303 (11.0) | 914,057 (5.7) | 760,793 (6.9) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 2,362,574 (10.0) | 5,498,707 (4.3) | 3,661,322 (6.4) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 16,818,290 (6.2) | 27,209,507 (3.9) | 25,931,973 (4.0) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 68,966,564 |
| 1,024 | — | — | 127,243,130 |
| 4,096 | — | — | 276,938,144 |
| 65,536 | — | — | 3,055,012,042 |
| 1,048,576 | — | — | 8,136,201,890 |
| 1,000 | — | — | 141,958,022 |
| 1,080 | — | — | 341,168,173 |
| 1,920 | — | — | 567,885,704 |
| 1,009 | — | — | 135,231,415 |
| 1,296 | — | — | 256,166,203 |
| 10,007 | — | — | 1,498,843,124 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
