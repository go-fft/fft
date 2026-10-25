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
| 256 (2⁸) | — (0.0) | 281 (36.5) | 6,821 (1.5) | 5,338 (1.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,546 (33.1) | 13,299 (3.8) | 9,423 (5.4) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 10,344 (23.8) | 43,470 (5.7) | 32,544 (7.6) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 303,340 (17.3) | 1,976,325 (2.7) | 906,701 (5.8) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 11,995,658 (8.7) | 25,750,098 (4.1) | 13,795,595 (7.6) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 2,031 (24.5) | 13,483 (3.7) | 9,821 (5.1) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 2,236 (24.3) | 14,308 (3.8) | 10,835 (5.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 3,892 (26.9) | 22,054 (4.7) | 15,933 (6.6) | — (0.0) | nan× | — |
| 1,009 (prime) | — (0.0) | 20,324 (2.5) | 61,529 (0.8) | 38,741 (1.3) | — (0.0) | nan× | — |
| 1,296 (2⁴·3⁴) | — (0.0) | 2,957 (22.7) | 16,094 (4.2) | 12,328 (5.4) | — (0.0) | nan× | — |
| 10,007 (prime) | — (0.0) | 207,844 (3.2) | 648,592 (1.0) | 473,304 (1.4) | — (0.0) | nan× | — |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 228 (22.4) | 6,005 (0.9) | 5,459 (0.9) | — (0.0) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 978 (26.2) | 9,519 (2.7) | 8,513 (3.0) | — (0.0) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 6,458 (19.0) | 25,506 (4.8) | 22,342 (5.5) | — (0.0) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 161,321 (16.2) | 434,759 (6.0) | 511,095 (5.1) | — (0.0) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 3,655,309 (14.3) | 8,371,765 (6.3) | 10,034,589 (5.2) | — (0.0) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,235 (20.2) | 11,255 (2.2) | 8,832 (2.8) | — (0.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,343 (20.3) | 11,138 (2.4) | 9,179 (3.0) | — (0.0) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,359 (22.2) | 16,289 (3.2) | 12,226 (4.3) | — (0.0) | nan× | — |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | — (0.0) | 318 (16.1) | nan× | — |
| 1,024 (2¹⁰) | — (0.0) | 1,260 (20.3) | nan× | — |
| 4,096 (2¹²) | — (0.0) | 5,902 (20.8) | nan× | — |
| 65,536 (2¹⁶) | — (0.0) | 167,187 (15.7) | nan× | — |
| 1,048,576 (2²⁰) | — (0.0) | 4,066,184 (12.9) | nan× | — |
| 1,000 (2³·5³) | — (0.0) | 1,554 (16.0) | nan× | — |
| 1,080 (2³·3³·5) | — (0.0) | 1,662 (16.4) | nan× | — |
| 1,920 (2⁷·3·5) | — (0.0) | 2,679 (19.5) | nan× | — |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | — (0.0) | — (0.0) | 10,647 (23.1) | 48,574 (5.1) | 37,534 (6.5) | nan× | — |
| 128x128 | — (0.0) | — (0.0) | 61,973 (18.5) | 151,129 (7.6) | 127,925 (9.0) | nan× | — |
| 256x256 | — (0.0) | — (0.0) | 285,708 (18.4) | 698,571 (7.5) | 463,794 (11.3) | nan× | — |
| 512x512 | — (0.0) | — (0.0) | 1,306,243 (18.1) | 3,004,652 (7.9) | 1,898,057 (12.4) | nan× | — |
| 1024x1024 | — (0.0) | — (0.0) | 6,493,991 (16.1) | 13,624,675 (7.7) | 9,229,788 (11.4) | nan× | — |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | — | — | 71,562,882 |
| 1,024 | — | — | 131,712,910 |
| 4,096 | — | — | 248,917,233 |
| 65,536 | — | — | 2,188,453,524 |
| 1,048,576 | — | — | 4,309,515,888 |
| 1,000 | — | — | 154,239,530 |
| 1,080 | — | — | 364,842,026 |
| 1,920 | — | — | 569,926,489 |
| 1,009 | — | — | 142,846,865 |
| 1,296 | — | — | 265,249,440 |
| 10,007 | — | — | 1,258,279,842 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 0/0 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 0/0 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 0/0 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
