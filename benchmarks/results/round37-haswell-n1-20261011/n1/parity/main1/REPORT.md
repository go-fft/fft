# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Neoverse-N1 (cfarm424), one pinned core.
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
| 256 (2⁸) | 1,005 (10.2) | 1,151 (8.9) | 9,573 (1.1) | 7,620 (1.3) | 5,779 (1.8) | 0.87× | **≥ parity** |
| 1,024 (2¹⁰) | 4,891 (10.5) | 6,320 (8.1) | 18,735 (2.7) | 14,563 (3.5) | 30,387 (1.7) | 0.77× | **≥ parity** |
| 4,096 (2¹²) | 23,916 (10.3) | 40,627 (6.0) | 59,110 (4.2) | 47,585 (5.2) | 146,551 (1.7) | 0.59× | **≥ parity** |
| 65,536 (2¹⁶) | 717,997 (7.3) | 1,254,255 (4.2) | 2,131,551 (2.5) | 1,407,294 (3.7) | 3,239,327 (1.6) | 0.57× | **≥ parity** |
| 1,048,576 (2²⁰) | 20,010,282 (5.2) | 44,100,021 (2.4) | 44,564,557 (2.4) | 32,713,106 (3.2) | 75,024,619 (1.4) | 0.45× | **≥ parity** |
| 1,000 (2³·5³) | 6,007 (8.3) | 7,180 (6.9) | 19,385 (2.6) | 15,163 (3.3) | 33,834 (1.5) | 0.84× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,871 (7.9) | 7,712 (7.1) | 21,638 (2.5) | 16,808 (3.2) | 41,448 (1.3) | 0.89× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,962 (8.8) | 13,542 (7.7) | 32,840 (3.2) | 25,981 (4.0) | 70,175 (1.5) | 0.88× | **≥ parity** |
| 1,009 (prime) | 21,718 (2.3) | 48,934 (1.0) | 96,594 (0.5) | 57,495 (0.9) | 2,240,076 (0.0) | 0.44× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,690 (7.7) | 11,330 (5.9) | 24,721 (2.7) | 19,712 (3.4) | 48,964 (1.4) | 0.77× | **≥ parity** |
| 10,007 (prime) | 419,286 (1.6) | 583,034 (1.1) | 1,032,470 (0.6) | 768,202 (0.9) | 218,607,665 (0.0) | 0.72× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 760 (6.7) | 719 (7.1) | 8,631 (0.6) | 7,686 (0.7) | 2,976 (1.7) | 1.06× | lags FFTW 1.06× |
| 1,024 (2¹⁰) | 3,086 (8.3) | 4,162 (6.2) | 14,191 (1.8) | 12,485 (2.1) | 15,184 (1.7) | 0.74× | **≥ parity** |
| 4,096 (2¹²) | 15,003 (8.2) | 19,652 (6.3) | 37,254 (3.3) | 32,993 (3.7) | 70,327 (1.7) | 0.76× | **≥ parity** |
| 65,536 (2¹⁶) | 392,479 (6.7) | 542,986 (4.8) | 719,969 (3.6) | 752,276 (3.5) | 1,776,311 (1.5) | 0.72× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,106,924 (5.2) | 18,271,852 (2.9) | 19,216,160 (2.7) | 18,845,860 (2.8) | 42,530,882 (1.2) | 0.55× | **≥ parity** |
| 1,000 (2³·5³) | 3,830 (6.5) | 3,963 (6.3) | 15,295 (1.6) | 12,548 (2.0) | 15,028 (1.7) | 0.97× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,359 (6.2) | 4,212 (6.5) | 15,998 (1.7) | 13,892 (2.0) | 17,392 (1.6) | 1.03× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,124 (7.3) | 7,328 (7.1) | 22,457 (2.3) | 19,388 (2.7) | 30,923 (1.7) | 0.97× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 764 (6.7) | 808 (6.3) | 0.95× | **≥ parity** |
| 1,024 (2¹⁰) | 3,103 (8.3) | 4,445 (5.8) | 0.70× | **≥ parity** |
| 4,096 (2¹²) | 14,628 (8.4) | 21,142 (5.8) | 0.69× | **≥ parity** |
| 65,536 (2¹⁶) | 389,113 (6.7) | 614,416 (4.3) | 0.63× | **≥ parity** |
| 1,048,576 (2²⁰) | 9,957,874 (5.3) | 23,456,423 (2.2) | 0.42× | **≥ parity** |
| 1,000 (2³·5³) | 3,823 (6.5) | 4,286 (5.8) | 0.89× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,396 (6.2) | 4,419 (6.2) | 0.99× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,120 (7.4) | 7,818 (6.7) | 0.91× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 24,616 (10.0) | 31,203 (7.9) | 30,852 (8.0) | 73,875 (3.3) | 50,774 (4.8) | 0.80× | **≥ parity** |
| 128x128 | 125,586 (9.1) | 152,009 (7.5) | 192,447 (6.0) | 258,437 (4.4) | 206,472 (5.6) | 0.65× | **≥ parity** |
| 256x256 | 624,979 (8.4) | 782,998 (6.7) | 1,267,600 (4.1) | 1,149,151 (4.6) | 940,990 (5.6) | 0.49× | **≥ parity** |
| 512x512 | 3,124,511 (7.6) | 4,149,720 (5.7) | 6,652,885 (3.5) | 5,639,707 (4.2) | 4,157,359 (5.7) | 0.47× | **≥ parity** |
| 1024x1024 | 18,642,109 (5.6) | 18,666,381 (5.6) | 43,981,799 (2.4) | 26,328,035 (4.0) | 20,903,446 (5.0) | 0.42× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 14,474 | 9,980 | 4,493,384 |
| 1,024 | 59,961 | 34,977 | 3,267,838 |
| 4,096 | 193,591 | 130,806 | 6,180,264 |
| 65,536 | 3,218,639 | 1,936,420 | 21,955,309 |
| 1,048,576 | 48,522,708 | 30,601,111 | 20,813,843 |
| 1,000 | 52,709 | 34,756 | 4,674,029 |
| 1,080 | 60,457 | 37,288 | 12,742,974 |
| 1,920 | 100,011 | 65,228 | 17,417,842 |
| 1,009 | 116,665 | 40,745 | 6,773,436 |
| 1,296 | 70,072 | 44,213 | 10,592,965 |
| 10,007 | 1,395,153 | 386,752 | 35,397,014 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 23/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.06×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
