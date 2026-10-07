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
| 256 (2⁸) | 1,105 (9.3) | 1,155 (8.9) | 9,483 (1.1) | 7,786 (1.3) | 5,834 (1.8) | 0.96× | **≥ parity** |
| 1,024 (2¹⁰) | 5,249 (9.8) | 6,441 (7.9) | 19,002 (2.7) | 14,932 (3.4) | 30,955 (1.7) | 0.81× | **≥ parity** |
| 4,096 (2¹²) | 27,618 (8.9) | 41,025 (6.0) | 58,676 (4.2) | 47,815 (5.1) | 143,171 (1.7) | 0.67× | **≥ parity** |
| 65,536 (2¹⁶) | 715,830 (7.3) | 1,233,616 (4.2) | 2,030,019 (2.6) | 1,439,566 (3.6) | 3,307,332 (1.6) | 0.58× | **≥ parity** |
| 1,048,576 (2²⁰) | 20,186,239 (5.2) | 44,102,620 (2.4) | 42,421,075 (2.5) | 31,273,598 (3.4) | 75,338,373 (1.4) | 0.46× | **≥ parity** |
| 1,000 (2³·5³) | 6,012 (8.3) | 7,219 (6.9) | 19,179 (2.6) | 15,269 (3.3) | 33,935 (1.5) | 0.83× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,879 (7.9) | 7,680 (7.1) | 22,133 (2.5) | 16,692 (3.3) | 41,856 (1.3) | 0.90× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,949 (8.8) | 13,549 (7.7) | 33,188 (3.2) | 26,096 (4.0) | 70,733 (1.5) | 0.88× | **≥ parity** |
| 1,009 (prime) | 21,562 (2.3) | 48,572 (1.0) | 97,371 (0.5) | 58,075 (0.9) | 2,225,823 (0.0) | 0.44× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,647 (7.7) | 11,464 (5.8) | 25,457 (2.6) | 19,818 (3.4) | 49,368 (1.4) | 0.75× | **≥ parity** |
| 10,007 (prime) | 419,006 (1.6) | 603,291 (1.1) | 1,040,777 (0.6) | 790,065 (0.8) | 220,516,350 (0.0) | 0.69× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 836 (6.1) | 722 (7.1) | 8,633 (0.6) | 7,839 (0.7) | 2,994 (1.7) | 1.16× | lags FFTW 1.16× |
| 1,024 (2¹⁰) | 3,564 (7.2) | 4,230 (6.1) | 14,571 (1.8) | 12,485 (2.1) | 15,187 (1.7) | 0.84× | **≥ parity** |
| 4,096 (2¹²) | 16,517 (7.4) | 19,858 (6.2) | 37,926 (3.2) | 33,392 (3.7) | 70,092 (1.8) | 0.83× | **≥ parity** |
| 65,536 (2¹⁶) | 413,805 (6.3) | 539,718 (4.9) | 729,815 (3.6) | 765,589 (3.4) | 1,766,820 (1.5) | 0.77× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,581,204 (5.0) | 18,678,518 (2.8) | 23,247,589 (2.3) | 18,879,134 (2.8) | 43,023,391 (1.2) | 0.57× | **≥ parity** |
| 1,000 (2³·5³) | 4,137 (6.0) | 3,945 (6.3) | 15,554 (1.6) | 12,780 (1.9) | 14,934 (1.7) | 1.05× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,636 (5.9) | 4,218 (6.5) | 16,641 (1.6) | 14,124 (1.9) | 17,505 (1.6) | 1.10× | lags FFTW 1.10× |
| 1,920 (2⁷·3·5) | 7,621 (6.9) | 7,329 (7.1) | 21,945 (2.4) | 18,868 (2.8) | 30,544 (1.7) | 1.04× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 968 (5.3) | 809 (6.3) | 1.20× | lags FFTW 1.20× |
| 1,024 (2¹⁰) | 3,866 (6.6) | 4,450 (5.8) | 0.87× | **≥ parity** |
| 4,096 (2¹²) | 17,585 (7.0) | 21,091 (5.8) | 0.83× | **≥ parity** |
| 65,536 (2¹⁶) | 434,183 (6.0) | 631,113 (4.2) | 0.69× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,992,241 (4.8) | 21,322,328 (2.5) | 0.52× | **≥ parity** |
| 1,000 (2³·5³) | 4,526 (5.5) | 4,276 (5.8) | 1.06× | lags FFTW 1.06× |
| 1,080 (2³·3³·5) | 5,069 (5.4) | 4,454 (6.1) | 1.14× | lags FFTW 1.14× |
| 1,920 (2⁷·3·5) | 8,350 (6.3) | 7,816 (6.7) | 1.07× | lags FFTW 1.07× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 34,633 (7.1) | 41,332 (5.9) | 30,970 (7.9) | 72,373 (3.4) | 50,853 (4.8) | 1.12× | lags FFTW 1.12× |
| 128x128 | 142,732 (8.0) | 162,869 (7.0) | 193,827 (5.9) | 259,018 (4.4) | 201,868 (5.7) | 0.74× | **≥ parity** |
| 256x256 | 755,928 (6.9) | 836,148 (6.3) | 1,309,668 (4.0) | 1,112,048 (4.7) | 921,728 (5.7) | 0.58× | **≥ parity** |
| 512x512 | 3,448,792 (6.8) | 4,159,138 (5.7) | 6,903,857 (3.4) | 5,152,756 (4.6) | 4,139,022 (5.7) | 0.50× | **≥ parity** |
| 1024x1024 | 17,924,219 (5.9) | 18,261,818 (5.7) | 44,166,117 (2.4) | 26,457,899 (4.0) | 21,042,111 (5.0) | 0.41× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 14,902 | 9,875 | 4,487,025 |
| 1,024 | 52,776 | 34,289 | 3,274,567 |
| 4,096 | 195,506 | 137,932 | 6,200,490 |
| 65,536 | 3,250,296 | 1,999,917 | 21,507,876 |
| 1,048,576 | 48,991,947 | 31,206,537 | 20,380,858 |
| 1,000 | 55,795 | 34,501 | 4,840,912 |
| 1,080 | 61,328 | 37,491 | 12,889,389 |
| 1,920 | 102,116 | 65,504 | 17,304,013 |
| 1,009 | 143,901 | 41,328 | 6,204,931 |
| 1,296 | 71,494 | 45,153 | 10,594,875 |
| 10,007 | 1,377,813 | 396,560 | 36,523,335 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 21/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- real 256 (2⁸): 1.16×
- 2-D 64x64: 1.12×
- real 1,080 (2³·3³·5): 1.10×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
