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
| 256 (2⁸) | 1,010 (10.1) | 1,162 (8.8) | 9,482 (1.1) | 7,718 (1.3) | 5,783 (1.8) | 0.87× | **≥ parity** |
| 1,024 (2¹⁰) | 4,863 (10.5) | 6,401 (8.0) | 18,824 (2.7) | 14,757 (3.5) | 30,491 (1.7) | 0.76× | **≥ parity** |
| 4,096 (2¹²) | 23,954 (10.3) | 41,300 (6.0) | 60,001 (4.1) | 48,260 (5.1) | 146,955 (1.7) | 0.58× | **≥ parity** |
| 65,536 (2¹⁶) | 719,587 (7.3) | 1,304,811 (4.0) | 2,241,211 (2.3) | 1,469,908 (3.6) | 3,421,300 (1.5) | 0.55× | **≥ parity** |
| 1,048,576 (2²⁰) | 20,431,493 (5.1) | 47,552,054 (2.2) | 46,227,090 (2.3) | 30,739,035 (3.4) | 73,985,644 (1.4) | 0.43× | **≥ parity** |
| 1,000 (2³·5³) | 6,099 (8.2) | 7,194 (6.9) | 19,588 (2.5) | 15,741 (3.2) | 33,942 (1.5) | 0.85× | **≥ parity** |
| 1,080 (2³·3³·5) | 6,905 (7.9) | 7,704 (7.1) | 22,263 (2.4) | 16,976 (3.2) | 41,631 (1.3) | 0.90× | **≥ parity** |
| 1,920 (2⁷·3·5) | 11,949 (8.8) | 13,503 (7.8) | 33,794 (3.1) | 26,583 (3.9) | 70,003 (1.5) | 0.88× | **≥ parity** |
| 1,009 (prime) | 21,363 (2.4) | 48,134 (1.0) | 96,995 (0.5) | 57,747 (0.9) | 2,224,320 (0.0) | 0.44× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 8,657 (7.7) | 11,409 (5.9) | 24,823 (2.7) | 19,777 (3.4) | 48,795 (1.4) | 0.76× | **≥ parity** |
| 10,007 (prime) | 419,629 (1.6) | 615,660 (1.1) | 1,049,646 (0.6) | 759,762 (0.9) | 216,535,365 (0.0) | 0.68× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 753 (6.8) | 720 (7.1) | 8,782 (0.6) | 7,880 (0.6) | 2,990 (1.7) | 1.05× | **≥ parity** |
| 1,024 (2¹⁰) | 3,131 (8.2) | 4,248 (6.0) | 14,192 (1.8) | 12,516 (2.0) | 15,289 (1.7) | 0.74× | **≥ parity** |
| 4,096 (2¹²) | 14,799 (8.3) | 20,021 (6.1) | 37,156 (3.3) | 33,160 (3.7) | 68,189 (1.8) | 0.74× | **≥ parity** |
| 65,536 (2¹⁶) | 396,407 (6.6) | 570,441 (4.6) | 727,700 (3.6) | 773,975 (3.4) | 1,743,830 (1.5) | 0.69× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,024,237 (5.2) | 19,647,184 (2.7) | 21,808,864 (2.4) | 19,221,289 (2.7) | 42,882,385 (1.2) | 0.51× | **≥ parity** |
| 1,000 (2³·5³) | 3,881 (6.4) | 3,947 (6.3) | 15,188 (1.6) | 12,713 (2.0) | 14,897 (1.7) | 0.98× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,364 (6.2) | 4,214 (6.5) | 15,976 (1.7) | 14,036 (1.9) | 17,461 (1.6) | 1.04× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,119 (7.4) | 7,325 (7.1) | 21,935 (2.4) | 18,981 (2.8) | 30,563 (1.7) | 0.97× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 764 (6.7) | 808 (6.3) | 0.95× | **≥ parity** |
| 1,024 (2¹⁰) | 3,069 (8.3) | 4,467 (5.7) | 0.69× | **≥ parity** |
| 4,096 (2¹²) | 14,685 (8.4) | 21,140 (5.8) | 0.69× | **≥ parity** |
| 65,536 (2¹⁶) | 387,312 (6.8) | 689,244 (3.8) | 0.56× | **≥ parity** |
| 1,048,576 (2²⁰) | 10,223,773 (5.1) | 23,217,274 (2.3) | 0.44× | **≥ parity** |
| 1,000 (2³·5³) | 3,894 (6.4) | 4,256 (5.9) | 0.91× | **≥ parity** |
| 1,080 (2³·3³·5) | 4,345 (6.3) | 4,426 (6.1) | 0.98× | **≥ parity** |
| 1,920 (2⁷·3·5) | 7,173 (7.3) | 7,810 (6.7) | 0.92× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 24,585 (10.0) | 29,720 (8.3) | 30,944 (7.9) | 71,506 (3.4) | 50,094 (4.9) | 0.79× | **≥ parity** |
| 128x128 | 127,707 (9.0) | 146,682 (7.8) | 193,707 (5.9) | 254,411 (4.5) | 199,726 (5.7) | 0.66× | **≥ parity** |
| 256x256 | 619,436 (8.5) | 713,526 (7.3) | 1,405,655 (3.7) | 1,088,201 (4.8) | 901,638 (5.8) | 0.44× | **≥ parity** |
| 512x512 | 3,218,036 (7.3) | 3,980,629 (5.9) | 7,759,418 (3.0) | 5,275,140 (4.5) | 3,934,539 (6.0) | 0.41× | **≥ parity** |
| 1024x1024 | 17,818,614 (5.9) | 18,444,570 (5.7) | 46,589,220 (2.3) | 25,750,725 (4.1) | 19,913,557 (5.3) | 0.38× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 14,386 | 9,839 | 4,677,936 |
| 1,024 | 52,232 | 34,164 | 3,282,787 |
| 4,096 | 196,052 | 129,636 | 6,341,649 |
| 65,536 | 3,139,719 | 1,992,199 | 21,475,801 |
| 1,048,576 | 45,766,995 | 29,028,375 | 19,964,809 |
| 1,000 | 52,105 | 34,247 | 4,768,180 |
| 1,080 | 59,236 | 37,205 | 13,195,869 |
| 1,920 | 101,368 | 66,067 | 17,474,838 |
| 1,009 | 117,459 | 40,308 | 6,124,925 |
| 1,296 | 93,014 | 44,614 | 10,341,290 |
| 10,007 | 1,382,620 | 403,821 | 36,495,866 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 24/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

None: every row is at or above parity.

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
