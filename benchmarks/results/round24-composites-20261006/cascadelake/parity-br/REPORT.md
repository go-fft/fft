# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Intel Xeon (Cascade Lake), cfarm151, 8 vCPUs, pinned to one core.
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
| 256 (2⁸) | 518 (19.8) | 380 (27.0) | 10,025 (1.0) | 8,022 (1.3) | 5,540 (1.8) | 1.36× | lags FFTW 1.36× |
| 1,024 (2¹⁰) | 2,833 (18.1) | 2,167 (23.6) | 20,845 (2.5) | 15,159 (3.4) | 29,150 (1.8) | 1.31× | lags FFTW 1.31× |
| 4,096 (2¹²) | 13,305 (18.5) | 13,277 (18.5) | 74,866 (3.3) | 54,509 (4.5) | 135,011 (1.8) | 1.00× | **≥ parity** |
| 65,536 (2¹⁶) | 500,803 (10.5) | 475,518 (11.0) | 2,270,224 (2.3) | 1,579,284 (3.3) | 3,502,937 (1.5) | 1.05× | lags FFTW 1.05× |
| 1,048,576 (2²⁰) | 24,014,434 (4.4) | 30,705,798 (3.4) | 66,975,820 (1.6) | 47,575,533 (2.2) | 103,842,046 (1.0) | 0.78× | **≥ parity** |
| 1,000 (2³·5³) | 3,415 (14.6) | 3,141 (15.9) | 21,683 (2.3) | 16,088 (3.1) | 30,932 (1.6) | 1.09× | lags FFTW 1.09× |
| 1,080 (2³·3³·5) | 4,161 (13.1) | 3,874 (14.0) | 23,343 (2.3) | 16,923 (3.2) | 36,048 (1.5) | 1.07× | lags FFTW 1.07× |
| 1,920 (2⁷·3·5) | 6,771 (15.5) | 6,104 (17.2) | 36,595 (2.9) | 26,736 (3.9) | 62,857 (1.7) | 1.11× | lags FFTW 1.11× |
| 1,009 (prime) | 19,031 (2.6) | 34,448 (1.5) | 102,168 (0.5) | 65,399 (0.8) | 1,454,985 (0.0) | 0.55× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 5,107 (13.1) | 4,860 (13.8) | 26,574 (2.5) | 19,805 (3.4) | 42,148 (1.6) | 1.05× | lags FFTW 1.05× |
| 10,007 (prime) | 344,210 (1.9) | 434,687 (1.5) | 1,015,021 (0.7) | 873,947 (0.8) | 141,886,288 (0.0) | 0.79× | **≥ parity** |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 482 (10.6) | 484 (10.6) | 8,331 (0.6) | 7,920 (0.6) | 2,672 (1.9) | 1.00× | **≥ parity** |
| 1,024 (2¹⁰) | 1,802 (14.2) | 1,813 (14.1) | 14,718 (1.7) | 12,794 (2.0) | 13,752 (1.9) | 0.99× | **≥ parity** |
| 4,096 (2¹²) | 9,337 (13.2) | 8,968 (13.7) | 41,848 (2.9) | 35,835 (3.4) | 62,351 (2.0) | 1.04× | **≥ parity** |
| 65,536 (2¹⁶) | 251,485 (10.4) | 226,794 (11.6) | 749,736 (3.5) | 828,308 (3.2) | 1,495,303 (1.8) | 1.11× | lags FFTW 1.11× |
| 1,048,576 (2²⁰) | 13,714,479 (3.8) | 11,429,464 (4.6) | 32,970,638 (1.6) | 24,646,589 (2.1) | 52,453,187 (1.0) | 1.20× | lags FFTW 1.20× |
| 1,000 (2³·5³) | 2,229 (11.2) | 2,164 (11.5) | 16,130 (1.5) | 13,158 (1.9) | 14,325 (1.7) | 1.03× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,574 (10.6) | 2,293 (11.9) | 17,416 (1.6) | 14,176 (1.9) | 16,633 (1.6) | 1.12× | lags FFTW 1.12× |
| 1,920 (2⁷·3·5) | 4,141 (12.6) | 4,092 (12.8) | 22,601 (2.3) | 19,477 (2.7) | 30,612 (1.7) | 1.01× | **≥ parity** |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 499 (10.3) | 650 (7.9) | 0.77× | **≥ parity** |
| 1,024 (2¹⁰) | 1,775 (14.4) | 2,163 (11.8) | 0.82× | **≥ parity** |
| 4,096 (2¹²) | 9,462 (13.0) | 9,658 (12.7) | 0.98× | **≥ parity** |
| 65,536 (2¹⁶) | 271,875 (9.6) | 253,897 (10.3) | 1.07× | lags FFTW 1.07× |
| 1,048,576 (2²⁰) | 13,784,418 (3.8) | 13,229,042 (4.0) | 1.04× | **≥ parity** |
| 1,000 (2³·5³) | 2,248 (11.1) | 2,634 (9.5) | 0.85× | **≥ parity** |
| 1,080 (2³·3³·5) | 2,520 (10.8) | 2,870 (9.5) | 0.88× | **≥ parity** |
| 1,920 (2⁷·3·5) | 4,101 (12.8) | 4,593 (11.4) | 0.89× | **≥ parity** |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. The go-fft column is a reused `PlanN` writing into a reused slice, as FFTW's plan writes into its reused array; the `FFT2` column is the convenience call, which returns a new slice each time and so also pays an allocation (and, on a many-core host, the garbage collector). ns/op (GFLOP/s).

| shape | go-fft | FFT2 | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|---:|:--|
| 64x64 | 17,814 (13.8) | 31,258 (7.9) | 16,204 (15.2) | 68,213 (3.6) | 50,359 (4.9) | 1.10× | lags FFTW 1.10× |
| 128x128 | 86,036 (13.3) | 152,306 (7.5) | 96,617 (11.9) | 240,451 (4.8) | 207,712 (5.5) | 0.89× | **≥ parity** |
| 256x256 | 464,771 (11.3) | 869,870 (6.0) | 598,778 (8.8) | 1,043,190 (5.0) | 895,086 (5.9) | 0.78× | **≥ parity** |
| 512x512 | 5,252,186 (4.5) | 6,129,349 (3.8) | 3,379,437 (7.0) | 6,870,316 (3.4) | 4,893,303 (4.8) | 1.55× | lags FFTW 1.55× |
| 1024x1024 | 19,462,524 (5.4) | 22,931,317 (4.6) | 27,938,306 (3.8) | 30,974,002 (3.4) | 28,913,758 (3.6) | 0.70× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 20,199 | 11,266 | 71,085,785 |
| 1,024 | 74,341 | 41,298 | 126,900,865 |
| 4,096 | 278,660 | 153,210 | 279,488,832 |
| 65,536 | 4,559,457 | 2,441,200 | 3,577,250,803 |
| 1,048,576 | 83,357,853 | 36,549,585 | 9,932,372,632 |
| 1,000 | 77,189 | 38,329 | 155,014,085 |
| 1,080 | 85,028 | 41,268 | 372,202,426 |
| 1,920 | 141,457 | 71,103 | 615,350,226 |
| 1,009 | 187,839 | 46,109 | 148,524,271 |
| 1,296 | 94,648 | 49,195 | 281,962,836 |
| 10,007 | 2,228,142 | 448,042 | 1,742,994,850 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 12/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 24/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 23/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

### Rows behind FFTW, worst first

- 2-D 512x512: 1.55×
- complex 256 (2⁸): 1.36×
- complex 1,024 (2¹⁰): 1.31×
- real 1,048,576 (2²⁰): 1.20×
- real 1,080 (2³·3³·5): 1.12×
- complex 1,920 (2⁷·3·5): 1.11×
- real 65,536 (2¹⁶): 1.11×
- 2-D 64x64: 1.10×
- complex 1,000 (2³·5³): 1.09×
- complex 1,080 (2³·3³·5): 1.07×
- complex 65,536 (2¹⁶): 1.05×
- complex 1,296 (2⁴·3⁴): 1.05×

Why each band trails, and the levers already measured against it (kept or dropped), are in the dated rounds of the repository's BENCHMARKS.md, not regenerated here.
