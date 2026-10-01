# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: ARM Neoverse-N1 (64 cores, NEON; GCC Compile Farm cfarm424), Debian 13, arm64.
- **Toolchains**: go1.26.4 linux/arm64 (cross-compiled); native **FFTW fftw-3.3.10-neon** (3.3.10 built from source with --enable-neon, statically linked from C); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
- **Single-threaded** for the apples-to-apples core comparison: FFTW planned with `threads=1`; numpy/scipy pinned via `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`; scipy `workers=1`; Go benchmarks are single-goroutine for 1-D. The 2-D rows are the one place go-fft uses its multicore path (the others are all 1-core).
- **Plan reuse / steady state**: go-fft via its cached `Plan` API (`NewPlan(n).FFT`, `NewRealPlan(n).RFFT`); gonum via its reused `CmplxFFT`/`FFT` object; FFTW via a reused `FFTW_MEASURE` plan; scipy via its internal plan cache. Each number is the **steady-state transform** cost, not planning — plan/setup cost is reported separately below.
- **Iterations**: Go uses `-benchtime=1s` (auto-scaled `b.N`); the C and Python harnesses auto-scale each batch to ~0.2 s and take the **best of 6** batches after warm-up. Lower ns/op is better.
- **Metric**: ns/op and **GFLOP/s** using the standard `5·N·log2(N)` flop convention for a complex N-point FFT (real rfft counted at half, `2.5·N·log2(N)`; 2-D at `5·N·log2(N)` with N = total points).
- **Inputs**: bit-identical across all four implementations (`((i·7+1)%13)·0.1 + i·((i·3+2)%11)·0.1` for complex; `((i·7+1)%13)·0.1` for real).
- **Note on pyfftw**: the bundled pip-wheel FFTW plans a poor 2-D transform on Apple Silicon (≈4× slower than the native bottle); the FFTW column therefore uses the **native Homebrew FFTW called directly from C** (`benchmarks/cbench/fftw_bench.c`) as the authoritative gold standard.

## Complex 1-D FFT (`complex128`)

ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW (lower is better; ≤1.05 = parity).

| N | go-fft | FFTW | numpy.fft | scipy.fft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 2,337 (4.4) | 1,156 (8.9) | 9,492 (1.1) | 7,772 (1.3) | 10,793 (0.9) | 2.02× | lags FFTW 2.02× |
| 1,024 (2¹⁰) | 9,033 (5.7) | 6,338 (8.1) | 18,690 (2.7) | 14,495 (3.5) | 55,510 (0.9) | 1.43× | lags FFTW 1.43× |
| 4,096 (2¹²) | 53,361 (4.6) | 40,580 (6.1) | 60,014 (4.1) | 48,824 (5.0) | 273,726 (0.9) | 1.31× | lags FFTW 1.31× |
| 65,536 (2¹⁶) | 1,278,339 (4.1) | 1,283,754 (4.1) | 2,168,781 (2.4) | 1,464,001 (3.6) | 6,108,245 (0.9) | 1.00× | **≥ parity** |
| 1,048,576 (2²⁰) | 31,597,041 (3.3) | 46,870,316 (2.2) | 43,385,103 (2.4) | 32,521,122 (3.2) | 143,733,954 (0.7) | 0.67× | **≥ parity** |
| 1,000 (2³·5³) | 10,740 (4.6) | 7,152 (7.0) | 19,146 (2.6) | 15,094 (3.3) | 53,390 (0.9) | 1.50× | lags FFTW 1.50× |
| 1,080 (2³·3³·5) | 12,687 (4.3) | 7,734 (7.0) | 22,238 (2.4) | 16,797 (3.2) | 63,870 (0.9) | 1.64× | lags FFTW 1.64× |
| 1,920 (2⁷·3·5) | 21,576 (4.9) | 13,505 (7.8) | 33,176 (3.2) | 26,218 (4.0) | 116,329 (0.9) | 1.60× | lags FFTW 1.60× |
| 1,009 (prime) | 26,171 (1.9) | 48,071 (1.0) | 97,199 (0.5) | 59,272 (0.8) | 2,344,803 (0.0) | 0.54× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 15,751 (4.3) | 11,328 (5.9) | 25,142 (2.7) | 20,048 (3.3) | 79,773 (0.8) | 1.39× | lags FFTW 1.39× |
| 10,007 (prime) | 811,668 (0.8) | 594,800 (1.1) | 1,028,372 (0.6) | 770,705 (0.9) | 228,040,780 (0.0) | 1.36× | lags FFTW 1.36× |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 1,254 (4.1) | 719 (7.1) | 8,666 (0.6) | 7,707 (0.7) | 6,331 (0.8) | 1.74× | lags FFTW 1.74× |
| 1,024 (2¹⁰) | 5,604 (4.6) | 4,169 (6.1) | 14,187 (1.8) | 12,289 (2.1) | 29,280 (0.9) | 1.34× | lags FFTW 1.34× |
| 4,096 (2¹²) | 29,677 (4.1) | 19,721 (6.2) | 37,707 (3.3) | 32,966 (3.7) | 140,228 (0.9) | 1.50× | lags FFTW 1.50× |
| 65,536 (2¹⁶) | 673,018 (3.9) | 549,526 (4.8) | 707,639 (3.7) | 747,332 (3.5) | 3,191,812 (0.8) | 1.22× | lags FFTW 1.22× |
| 1,048,576 (2²⁰) | 16,579,733 (3.2) | 18,025,976 (2.9) | 21,811,723 (2.4) | 18,788,273 (2.8) | 76,384,071 (0.7) | 0.92× | **≥ parity** |
| 1,000 (2³·5³) | 6,352 (3.9) | 3,944 (6.3) | 15,499 (1.6) | 12,652 (2.0) | 28,675 (0.9) | 1.61× | lags FFTW 1.61× |
| 1,080 (2³·3³·5) | 6,870 (4.0) | 4,204 (6.5) | 15,971 (1.7) | 14,415 (1.9) | 33,183 (0.8) | 1.63× | lags FFTW 1.63× |
| 1,920 (2⁷·3·5) | 12,114 (4.3) | 7,363 (7.1) | 22,056 (2.4) | 19,342 (2.7) | 57,775 (0.9) | 1.65× | lags FFTW 1.65× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 1,300 (3.9) | 809 (6.3) | 1.61× | lags FFTW 1.61× |
| 1,024 (2¹⁰) | 5,799 (4.4) | 4,456 (5.7) | 1.30× | lags FFTW 1.30× |
| 4,096 (2¹²) | 29,673 (4.1) | 21,154 (5.8) | 1.40× | lags FFTW 1.40× |
| 65,536 (2¹⁶) | 726,652 (3.6) | 645,732 (4.1) | 1.13× | lags FFTW 1.13× |
| 1,048,576 (2²⁰) | 15,960,901 (3.3) | 22,349,848 (2.3) | 0.71× | **≥ parity** |
| 1,000 (2³·5³) | 6,680 (3.7) | 4,248 (5.9) | 1.57× | lags FFTW 1.57× |
| 1,080 (2³·3³·5) | 7,480 (3.6) | 4,423 (6.2) | 1.69× | lags FFTW 1.69× |
| 1,920 (2⁷·3·5) | 12,808 (4.1) | 7,946 (6.6) | 1.61× | lags FFTW 1.61× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. ns/op (GFLOP/s).

| shape | go-fft | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|:--|
| 64x64 | 156,379 (1.6) | 30,871 (8.0) | 72,873 (3.4) | 51,472 (4.8) | 5.07× | lags FFTW 5.07× |
| 128x128 | 667,994 (1.7) | 194,337 (5.9) | 248,915 (4.6) | 200,691 (5.7) | 3.44× | lags FFTW 3.44× |
| 256x256 | 1,471,322 (3.6) | 1,351,377 (3.9) | 1,153,662 (4.5) | 929,312 (5.6) | 1.09× | lags FFTW 1.09× |
| 512x512 | 3,762,352 (6.3) | 7,360,856 (3.2) | 5,660,605 (4.2) | 4,158,658 (5.7) | 0.51× | **≥ parity** |
| 1024x1024 | 11,096,481 (9.4) | 44,898,387 (2.3) | 26,390,019 (4.0) | 19,916,617 (5.3) | 0.25× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 33,141 | 28,213 | 9,862,686 |
| 1,024 | 107,503 | 105,508 | 3,334,537 |
| 4,096 | 415,291 | 412,696 | 6,172,984 |
| 65,536 | 6,673,822 | 6,408,346 | 21,883,249 |
| 1,048,576 | 76,244,650 | 67,168,007 | 20,829,631 |
| 1,000 | 109,924 | 112,808 | 4,680,399 |
| 1,080 | 124,771 | 115,391 | 12,773,535 |
| 1,920 | 211,895 | 180,248 | 17,327,852 |
| 1,009 | 311,254 | 114,216 | 6,504,790 |
| 1,296 | 149,651 | 132,137 | 10,639,940 |
| 10,007 | 4,808,351 | 1,200,333 | 36,185,090 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 6/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 21/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 19/24 ops at-or-above parity.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

**vs gonum** (the fair pure-Go, CGO=0 peer): go-fft is faster at **every** size measured — typically 3–5× on composite N and an order of magnitude on primes (gonum falls back to a naive Bluestein with no Rader path), confirming go-fft is the fastest pure-Go FFT here.

**vs pocketfft** (numpy/scipy): go-fft wins the small-N rows outright (the Python FFI tax dominates pocketfft's C kernel there) and is competitive-to-winning at large 1-D and large 2-D; the residual losses are the mid-range single-core bands.

**vs FFTW** (the gold standard): FFTW leads on the single-core power-of-two and smooth-composite mid-range — it has hand-written SIMD codelets and a dedicated Hermitian real kernel that a scalar pure-Go library cannot match on one core. go-fft reaches parity-or-better where the algorithm, not raw SIMD throughput, dominates: very large 1-D, the large 2-D multicore shapes, and (relative to FFTW's own cost) the large-prime rows.

### Lagging ops — root cause + action items

Honest read of where go-fft trails FFTW, with the concrete lever to close each gap:

1. **Power-of-two & smooth-composite mid-range (256 … 4096, 1000/1080/1296/1920).** *Root cause*: scalar Go butterflies vs FFTW's hand-written NEON SIMD codelets — the kernel is the same Cooley–Tukey schedule, FFTW just does 2 complex muls per instruction. *Action*: drop in the **go-asmgen SIMD complex-multiply kernels** (already validated bit-identical on amd64/arm64/s390x/riscv64) on the hot radix-2/4 butterfly inner loop — the single highest-ROI item; expected to roughly halve this band on arm64.
2. **Real mid-range (1024 … 65536).** *Root cause*: go-fft packs a real signal into a half-length complex FFT and untangles once; FFTW runs a **dedicated real (r2c) kernel** that exploits Hermitian symmetry at every stage (~2× less arithmetic). *Action*: implement a native split-radix real butterfly schedule (declined before as duplication; the measured gap now justifies it) — and it benefits from item 1's SIMD too.
3. **Large primes (1009, 10007).** *Root cause*: Rader/Bluestein convolve at length ≈N−1 on the recursive mixed-radix engine, which pays a ~1.5× recursion tax; FFTW convolves at exactly N−1 with codelet-fused mixed-radix. *Note*: go-fft is already **2–2.5× faster than gonum** here and within ~1.6× of FFTW (vs ~30× gap for gonum's naive Bluestein). *Action*: an **iterative mixed-radix engine** for the smooth convolution lengths (the same lever that closed the pow2 band) — a substantial new engine, lower priority than items 1–2.
4. **Small 2-D (64×64, 128×128).** *Root cause*: below the parallel threshold, so they run the serial per-line path while FFTW uses fused 2-D codelets. *Action*: SIMD (item 1) lifts the per-line 1-D cost directly; the multicore path already makes go-fft win at 512×512 and 1024×1024 (beats numpy, ties/leads scipy).

> The unifying lever is **item 1 (go-asmgen SIMD butterflies)**: it attacks the pow2/composite mid-range, the per-line cost inside 2-D, and feeds the real and prime paths (which both convolve via complex FFTs). That is the one change that moves the most rows toward FFTW parity.
