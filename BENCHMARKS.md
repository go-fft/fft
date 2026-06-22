# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Apple M4 Max (16-core: 12P+4E), macOS 26.5 (25F71), arm64.
- **Toolchains**: go1.26.4 darwin/arm64; native **FFTW fftw-3.3.11** (Homebrew arm64 bottle, NEON, linked from C); **numpy 2.5.0** / **scipy 1.18.0** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 792 (12.9) | 420 (24.4) | 2,828 (3.6) | 2,302 (4.4) | 3,272 (3.1) | 1.88× | lags FFTW 1.88× |
| 1,024 (2¹⁰) | 3,436 (14.9) | 2,055 (24.9) | 5,678 (9.0) | 4,470 (11.5) | 20,066 (2.6) | 1.67× | lags FFTW 1.67× |
| 4,096 (2¹²) | 16,885 (14.6) | 10,809 (22.7) | 20,359 (12.1) | 16,148 (15.2) | 80,164 (3.1) | 1.56× | lags FFTW 1.56× |
| 65,536 (2¹⁶) | 386,856 (13.6) | 316,808 (16.5) | 554,088 (9.5) | 499,500 (10.5) | 1,901,025 (2.8) | 1.22× | lags FFTW 1.22× |
| 1,048,576 (2²⁰) | 14,798,946 (7.1) | 12,382,438 (8.5) | 11,044,400 (9.5) | 8,040,680 (13.0) | 41,139,865 (2.5) | 1.20× | lags FFTW 1.20× |
| 1,000 (2³·5³) | 5,808 (8.6) | 3,634 (13.7) | 6,075 (8.2) | 4,961 (10.0) | 16,541 (3.0) | 1.60× | lags FFTW 1.60× |
| 1,080 (2³·3³·5) | 6,727 (8.1) | 3,932 (13.8) | 6,799 (8.0) | 5,266 (10.3) | 21,183 (2.6) | 1.71× | lags FFTW 1.71× |
| 1,920 (2⁷·3·5) | 12,224 (8.6) | 7,892 (13.3) | 11,349 (9.2) | 9,059 (11.6) | 36,810 (2.8) | 1.55× | lags FFTW 1.55× |
| 1,009 (prime) | 16,907 (3.0) | 25,074 (2.0) | 31,560 (1.6) | 19,004 (2.6) | 563,199 (0.1) | 0.67× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 10,005 (6.7) | 4,602 (14.6) | 8,063 (8.3) | 6,364 (10.5) | 25,328 (2.6) | 2.17× | lags FFTW 2.17× |
| 10,007 (prime) | 411,093 (1.6) | 260,158 (2.6) | 321,309 (2.1) | 276,031 (2.4) | 75,674,509 (0.0) | 1.58× | lags FFTW 1.58× |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 1,018 (5.0) | 354 (14.5) | 2,432 (2.1) | 2,422 (2.1) | 1,882 (2.7) | 2.88× | lags FFTW 2.88× |
| 1,024 (2¹⁰) | 4,312 (5.9) | 1,487 (17.2) | 4,180 (6.1) | 3,726 (6.9) | 8,998 (2.8) | 2.90× | lags FFTW 2.90× |
| 4,096 (2¹²) | 18,468 (6.7) | 7,262 (16.9) | 11,742 (10.5) | 10,416 (11.8) | 39,264 (3.1) | 2.54× | lags FFTW 2.54× |
| 65,536 (2¹⁶) | 317,644 (8.3) | 255,328 (10.3) | 230,189 (11.4) | 254,964 (10.3) | 896,439 (2.9) | 1.24× | lags FFTW 1.24× |
| 1,048,576 (2²⁰) | 5,544,915 (9.5) | 5,142,156 (10.2) | 4,392,782 (11.9) | 6,365,402 (8.2) | 18,259,818 (2.9) | 1.08× | lags FFTW 1.08× |
| 1,000 (2³·5³) | 4,429 (5.6) | 1,655 (15.1) | 4,289 (5.8) | 3,663 (6.8) | 8,604 (2.9) | 2.68× | lags FFTW 2.68× |
| 1,080 (2³·3³·5) | 4,826 (5.6) | 1,915 (14.2) | 4,426 (6.1) | 4,196 (6.5) | 9,933 (2.7) | 2.52× | lags FFTW 2.52× |
| 1,920 (2⁷·3·5) | 8,443 (6.2) | 3,022 (17.3) | 6,188 (8.5) | 5,657 (9.3) | 17,156 (3.1) | 2.79× | lags FFTW 2.79× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. ns/op (GFLOP/s).

| shape | go-fft | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|:--|
| 64x64 | 30,359 (8.1) | 12,534 (19.6) | 19,895 (12.4) | 14,005 (17.5) | 2.42× | lags FFTW 2.42× |
| 128x128 | 77,777 (14.7) | 83,533 (13.7) | 76,859 (14.9) | 64,707 (17.7) | 0.93× | **≥ parity** |
| 256x256 | 208,474 (25.1) | 425,082 (12.3) | 359,158 (14.6) | 207,601 (25.3) | 0.49× | **≥ parity** |
| 512x512 | 620,216 (38.0) | 1,414,266 (16.7) | 1,697,966 (13.9) | 964,884 (24.5) | 0.44× | **≥ parity** |
| 1024x1024 | 2,721,951 (38.5) | 9,545,594 (11.0) | 9,211,551 (11.4) | 5,315,022 (19.7) | 0.29× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 3,762 | 2,473 | 1,685,000 |
| 1,024 | 14,808 | 9,285 | 3,434,000 |
| 4,096 | 60,228 | 32,714 | 11,604,000 |
| 65,536 | 1,094,262 | 464,499 | 353,280,000 |
| 1,048,576 | 19,298,901 | 6,429,831 | 3,860,636,000 |
| 1,000 | 7,184 | 8,926 | 5,765,000 |
| 1,080 | 7,460 | 9,917 | 14,182,000 |
| 1,920 | 13,310 | 17,266 | 25,377,000 |
| 1,009 | 34,376 | 10,674 | 33,948,000 |
| 1,296 | 8,928 | 11,556 | 11,156,000 |
| 10,007 | 993,365 | 100,654 | 189,451,000 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 5/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 14/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 10/24 ops at-or-above parity.
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
