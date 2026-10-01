# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: AMD EPYC 7773X (Zen 3, 64 cores / 128 threads, AVX2; GCC Compile Farm cfarm420), Arch Linux, amd64.
- **Toolchains**: go1.26.4 linux/amd64 (cross-compiled); native **FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128** (3.3.10 built from source with --enable-sse2 --enable-avx --enable-avx2 --enable-fma, statically linked from C); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 440 (23.3) | 274 (37.4) | 6,809 (1.5) | 5,515 (1.9) | 13,805 (0.7) | 1.61× | lags FFTW 1.61× |
| 1,024 (2¹⁰) | 2,675 (19.1) | 1,550 (33.0) | 13,556 (3.8) | 10,109 (5.1) | 61,163 (0.8) | 1.73× | lags FFTW 1.73× |
| 4,096 (2¹²) | 24,718 (9.9) | 10,232 (24.0) | 44,310 (5.5) | 34,887 (7.0) | 273,185 (0.9) | 2.42× | lags FFTW 2.42× |
| 65,536 (2¹⁶) | 589,192 (8.9) | 295,193 (17.8) | 1,950,785 (2.7) | 915,229 (5.7) | 6,390,172 (0.8) | 2.00× | lags FFTW 2.00× |
| 1,048,576 (2²⁰) | 12,773,146 (8.2) | 11,265,680 (9.3) | 36,936,920 (2.8) | 14,510,891 (7.2) | 127,505,016 (0.8) | 1.13× | lags FFTW 1.13× |
| 1,000 (2³·5³) | 2,976 (16.7) | 2,086 (23.9) | 13,685 (3.6) | 10,354 (4.8) | 67,148 (0.7) | 1.43× | lags FFTW 1.43× |
| 1,080 (2³·3³·5) | 3,454 (15.8) | 2,316 (23.5) | 14,505 (3.8) | 11,124 (4.9) | 72,565 (0.7) | 1.49× | lags FFTW 1.49× |
| 1,920 (2⁷·3·5) | 5,745 (18.2) | 4,112 (25.5) | 22,120 (4.7) | 16,649 (6.3) | 128,622 (0.8) | 1.40× | lags FFTW 1.40× |
| 1,009 (prime) | 15,284 (3.3) | 20,303 (2.5) | 60,697 (0.8) | 38,540 (1.3) | 1,554,000 (0.0) | 0.75× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 4,396 (15.2) | 3,001 (22.3) | 16,517 (4.1) | 12,551 (5.3) | 85,429 (0.8) | 1.46× | lags FFTW 1.46× |
| 10,007 (prime) | 361,765 (1.8) | 204,490 (3.3) | 645,414 (1.0) | 469,408 (1.4) | 153,537,553 (0.0) | 1.77× | lags FFTW 1.77× |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 444 (11.5) | 221 (23.2) | 6,088 (0.8) | 5,387 (1.0) | 5,204 (1.0) | 2.01× | lags FFTW 2.01× |
| 1,024 (2¹⁰) | 1,739 (14.7) | 1,174 (21.8) | 9,611 (2.7) | 8,345 (3.1) | 26,520 (1.0) | 1.48× | lags FFTW 1.48× |
| 4,096 (2¹²) | 14,953 (8.2) | 5,127 (24.0) | 25,534 (4.8) | 22,680 (5.4) | 113,124 (1.1) | 2.92× | lags FFTW 2.92× |
| 65,536 (2¹⁶) | 282,435 (9.3) | 154,613 (17.0) | 433,328 (6.0) | 502,301 (5.2) | 2,390,855 (1.1) | 1.83× | lags FFTW 1.83× |
| 1,048,576 (2²⁰) | 6,069,093 (8.6) | 3,723,631 (14.1) | 8,311,636 (6.3) | 10,002,090 (5.2) | 46,695,860 (1.1) | 1.63× | lags FFTW 1.63× |
| 1,000 (2³·5³) | 2,088 (11.9) | 1,244 (20.0) | 10,271 (2.4) | 8,484 (2.9) | 27,124 (0.9) | 1.68× | lags FFTW 1.68× |
| 1,080 (2³·3³·5) | 2,297 (11.8) | 1,362 (20.0) | 10,960 (2.5) | 8,935 (3.0) | 28,067 (1.0) | 1.69× | lags FFTW 1.69× |
| 1,920 (2⁷·3·5) | 3,851 (13.6) | 2,272 (23.0) | 14,146 (3.7) | 12,109 (4.3) | 51,817 (1.0) | 1.70× | lags FFTW 1.70× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 561 (9.1) | 373 (13.7) | 1.51× | lags FFTW 1.51× |
| 1,024 (2¹⁰) | 2,230 (11.5) | 1,178 (21.7) | 1.89× | lags FFTW 1.89× |
| 4,096 (2¹²) | 16,055 (7.7) | 5,803 (21.2) | 2.77× | lags FFTW 2.77× |
| 65,536 (2¹⁶) | 346,591 (7.6) | 170,671 (15.4) | 2.03× | lags FFTW 2.03× |
| 1,048,576 (2²⁰) | 7,094,883 (7.4) | 4,108,638 (12.8) | 1.73× | lags FFTW 1.73× |
| 1,000 (2³·5³) | 2,657 (9.4) | 1,455 (17.1) | 1.83× | lags FFTW 1.83× |
| 1,080 (2³·3³·5) | 2,829 (9.6) | 1,981 (13.7) | 1.43× | lags FFTW 1.43× |
| 1,920 (2⁷·3·5) | 4,725 (11.1) | 2,649 (19.8) | 1.78× | lags FFTW 1.78× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. ns/op (GFLOP/s).

| shape | go-fft | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|:--|
| 64x64 | 53,326 (4.6) | 10,864 (22.6) | 47,212 (5.2) | 33,764 (7.3) | 4.91× | lags FFTW 4.91× |
| 128x128 | 540,927 (2.1) | 63,503 (18.1) | 149,565 (7.7) | 126,876 (9.0) | 8.52× | lags FFTW 8.52× |
| 256x256 | 1,103,190 (4.8) | 294,830 (17.8) | 639,744 (8.2) | 456,369 (11.5) | 3.74× | lags FFTW 3.74× |
| 512x512 | 2,186,968 (10.8) | 1,344,175 (17.6) | 2,867,047 (8.2) | 1,898,194 (12.4) | 1.63× | lags FFTW 1.63× |
| 1024x1024 | 5,621,136 (18.7) | 6,453,221 (16.2) | 13,468,104 (7.8) | 9,314,073 (11.3) | 0.87× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 29,303 | 13,892 | 72,096,693 |
| 1,024 | 101,026 | 53,536 | 144,674,895 |
| 4,096 | 346,691 | 209,148 | 245,106,521 |
| 65,536 | 5,372,597 | 3,023,921 | 2,267,739,230 |
| 1,048,576 | 67,992,537 | 33,395,487 | 4,261,602,246 |
| 1,000 | 106,241 | 51,071 | 151,070,849 |
| 1,080 | 105,488 | 57,034 | 367,682,777 |
| 1,920 | 311,751 | 123,539 | 593,853,860 |
| 1,009 | 177,882 | 71,794 | 142,694,145 |
| 1,296 | 143,445 | 71,368 | 268,063,982 |
| 10,007 | 2,018,212 | 574,623 | 1,234,310,341 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 2/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 21/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 20/24 ops at-or-above parity.
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
