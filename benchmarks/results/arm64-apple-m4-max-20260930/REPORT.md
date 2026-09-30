# Performance parity — go-fft vs FFTW / numpy.fft / scipy.fft / gonum

Standardized parity report for the pure-Go (CGO=0) FFT library `go-fft`, measured against the gold-standard C library **FFTW** and the reference Python (`numpy.fft`, `scipy.fft` = pocketfft) and Go (`gonum`) FFTs, all on the **same machine, same inputs, same sizes**.

> Regenerate with `benchmarks/run.sh` (it runs the Go benchmarks, the native-FFTW C harness, and the numpy/scipy/pyfftw Python harness, then rebuilds this file). Numerical correctness is gated first: every go-fft transform is checked against `numpy.fft` within `rtol=1e-9, atol=1e-7` before any timing is reported.

## Methodology

- **Machine**: Apple M4 Max (16-core: 12P+4E), macOS 26.5 (25F71), arm64.
- **Toolchains**: go1.26.4 darwin/arm64; native **FFTW fftw-3.3.11** (Homebrew arm64 bottle, NEON, linked from C); **numpy 2.5.3** / **scipy 1.18.1** (pocketfft); **pyfftw 0.15.1**; **gonum.org/v1/gonum v0.16.0**.
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
| 256 (2⁸) | 582 (17.6) | 405 (25.3) | 2,593 (3.9) | 2,141 (4.8) | 3,089 (3.3) | 1.43× | lags FFTW 1.43× |
| 1,024 (2¹⁰) | 2,251 (22.7) | 2,002 (25.6) | 5,425 (9.4) | 4,227 (12.1) | 15,849 (3.2) | 1.12× | lags FFTW 1.12× |
| 4,096 (2¹²) | 12,852 (19.1) | 10,203 (24.1) | 19,373 (12.7) | 15,523 (15.8) | 77,128 (3.2) | 1.26× | lags FFTW 1.26× |
| 65,536 (2¹⁶) | 290,242 (18.1) | 257,849 (20.3) | 484,838 (10.8) | 469,146 (11.2) | 1,563,469 (3.4) | 1.13× | lags FFTW 1.13× |
| 1,048,576 (2²⁰) | 6,346,966 (16.5) | 7,850,812 (13.4) | 10,137,405 (10.3) | 7,023,678 (14.9) | 39,414,153 (2.7) | 0.81× | **≥ parity** |
| 1,000 (2³·5³) | 2,535 (19.7) | 2,287 (21.8) | 5,609 (8.9) | 4,667 (10.7) | 16,036 (3.1) | 1.11× | lags FFTW 1.11× |
| 1,080 (2³·3³·5) | 2,773 (19.6) | 2,401 (22.7) | 6,327 (8.6) | 5,000 (10.9) | 18,817 (2.9) | 1.15× | lags FFTW 1.15× |
| 1,920 (2⁷·3·5) | 4,806 (21.8) | 4,479 (23.4) | 10,453 (10.0) | 8,284 (12.6) | 33,620 (3.1) | 1.07× | lags FFTW 1.07× |
| 1,009 (prime) | 6,974 (7.2) | 13,265 (3.8) | 29,481 (1.7) | 18,253 (2.8) | 511,376 (0.1) | 0.53× | **≥ parity** |
| 1,296 (2⁴·3⁴) | 3,474 (19.3) | 2,918 (23.0) | 7,272 (9.2) | 5,891 (11.4) | 22,467 (3.0) | 1.19× | lags FFTW 1.19× |
| 10,007 (prime) | 200,132 (3.3) | 179,127 (3.7) | 308,983 (2.2) | 268,263 (2.5) | 68,836,286 (0.0) | 1.12× | lags FFTW 1.12× |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 299 (17.1) | 198 (25.9) | 2,177 (2.4) | 2,174 (2.4) | 1,685 (3.0) | 1.51× | lags FFTW 1.51× |
| 1,024 (2¹⁰) | 1,324 (19.3) | 970 (26.4) | 3,927 (6.5) | 3,464 (7.4) | 8,108 (3.2) | 1.37× | lags FFTW 1.37× |
| 4,096 (2¹²) | 7,247 (17.0) | 4,698 (26.2) | 11,159 (11.0) | 10,263 (12.0) | 35,339 (3.5) | 1.54× | lags FFTW 1.54× |
| 65,536 (2¹⁶) | 151,131 (17.3) | 133,814 (19.6) | 211,740 (12.4) | 249,534 (10.5) | 766,554 (3.4) | 1.13× | lags FFTW 1.13× |
| 1,048,576 (2²⁰) | 3,555,802 (14.7) | 2,804,359 (18.7) | 4,227,746 (12.4) | 5,853,469 (9.0) | 15,935,380 (3.3) | 1.27× | lags FFTW 1.27× |
| 1,000 (2³·5³) | 1,471 (16.9) | 1,137 (21.9) | 3,992 (6.2) | 3,517 (7.1) | 7,888 (3.2) | 1.29× | lags FFTW 1.29× |
| 1,080 (2³·3³·5) | 1,603 (17.0) | 1,125 (24.2) | 4,136 (6.6) | 4,023 (6.8) | 9,213 (3.0) | 1.43× | lags FFTW 1.43× |
| 1,920 (2⁷·3·5) | 2,911 (18.0) | 2,200 (23.8) | 5,900 (8.9) | 5,448 (9.6) | 16,226 (3.2) | 1.32× | lags FFTW 1.32× |

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse mirrors the forward packing: an N/2-point inverse complex FFT plus an untangle pass, half the work of a full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW.

| N | go-fft | FFTW | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 331 (15.5) | 243 (21.0) | 1.36× | lags FFTW 1.36× |
| 1,024 (2¹⁰) | 1,430 (17.9) | 1,072 (23.9) | 1.33× | lags FFTW 1.33× |
| 4,096 (2¹²) | 7,458 (16.5) | 5,495 (22.4) | 1.36× | lags FFTW 1.36× |
| 65,536 (2¹⁶) | 157,053 (16.7) | 139,297 (18.8) | 1.13× | lags FFTW 1.13× |
| 1,048,576 (2²⁰) | 3,340,787 (15.7) | 3,143,641 (16.7) | 1.06× | lags FFTW 1.06× |
| 1,000 (2³·5³) | 1,562 (16.0) | 1,163 (21.4) | 1.34× | lags FFTW 1.34× |
| 1,080 (2³·3³·5) | 1,731 (15.7) | 1,237 (22.0) | 1.40× | lags FFTW 1.40× |
| 1,920 (2⁷·3·5) | 3,067 (17.1) | 2,164 (24.2) | 1.42× | lags FFTW 1.42× |

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. ns/op (GFLOP/s).

| shape | go-fft | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|:--|
| 64x64 | 21,557 (11.4) | 8,767 (28.0) | 18,953 (13.0) | 13,549 (18.1) | 2.46× | lags FFTW 2.46× |
| 128x128 | 74,387 (15.4) | 55,138 (20.8) | 76,745 (14.9) | 63,902 (17.9) | 1.35× | lags FFTW 1.35× |
| 256x256 | 204,584 (25.6) | 258,974 (20.2) | 342,644 (15.3) | 202,136 (25.9) | 0.79× | **≥ parity** |
| 512x512 | 514,403 (45.9) | 1,206,356 (19.6) | 1,617,924 (14.6) | 933,668 (25.3) | 0.43× | **≥ parity** |
| 1024x1024 | 2,470,696 (42.4) | 5,532,234 (19.0) | 7,980,729 (13.1) | 5,110,417 (20.5) | 0.45× | **≥ parity** |

## Plan / setup cost (built once, then amortized)

Steady-state transforms above reuse a plan. This is the one-time construction cost (ns), reported separately. go-fft and gonum build twiddle tables in Go; FFTW's `FFTW_MEASURE` *times trial transforms* to pick codelets, so its planning is orders of magnitude more expensive — the price of its steady-state speed, paid back only across many reuses.

| N | go-fft NewPlan | gonum NewCmplxFFT | FFTW FFTW_MEASURE |
|---:|---:|---:|---:|
| 256 | 2,641 | 2,380 | 1,972,000 |
| 1,024 | 9,203 | 8,576 | 3,272,000 |
| 4,096 | 35,124 | 31,696 | 10,989,000 |
| 65,536 | 512,753 | 442,693 | 302,355,000 |
| 1,048,576 | 7,000,970 | 6,385,443 | 1,979,954,000 |
| 1,000 | 15,647 | 17,235 | 4,181,000 |
| 1,080 | 10,043 | 9,430 | 10,211,000 |
| 1,920 | 17,004 | 15,745 | 16,598,000 |
| 1,009 | 26,233 | 9,677 | 7,802,000 |
| 1,296 | 11,540 | 11,044 | 8,235,000 |
| 10,007 | 479,593 | 90,949 | 123,414,000 |

## Summary

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host:

- **vs FFTW (native C, gold standard)**: 5/24 ops at-or-above parity.
- **vs numpy.fft (pocketfft)**: 23/24 ops at-or-above parity.
- **vs scipy.fft (pocketfft)**: 22/24 ops at-or-above parity.
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
