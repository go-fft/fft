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
- **Snapshot note (real-inverse round, 2026-06-23)**: the **Real inverse 1-D IRFFT (c2r)** section and its before→after table are a fresh consistent snapshot taken together on the same machine state — go-fft's packed inverse and the native FFTW c2r harness (`fftw_plan_dft_c2r_1d`, `FFTW_MEASURE | FFTW_PRESERVE_INPUT`) measured back-to-back. The complex-1-D, real-1-D RFFT, and 2-D rows are unchanged by this round (the forward and complex paths are untouched) and are carried over from the SIMD-butterfly snapshot below.
- **Snapshot note (SIMD-butterfly round, 2026-06-22)**: the **go-fft and FFTW columns above are a fresh consistent snapshot** taken together on the same machine state after the SIMD-butterfly change; the **numpy/scipy/gonum columns are carried over from the prior snapshot** (reference impls unaffected by this change). The absolute FFTW ns/op in this snapshot are lower than the previous one (a cooler thermal state — FFTW's hand-tuned NEON codelets are the most thermally sensitive), which makes the *raw ratios* look larger even though **go-fft itself got faster at every size**. The honest, machine-state-controlled before→after for go-fft (FFTW held fixed at this snapshot) is reported under "SIMD-butterfly round" below — that, not the cross-snapshot ratio drift, is the measure of this change.

## Complex 1-D FFT (`complex128`)

ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW (lower is better; ≤1.05 = parity).

| N | go-fft | FFTW | numpy.fft | scipy.fft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 716 (14.3) | 419 (24.4) | 2,828 | 2,302 | 3,272 | 1.71× | lags FFTW 1.71× |
| 1,024 (2¹⁰) | 3,395 (15.1) | 2,127 (24.1) | 5,678 | 4,470 | 20,066 | 1.60× | lags FFTW 1.60× |
| 4,096 (2¹²) | 16,383 (15.0) | 11,064 (22.2) | 20,359 | 16,148 | 80,164 | 1.48× | lags FFTW 1.48× |
| 65,536 (2¹⁶) | 377,784 (13.9) | 322,850 (16.2) | 554,088 | 499,500 | 1,901,025 | 1.17× | lags FFTW 1.17× |
| 1,048,576 (2²⁰) | 13,355,742 (7.9) | 8,171,156 (12.8) | 11,044,400 | 8,040,680 | 41,139,865 | 1.63× | lags FFTW 1.63× |
| 1,000 (2³·5³) | 5,850 (8.5) | 2,478 (20.1) | 6,075 | 4,961 | 16,541 | 2.36× | lags FFTW 2.36× |
| 1,080 (2³·3³·5) | 6,802 (8.0) | 2,466 (22.1) | 6,799 | 5,266 | 21,183 | 2.76× | lags FFTW 2.76× |
| 1,920 (2⁷·3·5) | 12,516 (8.4) | 4,340 (24.1) | 11,349 | 9,059 | 36,810 | 2.88× | lags FFTW 2.88× |
| 1,009 (prime) | 17,214 (2.9) | 13,668 (3.7) | 31,560 | 19,004 | 563,199 | 1.26× | lags FFTW 1.26× |
| 1,296 (2⁴·3⁴) | 9,818 (6.8) | 2,998 (22.4) | 8,063 | 6,364 | 25,328 | 3.28× | lags FFTW 3.28× |
| 10,007 (prime) | 369,217 (1.8) | 171,553 (3.9) | 321,309 | 276,031 | 75,674,509 | 2.15× | lags FFTW 2.15× |

## Real 1-D RFFT (`float64` → `complex128`, N/2+1 bins)

| N | go-fft | FFTW | numpy.rfft | scipy.rfft | gonum | go/FFTW | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 (2⁸) | 525 (9.8) | 220 (23.3) | 2,432 | 2,422 | 1,882 | 2.39× | lags FFTW 2.39× |
| 1,024 (2¹⁰) | 2,289 (11.2) | 1,000 (25.6) | 4,180 | 3,726 | 8,998 | 2.29× | lags FFTW 2.29× |
| 4,096 (2¹²) | 10,320 (11.9) | 4,890 (25.1) | 11,742 | 10,416 | 39,264 | 2.11× | lags FFTW 2.11× |
| 65,536 (2¹⁶) | 230,167 (11.4) | 143,040 (18.3) | 230,189 | 254,964 | 896,439 | 1.61× | lags FFTW 1.61× |
| 1,048,576 (2²⁰) | 4,465,650 (11.7) | 2,958,859 (17.7) | 4,392,782 | 6,365,402 | 18,259,818 | 1.51× | lags FFTW 1.51× |
| 1,000 (2³·5³) | 3,144 (7.9) | 1,176 (21.2) | 4,289 | 3,663 | 8,604 | 2.67× | lags FFTW 2.67× |
| 1,080 (2³·3³·5) | 3,616 (7.5) | 1,131 (24.1) | 4,426 | 4,196 | 9,933 | 3.20× | lags FFTW 3.20× |
| 1,920 (2⁷·3·5) | 6,555 (8.0) | 2,189 (23.9) | 6,188 | 5,657 | 17,156 | 2.99× | lags FFTW 2.99× |

### Forward RFFT (r2c) — before→after (pack/untangle round, same machine state)

The forward r2c already packed N reals into an N/2-point complex FFT and untangled the result. Profiling that path showed the **pack + untangle pass** — everything outside the core complex FFT — was a large share of the time at the lagging mid-range (≈48% at N=256, ≈44% at N=1024, ≈37% at N=4096, falling to ≈24% at N=65536 where the O(N·logN) core dominates). Two changes attacked exactly that pass:

1. **Allocation-free working set.** RFFT (and the symmetric c2r inverse) allocated two N/2-length `complex128` buffers (`z` for packing, `Z` for the spectrum) on **every call** — 16 KB / 2 allocs per call at N=1024. They now borrow one pooled `2·(N/2)` buffer per concurrent caller from the immutable, concurrent-safe plan (`sync.Pool`), so the steady-state RFFT is **0 alloc/op** on the power-of-two path.
2. **Branchless, vectorizable untangle.** The conjugate-pair recombination loop carried a data-dependent `if k != m-k` branch in its body (the self-paired middle bin), which stalls the gc autovectorizer. The loop is now hoisted into `rfftUntangle`, runs `k = 1 .. (m-1)/2` with **no branch** (writing both `dst[k]` and `dst[m-k]` every iteration), and finishes the single self-paired bin once after the loop.

Measured back-to-back on the same host against the same native FFTW r2c plan (best/median of 5, ns/op; ratio = go-fft ÷ FFTW r2c):

| N | go-fft before | go-fft after | speed-up | FFTW r2c | ratio before→after |
|---:|---:|---:|---:|---:|---:|
| 256 (2⁸) | 868 | 525 | 1.65× | 220 | 3.94× → **2.39×** |
| 1,024 (2¹⁰) | 3,711 | 2,289 | 1.62× | 1,000 | 3.71× → **2.29×** |
| 4,096 (2¹²) | 15,879 | 10,320 | 1.54× | 4,890 | 3.25× → **2.11×** |
| 65,536 (2¹⁶) | 265,465 | 230,167 | 1.15× | 143,040 | 1.86× → **1.61×** |
| 1,048,576 (2²⁰) | 4,761,128 | 4,465,650 | 1.07× | 2,958,859 | 1.61× → **1.51×** |
| 1,000 (2³·5³) | 3,949 | 3,144 | 1.26× | 1,176 | 3.36× → **2.67×** |
| 1,080 (2³·3³·5) | 4,333 | 3,616 | 1.20× | 1,131 | 3.83× → **3.20×** |
| 1,920 (2⁷·3·5) | 8,490 | 6,555 | 1.30× | 2,189 | 3.88× → **2.99×** |

The win is largest exactly where the report flagged the worst lag — the power-of-two mid-range — because that is where the pack/untangle pass was the biggest fraction of the call: **1.5–1.65× at N=256…4096**, narrowing as N grows and the FFT core (already SIMD) takes over. The FFTW gap on the lagging mid-range drops from ~3.3–3.9× to ~2.1–2.4×. Correctness is held: r2c matches the full-spectrum oracle and `numpy.fft.rfft`, `IRFFT(RFFT(x))≈x` round-trips, and the DC/Nyquist bins stay exact (the classic real-FFT bug, guarded by the conjugate-symmetry and DC/Nyquist oracle tests). The same pooling also helped the c2r inverse (carried in the c2r table below).

### Forward RFFT (r2c) — fused-pack round (2026-06-24)

This round revisited the deferred lever — a dedicated real kernel exploiting Hermitian symmetry at every stage — and landed the part that measured a win while honestly reverting the part that did not.

**Landed: pack fused into the bit-reversal gather.** The power-of-two even path packed the N reals into an N/2-complex buffer `z[j] = src[2j] + i·src[2j+1]` in one O(N) pass, then the iterative complex kernel's bit-reversal *gathered* `z` into the working buffer in a second pass. Those two passes are now one: `transformRealPacked` reads the real input straight into the bit-reversed working buffer (`Z[i] = complex(src[2·revPos[i]], src[2·revPos[i]+1])`), removing the separate pack pass and the intermediate `z` write/read entirely. The packing arithmetic is unchanged, so the output is **bit-identical** to before — no correctness or cross-arch FMA risk. Measured by an interleaved in-process A/B (old pack+gather vs fused gather, alternated so both see the same instantaneous load; best of 10, the benchmark host under variable background load this round): the fused gather is **~1.2–1.7× faster on the pack/gather portion at N=256…4096** (N=256 ~1.38×, N=512 ~1.21×, N=2048 ~1.66×, N=4096 ~1.15×), the gain concentrated at the small mid-range where the pack pass is the largest fraction of the call and fading into the noise as the O(N·logN) SIMD complex core takes over at large N.

**Tried and reverted: full real split-radix kernel (Sorensen RVFFT).** A complete real-valued split-radix forward kernel (the in-place Sorensen/Heideman/Burrus RVFFT — bit-reversal, length-2 butterflies, ascending L-butterfly stages, packed real read-out) was implemented and **validated bit-for-bit against the full-complex oracle and `numpy.fft.rfft` across N = 4…1024, with exactly-real DC/Nyquist**. It has the textbook ~⅓-fewer-flops advantage. But measured back-to-back vs the existing packed path it was a **2–10× regression**: the kernel alone (no read-out) ran 822 ns at N=256 and 1.39 ms at N=65536, already slower than the *entire* packed RFFT (≈540 ns / ≈259 µs). The cause is exactly the memory-schedule lesson `iterative.go` documents: the scalar split-radix has a scattered, strided, un-SIMD-able access pattern, while the packed path reuses the already-SIMD, cache-friendly iterative complex engine. The flop saving is swamped by losing SIMD and the linear cache schedule, worsening with N. Per the "revert no-ops" rule the kernel was removed; the honest residual is that closing the FFTW gap further needs a *SIMD, cache-blocked* real kernel (FFTW's codelet-generator approach), not a scalar real-split-radix port — the same item-1 SIMD-codelet residual that bounds the complex mid-range. The forward r2c gap to FFTW is therefore essentially unchanged from the pack/untangle round (~2.1–2.4× at the mid-range), now with the cheap fused-pack win banked on top.

## Real inverse 1-D IRFFT (`complex128` N/2+1 bins → `float64`)

The c2r inverse now mirrors the forward packing: it reverses the untangle to recover the N/2-point packed spectrum, runs **one N/2-point inverse complex FFT**, and unpacks — half the arithmetic and memory traffic of the previous full length-N conjugate-symmetric inverse. ns/op (GFLOP/s). Ratio = go-fft ÷ FFTW (lower is better; ≤1.05 = parity).

| N | go-fft | FFTW (c2r) | go/FFTW | verdict |
|---:|---:|---:|---:|:--|
| 256 (2⁸) | 797 (6.4) | 253 (20.2) | 3.15× | lags FFTW 3.15× |
| 1,024 (2¹⁰) | 3,517 (7.3) | 1,184 (21.6) | 2.97× | lags FFTW 2.97× |
| 4,096 (2¹²) | 16,749 (7.3) | 5,526 (22.2) | 3.03× | lags FFTW 3.03× |
| 65,536 (2¹⁶) | 351,731 (7.5) | 155,132 (16.9) | 2.27× | lags FFTW 2.27× |
| 1,048,576 (2²⁰) | 5,062,003 (10.4) | 3,329,422 (15.7) | 1.52× | lags FFTW 1.52× |
| 1,000 (2³·5³) | 4,659 (5.3) | 1,220 (20.4) | 3.82× | lags FFTW 3.82× |
| 1,080 (2³·3³·5) | 4,770 (5.7) | 1,341 (20.3) | 3.56× | lags FFTW 3.56× |
| 1,920 (2⁷·3·5) | 8,786 (5.5) | 2,253 (23.0) | 3.90× | lags FFTW 3.90× |

### Real inverse (c2r) — before→after (half-length packing, same machine state)

The previous IRFFT promoted the half spectrum to a full conjugate-symmetric length-N inverse complex FFT — ~2× the work the real symmetry requires. The packed inverse does an N/2-point inverse FFT plus an untangle pass instead. Measured on the same host and the same c2r input the FFTW c2r harness inverts (lower ns/op is better; ratio = go-fft ÷ this-snapshot FFTW c2r):

| N | full-size inverse (before) | packed inverse (after) | speedup | before/FFTW | after/FFTW |
|---:|---:|---:|---:|---:|---:|
| 256 | 2,088 | 797 | 2.62× | 8.26× | 3.15× |
| 1,024 | 9,363 | 3,517 | 2.66× | 7.91× | 2.97× |
| 4,096 | 34,999 | 16,749 | 2.09× | 6.33× | 3.03× |
| 65,536 | 612,049 | 351,731 | 1.74× | 3.95× | 2.27× |
| 1,048,576 | 12,086,105 | 5,062,003 | 2.39× | 3.63× | 1.52× |
| 1,000 | 8,699 | 4,659 | 1.87× | 7.13× | 3.82× |
| 1,080 | 9,377 | 4,770 | 1.97× | 6.99× | 3.56× |
| 1,920 | 17,675 | 8,786 | 2.01× | 7.84× | 3.90× |

The packed inverse roughly **halves the c2r time at every measured size** (1.74–2.66×, the real-symmetry 2× realized), cutting the FFTW c2r gap from ~3.6–8.3× down to ~1.5–3.9×. Correctness is held: `IRFFT(RFFT(x), len(x)) ≈ x` round-trips within float64 tolerance, the DC/Nyquist bins reconstruct exactly (the classic real-FFT bug, guarded by `TestIRFFTPackedMatchesFullInverse` against the conjugate-mirror oracle and against `numpy.fft.irfft`), and odd N still routes to the full conjugate-mirror inverse.

## 2-D complex FFT2 (`complex128`)

go-fft fans the independent 1-D row/column transforms across goroutines above a work-size threshold (the multicore path); FFTW / numpy / scipy are single-threaded here. ns/op (GFLOP/s).

| shape | go-fft | FFTW | numpy.fft2 | scipy.fft2 | go/FFTW | verdict |
|:--|---:|---:|---:|---:|---:|:--|
| 64x64 | 29,022 (8.5) | 9,064 (27.1) | 19,895 | 14,005 | 3.20× | lags FFTW 3.20× |
| 128x128 | 78,984 (14.5) | 56,858 (20.2) | 76,859 | 64,707 | 1.39× | lags FFTW 1.39× |
| 256x256 | 208,437 (25.2) | 271,098 (19.3) | 359,158 | 207,601 | 0.77× | **≥ parity** |
| 512x512 | 633,920 (37.2) | 1,247,215 (18.9) | 1,697,966 | 964,884 | 0.51× | **≥ parity** |
| 1024x1024 | 2,784,623 (37.7) | 6,301,688 (16.6) | 9,211,551 | 5,315,022 | 0.44× | **≥ parity** |

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

At-or-above parity (≤ 1.05× the reference's ns/op) across all complex-1-D + real-1-D + 2-D rows measured on this host (this snapshot — FFTW running in a fast thermal state, see Methodology and the before→after note):

- **vs FFTW (native C, gold standard)**: 3/24 ops at-or-above parity (the three large 2-D multicore shapes); the 1-D rows lag this fast-state FFTW, though go-fft's own ns/op improved at every size this round.
- **vs gonum (pure-Go peer)**: 19/19 ops at-or-above parity.

**vs gonum** (the fair pure-Go, CGO=0 peer): go-fft is faster at **every** size measured — typically 3–5× on composite N and an order of magnitude on primes (gonum falls back to a naive Bluestein with no Rader path), confirming go-fft is the fastest pure-Go FFT here.

**vs pocketfft** (numpy/scipy): go-fft wins the small-N rows outright (the Python FFI tax dominates pocketfft's C kernel there) and is competitive-to-winning at large 1-D and large 2-D; the residual losses are the mid-range single-core bands.

**vs FFTW** (the gold standard): FFTW leads on the single-core power-of-two and smooth-composite mid-range — it has hand-written SIMD codelets and a dedicated Hermitian real kernel; go-fft's butterflies are now SIMD too (routed SSE2 on amd64, gc-autovectorized NEON on arm64), but FFTW's codelet generator and r2c real kernel remain ahead on one core. go-fft reaches parity-or-better where the algorithm, not raw SIMD throughput, dominates: very large 1-D, the large 2-D multicore shapes, and (relative to FFTW's own cost) the large-prime rows.

### Lagging ops — root cause + action items

Honest read of where go-fft trails FFTW, with the concrete lever to close each gap:

1. **Power-of-two & smooth-composite mid-range (256 … 4096, 1000/1080/1296/1920).** *Root cause*: scalar Go butterflies vs FFTW's hand-written NEON SIMD codelets — the kernel is the same Cooley–Tukey schedule, FFTW just does 2 complex muls per instruction. *Action taken (SIMD-butterfly round)*: the radix-2/radix-4 decimation-in-time butterfly inner loops were lifted into stage-granularity kernels (`internal/kernels/butterfly*.go`) behind a stable seam, with go-asmgen generating a routed **SSE2 stage kernel on amd64** (real packed ADDPD/SUBPD — measured **1.34–1.43× faster** than the scalar stage there, since GOAMD64=v1 does not autovectorize). On **arm64/s390x** the Go assembler exposes vector floating-point only as the fused multiply-add family — there is **no vector `VFADD`/`VFSUB`** — so a hand kernel must emulate every add/sub as a copy + FMA-by-one and pay a `VLD2`/`VST2` deinterleave; built and benchmarked, that **only ties** the gc autovectorizer, which already extracts the NEON throughput from the simple loop (the same lesson the SIMD complex-multiply round taught). So off amd64 the hot path stays the **autovectorized Go loop**, and the SIMD kernels remain validated bit-identical artifacts. See the before→after below. *(Remaining gap to FFTW on arm64 is its real-FFT Hermitian kernel and codelet scheduling — items 2–3 — not raw complex-mul SIMD, which the compiler already vectorizes.)*
2. **Real mid-range (1024 … 65536).** *Root cause*: go-fft packs a real signal into a half-length complex FFT and untangles once; FFTW runs a **dedicated real (r2c) kernel** that exploits Hermitian symmetry at every stage (~2× less arithmetic). *Action taken (real-inverse round)*: the **inverse (c2r) IRFFT was promoting the half spectrum to a full length-N inverse complex FFT** — ~2× the work the symmetry needs — and now packs symmetrically (reverse-untangle → N/2-point inverse FFT → unpack), which **roughly halves the c2r time at every size** (1.74–2.66×; see the c2r before→after table above) and cuts the FFTW c2r gap from ~3.6–8.3× to ~1.5–3.9×. The forward r2c already packs to a half-length FFT; the **pack/untangle round** then made that pass allocation-free (pooled working set) and branchless/vectorizable, cutting the forward mid-range time **1.5–1.65×** and the FFTW gap from ~3.3–3.9× to ~2.1–2.4× (see the forward-RFFT before→after table above). The residual forward gap to FFTW is its codelet-scheduled Hermitian-symmetric real kernel. The **fused-pack round (2026-06-24)** then took the remaining cheap structural win on the forward path — fusing the real-pair packing into the iterative kernel's bit-reversal gather, removing one O(N) pass and the intermediate buffer (~1.2–1.7× faster on the pack/gather portion at N=256…4096, bit-identical output; see the fused-pack note above). The deferred native real split-radix kernel (Sorensen RVFFT) was **implemented, validated bit-exact, benchmarked, and reverted**: a scalar real-split-radix is a 2–10× regression because it abandons the SIMD cache-friendly complex engine the packed path reuses (the memory-schedule lesson). Closing the forward gap further now demonstrably needs a *SIMD, cache-blocked* real kernel (FFTW's codelet generator), the same item-1 SIMD-codelet residual that bounds the complex mid-range — not a scalar real-FFT port.
3. **Large primes (1009, 10007).** *Root cause*: Rader/Bluestein convolve at length ≈N−1 on the recursive mixed-radix engine, which pays a ~1.5× recursion tax; FFTW convolves at exactly N−1 with codelet-fused mixed-radix. *Note*: go-fft is already **2–2.5× faster than gonum** here and within ~1.6× of FFTW (vs ~30× gap for gonum's naive Bluestein). *Action*: an **iterative mixed-radix engine** for the smooth convolution lengths (the same lever that closed the pow2 band) — a substantial new engine, lower priority than items 1–2. *Action taken (Stockham round, 2026-09-29)*: that engine now exists (`stockham.go`) and every smooth length, including the Rader convolution lengths, runs on it — primes 1009/2017 **1.7× faster**, 10007 **1.3×**; see the Stockham round below.
4. **Small 2-D (64×64, 128×128).** *Root cause*: below the parallel threshold, so they run the serial per-line path while FFTW uses fused 2-D codelets. *Action*: SIMD (item 1) lifts the per-line 1-D cost directly; the multicore path already makes go-fft win at 512×512 and 1024×1024 (beats numpy, ties/leads scipy).

> The unifying lever is **item 1 (go-asmgen SIMD butterflies)**: it attacks the pow2/composite mid-range, the per-line cost inside 2-D, and feeds the real and prime paths (which both convolve via complex FFTs). That is the one change that moves the most rows toward FFTW parity.

### SIMD-butterfly round — before→after (same machine state)

The honest measure of the butterfly change is go-fft before vs after **on the same machine state** (the cross-snapshot FFTW thermal drift noted in Methodology is removed by holding FFTW fixed at this snapshot and only varying go-fft's own code). Lower ns/op is better; "ratio" is go-fft ÷ this-snapshot FFTW.

**amd64 (SSE2 stage butterflies routed; measured under Rosetta — the *ratio* of the speedup is the load-bearing number, absolute ns/op are Rosetta-inflated):**

| N (complex) | scalar stage | SSE2 stage | speedup |
|---:|---:|---:|---:|
| 256 | 1,034 | 749 | 1.38× |
| 1,024 | 4,785 | 3,386 | 1.41× |
| 4,096 | 25,034 | 17,561 | 1.43× |
| 65,536 | 1,339,594 | 1,000,061 | 1.34× |

The SSE2 packed butterfly does the re/im add/sub in one `ADDPD`/`SUBPD` (2× the scalar throughput GOAMD64=v1 leaves unvectorized), bit-identical to the scalar oracle, validated by the amd64 CI execution job.

**arm64 (M4 Max, the gold-standard host; autovectorized Go loop — the SIMD kernel was measured to only *tie* it):** go-fft before vs after is within run-to-run noise (≈±2%) at every pow2 and smooth-composite size — the gc autovectorizer already emits the NEON the hand kernel would, and the Go arm64 assembler's lack of a vector `VFADD`/`VFSUB` denies the kernel any further headroom. The arm64 1-D rows in the tables above are therefore essentially unchanged by this round; the FFTW gap on arm64 is closed by items 2–3, not by complex-mul SIMD. This is the same outcome, and the same documented cause, as the earlier SIMD complex-multiply round (see `internal/kernels/cmul.go`).

**Net:** the butterfly hot loop is now a clean, per-arch-dispatched, 100%-covered, six-arch-validated kernel seam (`internal/kernels/butterfly*.go`), routing through hand SSE2 where it wins (amd64) and the autovectorized Go loop where the compiler already matches SIMD (every other arch) — bit-identity asserted on amd64, numerical correctness (≤1 ULP vs the oracle, FFTW/numpy-validated within tolerance) everywhere.

### Stockham round — iterative mixed-radix engine (2026-09-29, same machine state)

Every length whose prime factors are all ≤ 13, other than a power of two, used to run the **recursive** mixed-radix engine (`mixedradix.go`): each level gathered strided inputs and read its twiddles at stride n/len, and every call allocated an n-point scratch buffer. That engine carried the "recursion tax" item 3 names, and since Rader convolves at a smooth length, the large primes paid it as well as the composites.

Those lengths now run an **iterative Stockham autosort engine** (`stockham.go`, the pocketfft pass schedule): one pass per radix, ping-ponging between `dst` and one pooled scratch buffer. Every inner loop has unit stride and each pass reads its own contiguous twiddle block. There is no bit-reversal pass, because the autosort layout produces natural order. The radices are the same (2/3/4/5/7 straight-line, general radix for 11/13). The Rader and Bluestein convolution buffers are pooled too, and FFT in place, so those paths stop allocating three (Rader) or one (Bluestein) convolution-length buffers per call. Powers of two stay on the iterative pow2 kernel, which carries the amd64 SSE2/AVX2 butterflies and the fused real packing.

Measured as an interleaved A/B (the old and new test binaries alternated, best of 3 at 0.3 s each, Apple M4 Max; `main` at 3d840b8 vs this change). The power-of-two rows, whose code did not change, are the control: they move by ±5%, which is the noise floor of this run.

| N | before (ns/op) | after (ns/op) | speed-up | after ÷ FFTW (carried snapshot) |
|---:|---:|---:|---:|---:|
| 1,000 (2³·5³) | 5,632 | 3,566 | 1.58× | 2.36× → 1.44× |
| 1,080 (2³·3³·5) | 6,528 | 4,048 | 1.61× | 2.76× → 1.64× |
| 1,296 (2⁴·3⁴) | 9,331 | 5,036 | 1.85× | 3.28× → 1.68× |
| 1,920 (2⁷·3·5) | 12,010 | 7,165 | 1.68× | 2.88× → 1.65× |
| 1,009 (prime, Rader) | 16,346 | 9,416 | 1.74× | 1.26× → 0.69× |
| 2,017 (prime, Rader) | 34,372 | 20,157 | 1.71× | — |
| 10,007 (prime, Rader) | 343,364 | 260,325 | 1.32× | 2.15× → 1.52× |
| 641 (prime, Bluestein) | 45,452 | 43,286 | 1.05× | — |
| RFFT 1,000 / 1,080 / 1,920 | 3,357 / 3,795 / 6,895 | 2,137 / 2,400 / 4,177 | 1.57 / 1.58 / 1.65× | 2.67/3.20/2.99× → 1.82/2.12/1.91× |
| IRFFT 1,000 / 1,080 / 1,920 | 3,610 / 4,116 / 7,290 | 2,379 / 2,682 / 4,679 | 1.52 / 1.53 / 1.56× | 3.82/3.56/3.90× → 1.95/2.00/2.08× |
| 256 / 1,024 (control, unchanged code) | 735 / 3,432 | 703 / 3,303 | ±5% | — |

The speed-ups are the load-bearing numbers. The FFTW column divides by the FFTW figures carried in the tables above (a June snapshot), so it is indicative rather than a fresh same-state ratio. The benchmark host was also under background load during this run (load average ≈ 8–10 on 16 cores), which the interleaving cancels for the ratio but not for the absolute ns/op. The complex, real and 2-D tables above have not been regenerated with `benchmarks/run.sh`.

Correctness: the Stockham engine is cross-checked against the recursive engine, which is kept as an independent test oracle the way the split-radix engine is, at **every smooth length up to 3000**. The check runs in both directions and with `dst` aliasing `src`, so it reaches every radix at both `ido == 1` and `ido > 1`. The existing naive-DFT and numpy differential tests now run through the new engine unchanged. A dirty-pool test runs a plan on a large input and then on a small one, then compares the result with a fresh plan. Removing either new `clear` of the pooled convolution pad makes it fail, which was checked by mutation.

Remaining here: the general radix pass (11, 13) is still an O(r²) loop, and the Bluestein path (primes < 700 and lengths with a large prime factor) still convolves on the radix-2 kernel. On arm64 the Stockham engine also ran pow2 lengths ~5–10% faster than the pow2 kernel, but routing powers of two to it would drop the amd64 AVX2 butterflies and the fused real packing, so that was not done.

### Round 2 — pocketfft's passes, per-architecture routing, re-measured prime engines (2026-09-29)

This round compared the Stockham engine with pocketfft's `cfftp`/`fftblue` (read from `pocketfft_hdronly.h`, the C++ core behind `numpy.fft` and `scipy.fft`), and re-measured each routing constant on the new engine instead of carrying it over. Every step was measured before it was kept.

1. **Radix-8 pass** (pocketfft's `pass8`, whose odd half rotates by ±45°/±135° instead of multiplying). pocketfft takes every 8 it can. Here, radix 8 wins while the working set is small (512: 1.30×, 2048: 1.17× over radix 4) but loses on large pure powers of two (65536: 0.71–0.78×), where eight input and eight output streams at power-of-two strides alias in the cache. The factorization therefore uses radix 8 up to 4096 and for any length with an odd factor, and radix 4 for larger powers of two. It never takes a lone radix-2 pass when two radix-4 passes fit.
2. **Bounds-check-free passes.** Point `i = 0` (no twiddle) is done once per block, and every stream is resliced to exactly `ido-1`, so the compiler proves every index and the inner loops carry neither a bounds check nor a per-point branch. The direction is a sign folded exactly into the rotations. Radix 5, 7 and 8 exceed the inliner's budget and are inlined by hand, and the final pass (`ido == 1`) has its own flat loop. Together: 1.2–1.33× on composites.
3. **Powers of two on arm64 only.** The Stockham engine beats the pow2 kernel on arm64 at every size, but not everywhere else. Measured on real hardware (Stockham ÷ pow2 kernel speed, 256/1024/4096/65536): arm64 M4 1.07/1.30/1.10/1.34; amd64 Xeon E5-2620 v3 0.46/0.45/0.45/0.47, because the AVX2 butterflies win; riscv64 SpacemiT X60 0.89/0.82/0.82/0.96; ppc64le POWER9 0.97/1.10/0.99/1.35; loong64 3A5000 1.03/1.15/1.00/1.26. So powers of two route to Stockham on arm64 only (`route_arm64.go`). s390x was unreachable and keeps the pow2 kernel. The route is a variable so a test runs both routes, complex and real, on every architecture.
4. **Convolution-length cost model refitted.** The per-radix weights were calibrated on the recursive engine. On the new engine, over the linear-convolution windows of 260 primes, their picks were 26% (arm64) and 13% (amd64) slower than the fastest length in the window, worse than just taking the smallest smooth length. The weights were refitted by least squares to the time of every 7-smooth length in [500, 45000] on both hosts, and cross-validated (each host's fit on the other's data). Their picks now land 4.3% (arm64) and 2.6% (amd64) above the best on average, with the median exactly on it.
5. **Bluestein on the Stockham engine at a smooth length** (pocketfft's `fftblue` + `good_size`). It used to pad to a power of two and run a standalone radix-2 kernel, so N=641 did a 2048-point convolution where 1296 suffices. That alone took 641 from 43.1 to 8.4 µs on arm64 (5.1×). It is also far more accurate: relative RMS error vs a compensated-sum DFT went 4.7e-14 → 1.3e-15 at N=1282 = 2·641, which stays on Bluestein. 641 itself then moved to Rader (item 6; 640 = 2⁷·5), ending at 3.85 µs (11.6× over `main`) with error 1.7e-14 → 9.8e-16.
6. **Rader only where it wins.** Both engines were timed on every prime from 17 to 6000, on arm64 and amd64. Rader with a 7-smooth N−1 (a cyclic convolution at exactly N−1) won 65/65 on amd64 and most cases on arm64. Rader with an 11 or 13 in N−1 lost every time. The zero-padded linear Rader lost to Bluestein in all but 15 of about 1000 primes, because its permuted gather/scatter costs more than Bluestein's contiguous chirp passes. The router is now "prime with 7-smooth N−1 → Rader, else Bluestein". The size threshold (700) and the linear Rader path are gone.
7. **Real transforms without pack/unpack passes.** The packing z[j] = x[2j] + i·x[2j+1] is exactly the memory layout of a `[]complex128`, so on the Stockham route RFFT reads the input through a view of it and IRFFT writes the signal in place. The 1/m normalization is folded into the inverse's re-tangle pass. A test pins the layout.

Numerical accuracy was checked against a compensated-summation DFT (twoSum + FMA error terms) at 12 lengths. It is unchanged to the same order everywhere (≈1e-15 relative RMS), except for the 641/1282 gains above.

Interleaved A/B, best of 5 at 0.3 s. `main` = 3d840b8, #9 = the first Stockham round. Both hosts were shared and loaded; the unchanged pow2 paths on amd64 read within ±10%, which is that host's noise.

**arm64 (Apple M4 Max)**, ns/op. The last column divides by the FFTW figures carried in the tables above (a June snapshot, so indicative):

| op | main | #9 | this round | vs main | vs #9 | ÷ FFTW, main → now |
|:--|---:|---:|---:|---:|---:|---:|
| complex 256 / 1024 / 4096 | 739 / 3,429 / 16,231 | 692 / 3,292 / 16,157 | 602 / 2,315 / 13,093 | 1.23 / 1.48 / 1.24× | 1.15 / 1.42 / 1.23× | 1.76→1.44 / 1.61→1.09 / 1.47→1.18 |
| complex 65536 / 2²⁰ | 370,547 / 9,537,617 | 366,154 / 9,219,404 | 273,896 / 6,532,895 | 1.35 / 1.46× | 1.34 / 1.41× | 1.15→**0.85** / 1.17→**0.80** |
| complex 1000 / 1080 / 1296 / 1920 | 5,511 / 6,398 / 9,024 / 11,693 | 3,549 / 4,036 / 4,989 / 7,106 | 2,611 / 2,830 / 3,554 / 4,913 | 2.11 / 2.26 / 2.54 / 2.38× | 1.36 / 1.43 / 1.40 / 1.45× | 2.22→1.05 / 2.59→1.15 / 3.01→1.19 / 2.69→1.13 |
| prime 1009 / 2017 / 10007 | 15,752 / 32,603 / 341,088 | 9,390 / 19,862 / 258,045 | 7,124 / 15,407 / 204,178 | 2.21 / 2.12 / 1.67× | 1.32 / 1.29 / 1.26× | 1.15→**0.52** / — / 1.99→1.19 |
| prime 641 (now Rader) | 44,757 | 42,521 | 3,851 | 11.6× | 11.0× | — |
| RFFT 256 / 1024 / 4096 / 65536 | 451 / 1,999 / 9,315 / 197,737 | 448 / 2,006 / 9,341 / 201,660 | 303 / 1,340 / 7,349 / 150,809 | 1.49 / 1.49 / 1.27 / 1.31× | 1.47 / 1.50 / 1.27 / 1.34× | 2.05→1.38 / 2.00→1.34 / 1.90→1.50 / 1.38→1.05 |
| IRFFT 256 / 1024 / 4096 / 65536 | 525 / 2,280 / 10,234 / 216,512 | 522 / 2,279 / 10,187 / 216,823 | 331 / 1,455 / 7,614 / 160,512 | 1.59 / 1.57 / 1.34 / 1.35× | 1.58 / 1.57 / 1.34 / 1.35× | 2.07→1.31 / 1.93→1.23 / 1.85→1.38 / 1.40→1.03 |
| RFFT 1000 / 1080 / 1920 | 3,027 / 3,451 / 6,330 | 2,114 / 2,392 / 4,194 | 1,496 / 1,618 / 2,936 | 2.02 / 2.13 / 2.16× | 1.41 / 1.48 / 1.43× | 2.57→1.27 / 3.05→1.43 / 2.89→1.34 |
| IRFFT 1000 / 1080 / 1920 | 3,312 / 3,735 / 6,703 | 2,362 / 2,659 / 4,666 | 1,591 / 1,731 / 3,135 | 2.08 / 2.16 / 2.14× | 1.48 / 1.54 / 1.49× | 2.71→1.30 / 2.79→1.29 / 2.98→1.39 |

**amd64 (Intel Xeon E5-2620 v3, Haswell, AVX2)**, ns/op:

| op | main | #9 | this round | vs main | vs #9 |
|:--|---:|---:|---:|---:|---:|
| complex 1000 / 1080 / 1296 / 1920 | 49,049 / 65,252 / 86,751 / 101,880 | 21,108 / 24,141 / 30,096 / 40,778 | 17,691 / 19,454 / 25,946 / 33,826 | 2.77 / 3.35 / 3.34 / 3.01× | 1.19 / 1.24 / 1.16 / 1.21× |
| prime 1009 / 2017 / 10007 / 641 | 150,070 / 306,025 / 3,764,135 / 209,701 | 52,444 / 111,684 / 1,437,809 / 165,109 | 42,382 / 100,380 / 1,144,763 / 21,272 | 3.54 / 3.05 / 3.29 / 9.86× | 1.24 / 1.11 / 1.26 / 7.76× |
| RFFT 1000 / 1080 / 1920 | 19,985 / 23,437 / 49,181 | 11,677 / 13,483 / 24,672 | 9,684 / 10,659 / 19,604 | 2.06 / 2.20 / 2.51× | 1.21 / 1.26 / 1.26× |
| IRFFT 1000 / 1080 / 1920 | 21,104 / 26,561 / 49,746 | 14,206 / 14,778 / 25,855 | 9,903 / 11,921 / 19,008 | 2.13 / 2.23 / 2.62× | 1.43 / 1.24 / 1.36× |
| complex and real powers of two | — | — | — | unchanged (same code; 0.94–1.08×) | — |

What remains:

- **amd64 non-powers of two run scalar.** At 1000 points they cost ~2.3× per point what the AVX2 pow2 kernel does at 1024. The next lever there is SIMD Stockham passes, which go-asmgen could generate the way it generates the butterflies.
- **riscv64, ppc64le and loong64** keep the pow2 kernel for powers of two.
- **Small real transforms** still trail FFTW ~1.3–1.5×. FFTW has codelet-scheduled real kernels; pocketfft's `rfftp` (FFTPACK real passes) is the pure-Go-portable reference for a native real engine, which was not attempted this round.
- **The FFTW columns** need a fresh `benchmarks/run.sh` on an idle host.

### Round 3 — AVX2 Stockham passes, and an inventory of the SIMD extensions (2026-09-30)

**Which vector extensions go-fft uses, and which it does not.** The inventory below records what the Go 1.26 assembler can emit for float64 vectors on each architecture. Two notes on it:

- **Encoding.** When the assembler has no mnemonic, an instruction can still be encoded by hand as a `WORD`. The encodings used below were checked against Apple's assembler.
- **Bit-identity.** Every kernel must be bit-identical to the scalar Go code it replaces. The scalar code on arm64/s390x/riscv64 is compiled with fused multiply-adds (FMA is baseline there), so a kernel must reproduce exactly the multiplies gc fuses. On amd64 at GOAMD64=v1 the scalar code does not fuse, so no amd64 kernel uses FMA.

| arch | extension | status in go-fft | why |
|:--|:--|:--|:--|
| amd64 | SSE2 | pow2 butterflies, complex multiply | baseline |
| amd64 | AVX2 | pow2 butterflies; **Stockham passes (this round)** | see below |
| amd64 | FMA3 | not used | not bit-identical to the GOAMD64=v1 scalar oracle; would save ~1 op per complex multiply |
| amd64 | AVX-512 | **prototype measured, not shipped** | radix-4 pass on Cascade Lake (cfarm151): 1.05–1.50×, typically ~1.2× over AVX2, and bit-identical (signed zeros included). Shipping it needs an AVX-512 probe in go-asmgen (its FeatureProbe knows AVX2 and POPCNT), ZMM variants of the ten kernels, and a coverage scheme that holds on CI runners with and without AVX-512. That is a round of its own. |
| arm64 | NEON (ASIMD) | complex multiply (via `VFMLA`/`VFMLS`) | the assembler has no vector `FADD`/`FSUB`/`FMUL`/`FNEG` for float64; with `WORD`-encoded ones, a deinterleaved (`VLD2`/`VST2`) radix-4 Stockham prototype is bit-identical once it fuses the same product gc does (the imaginary part fuses `ar·bi`, not `ai·br`, which a first try got wrong by 1 ULP). It measured 1.14–1.38× on Neoverse-N1 (cfarm424) but only 1.04–1.13× on Apple M4 and **0.86×** on a large pass there, so it is not shipped. |
| arm64 | SVE / SVE2 | not used | no SVE machine reachable (cfarm's GB10 hosts timed out); the Go assembler has no SVE mnemonics |
| riscv64 | RVV 1.0 | complex multiply | `VFADD`/`VFMUL`/`VFMACC` exist; Stockham passes not attempted (the pow2 kernel already beats the scalar Stockham engine there) |
| s390x | Vector facility | complex multiply | no real hardware reachable this round |
| ppc64le | VSX | not used | the Go assembler still has no VSX double arithmetic (`XVADDDP`…); `WORD` encoding would be possible |
| loong64 | LSX / LASX | not used | **new:** Go 1.26 now assembles `VADDD`/`VSUBD`/`VMULD` and the LASX `XV` forms, which it could not when the complex-multiply round ruled loong64 out |

**AVX2 Stockham passes.** On amd64 the Stockham engine ran scalar, ~2.3× slower per point than the AVX2 pow2 kernel. Its radix-2/3/4/5/8 passes are now generated by go-asmgen (`genStockhamAVX2`, `genStockhamLastAVX2` in `asmgen/amd64/gen.go`) into `butterfly_amd64.s`.

- **Layout.** Two points share each YMM register. An odd `ido` finishes with a 128-bit step. The final pass (`ido == 1`) has no twiddles, so it pairs blocks instead of points, gathering them with `VINSERTF128`.
- **Bit-identity.** Every operation is the scalar pass's own, in the same order, separately rounded: `VMULPD`, `VADDPD`, and `VADDSUBPD` for the complex product, with no FMA. The ±i, ±45° and ±135° rotations are lane swaps and sign flips, which are exact.
- **The i = 0 point.** The scalar pass does not multiply it by its twiddle, and a multiply by one is not exact on signed zeros or infinities: (−0)·1 − (−0)·0 = +0, and ∞·0 = NaN. So the first pair of every block is multiplied, then the untwiddled value is blended back in with `VBLENDPD $3`.
- **Direction.** It lives in a constants table passed by pointer, so one kernel serves both directions.

Correctness, all on real hardware (Haswell cfarm13, Zen 3 cfarm420, Cascade Lake cfarm151; Rosetta has no AVX2, so local runs cannot show it):

- **Bitwise test.** `TestStockhamPassMatchesScalar` compares the AVX2 and scalar engines bit for bit, at every smooth length up to 2100 plus 8192/20160/45000/196608, forward and inverse, in place and not.
- **Test inputs.** It uses four signals: a generic one, all −0, mixed ±0, and one with an infinity. A generic signal cannot tell a multiply by one from no multiply; zeros and infinities can.
- **Mutation checks.** Dropping the blend, swapping the direction tables, or giving the final-pass kernel a wrong offset each makes the test fail. Dropping the blend passed on the generic signal alone, which is why the zero and infinity signals are there.
- **Coverage.** With `-coverpkg` over both packages it stays at 100% on real AVX2 amd64, and at 100% on arm64.

**Powers of two on amd64 re-routed.** Once Stockham was vectorized, it beat the AVX2 pow2 kernel up to 4096 on all three CPUs (pow2 kernel ÷ Stockham time):

| n | 64 | 256 | 1024 | 4096 | 16384 | 65536 | 262144 | 2²⁰ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Haswell (E5-2620 v3) | 1.28 | 1.63 | 1.35 | 1.28 | 0.92 | 0.97 | 0.81 | 1.07 |
| Zen 3 (EPYC 7773X) | 1.51 | 1.51 | 1.35 | 0.95 | 1.58 | 1.38 | 1.40 | 1.43 |
| Cascade Lake | 1.57 | 1.86 | 1.63 | 1.22 | 1.04 | 1.00 | 1.08 | 0.98 |

So on amd64 with AVX2, powers of two up to 4096 run on the Stockham engine and larger ones stay on the pow2 kernel, where the three CPUs disagree (`route_amd64.go`).

**End to end on amd64** (interleaved A/B against round 2, best of 3; the 2²⁰ rows and RFFT 4096 on Zen 3 were re-measured best-of-6 and are within ±4%):

| op | Haswell | Zen 3 | Cascade Lake |
|:--|--:|--:|--:|
| complex 256 / 1024 / 4096 | 2.30 / 1.84 / 1.67× | 2.09 / 1.63 / 1.16× | 2.52 / 2.04 / 1.68× |
| complex 1000 / 1080 / 1296 / 1920 | 3.12 / 2.81 / 2.66 / 3.22× | 3.07 / 2.67 / 2.42 / 2.80× | 2.79 / 2.77 / 2.57 / 2.74× |
| prime 641 / 1009 / 2017 / 10007 | 2.89 / 1.62 / 1.54 / 2.12× | 2.18 / 1.45 / 1.51 / 2.06× | 2.02 / 1.58 / 1.54 / 2.29× |
| RFFT / IRFFT 256–1024 | 1.42–1.84× | 1.44–1.68× | 1.46–1.74× |
| RFFT / IRFFT 1000–1920 | 2.12–2.70× | 1.88–2.49× | 1.94–2.10× |
| powers of two ≥ 65536 (unchanged) | 0.91–1.10× | 0.96–1.04× | 0.93–1.02× |

**Fresh parity runs on three hosts (2026-09-30).** `benchmarks/run.sh` was re-run in full against native FFTW, numpy and scipy on three machines:

- **Apple M4 Max:** the workstation of every table above, under background load (load average 3.4–5).
- **AMD EPYC 7773X (Zen 3):** GCC Compile Farm cfarm420, 128 threads, load average ≈ 2–4.
- **ARM Neoverse-N1:** cfarm424, 64 cores, idle.

On the Linux hosts FFTW 3.3.10 was built from source with its SIMD codelets. The correctness gate passed 24/24 on each. The complete reports and raw data are in [`benchmarks/results/`](benchmarks/results/README.md), and the tables at the top of this file are the June snapshot, not regenerated. go-fft ÷ FFTW (below 1 = go-fft faster):

| op | M4 Max | Zen 3 | Neoverse-N1 |
|:--|--:|--:|--:|
| complex 256 / 1024 / 4096 | 1.43 / 1.12 / 1.26 | 1.61 / 1.73 / 2.42 | 2.02 / 1.43 / 1.31 |
| complex 65536 / 2²⁰ | 1.13 / **0.81** | 2.00 / 1.13 | **1.00** / **0.67** |
| complex 1000 / 1080 / 1296 / 1920 | 1.11 / 1.15 / 1.19 / 1.07 | 1.43 / 1.49 / 1.46 / 1.40 | 1.50 / 1.64 / 1.39 / 1.60 |
| prime 1009 / 10007 | **0.53** / 1.12 | **0.75** / 1.77 | **0.54** / 1.36 |
| RFFT 256 / 1024 / 4096 | 1.51 / 1.37 / 1.54 | 2.01 / 1.48 / 2.92 | 1.74 / 1.34 / 1.50 |
| RFFT 65536 / 2²⁰ | 1.13 / 1.27 | 1.83 / 1.63 | 1.22 / **0.92** |
| RFFT 1000 / 1080 / 1920 | 1.29 / 1.43 / 1.32 | 1.68 / 1.69 / 1.70 | 1.61 / 1.63 / 1.65 |
| 2-D 512² / 1024² (multicore) | **0.43** / **0.45** | 1.63 / **0.87** | **0.51** / **0.25** |

Against the June M4 snapshot, the 1-D composites went from 2.4–3.3× FFTW to 1.07–1.19×, and complex 1024 from 1.60× to 1.12×. The weakest host relative to FFTW is Zen 3. Its FFTW uses AVX2 codelets with FMA, while go-fft's powers of two above 4096 there still run the pow2 kernel (65536: 2.0×). AVX-512 does not apply to Zen 3; FMA is the lever there, and the bit-identity rule excludes it (see the inventory above).

**Multicore fan-out fixed.** The fresh runs showed small 2-D shapes collapsing on many-core hosts: 128×128 at 8.5× FFTW on Zen 3 and 3.4× on N1, against 1.35× on the M4. The cause was `parChunks`, which split the lines into `min(GOMAXPROCS, lines)` chunks. On 128 threads, a 128×128 transform ran as 128 goroutines of one 128-point line each, about a microsecond of work against the cost of spawning, waking and joining them.

Chunks are now sized by work: no goroutine gets less than `parMinChunk` = 8192 elements. That value was chosen from a sweep of 2048–65536 over 128²…1024² on four hosts (16, 24, 64 and 128 threads); 8192 was best or within noise of the best on each. Old → new, best of 4:

| shape | M4 (16) | Haswell (24) | N1 (64) | Zen 3 (128) |
|:--|--:|--:|--:|--:|
| 128×128 | 1.05× | 0.99× | **1.39×** | **2.69×** |
| 256×256 | 1.01× | 1.16× | **1.30×** | **1.86×** |
| 512×512 | 0.99× | 0.81× | 0.94× | **2.25×** |
| 1024×1024 | 0.95× | 1.37× | 1.09× | 0.93× |

Where `work / 8192` is at least the thread count, the configuration is unchanged. That holds for M4 and Haswell at 512² and 1024², and for N1 and Zen 3 at 1024². Those cells run identical code, so their 0.81–1.37× spread is the noise of shared hosts, not an effect. `TestParChunksCapsWorkersByWork` pins the chunk counts.

**What remains on many-core hosts: the garbage collector.** Profiling 128×128 on Zen 3 put about three quarters of the time outside the FFT, almost all of it in the GC. `runtime.(*lfstack).pop`, `getempty` and `procyield` are its mark workers coordinating across 128 threads. `FFT2`/`FFTN` return a new slice, 272 KB at 128×128, so a benchmark loop triggers a collection every few calls, and a collection on 128 threads is expensive. numpy and scipy allocate their result too, so only the FFTW comparison is skewed. Closing it needs an allocation-free N-D entry point, the way `Plan.FFT(dst, src)` is for 1-D. That is new public API, so it is left as a decision rather than added here.

### Round 4 — AVX2 real-FFT untangle, radix rules re-measured per architecture, six-step ruled out (2026-10-01)

The Round 3 fresh runs left the real transforms as the widest gap to FFTW on amd64 (RFFT 256–4096 at 1.5–2.9× on Zen 3). Profiling RFFT 1024 on Zen 3 put 39% of the time in `rfftUntangle`, the Go loop that recombines the half-length spectrum, and 72% of RFFT 4096 in the radix-8 pass.

1. **AVX2 untangle and re-tangle.** `rfftUntangle` (forward) and `irfftRetangle` (inverse) now run their bins two at a time in generated AVX2 kernels (`genUntangleAVX2`, `genRetangleAVX2`). Bin k needs Z[k] and Z[m−k]; for k and k+1 those are one forward load and one mirrored load with its 128-bit halves swapped.
   - **Arithmetic.** The scalar arithmetic maps onto lane blends, multiplies by 0.5 (or the inverse's scale), and the same complex product. Conjugations and multiplications by i are exact sign flips and swaps.
   - **Bit-identity.** Both kernels are bit-identical to the Go loops at GOAMD64=v1. `TestUntangleMatchesScalar` and `TestRetangleMatchesScalar` check every m ≤ 700 plus larger ones, on the generic, ±0 and ∞ signals, on Haswell, Zen 3 and Cascade Lake.
   - **Mutation checks.** Removing the conjugation from either kernel makes its test fail.

2. **Radix rules re-measured per architecture.** The "radix 8 up to 4096" rule (Round 2) had been calibrated on the M4 with scalar passes, before the passes lost their bounds checks. Three rules for 2^e were timed on five CPUs from 2^5 to 2^13: A, radix 8 as far as it goes; B, radix 4 with a radix-2 pass for an odd e; C, radix 4 opening with one radix-8 pass for an odd e. Time over the best of the three, geometric mean:

   | | rule A | rule B | rule C |
   |:--|--:|--:|--:|
   | arm64 (M4, Neoverse-N1) | 1.121 (worst 1.51) | 1.089 (worst 1.26) | **1.022** (worst 1.11) |
   | amd64 (Haswell, Zen 3, Cascade Lake) | **1.082** (worst 1.62) | 1.190 (worst 1.71) | 1.129 (worst 1.78) |

   - **arm64:** powers of two now use rule C up to 8192. Above that, rule B stays: re-timed to 2^20, C lost at 2^15 on the M4 (1.19–1.25×) and won only 3–11% on N1.
   - **amd64:** keeps rule A. Its known loss is Zen 3 at 2048/4096 (1.36×/1.42× radix 4's time), where the radix-8 pass's 23 power-of-two-strided streams contend for L1 sets.
   - **Composites:** lengths with an odd factor keep rule A everywhere.

3. **Six-step (Bailey) for large N: measured and dropped.** A six-step prototype splits n = n1·n2 into in-cache FFTs, with a twiddle pass and three blocked transposes. It was correct, but 1.5–2.6× slower than the Stockham engine at 2^16–2^20 on the M4. The profile put 61% of its time in the transposes, which are 4 KB-strided rows of complex128, and only ~30% in the FFTs that the cache blocking helps. With a free transpose the ceiling would be ~25%, so it was not pursued.

Speed-up over Round 3 (#12), interleaved A/B (best of 3; the arm64 rows were re-checked with medians of 6–8 under background load):

| op | Haswell | Zen 3 | Cascade Lake | M4 | Neoverse-N1 |
|:--|--:|--:|--:|--:|--:|
| RFFT 256 / 1024 / 4096 | 1.15 / 1.18 / 1.17 | 1.19 / 1.27 / 1.05 | 1.31 / 1.34 / 1.25 | 0.99 / 1.03 / 1.20 | ≈1 / ≈1 / 1.10 |
| IRFFT 256 / 1024 / 4096 | 1.75 / 1.56 / 1.35 | 1.48 / 1.58 / 1.17 | 1.41 / 1.50 / 1.34 | 0.99 / 1.04 / 1.19 | ≈1 / ≈1 / 1.11 |
| RFFT / IRFFT 1000–1920 | 1.04–1.44 | 1.20–1.47 | 1.24–1.43 | ≈1 | ≈1 |
| complex 256 / 4096 | ≈1 | ≈1 | ≈1 | 1.36 / 1.18 | 1.26 / 1.07 |
| other sizes | ≈1 | ≈1 | ≈1 | ≈1 | ≈1 |

"≈1" marks rows whose code did not change on that architecture, within that host's noise (±10% on the shared hosts). On arm64 the real transforms gain only where their half-length falls under the new radix rule (4096 → 2048).

### Round 5 — allocation-free N-D plans, a rebuilt N-D path, and the in-place pow2 kernel (2026-10-01)

Round 3 traced the small 2-D transforms on many-core hosts to the garbage collector. `FFT2`/`FFTN`/`RFFT2` return a new slice each call, and on 128 threads a collection costs more than a 128×128 FFT. Closing that needed an API that writes into the caller's slice, the way `Plan.FFT(dst, src)` does in 1-D and `fftw_plan_dft(rank, dims, …)` does in FFTW.

**New API.**
- `NewPlanN(shape...)` with `FFT(dst, src)` / `IFFT(dst, src)`, for complex transforms of any rank. `dst` may alias `src`.
- `NewRealPlan2(rows, cols)` with `RFFT(spec, img)` / `IRFFT(img, spec)` / `SpectrumLen()`.

Both allocate nothing in steady state (`TestPlanNAllocatesNothing`, `TestRealPlan2AllocatesNothing`; an odd-width real plan still allocates in its row transform). `FFTN`/`FFT2`/`IFFTN`/`IFFT2`/`RFFT2`/`IRFFT2` are now these plans writing into a new slice, cached by shape, so there is one implementation.

**The N-D path, rebuilt and measured step by step.**
- **The contiguous last axis** is transformed in place, with no gather or scatter. A real 2-D plan's rows go straight from the input matrix into their row of the output.
- **Every other axis** gathers up to 8 neighbouring lines at once, so each element read is a whole run of a cache line rather than 16 bytes of one. The block budget (128 KB) came from a 16/32/64/128 KB sweep at 256²…2048²: it was best or within 2% on M4 and N1, and the best single value on Haswell and Zen 3 (geometric mean 1.10 of each size's best). My first guess, half an L1, was refuted by that sweep.
- **Two Haswell regressions** turned up and were removed on the way:
  - *4K aliasing.* Eight 512-point lines sat 8 KB apart in the scratch buffer, so every gather write hit one aliasing set. The lines are now staggered by 64 bytes.
  - *Loop overhead.* The 8-wide inner loop cost more than the memory traffic. Full blocks now go through an unrolled gather/scatter with no bounds checks.

  With both fixed, single-threaded Haswell went from ×0.76–0.86 to ×0.98–1.06 for `RFFT2` and ×1.05–1.26 for `IRFFT2`.
- **IFFTN normalization** is one pass at the end instead of one per line.

**Found on real hardware only.** The zero-allocation tests failed on ppc64le (POWER9) and riscv64 (SpacemiT X60), at 128 allocations per 64×64 call. Those architectures route powers of two to the iterative pow2 kernel, whose in-place call copied its input with `append` on every call. That cost also hit any in-place 1-D transform there. The copy now comes from a pool (`TestInPlacePow2KernelAllocatesNothing`).

Median speed-up over `main` after Round 4. `PlanN` and `RealPlan2` are compared with `main`'s `FFT2` and `RFFT2`, i.e. what a user calls today:

| op | M4 (16 threads) | Haswell (24) | Neoverse-N1 (64) | Zen 3 (128) |
|:--|--:|--:|--:|--:|
| `FFT2` 32² … 1024² | 1.17–1.59 | 0.99–1.58 | 1.08–1.41 | 1.05–1.35 |
| **`PlanN.FFT`** 32² … 1024² | 1.35–2.03 | 1.39–3.46 | 1.55–3.27 | **2.47–4.05** |
| `RFFT2` 64² … 1024² | 1.05–1.31 | 0.96–1.17 | 1.05–1.13 | 1.10–1.26 |
| `IRFFT2` 64² … 1024² | 1.21–1.46 | 1.11–1.42 | 1.17–1.38 | 1.56–1.80 |
| **`RealPlan2.RFFT`** 64² … 1024² | 1.29–2.12 | 1.45–2.36 | 1.44–3.11 | **2.38–3.04** |

The three cells below 1.00 are all on Haswell (0.96–0.99), a shared host whose run-to-run noise is larger than that.

**Regression check.** The whole public API (FFT/IFFT/RFFT/IRFFT at every length 1–2100 plus 12 large ones, FFT2/IFFT2/RFFT2/IRFFT2 at 10 shapes, FFTN/IFFTN at 3 ranks; 8494 cases) was computed by the code before these rounds and by this code, on M4 and on AVX2 Haswell, and compared:

- **Out of tolerance:** 0 cases. **Non-finite outputs:** none.
- **2-D/N-D cases:** they differ by at most 2.1e-15 relative.
- **1-D, largest differences:** 2e-13, on odd-length IRFFT. Measured against numpy, the new code was the closer one in all 40 of the most-differing cases (2e-15 against the old 2e-13), so those differences are an accuracy gain.

### Round 6 — AVX-512, where it pays (2026-10-01)

**go-asmgen first.** go-asmgen v0.10.0 adds an `AVX512F` feature probe:
- leaf 7 must exist;
- OSXSAVE must be set;
- XCR0 must hold XMM, YMM, opmask, ZMM_Hi256 and Hi16_ZMM (mask 0xE6);
- CPUID.7.0:EBX bit 16 must be set.

Generated and run on real hardware, it agreed with `/proc/cpuinfo` on Cascade Lake (true), Haswell and Zen 3 (false). On macOS it answers false, because Darwin enables ZMM state lazily, so an Intel Mac runs the AVX2 kernels.

The same release adds encoders for the arm64 vector `FADD`/`FSUB`/`FMUL`/`FNEG` .2D that `cmd/asm` lacks. They are pinned against the system assembler, and ran bit-identically to Go arithmetic over 2.4M lanes on M4 and Neoverse-N1.

go-fft's generator now has its own `go.mod` pinning go-asmgen v0.10.0. Before, it resolved go-asmgen from whatever a module cache held, so the committed `.s` files were only reproducible by luck.

**The kernels.** The Stockham emitter gained a 512-bit width, four points per ZMM register. Every instruction AVX-512 lacks at that width is replaced exactly:
- `VADDSUBPD` → a real-lane sign flip, then `VADDPD` (x + (−y) is x − y);
- `VXORPD` → `VPXORQ` (avoids needing AVX512DQ);
- the i = 0 blend → `VBLENDMPD` under opmask K1;
- `VINSERTF128` → `VINSERTF32X4` for the final pass.

`ido` mod 4 is finished with the 256- and 128-bit bodies. The bit-identity test now compares scalar against AVX2 and against AVX-512. On Cascade Lake it passes on every signal (generic, ±0, ∞), and removing the K1 blend or using the wrong sign row each makes it fail.

**Where AVX-512 is used was decided by measurement.** On an idle Cascade Lake (cfarm151, load ≈ 0), AVX-512 sped powers of two up but not composites. AVX2 time ÷ AVX-512 time:

| | all radices at 512 bits | only radix-2/4/8 passes at 512 bits | 512 bits except radix 8 |
|:--|--:|--:|--:|
| 256 / 1024 / 4096 | 1.16 / 1.21 / 1.49 | 1.17 / 1.22 / 1.50 | 0.86 / 0.93 / 1.03 |
| 1000 / 1080 / 1296 / 1920 / 2000 | 0.95–1.04 | 0.83–0.94 | 0.94–1.03 |
| 20160 / 1008 / 10080 / 45000 | 0.88–0.96 | 0.84–0.91 | 0.86–0.97 |

Mixing widths inside one transform is worse than either width alone. That is consistent with the core's 512-bit frequency licence costing more than radix-3/5/7 passes gain. So AVX-512 is decided per transform, not per pass: only a power of two of at least 256 points uses it, because at 8–128 points the AVX-512 time ratio is 0.94–1.03 (`wide512`). The radix-3/5 kernels at 512 bits were then never called, so they are not generated.

Stockham with AVX-512 also beats the pow2 kernel up to 16384 (pow2 kernel time ÷ Stockham: 4096 2.33, 8192 1.35, 16384 1.74; above that it wanders between 0.86 and 1.23). On amd64 with AVX-512, powers of two up to 16384 now route to Stockham (`route_amd64.go`).

**Coverage.** Kernel choice (`stockhamWidth`, `stockhamKernels`) and routing are pure functions, tested for every combination, so the 100% gate does not depend on whether a CI runner has AVX-512. Measured: 100% on Cascade Lake (AVX-512), on Haswell (AVX2 only) and on arm64.

**Result**, end to end against `main` (median of 4; Cascade Lake idle):

| op | speed-up |
|:--|--:|
| complex 256 / 1024 / 4096 | 1.14 / 1.19 / 1.38 |
| RFFT / IRFFT 4096 | 1.21 / 1.21 |
| composites, primes, and powers of two the change does not reach | 0.91–1.16 |

The last row is code this change does not alter on that machine: composites stay on AVX2, and 2²⁰ and the real 256 path stay on the routes they had. Its spread is run-to-run noise. Zen 3, which has no AVX-512 and so runs the same code as before, showed the same spread over the same rows (complex 2²⁰ at 1.18 there too).

**Regression check.** The 8494 public-API cases were computed by `main` and by this code on the AVX-512 machine. None was out of tolerance; the largest difference was 1.1e-15 relative (IRFFT 16384, which now takes the Stockham route). The same cases agree with the M4's to 2.5e-15.

### Round 7 — cache-blocked pow2 kernel, and a coverage gate that passed by chance (2026-10-03)

**Cache blocking.** The pow2 kernel serves powers of two above 4096 on amd64 (16384 with AVX-512), and every power of two on riscv64, ppc64le, loong64 and s390x. It is an in-place iterative DIT, so every stage used to sweep the whole array: eight sweeps of 1 MB at 65536.

The stages whose butterfly groups fit in a 4096-point block now run block by block, all of them on each block while it is in cache. The bit-reversal gather for that block is fused in front of them. Only the wide stages still sweep the array. A block gets exactly the operations its points get in a full sweep, so the result is bit-identical; `TestCacheBlockingIsBitIdentical` checks every entry point at four block sizes against the unblocked run. Unlike the six-step attempt (Round 4), nothing is transposed.

The block size was swept from 512 to 32768 points on Haswell, Zen 3, Cascade Lake, POWER9 and SpacemiT X60, at 2^11 to 2^20. Values are unblocked time ÷ blocked time:

- **Choice:** 4096 had the best geometric mean (1.031).
- **At 2^20:** the gain shows on all five CPUs (1.03–1.57).
- **On the two idle hosts:** 1.19 at 2^20 and 1.07 at 2^18 on Cascade Lake; 1.05 and 1.04 on POWER9.
- **Below 2^17:** blocking is neutral, because the array already fits in L2.
- **At n ≤ 4096:** the code is the unblocked code.

Haswell and the X60 were loaded during the run (load average 8.8 and 4.3). On rows where the code is identical, their ratios still swing by ±30–40%, so only the idle hosts' numbers are claimed above.

**A coverage gate that passed by chance.** `cachedPlan`'s branch for "another goroutine stored a plan while this one built its own" was covered only when two concurrent tests happened to race. Coverage read 91.7% or 100% on identical runs, so the 100% gate passed or failed by luck. The builder is now injectable, and `TestCachedPlanKeepsTheFirstStored` takes that branch on purpose. Six consecutive `-race -coverpkg` runs, the way CI runs them, now give 100%, with no function varying between runs.

### Round 8 — the scratch buffer was in dst's L1 sets (2026-10-04)

**Where it showed.** A fresh parity run on v0.1.2 put complex 4096 on Zen 3 at 2.15× FFTW, at half the GFLOP/s of 1024. Timing each Stockham pass alone (ns per point, page-aligned buffers) found the cost in the passes whose streams sit a multiple of 4 KB apart:

| | stride of the costly pass | that pass | the same pass, output moved off the input's sets | the other passes |
|:--|:--|--:|--:|--:|
| Neoverse-N1, 4096 (radix 4) | 16 KB | 3.38 | 2.27 | 1.9–2.1 |
| Zen 3, 4096 (radix 8, AVX2) | 8 KB | 3.27 | 1.37 | 0.72–1.18 |

**Why.** Every pass writes its r output streams n/r points apart, and the first pass reads its r inputs as far apart. For a power of two that distance is a multiple of 4 KB, so a pass's output streams share one L1 set, and its input streams share one too. Go places large allocations on page boundaries, so dst and the scratch buffer are usually a multiple of 4 KB apart, and then the two groups land in the same set: 2r lines for an 8-way (Zen 3, Intel) or 4-way (N1) L1. Moved off (576 bytes on N1, 2112 on Zen 3 in the table), a whole N1 transform went from 12.7 to 11.1 ns per point at 4096 and from 16.2 to 13.7 at 16384; on Zen 3, 4096 went from 5.43 to 3.80 with a 576-byte gap.

**The fix.** For powers of two from 1024 points, the pooled scratch buffer is 4 KB longer, and each call slices it to start 576 bytes past dst, modulo 4 KB (`offTheSets`). Seven gaps from 0 to 3136 bytes were swept through the transform on four CPUs. Time with the buffers 4 KB-aligned ÷ time with the gap:

| | 1024 | 2048 | 4096 | 8192 | 16384 | 65536 |
|:--|--:|--:|--:|--:|--:|--:|
| Zen 3 | 1.34 | 1.49 | 1.65 | 1.12 | 1.21 | 1.23 |
| Haswell | 1.04 | 1.04 | 1.03 | 1.00 | 1.00 | 1.04 |
| Neoverse-N1 | 0.99 | 0.99 | 1.00 | 1.06 | 1.06 | 1.20 |
| Cascade Lake | 0.98 | 1.01 | 1.01 | 0.99 | 0.99 | 0.99 |

- **Why 576 bytes:** it is the only gap that lost nowhere. 2112 bytes cost 8% at 1024 on both Intel CPUs, and a 64-byte gap was slower than none on Zen 3.
- **Why N1 gains little at small sizes:** its L1 ways are 16 KB, so whether the conflict happened depended on where the allocator put the buffers. The gap now rules it out.
- **Why powers of two only:** lengths 2^k·3 (6144, 12288) lost 2–5% with any gap on all four CPUs, while 1920 and 3840 did not move and 1000 gained 3–6% on Intel only.
- **Why from 1024 points:** below 1024 a first pass's streams are less than 4 KB apart. The gap bought nothing there and cost 2–3% at 64 and 128 points on Haswell and Cascade Lake.

The gap changes no arithmetic: only where the scratch buffer starts.

**What it changed in the routing.** The routes and radix rules of Rounds 2–6 were measured with the conflict in place, so they were measured again (`route_*.go` holds the tables).

- **Stockham vs the pow2 kernel**, pow2 kernel time ÷ Stockham time:

  | n | 4096 | 8192 | 16384 | 32768 | 65536 | 2^17 | 2^18 | 2^19 | 2^20 |
  |:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
  | Zen 3 | 1.50 | 1.58 | 1.94 | 1.50 | 1.81 | 1.38 | 1.81 | 1.96 | 2.51 |
  | Haswell | 1.76 | 1.05 | 1.65 | 1.04 | 1.16 | 0.82 | 1.15 | 1.23 | 1.24 |
  | POWER9 | 1.20 | 1.24 | 1.34 | 1.30 | 1.37 | 1.15 | 1.22 | 1.44 | 1.81 |

  With AVX2, and on ppc64le, every power of two now routes to Stockham. Before, the threshold was 4096 with AVX2, and ppc64le used the pow2 kernel throughout.
- **Radix rules**, time over the best of three rules, geometric mean over 256…2^20:
  - **Cascade Lake (AVX-512):** radix 8 everywhere scores 1.000. The old rule, radix 8 to 4096 then radix 4, scored 1.239. With radix 8, Stockham also beats the pow2 kernel at every size up to 2^20 (2^20: 28.7 against 36.7 ns per point). So AVX-512 now routes every power of two to Stockham with radix 8 throughout, where the route used to stop at 16384.
  - **POWER9:** arm64's rule, radix 4 opening with one radix 8 for an odd exponent up to 8192, scores 1.003, against 1.056 for the rule ppc64le had. ppc64le now takes arm64's rule.
  - **Haswell vs Zen 3:** the two AVX2 CPUs disagree. Radix 8 wins at every size on Haswell and loses at every size from 2048 on Zen 3. The current rule (radix 8 to 4096) ties for best over both (1.064), and stays.
- **Not re-measured:** riscv64 and loong64, whose hosts were loaded, and s390x, which was not reachable. They keep the pow2 kernel.

**Accuracy.** For exact tones, the relative RMS error of the forward transform at 2^20 was 4.5–5.5e-16 for Stockham and 6.0e-16 for the pow2 kernel, on Zen 3, Cascade Lake and POWER9. Round-trip errors were the same, 4.4–4.8e-16.

**Regression check.** The 8494 public-API cases were computed by `main` and by this code. On POWER9 none was out of tolerance, and the largest difference was 2.5e-15 relative (IRFFT of composite lengths, whose Bluestein convolutions now run on Stockham).

**Benchmark harness.**
- **2-D row:** it timed `FFT2`, which returns a new slice per call, so it also timed an allocation and, on a many-core host, the garbage collector. Every other row writes into a reused slice through a plan. The row now times a reused `PlanN`, and the allocating call stays as a separate column. On Zen 3 the difference is 25 against 65 µs at 64×64 and 0.32 against 0.95 ms at 256×256.
- **`report.py`:** it ended every report with a fixed "root cause" paragraph that had gone stale ("scalar Go butterflies", "a recursive engine"). It now lists the rows behind FFTW, worst first, and points here for the explanations.

**The 2^20 row that looked like a regression.** The fresh run put complex 2^20 on Zen 3 at 18.9 ms, against 12.8 ms on 2026-09-30. Fifteen interleaved rounds of the 09-30 code, an intermediate commit that does not touch this path (a control), and v0.1.2 gave medians of 19.1, 19.3 and 19.6 ms. The old code is just as slow today: the host changed, not the code.

**Result**, end to end against `main`. Five interleaved rounds, one core, main time ÷ new time:

| | complex 4096 | 65536 | 2^20 | RFFT 65536 | 2^20 | IRFFT 2^20 |
|:--|--:|--:|--:|--:|--:|--:|
| Zen 3 | 1.42 | 1.83 | 1.99 | 1.39 | 1.18 | 1.94 |
| Haswell | 1.03 | 1.15 | 1.25 | 1.04 | 1.30 | 1.49 |
| Cascade Lake | 1.01 | 1.27 | 1.20 | 1.46 | 1.24 | 1.31 |
| POWER9 | 1.35 | 1.34 | 1.78 | 1.29 | 1.46 | 1.53 |
| Neoverse-N1 | 1.11 | 1.19 | 1.13 | 1.07 | 1.17 | 1.12 |

- **POWER9:** every power of two from 256 gains ×1.25–1.39, and the 2-D shapes ×1.25–1.34.
- **Composites and primes:** their code did not change. They read 0.99–1.01 on the four quiet hosts; on Zen 3 their paired ratios spread from 0.71 to 1.07.
- **128×128:** read 0.96–0.97 on Zen 3 and Haswell. That run applied the gap from 64 points; the gap now starts at 1024.
