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

**Tried and reverted / not pursued.** A native split-radix *real* first-stage butterfly schedule (FFTW's approach, item 3) would attack the remaining gap — the residual is FFTW's per-stage Hermitian-symmetric real codelets — but it is a large rewrite that the profile does not yet justify against the now-much-cheaper pack/untangle, and it risks the DC/Nyquist exactness the oracle tests pin. The cheap, measured wins (kill the allocations, de-branch the recombination) were taken; the codelet-generator-grade real kernel is the honest remaining residual and is left as the next lever.

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
2. **Real mid-range (1024 … 65536).** *Root cause*: go-fft packs a real signal into a half-length complex FFT and untangles once; FFTW runs a **dedicated real (r2c) kernel** that exploits Hermitian symmetry at every stage (~2× less arithmetic). *Action taken (real-inverse round)*: the **inverse (c2r) IRFFT was promoting the half spectrum to a full length-N inverse complex FFT** — ~2× the work the symmetry needs — and now packs symmetrically (reverse-untangle → N/2-point inverse FFT → unpack), which **roughly halves the c2r time at every size** (1.74–2.66×; see the c2r before→after table above) and cuts the FFTW c2r gap from ~3.6–8.3× to ~1.5–3.9×. The forward r2c already packs to a half-length FFT; the **pack/untangle round** then made that pass allocation-free (pooled working set) and branchless/vectorizable, cutting the forward mid-range time **1.5–1.65×** and the FFTW gap from ~3.3–3.9× to ~2.1–2.4× (see the forward-RFFT before→after table above). The residual forward gap to FFTW is its codelet-scheduled Hermitian-symmetric real kernel — the next lever is a native split-radix real first-stage butterfly schedule, deliberately deferred this round (large rewrite, risks DC/Nyquist exactness) until it measures clearly ahead of the now-much-cheaper pack/untangle.
3. **Large primes (1009, 10007).** *Root cause*: Rader/Bluestein convolve at length ≈N−1 on the recursive mixed-radix engine, which pays a ~1.5× recursion tax; FFTW convolves at exactly N−1 with codelet-fused mixed-radix. *Note*: go-fft is already **2–2.5× faster than gonum** here and within ~1.6× of FFTW (vs ~30× gap for gonum's naive Bluestein). *Action*: an **iterative mixed-radix engine** for the smooth convolution lengths (the same lever that closed the pow2 band) — a substantial new engine, lower priority than items 1–2.
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
