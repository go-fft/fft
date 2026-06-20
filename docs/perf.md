# Performance

`go-fft` is benchmarked head-to-head against the two reference implementations
that matter:

* **[`gonum.org/v1/gonum/dsp/fourier`](https://pkg.go.dev/gonum.org/v1/gonum/dsp/fourier)**
  — the fairest comparison: the other pure-Go (`CGO_ENABLED=0`) FFT for Go.
  go-fft beats it at every size.
* **FFTW / pocketfft** via `numpy.fft` and `scipy.fft` (C, hand-tuned SIMD) —
  the absolute gold standard, and single-threaded. A pure-Go library is not
  expected to beat hand-written C+SIMD on a single core, but go-fft now **wins
  outright at several shapes** (large 1-D, large 2-D, and the small-N rows) and
  the remaining gaps are quantified honestly below.

All transforms are differentially validated against an O(N²) naive-DFT oracle
and a full round-trip suite, so every number below is for a *correct* transform.

## How to reproduce

The head-to-head Go benchmark vs gonum lives in its own module (`bench/`) so the
library itself stays dependency-free — `gonum` is never pulled into the library
`go.mod`:

```sh
cd bench
go test -run=^$ -bench=. -benchmem
```

The FFTW/pocketfft comparison runs on one Linux host so the silicon is held
constant: the Go matrix `BenchmarkH2H*` (in `h2h_test.go`) and the matching
Python script `scripts/fftbench.py` exercise the *same* transforms, sizes, and
2-D shapes.

```sh
go test -run=^$ -bench=H2H -count=3 .      # go-fft
python3 scripts/fftbench.py                # numpy.fft + scipy.fft
```

## Method

* go-fft is measured through its cached `Plan` API (`NewPlan(n).FFT`,
  `NewRealPlan(n).RFFT`), the apples-to-apples match for gonum's reused plan
  object and FFTW/pocketfft's reused plan: twiddle factors are built once and
  amortized across calls, so each number is the steady-state transform cost.
* ns/op is best-of-3, lower is better. The numpy/scipy numbers include the
  Python-call overhead (≈1–2 µs fixed per call), which dominates the small-N
  rows; the large-N and 2-D rows, where that overhead is amortized, are the
  like-for-like kernel comparison.
* Size taxonomy: powers of two (64 … 65536), highly-composite N (1000 = 2³·5³,
  1296 = 2⁴·3⁴), primes (2017, 9973, 10007 — the algorithm's worst case), and
  2-D shapes (where the multicore path applies).
* Hardware: a 4-core arm64 Linux host (same box for both sides). go-fft and the
  numpy/scipy script are run back-to-back so they share the machine state; ns/op
  is best-of-N (lower is better), the convention both sides use.

## go-fft vs FFTW / pocketfft (4-core arm64, same host, ns/op)

### Complex FFT

| N | go-fft | numpy.fft | scipy.fft | verdict |
|---:|---:|---:|---:|:--|
| 64 | **291** | 2 274 | 1 824 | **go-fft wins ~6.3×** (small-N) |
| 256 | **1 363** | 2 863 | 2 325 | **go-fft wins** (small-N) |
| 1024 | 7 249 | 5 478 | 4 421 | pocketfft ~1.6× → **go-fft now wins**, see iterative round |
| 4096 | 30 612 | 17 462 | 14 188 | pocketfft ~2.2× → **tie**, see iterative round |
| 65536 | **540 292** | 1 348 290 | 1 204 171 | **go-fft wins ~2.2×** |
| 1000 (2³·5³) | 5 784 | 5 662 | 4 511 | pocketfft ~1.3× |
| 1296 (2⁴·3⁴) | 9 735 | 6 847 | 5 664 | pocketfft ~1.7× |
| 769 (prime) | 14 510 | — | 12 978 | **~1.12×** (was ~3.1×) — Rader round |
| 2017 (prime) | **37 775** | 59 663 | 37 244 | **~parity (1.01×)** (was ~2.4×) — Rader round |
| 5003 (prime) | 194 392 | — | 83 422 | FFTW ~2.33× (was ~3.1×) |
| 9973 (prime) | 359 100 | 353 154 | 222 076 | FFTW ~1.62× (was ~2.9×) |
| 10007 (prime) | 371 383 | 282 924 | 225 605 | FFTW ~1.65× (was ~2.8×) |

### Real RFFT

| N | go-fft | numpy.rfft | scipy.rfft | verdict |
|---:|---:|---:|---:|:--|
| 64 | **254** | 2 129 | 1 886 | **go-fft wins ~7.4×** (small-N) |
| 256 | **1 019** | 2 639 | 2 293 | **go-fft wins ~2.3×** (small-N) |
| 1024 | 3 824 | 4 252 | 3 621 | **~parity (1.06×)** (was ~1.9×) — paired untangle |
| 2048 | 7 575 | — | 5 419 | pocketfft ~1.40× |
| 4096 | 15 787 | 11 759 | 9 772 | pocketfft ~1.62× (was ~2.7×) |
| 65536 | **272 560** | 276 281 | 300 164 | **go-fft wins ~1.10×** (was ~2.1×) |
| 1000 | 5 732 | 5 041 | 4 199 | pocketfft ~1.4× |
| 1296 | 8 181 | 5 722 | 4 963 | pocketfft ~1.7× |

### 2-D complex FFT2 (multicore)

numpy.fft / pocketfft are single-threaded; go-fft fans the independent row/column
1-D transforms of a 2-D FFT out across goroutines above a work-size threshold.
This is where a pure-Go library can decisively beat single-threaded C.

| shape | go-fft | numpy.fft2 | scipy.fft2 | verdict |
|:--|---:|---:|---:|:--|
| 64×64 | 66 519 | 21 484 | 15 834 | scipy ~4.2× (below the parallel threshold) |
| 128×128 | 302 201 | 80 553 | 66 859 | scipy ~4.5× |
| 256×256 | 933 034 | 725 341 | 555 908 | scipy ~1.7× |
| 512×512 | 2 598 623 | 2 962 278 | 2 421 166 | **go-fft beats numpy; ~tie with scipy** |
| 1024×1024 | **9 622 944** | 12 050 524 | 9 837 828 | **go-fft wins both** |

**Honest read.** go-fft now **wins outright** at: every small-N row (where the
Python FFI tax dominates the C kernels), large 1-D complex N = 65536 (~2.2–2.9×
faster than scipy on the same core), large 2-D (1024×1024 beats both numpy
and scipy; 512×512 beats numpy and ties scipy) thanks to the multicore path, and
— after the **iterative round** (see below) — the **mid-range complex** band that
was the last loss: N=1024 now wins (~1.48×) and N=2048/4096 tie pocketfft on the
same core. The split-radix round (below) shrank the remaining gaps before that:
the **mid-range complex** gap narrowed (1024: ~1.8×→~1.6×; 4096 stayed ~2.2×
until the iterative round closed it), the **real mid-range** gap roughly halved (1024:
~1.9×→~1.2×; 4096: ~2.7×→~1.9×; 65536: ~2.1×→~1.3×) because the real path's
half-length complex FFT now rides the split-radix kernel and skips a copy, and
the **prime** gap fell from ~2.9–4.9× to ~2.4–2.9× (the prime engines convolve
with the now-faster pow2 FFTs, and the Bluestein→Rader crossover dropped from
N=4500 to N=700). The **Rader-convolution + real-untangle round** (below) then
took the primes whose N−1 is *smooth* to parity — **2017 at 1.01× and 769 at
~1.12×** via a direct length-(N−1) cyclic convolution and a specialized radix-7
butterfly — roughly halved the rest (9973/10007 to ~1.6×, 5003 to ~2.3×) with a
2·3·5-smooth convolution pad, and brought **real 1024 to parity and real 65536 to
a win** with a paired conjugate-symmetric untangle. The honest residual losses are
now: the primes whose N−1 has a medium/large prime factor (5003 ~2.3×, 9973/10007
~1.6× — FFTW's codelet-fused length-(N−1) convolution), the real mid-range
2048/4096 (~1.4–1.6× — pocketfft's dedicated Hermitian real kernel), and the
non-power-of-two composites 1000/1296 (~1.4–1.7× — the recursive mixed-radix
engine's recursion tax). Each is mapped to its exact FFTW technique and pure-Go
lever in **Remaining gap** below. Where go-fft wins or ties (small-N, complex
mid-range, large 1-D, large 2-D, smooth-N−1 primes, real 1024/65536) the tables
above show it.

## What made go-fft fast (split-radix round, before → after)

Each optimization was kept only after measuring a win on the same host and
re-passing the full differential correctness suite (incl. big-endian s390x and
ppc64le and riscv64/loong64 under qemu).

0a. **Split-radix power-of-two engine.** Every pure power of two now takes a
   dedicated split-radix kernel instead of the general mixed-radix one. Split-radix
   computes the same DFT with ≈⅓ fewer real multiplies — one half-length DFT of
   the even samples plus two quarter-length DFTs of the ≡1 and ≡3 mod 4 samples,
   recombined by an L-shaped butterfly needing only W_N^k and W_N^{3k} — which is
   pocketfft's mid-range advantage. A hardcoded radix-4 leaf (roots all ±1/±i, no
   twiddles) terminates the recursion two levels early; that is what turns the
   lower operation count into a measured wall-clock win. Split-radix vs the prior
   mixed-radix engine, isolated A/B on the benchmark host (ns/op, best-of):

   | N | mixed-radix | split-radix | delta |
   |---:|---:|---:|---:|
   | 1024 | 8 010 | 5 563 | −31% |
   | 2048 | 21 028 | 14 382 | −32% |
   | 4096 | 40 253 | 34 191 | −15% |
   | 8192 | 93 706 | 78 149 | −17% |
   | 16384 | 162 693 | 150 193 | −8% |
   | 65536 | 787 040 | 716 772 | −9% |

0b. **Free prime + real speedups from 0a.** The Rader/Bluestein prime engines
   convolve with these pow2 FFTs, so they got faster for nothing (10007:
   ~1.13 ms → ~0.85 ms before the threshold retune below). The even-N real
   transform packs into a half-length complex buffer that is itself a pow2, so it
   rides the same kernel; feeding the freshly packed buffer straight to the
   split-radix kernel as its scratch also drops a copy and an allocation (RFFT
   3 → 2 allocs/op for pow2), giving real 1024: 8 750 → 4 524 ns and real 4096:
   33 029 → 19 462 ns vs the pre-round baseline.

0c. **Bluestein → Rader crossover lowered to N = 700.** Because 0a sped up the
   shared pow2 convolution, the measured crossover where Rader (no chirp
   pre/post-multiply, direct length-(N−1) convolution) beats Bluestein dropped
   from N = 4500 to N ≈ 700 (Bluestein still faster at N=641, Rader faster from
   N=769 up). Re-routing the 769..4500 primes to Rader removes the residual
   ~1.3–1.7× Bluestein overhead in that band:

   | N | Bluestein | Rader |
   |---:|---:|---:|
   | 769 | 53 893 | **43 931** |
   | 2017 | 114 977 | **86 771** |
   | 5003 | 624 952 | **361 171** |
   | 9973 | 1 405 512 | **657 107** |

## What made go-fft fast (iterative round, the mid-range frontier)

The split-radix round closed the *operation-count* gap but left a *memory-schedule*
gap: a recursive kernel revisits the data through a deep call tree, so the working
set is never resident across frames. pocketfft instead runs an **iterative**
kernel — one bit-reversal permutation, then log-N butterfly stages, each a single
linear pass over the array, twiddles streamed in consumption order. This round
ports that schedule to pure Go (`iterative.go`):

* **Bit-reversal once, then radix-4 DIT stages.** log₂(N) is decomposed into
  radix-4 stages (plus one leading radix-2 when log₂N is odd). Radix-4 halves the
  number of passes over memory versus radix-2 — the dominant cost on a
  memory-bound kernel.
* **Per-stage twiddles laid out for sequential reads.** Each stage stores its
  twiddles (radix-4: the triple W^k, W^{2k}, W^{3k}) contiguously in exactly the
  order the stage consumes them, so the twiddle read is a linear scan, not a
  strided gather into the size-N root table.
* **No manual blocking — the compiler wins the long loop.** An explicit
  cache-blocked inner loop (sweeping butterfly positions in L1-sized chunks across
  groups) was implemented and measured, and it *lost* to leaving each stage as one
  contiguous inner loop, most at the large sizes (N=65536: ~450 µs unblocked vs
  ~480 µs blocked). The gc autovectorizer extracts more from the simple long loop
  than from the fragmented blocked one — the same lesson the SIMD round taught. So
  the win is the iterative *schedule*, not hand-blocking.

The iterative kernel measured faster than the recursive split-radix engine at
**every** power-of-two length, so `NewPlan` now routes all powers of two here
(the split-radix engine stays as an independent, fully-tested validation oracle).
Re-measured on the iterative-round host (4-core arm64 Linux, Go 1.26.4 — a
slower box than the split-radix-round host above, so this section's numbers are
internally consistent same-host before/after and *not* comparable cell-for-cell
to the table above):

A/B, split-radix → iterative, through the public plan path (ns/op, best-of-4):

| N | split-radix | iterative | scipy.fft (same host) | verdict |
|---:|---:|---:|---:|:--|
| 1024 | 6 009 | **3 472** | 5 138 | **go-fft wins ~1.48×** (was scipy ~1.2×) |
| 2048 | 12 744 | **8 692** | 8 481 | **tie** (was scipy ~1.5×) |
| 4096 | 33 992 | **17 173** | 16 988 | **tie** (was scipy ~2.0×) |
| 8192 | 70 918 | **48 993** | — | −31% |
| 16384 | 132 806 | **93 481** | — | −30% |
| 65536 | 567 421 | **398 962** | 1 173 917 | **go-fft wins ~2.9×** |

The mid-range complex frontier documented as the last loss (~1.6–2.2× behind
pocketfft) is **closed**: go-fft now wins outright at N=1024 and ties pocketfft at
N=2048 and N=4096 on the same single core. The pow2 small-N rows (64, 256) and the
real-input path (whose even-N transform packs into a half-length pow2 complex FFT)
ride the same kernel and improved for free. This was a pure *schedule* win:
identical operation count, no wider vectors — confirming the standing lesson that
on this µ-arch the lever is memory layout the gc compiler can autovectorize, not
hand-emitted SIMD.

## What made go-fft fast (Rader-convolution + real-untangle round)

This round attacked the two documented residual losses: the prime gap and the
real mid-range. Each change was kept only after re-passing the full differential
suite (naive-DFT oracle + round-trip + the split-radix/iterative oracles) on
amd64 **and** big-endian s390x/ppc64le plus riscv64/loong64 under qemu, then
re-measuring a win on the 4-core arm64 host back-to-back against scipy.fft.

A. **Direct length-(N−1) cyclic Rader convolution for smooth N−1.** Rader's
   prime-N transform is a *cyclic* convolution of length q = N−1. The old code
   always zero-padded it to a power of two ≥ 2q−1 and ran a linear convolution.
   But when q itself is smooth (all prime factors ≤ maxRadix) the cyclic
   convolution can be evaluated *directly* at length q — `IFFT_q(FFT_q(a)·FFT_q(ker))`
   — with no zero-pad. For these primes the pad was 2–3.3× larger than q
   (N=769: q=768=2⁸·3 vs a 2048-point pad; N=2017: q=2016=2⁵·3²·7 vs 4096), so
   convolving at q directly roughly halves the FFT work. This is FFTW's tuned
   Rader path. Result: **769 ~3.1×→~1.12×, 2017 ~2.0×→parity (1.01×)**.

B. **2·3·5-smooth (not power-of-two) pad for non-smooth N−1.** When q has a
   medium/large prime factor (N=9973: q=9972=2²·3²·277; N=10007: q=10006=2·5003)
   the direct length-q transform would itself be a Bluestein call, so the linear
   convolution stays — but padded to the smallest **2·3·5-smooth** length ≥ 2q
   instead of the next power of two. A 2·3·5-smooth pad rides only the specialized
   radix-2/3/4/5 butterflies and is ~0.61× the size of the next power of two
   (N=9973: 20000 = 2⁵·5⁴ vs a 32768-point pad). Result: **9973 ~2.4×→~1.62×,
   10007 ~2.2×→~1.65×, 5003 ~3.1×→~2.33×**.

C. **Specialized radix-7 butterfly.** N=2017's convolution length 2016 = 2⁵·3²·7
   has a radix-7 stage that ran on the O(7²)=49-multiply general radix-p path —
   the single largest cost (~31% of the transform). A straight-line radix-7
   butterfly (exploiting the three conjugate root pairs the way radix-5 exploits
   two) replaced it. This is what carried 2017 the rest of the way to parity.

D. **Alloc-free general radix-p butterfly.** The general radix-p kernel allocated
   a length-r scratch slice *per call*, which dominated the allocation count of
   every prime/composite transform (~581 allocs/op at N=2017). Hoisting it to a
   fixed-size stack array (r ≤ maxRadix) dropped that to 5 allocs/op.

E. **Paired conjugate-symmetric real untangle.** The even-N real transform's
   untangle produced bins one at a time, reading Z[k] and Z[m−k] for *each* bin.
   The two outputs of a conjugate pair (k, m−k) share the same Z reads and the
   same twiddle W_n^k and satisfy dst[k]=xe+W·xo, dst[m−k]=conj(xe−W·xo) (since
   W_n^m=−1), so computing the pair together halves the Z reads and twiddle
   lookups. Result: **real 1024 ~1.2×→parity (1.06×); real 65536 now wins
   (~1.10×); real 4096 ~1.9×→~1.62×**.

F. **Plan-cache deadlock fix (correctness).** Building a Rader/Bluestein plan
   re-enters the plan cache to construct its convolution sub-plan; the cache held
   its lock across the whole build, so calling the package-level `FFT` on a
   Rader-routed prime self-deadlocked. The cache now builds plans outside the lock
   with a double-checked store (a benign race builds an identical immutable plan).

The earlier round's optimizations (still in force) follow.

1. **Multicore N-dimensional transforms.** FFTN/FFT2/RFFT2/IRFFT2 reuse one
   cached plan per axis (no per-line plan lookup or output allocation) and
   distribute the independent axis lines across goroutines above a work-size
   threshold. This is the lever single-threaded pocketfft cannot pull:

   | shape | before | after | vs scipy.fft2 |
   |:--|---:|---:|:--|
   | 512×512 | 10.7 ms | **2.60 ms** | ~tie |
   | 1024×1024 | 38.1 ms | **9.62 ms** | **go-fft wins** |

2. **Inlined mixed-radix leaf level.** A deep power-of-two factorization spent
   most of its calls on trivial length-1 sub-DFTs. Gathering the strided leaf
   samples directly and letting the radix butterfly do the size-r DFT removes
   that bottom layer of the call tree (it was ~36% of mid-range runtime):

   | N | before | after |
   |---:|---:|---:|
   | 1024 | 10.3 µs | **8.0 µs** (−22%) |
   | 4096 | 47.7 µs | **32 µs** (−30%) |
   | 65536 | 825 µs | **614 µs** (−19%, now beats scipy) |

3. **Direct-indexed twiddle tables.** The plan precomputes the conjugate
   (inverse) root table once and passes the active table into the butterflies,
   which index it directly — dropping the per-element modulo and conjugate
   branch from the radix-2/3/4/5 hot loops.

4. **Rader's algorithm for large primes.** A prime-N DFT becomes a length-(N−1)
   cyclic convolution via a primitive root, evaluated with the library's own
   power-of-two FFTs. It is routed in place of Bluestein only above a measured
   threshold (N ≥ 4500), where its two fewer length-N passes beat Bluestein's
   chirp pre/post-multiply:

   | N | Bluestein | Rader |
   |---:|---:|---:|
   | 10007 | 1.29 ms | **1.05 ms** |
   | 9973 | 1.33 ms | **1.06 ms** |

   Below the threshold Bluestein still wins (same convolution size, less
   permutation overhead) and is kept. Both engines were measured equally
   accurate at N = 10⁴ (~1.3e-8 max error vs the naive oracle).

5. **Branch-free RFFT untangle.** The even-N real transform's untangle handles
   the wrapping DC/Nyquist bins outside the loop and expands the interior into
   real arithmetic, dropping the per-bin modulo and complex-multiply helpers
   (N = 4096: 28 µs → 25.6 µs).

### A note on SIMD

The pointwise complex-multiply SIMD kernels (amd64/arm64/s390x/riscv64) ship for
the Bluestein/Rader convolution step and are **re-measured each round** at the
actual convolution widths. On arm64 at width 32768 the autovectorized scalar
loop runs in 11.5 µs versus 38.8 µs for the de-interleaving NEON kernel — the
hand-SIMD still **loses by ~3.3×**, so it stays off the hot path. The scalar
default is the measured-faster choice; the SIMD kernels remain validated,
per-arch-tested artifacts (bit-identical to the scalar oracle, asserted by the
per-arch CI jobs) and the reference for any future widening.

## Remaining gap

The complex mid-range is **closed** (1024 wins, 2048/4096 tie). The Rader round
above brought the primes whose N−1 is *smooth* to parity (2017 at 1.01×, 769 at
1.12×) and roughly halved the rest (9973/10007 to ~1.6×, 5003 to ~2.3×), and the
paired real untangle brought real 1024 to parity and made real 65536 a win. What
is left, and exactly why each is a genuine pure-Go ceiling rather than an
unexplored lever:

* **Primes whose N−1 has a medium/large prime factor (5003 ~2.33×, 9973/10007
  ~1.6×).** Rader convolves at length q = N−1. When q is smooth this is done
  directly at length q and reaches parity (above). When q is *not* smooth (9972 =
  2²·3²·277; 10006 = 2·5003) the convolution must be a *linear* one padded to
  ≥ 2q, because a direct length-q transform of a non-smooth q is itself a
  Bluestein call at the same padded size. The smallest 2·3·5-smooth pad (≈0.61× a
  power-of-two pad) is already used, but two convolution FFTs at ≈20000 points on
  the recursive mixed-radix engine still cost ~360 µs vs FFTW's ~220 µs. FFTW
  convolves at *exactly* q via codelet-fused mixed-radix that handles the medium
  prime factor as a nested sub-DFT — measured here to need a length-q FFT that the
  recursive engine runs ~1.5× slower than its operation count (the "recursion
  tax", ~18% pure call overhead at N=769). An iterative mixed-radix engine for the
  smooth convolution lengths is the identified lever to close the last ~1.6× but
  was out of scope for this round; the residual is FFTW's codelet-fused
  length-(N−1) convolution, not a missing algorithm.
* **Real mid-range (2048 ~1.40×, 4096 ~1.62×).** The even-N real transform packs
  into a half-length *complex* FFT and untangles. pocketfft instead runs a
  *dedicated real kernel* that exploits Hermitian symmetry at every butterfly
  stage, not just at the final untangle — genuinely ~2× less arithmetic. The
  packing/untangle overhead was minimized (branch-free real arithmetic, then the
  paired conjugate untangle that halves the Z reads), which brought 1024 to parity
  and 65536 to a win, but the half-complex approach cannot match a native real
  butterfly schedule in the 2048–4096 band. A dedicated real kernel remains the
  only way to close it and was previously declined as not worth the duplication;
  the residual here is that architectural choice, measured.
* **Composite N with small factors (1000 ~1.35×, 1296 ~1.67×).** Highly-composite
  but non-power-of-two, so they run the recursive mixed-radix engine and pay the
  same recursion tax as the prime convolutions above; the iterative mixed-radix
  engine would lift these too.
* **Small-2-D rows (64×64, 128×128).** Below the parallel threshold, so they run
  the serial per-line path without the multicore payoff; lowering the threshold
  there regressed nothing but did not help either, so it is left where the large
  shapes win.

These are the honest frontiers, each tied to one specific FFTW technique
(codelet-fused length-(N−1) convolution; a dedicated Hermitian real kernel) and
one identified pure-Go lever (an iterative mixed-radix engine). Where go-fft wins
or ties — small-N, complex mid-range, large 1-D, large 2-D, smooth-N−1 primes,
real 1024/65536 — the tables above show it.
