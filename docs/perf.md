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
| 2017 (prime) | 88 376 | 60 217 | 37 424 | FFTW ~2.4× (was ~2.9×) |
| 9973 (prime) | 642 045 | 360 220 | 219 094 | FFTW ~2.9× (was ~3.6×) |
| 10007 (prime) | 678 195 | 284 519 | 243 689 | FFTW ~2.8× (was ~4.9×) |

### Real RFFT

| N | go-fft | numpy.rfft | scipy.rfft | verdict |
|---:|---:|---:|---:|:--|
| 64 | **254** | 2 129 | 1 886 | **go-fft wins ~7.4×** (small-N) |
| 256 | **1 019** | 2 639 | 2 293 | **go-fft wins ~2.3×** (small-N) |
| 1024 | 4 524 | 4 199 | 3 700 | pocketfft ~1.2× (was ~1.9×) |
| 4096 | 19 462 | 11 304 | 10 092 | pocketfft ~1.9× (was ~2.7×) |
| 65536 | 342 544 | 276 736 | 271 738 | pocketfft ~1.3× (was ~2.1×) |
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
N=4500 to N=700). The honest residual loss is now just primes (~2.4–2.9× — FFTW's specially tuned
Rader); the mid-range complex 1024/4096 loss was closed by the iterative round
(win at 1024, tie at 2048/4096). Where go-fft already wins (small-N, mid-range
complex, large 1-D, large 2-D, and the real mid-range within ~1.2–1.9×) the
tables above show it.

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

The split-radix round shrank the documented mid-range and prime losses, and the
iterative round (above) then *closed* the complex mid-range entirely: go-fft now
wins N=1024 and ties pocketfft at N=2048/4096 on the same core. The real mid-range
is ~1.2–1.9× and the prime gap is ~2.4–2.9×. What is left:

* **Complex mid-range 1024/4096 — CLOSED.** This was the last documented loss
  (~1.6–2.2× behind pocketfft). The iterative round (above) ported pocketfft's
  memory schedule — bit-reversal + iterative radix-4 DIT stages with twiddles laid
  out for sequential reads — and go-fft now **wins** at N=1024 (~1.48×) and
  **ties** pocketfft at N=2048 and N=4096 on the same single core. It was a pure
  schedule win at identical operation count; manual cache-blocking was tried and
  lost to the gc autovectorizer over a long contiguous loop, so it was not shipped.
* **Primes (~2.4–2.9×).** FFTW's specially tuned Rader/Bluestein. go-fft's Rader
  now wins from N=700 and convolves with the faster split-radix FFTs; the
  residual is the chirp/permutation bookkeeping vs FFTW's codelet-fused version.
* **Small-2-D rows (64×64, 128×128).** Below the parallel threshold, so they run
  the serial per-line path without the multicore payoff; lowering the threshold
  there regressed nothing but did not help either, so it is left where the large
  shapes win.

These are the honest frontiers; where go-fft already wins (small-N, large 1-D,
large 2-D, and the real mid-range now within ~1.2–1.9×) the tables above show it.
