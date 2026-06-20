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
* Hardware: a 4-core arm64 Linux host (same box for both sides).

## go-fft vs FFTW / pocketfft (4-core arm64, same host, ns/op)

### Complex FFT

| N | go-fft | numpy.fft | scipy.fft | verdict |
|---:|---:|---:|---:|:--|
| 64 | **369** | 2 266 | 1 880 | **go-fft wins** (small-N) |
| 256 | **1 539** | 3 042 | 2 398 | **go-fft wins** (small-N) |
| 1024 | 8 253 | 5 657 | 4 604 | pocketfft ~1.8× |
| 4096 | 30 005 | 18 198 | 14 705 | pocketfft ~2.0× |
| 65536 | **613 804** | 1 294 854 | 1 137 688 | **go-fft wins ~1.85×** |
| 1000 (2³·5³) | 6 162 | 5 790 | 4 707 | pocketfft ~1.3× |
| 1296 (2⁴·3⁴) | 10 371 | 7 410 | 6 152 | pocketfft ~1.7× |
| 2017 (prime) | 113 178 | 63 239 | 38 474 | FFTW ~2.9× |
| 9973 (prime) | 839 247 | 373 870 | 230 158 | FFTW ~3.6× |
| 10007 (prime) | 869 931 | 293 850 | 175 897 | FFTW ~4.9× |

### Real RFFT

| N | go-fft | numpy.rfft | scipy.rfft | verdict |
|---:|---:|---:|---:|:--|
| 64 | **438** | 2 202 | 1 968 | **go-fft wins** (small-N) |
| 256 | **1 693** | 2 664 | 2 324 | **go-fft wins** (small-N) |
| 1024 | 7 153 | 4 352 | 3 787 | pocketfft ~1.9× |
| 4096 | 28 019 | 11 854 | 10 520 | pocketfft ~2.7× |
| 65536 | 465 348 | 250 078 | 224 177 | pocketfft ~2.1× |
| 1000 | 5 940 | 4 534 | 3 952 | pocketfft ~1.5× |
| 1296 | 8 504 | 5 092 | 4 482 | pocketfft ~1.9× |

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
Python FFI tax dominates the C kernels), large 1-D complex N = 65536 (~1.85×
faster than scipy on the same core), and large 2-D (1024×1024 beats both numpy
and scipy; 512×512 beats numpy and ties scipy) thanks to the multicore path. It
**still loses** in the mid-range 1-D (N ≈ 1k–4k, ~1.7–2.7×) — the hand-tuned-C
split-radix + SIMD margin — and on primes (~2.9–4.9×), where FFTW's specially
tuned Rader/Bluestein kernels remain ahead even though go-fft's own Rader engine
closed the prime gap from ~8× to ~5×. The real path keeps a ~1.9–2.7× pocketfft
lead in the mid-range; it uses the half-length-complex packing trick rather than
a hand-tuned real kernel.

## What made go-fft fast (this round, before → after)

Each optimization was kept only after measuring a win on the same host and
re-passing the full differential correctness suite (incl. big-endian s390x and
riscv64 under qemu).

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

The mid-range 1-D gap (~1.7–2.7×) is the split-radix + hand-SIMD-butterfly
margin of pocketfft; a true split-radix power-of-two kernel and a SIMD radix
butterfly that beats the gc autovectorizer are the next levers. The prime gap
(~3–5×) is FFTW's specially tuned Rader/Bluestein. The small-2-D rows
(64×64, 128×128) sit below the parallel threshold and run the serial path, so
they pay the per-line gather without the multicore payoff — lowering the
threshold there regressed nothing but did not help either, so it was left where
the large shapes win. These are the honest frontiers; where go-fft already wins
(small-N, large 1-D, large 2-D) the tables above show it.
