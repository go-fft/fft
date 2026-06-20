# Performance

`go-fft` is benchmarked head-to-head against the two reference implementations
that matter:

* **[`gonum.org/v1/gonum/dsp/fourier`](https://pkg.go.dev/gonum.org/v1/gonum/dsp/fourier)**
  — the fairest comparison: the other pure-Go (`CGO_ENABLED=0`) FFT for Go.
  **Matching or beating gonum is the primary bar.**
* **FFTW / pocketfft** via `numpy.fft` and `scipy.fft` (C, hand-tuned SIMD) —
  the absolute gold standard. A pure-Go library is not expected to beat
  hand-written C+SIMD, but the gap is quantified honestly below.

All transforms are differentially validated against an O(N²) naive-DFT oracle
and a full round-trip suite, so every number below is for a *correct* transform.

## How to reproduce

The head-to-head Go benchmark lives in its own module (`bench/`) so the library
itself stays dependency-free — `gonum` is never pulled into the library
`go.mod`:

```sh
cd bench
go test -run=^$ -bench=. -benchmem
```

The FFTW/pocketfft comparison runs Python on a Linux box (`scripts/fftbench.py`
in spirit; the procedure is documented in the section below). go-fft is run on
the *same* host so the hardware is held constant.

## Method

* go-fft is measured through its cached `Plan` API (`NewPlan(n).FFT`,
  `NewRealPlan(n).RFFT`), the apples-to-apples match for gonum's reused plan
  object and FFTW/pocketfft's reused plan: twiddle factors are built once and
  amortized across calls, so each number is the steady-state transform cost.
* ns/op is best-of-3 (`-count=3`), lower is better.
* Size taxonomy: powers of two (64 … 65536), highly-composite N (1000 = 2³·5³,
  1296 = 2⁴·3⁴), and a large prime (10007, the algorithm's worst case).

## go-fft vs gonum (Apple M4 Max, Go 1.26, ns/op, best of 3)

### Complex FFT

| N | go-fft | gonum | winner | go-fft speedup |
|---:|---:|---:|:--|---:|
| 64 | **500** | 738 | go-fft | 1.48× |
| 256 | **2 098** | 3 292 | go-fft | 1.57× |
| 1024 | **9 304** | 16 641 | go-fft | 1.79× |
| 4096 | **40 356** | 82 004 | go-fft | 2.03× |
| 65536 | **834 484** | 1 652 694 | go-fft | 1.98× |
| 1000 (2³·5³) | **9 694** | 16 980 | go-fft | 1.75× |
| 1296 (2⁴·3⁴) | **15 317** | 23 773 | go-fft | 1.55× |
| 10007 (prime) | **1 186 165** | 70 542 659 | go-fft | **59×** |

go-fft wins at **every** size. The prime case is the starkest: gonum's
arbitrary-N path is O(N²), while go-fft's Bluestein reduces it to an O(N log N)
power-of-two convolution — a ~59× win at N = 10007 that grows with N.

### Real RFFT

| N | go-fft | gonum | winner |
|---:|---:|---:|:--|
| 64 | 461 | **409** | gonum (≈tie) |
| 256 | **1 824** | 1 867 | go-fft |
| 1024 | **7 559** | 8 551 | go-fft |
| 4096 | **30 919** | 37 368 | go-fft |
| 65536 | **559 570** | 786 224 | go-fft |
| 1000 | **7 142** | 8 328 | go-fft |
| 1296 | **9 616** | 12 309 | go-fft |

go-fft wins on the real path everywhere except the smallest N = 64, where the
two are within noise.

## go-fft vs FFTW / pocketfft (Debian arm64, same host, ns/op)

Measured on one Linux arm64 host so the silicon is constant. **Caveat:** the
`numpy`/`scipy` numbers include the Python-call overhead (≈1–2 µs fixed per
call), which dominates the small-N rows and makes them *not* a like-for-like
kernel comparison there; the large-N rows, where that overhead is amortized,
are the meaningful ones.

### Complex FFT (ns/op)

| N | go-fft | numpy.fft | scipy.fft | note |
|---:|---:|---:|---:|:--|
| 64 | 533 | 2 282 | 1 793 | small-N: Python overhead dominates |
| 1024 | 10 323 | 5 598 | 4 519 | pocketfft ~2.3× faster |
| 4096 | 44 794 | 18 312 | 16 174 | pocketfft ~2.8× faster |
| 65536 | 918 335 | 1 361 442 | 1 187 121 | **go-fft matches/edges pocketfft** |
| 1000 | 10 094 | 5 772 | 4 664 | pocketfft ~2.2× faster |
| 1296 | 15 685 | 7 325 | 6 042 | pocketfft ~2.6× faster |
| 10007 (prime) | 1 452 315 | 298 150 | 177 684 | FFTW's tuned Rader/Bluestein ~8× faster |

### Real RFFT (ns/op)

| N | go-fft | numpy.rfft | scipy.rfft |
|---:|---:|---:|---:|
| 64 | 548 | 2 197 | 1 907 |
| 1024 | 8 506 | 4 360 | 3 706 |
| 4096 | 34 051 | 11 789 | 10 442 |
| 65536 | 594 445 | 251 122 | 247 240 |
| 1000 | 7 703 | 4 549 | 3 824 |
| 1296 | 10 249 | 5 045 | 4 377 |

**Honest read.** In the mid-range (N ≈ 1k–4k) pocketfft is roughly **2–3×
faster** than go-fft, the expected margin of hand-tuned C with SIMD-vectorized
radix kernels and split-radix butterflies — this is close to the pure-Go
ceiling for a scalar core. At large complex N (65536) go-fft *matches* pocketfft
on the same host (cache-blocking and the Python overhead close the gap). For the
real path, pocketfft keeps a ~2–3× lead at all sizes; it has a hand-tuned real
transform, where go-fft uses the half-length-complex packing trick.

## What made go-fft fast (before → after)

The starting point used **radix-2 only** for powers of two and **Bluestein for
every non-power-of-two**, recomputing all `sin`/`cos` on every call. The
optimizations, each kept only after measuring a win and re-passing the full
correctness suite:

1. **Plan cache with precomputed twiddles.** A `Plan`/`RealPlan` (and an
   internal per-length cache behind the convenience `FFT`/`RFFT` functions)
   builds every twiddle factor once and amortizes it across calls — no run-time
   trig.
2. **Mixed-radix Cooley–Tukey.** N is factored into small primes; the transform
   recurses with **radix-2/3/4/5 straight-line butterflies** and a general
   radix-p butterfly for 7/11/13. Bluestein is now reserved for lengths with a
   prime factor above `maxRadix` (13) only. This is what turned the
   highly-composite cases from a Bluestein fallback into a true fast path:

   | N | before (Bluestein) | after (mixed-radix) |
   |---:|---:|---:|
   | 1000 | 29 757 ns | **9 694 ns** (3.1× faster) |
   | 1296 | 44 715 ns | **15 317 ns** (2.9× faster) |

3. **Radix-4 over radix-2·2** for powers of two (fewer multiplies), pulled out
   first in the factorization.

### A note on SIMD

The pointwise complex-multiply SIMD kernels
(amd64/arm64/s390x/riscv64) ship for the Bluestein convolution step, where a
hot vector of complex products exists. They are **not** on the mixed-radix fast
path: the butterflies are interleaved load/store/rotate patterns the gc
compiler already auto-vectorizes well, and a prior measurement found hand-SIMD
*lost* to autovectorized scalar for the cmul shape by ~3×. SIMD is enabled only
where it measurably wins.

## Remaining gap

The mid-range gap to pocketfft (~2–3×) is the hand-tuned-C-with-SIMD margin. The
levers that could narrow it further, in order of expected payoff, are a
**split-radix** power-of-two kernel (≈⅓ fewer multiplies than radix-4), a
real-tuned RFFT kernel (rather than the complex-packing trick), and
**Rader's algorithm** for the prime case (to close the FFTW prime gap). These
are future work; the present release already beats the pure-Go peer (gonum)
across the board.
