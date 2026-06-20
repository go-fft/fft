<p align="center"><img src="https://raw.githubusercontent.com/go-fft/brand/main/social/go-fft.png" alt="go-fft/fft" width="720"></p>

# fft — go-fft

[![Docs](https://img.shields.io/badge/docs-mkdocs--material-9B1C2E)](https://go-fft.github.io/docs/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![Go](https://img.shields.io/badge/go-1.26.4%2B-00ADD8)](https://go.dev/dl/)
[![Status](https://img.shields.io/badge/status-phase%204-9a6700)](docs/plan-fft.md)

**A pure-Go (no cgo) FFT library** — the `numpy.fft` / `scipy.fft` equivalent for
Go. It computes the discrete Fourier transform of complex and real signals of
**any length**, with no dependency on the native FFTW3 C library.

Ruby has no cgo-free FFT (every option wraps FFTW3); `gonum/dsp/fourier` is pure
Go but its optimized assembly is amd64-only. This module is a fully portable
scalar core, with SIMD kernels generated across the six 64-bit Go targets
(amd64, arm64, riscv64, loong64, ppc64le, s390x) via
[go-asmgen](https://github.com/go-asmgen).

> Status: **Phase 4** — a correct pure-Go complex FFT (radix-2 Cooley–Tukey for
> power-of-two lengths, Bluestein's chirp-z for arbitrary lengths), the
> real-optimized `RFFT`/`IRFFT`, the multi-dimensional transforms
> (`FFT2`/`IFFT2`, `FFTN`/`IFFTN`, `RFFT2`/`IRFFT2`), the windowing / spectral
> helpers (windows, `FFTFreq`/`RFFTFreq`, `PSD`, `Spectrogram`), and go-asmgen
> SIMD kernels (bit-identical pointwise complex multiply: SSE2 on amd64, 2-wide
> NEON on arm64) behind a validated per-arch split CI; the other four 64-bit
> targets (riscv64/loong64/ppc64le/s390x) run the validated scalar path. See
> **[docs/plan-fft.md](docs/plan-fft.md)** for the phased roadmap.

## API

```go
import "github.com/go-fft/fft"

X := fft.FFT(x)          // forward DFT of []complex128, any length
y := fft.IFFT(X)         // inverse DFT, normalized by N (IFFT(FFT(x)) ≈ x)
S := fft.FFTReal(r)      // forward DFT of a []float64 signal (full spectrum)

R := fft.RFFT(r)         // real-input DFT, non-redundant N/2+1 bins (numpy.fft.rfft)
z := fft.IRFFT(R, len(r)) // back to []float64 (numpy.fft.irfft)

// Multi-dimensional, on flat row-major (C-order) data plus an explicit shape:
F2 := fft.FFT2(data, [2]int{rows, cols})   // 2-D DFT (numpy.fft.fft2)
d2 := fft.IFFT2(F2, [2]int{rows, cols})    // inverse 2-D DFT (numpy.fft.ifft2)
FN := fft.FFTN(data, shape)                // N-D DFT (numpy.fft.fftn)
dN := fft.IFFTN(FN, shape)                 // inverse N-D DFT (numpy.fft.ifftn)

// Real image-style 2-D transforms (last axis keeps cols/2+1 bins per row):
RG := fft.RFFT2(img, [2]int{rows, cols})   // numpy.fft.rfft2
zG := fft.IRFFT2(RG, [2]int{rows, cols})   // numpy.fft.irfft2

// Windowing and spectral helpers:
w  := fft.Hann(n)             // also Hamming, Blackman, BlackmanHarris, Bartlett
f  := fft.FFTFreq(n, d)       // bin frequencies (numpy.fft.fftfreq)
rf := fft.RFFTFreq(n, d)      // real-FFT bin frequencies (numpy.fft.rfftfreq)
p  := fft.PSD(sig, d)         // one-sided power spectral density (periodogram)
S  := fft.Spectrogram(sig, segment, overlap, fft.Hann(segment), d) // PSD frames

// Reusable plans — precompute the twiddle tables once, amortize across calls
// (no per-call sin/cos). The convenience FFT/RFFT functions above use an
// internal per-length plan cache, so they get this for free too.
p   := fft.NewPlan(n)          // complex transform plan of length n
p.FFT(dst, src)               // dst and src are []complex128 of length n (may alias)
p.IFFT(dst, src)              // normalized inverse
rp  := fft.NewRealPlan(n)      // real-input transform plan
rp.RFFT(dst, src)             // src []float64 (len n), dst []complex128 (len n/2+1)
rp.IRFFT(out, spec)           // out []float64 (len n), spec the half spectrum
```

The multi-dimensional transforms are separable: the 1-D FFT is applied along
each axis in turn. The forward transforms are unnormalized; the inverses divide
by the product of the transformed axis lengths. The shape must be positive and
its product must equal `len(data)`, else the call panics (numpy semantics).

Empty input returns an empty slice; length 1 returns a copy. `RFFT` keeps only
the lower `N/2+1` bins because a real signal's spectrum is conjugate-symmetric
(`X[N-k] = conj(X[k])`); `IRFFT` takes the target length `n` explicitly.

## Performance

`go-fft` uses a **split-radix** kernel for powers of two (≈⅓ fewer real
multiplies than radix-4), **mixed-radix Cooley–Tukey** for the other
highly-composite lengths (radix-2/3/4/5 straight-line butterflies plus a general
radix-p kernel for small primes), **Rader's algorithm** for primes (from N=700)
and **Bluestein's chirp-z** otherwise, with all twiddle factors cached per
length. It beats the pure-Go peer `gonum/dsp/fourier` (both `CGO_ENABLED=0`) at
**every** size — up to ~59× on primes (gonum's arbitrary-N path is O(N²)).

Against the C gold standard (FFTW/pocketfft via `numpy.fft` / `scipy.fft`,
single-threaded) on a 4-core arm64 Linux host, go-fft **wins outright** at
several shapes:

| transform | go-fft | scipy.fft | verdict |
|:--|---:|---:|:--|
| complex 65536 | **0.54 ms** | 1.20 ms | **go-fft ~2.2×** |
| complex 256 / 64 | **1.4 µs / 0.29 µs** | 2.3 µs / 1.8 µs | **go-fft wins** (small-N) |
| real 64 / 256 | **0.25 µs / 1.0 µs** | 1.9 µs / 2.3 µs | **go-fft wins** (small-N) |
| FFT2 1024×1024 | **9.6 ms** | 9.8 ms | **go-fft wins** (multicore) |
| FFT2 512×512 | **2.6 ms** | 2.4 ms | ~tie (beats numpy) |
| real 1024 | 4.5 µs | 3.7 µs | scipy ~1.2× |
| complex 1024 | 7.2 µs | 4.4 µs | scipy ~1.6× |
| complex 10007 (prime) | 0.68 ms | 0.24 ms | FFTW ~2.8× |

go-fft wins on large 1-D, large 2-D (the goroutine-parallel separable path
single-threaded pocketfft can't match), and the small-N rows. A split-radix
power-of-two engine (≈⅓ fewer real multiplies) plus a lowered Bluestein→Rader
prime crossover narrowed the rest: the real mid-range gap roughly halved (now
~1.2–1.9×), complex mid-range to ~1.6–2.2×, and primes from ~3–5× to ~2.4–2.9×.
Full methodology, every size, and the before/after optimization deltas are in
**[docs/perf.md](docs/perf.md)**. Reproduce the gonum head-to-head with
`cd bench && go test -bench=.` (gonum is isolated in a separate `bench/` module
so the library stays dependency-free) and the FFTW comparison with
`go test -bench=H2H .` + `python3 scripts/fftbench.py`.

## Why not cgo / FFTW?

FFTW3 is a C library: binding it reintroduces a C toolchain, cross-compilation
pain, and a non-Go build. A pure-Go implementation cross-compiles to every Go
target for free and is `CGO_ENABLED=0` clean — which is what makes it usable as
the FFT backend for an embedded Ruby (`go-embedded-ruby`) and for the wider
go-* ecosystem.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
