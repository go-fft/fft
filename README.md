<p align="center"><img src="https://raw.githubusercontent.com/go-fft/brand/main/social/go-fft.png" alt="go-fft/fft" width="720"></p>

# fft — go-fft

[![Docs](https://img.shields.io/badge/docs-mkdocs--material-9B1C2E)](https://go-fft.github.io/docs/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![Go](https://img.shields.io/badge/go-1.26.4%2B-00ADD8)](https://go.dev/dl/)
[![Status](https://img.shields.io/badge/status-phase%205-1a7f37)](docs/plan-fft.md)

**A pure-Go (no cgo) FFT library** — the `numpy.fft` / `scipy.fft` equivalent for
Go. It computes the discrete Fourier transform of complex and real signals of
**any length**, with no dependency on the native FFTW3 C library.

Ruby has no cgo-free FFT (every option wraps FFTW3); `gonum/dsp/fourier` is pure
Go but its optimized assembly is amd64-only. This module is a fully portable
scalar core, with SIMD kernels generated across the six 64-bit Go targets
(amd64, arm64, riscv64, loong64, ppc64le, s390x) via
[go-asmgen](https://github.com/go-asmgen).

> Status: **Phase 5** — a correct pure-Go complex FFT (radix-2 Cooley–Tukey for
> power-of-two lengths, Bluestein's chirp-z for arbitrary lengths), the
> real-optimized `RFFT`/`IRFFT`, the multi-dimensional transforms
> (`FFT2`/`IFFT2`, `FFTN`/`IFFTN`, `RFFT2`/`IRFFT2`), the windowing / spectral
> helpers (windows, `FFTFreq`/`RFFTFreq`, `PSD`, `Spectrogram`), and go-asmgen
> SIMD kernels (bit-identical pointwise complex multiply: SSE2 on amd64, NEON on
> arm64, RVV on riscv64, the vector facility on s390x — **four of the six
> targets**) behind a validated per-arch split CI, with loong64 and ppc64le on
> the validated scalar path (the Go assembler lacks the vector-double ops they
> need). The transform is also exposed to Ruby through the
> [go-embedded-ruby](https://github.com/go-embedded-ruby/ruby) `FFT` module
> (Phase 5). See **[docs/plan-fft.md](docs/plan-fft.md)** for the phased roadmap.

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
**every** size measured — typically 3–5× on composite N and ~30×–200× on
primes (gonum's arbitrary-N path is a naive Bluestein with no Rader path).

Against the C gold standard — native **FFTW 3.3.11** plus pocketfft via
`numpy.fft` / `scipy.fft`, all single-threaded — on an **Apple M4 Max** (macOS
26.5), go-fft **wins outright** at the large 2-D shapes and the small-N rows,
and is at-or-near parity on very large 1-D transforms:

| transform | go-fft | FFTW | scipy.fft | verdict |
|:--|---:|---:|---:|:--|
| complex 256 | **0.72 µs** | 0.42 µs | 2.30 µs | **beats pocketfft ~3.2×** |
| FFT2 1024×1024 | **2.78 ms** | 6.30 ms | 5.32 ms | **beats all** (multicore) |
| FFT2 512×512 | **0.63 ms** | 1.25 ms | 0.96 ms | **beats all** (multicore) |
| complex 65,536 | 0.38 ms | 0.32 ms | 0.50 ms | **near parity with FFTW** (1.17×), beats scipy |
| real 1,048,576 | 4.47 ms | 2.96 ms | 6.37 ms | beats scipy ~1.4×, lags FFTW ~1.5× |
| complex 1024 | 3.40 µs | 2.13 µs | 4.47 µs | lags FFTW ~1.6×, beats scipy ~1.3× |
| complex 1009 (prime) | 17.2 µs | 13.7 µs | 19.0 µs | lags FFTW ~1.3×, near parity with scipy |
| complex 10007 (prime) | 0.37 ms | 0.17 ms | 0.28 ms | lags FFTW ~2.2× |

go-fft wins outright on the large 2-D shapes (the goroutine-parallel separable
path single-threaded FFTW/pocketfft can't match) and the small-N rows (the
Python FFI tax dominates pocketfft there), and is at-or-near parity with FFTW
on very large 1-D transforms. FFTW still leads the single-core power-of-two and
smooth-composite mid-range with its hand-written NEON SIMD codelets and
dedicated Hermitian real kernel; go-fft's own SIMD butterfly kernels
(go-asmgen, amd64/arm64/s390x/riscv64) are built and validated bit-identical
but are *not* routed on the hot path off amd64 — measured to only tie or lose
to the gc autovectorizer there (see `docs/plan-fft.md` Phase 4). The remaining
identified levers are a SIMD/cache-blocked real (r2c) kernel and an iterative
mixed-radix engine for the large-prime convolution — see **[BENCHMARKS.md](BENCHMARKS.md)**'s
"Lagging ops" section for the full, per-op root-cause breakdown. Full
methodology, every size, and GFLOP/s are also in BENCHMARKS.md. Reproduce the
whole sweep with `benchmarks/run.sh` (go-fft + gonum via `go test -bench`, native
FFTW via a C harness, numpy/scipy via Python; correctness-gated; gonum is
isolated in the separate `benchmarks/` module so the library stays
dependency-free).

## Why not cgo / FFTW?

FFTW3 is a C library: binding it reintroduces a C toolchain, cross-compilation
pain, and a non-Go build. A pure-Go implementation cross-compiles to every Go
target for free and is `CGO_ENABLED=0` clean — which is what makes it usable as
the FFT backend for an embedded Ruby (`go-embedded-ruby`) and for the wider
go-* ecosystem.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
