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

`go-fft` uses **mixed-radix Cooley–Tukey** (radix-2/3/4/5 straight-line
butterflies plus a general radix-p kernel for small primes), reserving
**Bluestein's chirp-z** for lengths with a large prime factor, with all twiddle
factors cached per length. Benchmarked against the pure-Go peer
`gonum/dsp/fourier` (the fairest head-to-head, both `CGO_ENABLED=0`) on an Apple
M4 Max, best of 3, **go-fft wins at every complex-FFT size**:

| N | go-fft | gonum | go-fft speedup |
|---:|---:|---:|---:|
| 1024 | **9.3 µs** | 16.6 µs | 1.79× |
| 4096 | **40 µs** | 82 µs | 2.03× |
| 65536 | **0.83 ms** | 1.65 ms | 1.98× |
| 1000 (2³·5³) | **9.7 µs** | 17.0 µs | 1.75× |
| 1296 (2⁴·3⁴) | **15 µs** | 24 µs | 1.55× |
| 10007 (prime) | **1.19 ms** | 70.5 ms | **59×** |

Against the C gold standard (FFTW/pocketfft via `scipy.fft`) on the same Linux
arm64 host, go-fft is within ~2–3× in the mid-range and *matches* pocketfft at
large complex N — close to the pure-Go scalar ceiling. Full methodology, the
before/after optimization deltas, and the honest FFTW comparison table are in
**[docs/perf.md](docs/perf.md)**. Reproduce with `cd bench && go test -bench=.`
(gonum is isolated in a separate `bench/` module so the library stays
dependency-free).

## Why not cgo / FFTW?

FFTW3 is a C library: binding it reintroduces a C toolchain, cross-compilation
pain, and a non-Go build. A pure-Go implementation cross-compiles to every Go
target for free and is `CGO_ENABLED=0` clean — which is what makes it usable as
the FFT backend for an embedded Ruby (`go-embedded-ruby`) and for the wider
go-* ecosystem.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
