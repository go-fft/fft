# fft — go-fft

[![Docs](https://img.shields.io/badge/docs-mkdocs--material-9B1C2E)](https://go-fft.github.io/docs/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![Go](https://img.shields.io/badge/go-1.26.4%2B-00ADD8)](https://go.dev/dl/)
[![Status](https://img.shields.io/badge/status-phase%202-9a6700)](docs/plan-fft.md)

**A pure-Go (no cgo) FFT library** — the `numpy.fft` / `scipy.fft` equivalent for
Go. It computes the discrete Fourier transform of complex and real signals of
**any length**, with no dependency on the native FFTW3 C library.

Ruby has no cgo-free FFT (every option wraps FFTW3); `gonum/dsp/fourier` is pure
Go but its optimized assembly is amd64-only. This module is a fully portable
scalar core today, with SIMD kernels planned across all six 64-bit Go targets
(amd64, arm64, riscv64, loong64, ppc64le, s390x) via
[go-asmgen](https://github.com/go-asmgen).

> Status: **Phase 2** — a correct pure-Go complex FFT (radix-2 Cooley–Tukey for
> power-of-two lengths, Bluestein's chirp-z for arbitrary lengths), the
> real-optimized `RFFT`/`IRFFT`, and the multi-dimensional transforms
> (`FFT2`/`IFFT2`, `FFTN`/`IFFTN`, `RFFT2`/`IRFFT2`). See
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
```

The multi-dimensional transforms are separable: the 1-D FFT is applied along
each axis in turn. The forward transforms are unnormalized; the inverses divide
by the product of the transformed axis lengths. The shape must be positive and
its product must equal `len(data)`, else the call panics (numpy semantics).

Empty input returns an empty slice; length 1 returns a copy. `RFFT` keeps only
the lower `N/2+1` bins because a real signal's spectrum is conjugate-symmetric
(`X[N-k] = conj(X[k])`); `IRFFT` takes the target length `n` explicitly.

## Why not cgo / FFTW?

FFTW3 is a C library: binding it reintroduces a C toolchain, cross-compilation
pain, and a non-Go build. A pure-Go implementation cross-compiles to every Go
target for free and is `CGO_ENABLED=0` clean — which is what makes it usable as
the FFT backend for an embedded Ruby (`go-embedded-ruby`) and for the wider
go-* ecosystem.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
