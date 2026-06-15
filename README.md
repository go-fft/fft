# fft — go-fft

[![Docs](https://img.shields.io/badge/docs-mkdocs--material-9B1C2E)](https://go-fft.github.io/docs/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![Go](https://img.shields.io/badge/go-1.26.4%2B-00ADD8)](https://go.dev/dl/)
[![Status](https://img.shields.io/badge/status-phase%201-9a6700)](docs/plan-fft.md)

**A pure-Go (no cgo) FFT library** — the `numpy.fft` / `scipy.fft` equivalent for
Go. It computes the discrete Fourier transform of complex and real signals of
**any length**, with no dependency on the native FFTW3 C library.

Ruby has no cgo-free FFT (every option wraps FFTW3); `gonum/dsp/fourier` is pure
Go but its optimized assembly is amd64-only. This module is a fully portable
scalar core today, with SIMD kernels planned across all six 64-bit Go targets
(amd64, arm64, riscv64, loong64, ppc64le, s390x) via
[go-asmgen](https://github.com/go-asmgen).

> Status: **Phase 1** — a correct pure-Go complex FFT (radix-2 Cooley–Tukey for
> power-of-two lengths, Bluestein's chirp-z for arbitrary lengths) plus the
> real-optimized `RFFT`/`IRFFT`. See **[docs/plan-fft.md](docs/plan-fft.md)** for
> the phased roadmap.

## API

```go
import "github.com/go-fft/fft"

X := fft.FFT(x)          // forward DFT of []complex128, any length
y := fft.IFFT(X)         // inverse DFT, normalized by N (IFFT(FFT(x)) ≈ x)
S := fft.FFTReal(r)      // forward DFT of a []float64 signal (full spectrum)

R := fft.RFFT(r)         // real-input DFT, non-redundant N/2+1 bins (numpy.fft.rfft)
z := fft.IRFFT(R, len(r)) // back to []float64 (numpy.fft.irfft)
```

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
