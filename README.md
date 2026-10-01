<p align="center"><img src="https://raw.githubusercontent.com/go-fft/brand/main/social/go-fft.png" alt="go-fft/fft" width="720"></p>

# fft — go-fft

[![Docs](https://img.shields.io/badge/docs-hugo%20%2B%20relearn-FF4088)](https://go-fft.github.io/docs/)
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
> and go-asmgen SIMD kernels, of two kinds. The **pointwise complex multiply**
> is bit-identical on four of the six targets (SSE2 on amd64, NEON on arm64,
> RVV on riscv64, the vector facility on s390x), with loong64 and ppc64le on
> the validated scalar path because the Go assembler lacks the vector-double
> ops they need. The **butterfly stage kernels** — whole radix-2 and radix-4
> passes rather than one multiply — are amd64, SSE2 across the baseline and
> **AVX2** where the CPU and the OS both allow it, selected at run time. All of
> it sits behind a validated per-arch split CI. The transform is also exposed to Ruby through the
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

// N-D and real 2-D plans: write into your slice, allocate nothing in steady
// state. FFTN/FFT2/RFFT2 return a new slice each call; on a many-core host
// that allocation, not the FFT, dominates a small repeated transform.
pn  := fft.NewPlanN(rows, cols)      // any rank: NewPlanN(d0, d1, d2, ...)
pn.FFT(dst, src)                     // row-major, len = product of the shape (may alias)
pn.IFFT(dst, src)                    // normalized inverse (matches IFFTN)
r2  := fft.NewRealPlan2(rows, cols)  // real 2-D plan (matches RFFT2/IRFFT2)
r2.RFFT(spec, img)                   // spec len r2.SpectrumLen() = rows*(cols/2+1)
r2.IRFFT(img, spec)                  // normalized inverse
```

The multi-dimensional transforms are separable: the 1-D FFT is applied along
each axis in turn. The forward transforms are unnormalized; the inverses divide
by the product of the transformed axis lengths. The shape must be positive and
its product must equal `len(data)`, else the call panics (numpy semantics).

Empty input returns an empty slice; length 1 returns a copy. `RFFT` keeps only
the lower `N/2+1` bins because a real signal's spectrum is conjugate-symmetric
(`X[N-k] = conj(X[k])`); `IRFFT` takes the target length `n` explicitly.

## Performance

`go-fft` transforms every length whose prime factors are all ≤ 13 with an
**iterative mixed-radix (Stockham)** engine (radix-8/4/2/3/5/7 straight-line
passes plus a general radix-p pass for 11 and 13). On amd64 with AVX2 its
radix-2/3/4/5/8 passes run as generated **AVX2 kernels**, and a power of two of
256 points or more runs **AVX-512 kernels** where the CPU and OS support them;
both are bit-identical to the Go passes. Powers of two above 4096 on amd64 (16384 with AVX-512), and every power of two on
riscv64/ppc64le/loong64/s390x, use an **iterative radix-4 kernel** instead
(with SIMD butterflies on amd64). A prime whose N−1 is 7-smooth uses **Rader's algorithm**, and every
other length **Bluestein's chirp-z**, both convolving on the Stockham engine,
with all twiddle factors cached per
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
dedicated Hermitian real kernel.

### The amd64 butterfly kernels

The **pointwise complex multiply** kernels (go-asmgen; amd64, arm64, riscv64,
s390x) are built and validated bit-identical, and are *not* routed on the hot
path off amd64: measured, they only tie or lose to the gc autovectorizer there
(see `docs/plan-fft.md` Phase 4).

The **butterfly stage kernels** are a different thing and they are routed. They
run a whole radix-4 or radix-2 pass — both the loop over groups and the loop
over positions — inside one call. On amd64 that is SSE2 across the baseline,
and AVX2, processing two butterflies per YMM register, where both the CPU and
the operating system support it. `internal/kernels` picks between them at run
time; everything else keeps the SSE2 path.

    Intel Core i5-14600K, GOAMD64=v1, one thread, process affinity to CPU 0.
    Median of five 400 ms repetitions, reusable RealPlan, 0 B/op and 0 allocs/op
    on every row. Raw runs in benchmarks/results/amd64-avx2-20260922/.

| real input | SSE2 | AVX2 | time |
| ---: | ---: | ---: | ---: |
| 64 | 135.5 ns | 95.99 ns | −29.2% |
| 256 | 562.3 ns | 402.0 ns | −28.5% |
| 512 | 1,145 ns | 919.2 ns | −19.7% |
| 1,024 | 2,521 ns | 1,734 ns | −31.2% |
| 2,048 | 5,230 ns | 4,256 ns | −18.6% |
| 4,096 | 11,349 ns | 7,398 ns | −34.8% |
| 16,384 | 50,527 ns | 32,676 ns | −35.3% |

⛔ The arithmetic order is preserved exactly: separately rounded multiply, add
and subtract, no FMA and no reassociation, so the AVX2 result is **bit-identical**
to the SSE2 one and to the scalar oracle. That is what
`rfft_bitexact_amd64_test.go` and `butterfly_avx2_amd64_test.go` assert, the
second by running every shape down BOTH paths and comparing `math.Float64bits`.
A faster transform that answers differently is not the same transform. The remaining
identified lever is a SIMD/cache-blocked real (r2c) kernel (the iterative
mixed-radix engine for smooth lengths and the large-prime convolution has landed) — see **[BENCHMARKS.md](BENCHMARKS.md)**'s
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
