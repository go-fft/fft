// Package fft is a pure-Go (cgo-free) fast Fourier transform library — the
// numpy.fft / scipy.fft equivalent for Go.
//
// It computes the discrete Fourier transform (DFT) of complex and real signals
// of any length, with no dependency on the native FFTW3 C library. A length
// whose prime factors are all small uses an iterative mixed-radix (Stockham)
// engine (radix-8/4/2/3/5/7 straight-line passes plus a general radix-p pass
// for 11/13), whose radix-2/3/4/5/8 passes run as AVX2 (or, for powers of two,
// AVX-512) kernels on amd64, bit-identical to the Go passes; a power of two on
// riscv64/loong64/s390x, or on amd64 without AVX2, uses an iterative radix-4
// kernel instead (with SSE2 butterflies on amd64). A prime whose N-1 is
// 7-smooth uses Rader's algorithm and any other length Bluestein's chirp-z
// algorithm, so any length transforms correctly and fast. Twiddle factors are
// precomputed and cached per length (see Plan / NewPlan), so repeated
// transforms of one length recompute no sin/cos.
//
// The multi-dimensional transforms (FFT2/FFTN and their real and inverse forms)
// are separable: they apply a 1-D transform along each axis, and the
// independent lines of a large axis run in parallel across goroutines — a
// multicore path single-threaded references such as pocketfft cannot take.
// NewPlanN and NewRealPlan2 build reusable N-D plans that write into the
// caller's slice and allocate nothing in steady state. See BENCHMARKS.md for
// the head-to-head benchmarks against FFTW, numpy.fft and scipy.fft.
//
// The forward transform follows the unnormalized convention
//
//	X[k] = sum_{n=0}^{N-1} x[n] * exp(-2πi·k·n/N)
//
// and the inverse divides by N, so IFFT(FFT(x)) ≈ x to floating-point
// tolerance.
//
// RFFT returns only the non-redundant N/2+1 bins of a real signal's spectrum
// (numpy.fft.rfft) and IRFFT inverts it back to a real signal of a
// caller-specified length (numpy.fft.irfft). See docs/plan-fft.md for the
// roadmap.
package fft
