// Package fft is a pure-Go (cgo-free) fast Fourier transform library — the
// numpy.fft / scipy.fft equivalent for Go.
//
// It computes the discrete Fourier transform (DFT) of complex and real signals
// of any length, with no dependency on the native FFTW3 C library. A length
// whose prime factors are all small uses an iterative mixed-radix (Stockham)
// engine (radix-8/4/2/3/5/7 straight-line passes plus a general radix-p pass
// for 11/13), whose radix-2/3/4/5/8 passes run as AVX2 (or, for powers of two,
// AVX-512) kernels on amd64, bit-identical to the Go passes; a power of two on
// loong64/s390x, above 65536 on riscv64, or on amd64 without AVX2, uses an
// iterative radix-4 kernel instead (with SSE2 butterflies on amd64). A prime whose N-1 is
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
//
// # Fused multiply-add (Options.FMA)
//
// By default every amd64 kernel is bit-identical to the package's Go passes
// compiled without fused multiply-add, so a result does not depend on which
// kernel ran. Options{FMA: true} (FFTWith and the other ...With functions,
// NewPlanWith, NewRealPlanWith, NewPlanNWith, NewRealPlan2With) gives that up
// for speed on amd64 CPUs with FMA3: every twiddle product between passes,
// the √2/2 products of the radix-8 butterfly and the internal twiddles of
// the radix-9 and radix-16 butterflies are fused, as FFTW's FMA codelets
// fuse them. The results then differ from the default's in the last bits,
// and are as accurate: against scipy.fft in long double on the same inputs,
// the relative RMS error of 90 transforms (complex 16 to 2^20, composites,
// Rader and Bluestein primes, real, inverse and 2-D) was 0.90 to 1.04 times
// the default's, 1.1e-15 at most (BENCHMARKS.md, Round 41). The
// float32 transforms ignore the option; elsewhere than amd64 with FMA3 it
// builds the default plans. Each FMA kernel is bit-identical to a Go
// reference written with math.FMA, and FMA plans are cached apart from the
// default ones.
//
// An FMA plan fuses only where fusing was measured to pay (Round 42): on
// Intel CPUs with the AVX-512 split layout (Cascade Lake) and on AMD CPUs
// with the AVX2 one (Zen 3), a table gives, per length, the passes that
// fuse (the others run the default kernels) and, on the former, radix-32
// passes for 1,024 and 2,048 points, which only FMA plans have. Elsewhere
// every pass fuses. A transform of a length the table leaves unfused is the
// default transform, bit for bit.
package fft
