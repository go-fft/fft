package fft

import (
	"math"

	"github.com/go-fft/fft/internal/kernels"
)

// FFT returns the forward discrete Fourier transform of x, using the
// unnormalized convention
//
//	X[k] = sum_{n=0}^{N-1} x[n] * exp(-2πi·k·n/N).
//
// Power-of-two lengths use radix-2 Cooley–Tukey; any other length uses
// Bluestein's chirp-z algorithm. The input is not modified. Empty input
// returns an empty (non-nil) slice; length 1 returns a copy.
func FFT(x []complex128) []complex128 {
	return transform(x, false)
}

// IFFT returns the inverse discrete Fourier transform of x, normalized by N so
// that IFFT(FFT(x)) ≈ x to floating-point tolerance:
//
//	x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2πi·k·n/N).
//
// The input is not modified.
func IFFT(x []complex128) []complex128 {
	out := transform(x, true)
	n := len(out)
	if n == 0 {
		return out
	}
	inv := complex(1/float64(n), 0)
	for i := range out {
		out[i] *= inv
	}
	return out
}

// FFTReal returns the forward discrete Fourier transform of a real-valued
// signal as the full complex spectrum (length len(x)). It is a convenience
// wrapper that promotes x to complex128 and calls FFT.
func FFTReal(x []float64) []complex128 {
	c := make([]complex128, len(x))
	for i, v := range x {
		c[i] = complex(v, 0)
	}
	return FFT(c)
}

// transform dispatches to the radix-2 or Bluestein core and never mutates the
// caller's slice.
func transform(x []complex128, inverse bool) []complex128 {
	n := len(x)
	out := make([]complex128, n)
	copy(out, x)
	if n <= 1 {
		return out
	}
	if kernels.IsPowerOfTwo(n) {
		kernels.BitReverse(out)
		kernels.Radix2(out, inverse)
		return out
	}
	return bluestein(out, inverse)
}

// bluestein computes a length-N DFT for arbitrary N via the chirp-z transform,
// reducing it to a power-of-two convolution evaluated with radix-2 FFTs. The
// sign of the transform follows inverse (normalization is handled by IFFT).
func bluestein(x []complex128, inverse bool) []complex128 {
	n := len(x)
	sign := -1.0
	if inverse {
		sign = 1.0
	}

	// Smallest power of two >= 2N-1 for a linear convolution of length-N chirps.
	m := 1
	for m < 2*n-1 {
		m <<= 1
	}

	// Chirp w[j] = exp(sign·πi·j²/N). j² mod 2N keeps the angle accurate for
	// large N.
	w := make([]complex128, n)
	for j := 0; j < n; j++ {
		jj := (j * j) % (2 * n)
		ang := sign * math.Pi * float64(jj) / float64(n)
		w[j] = complex(math.Cos(ang), math.Sin(ang))
	}

	// a[j] = x[j]·w[j], zero-padded to length m.
	a := make([]complex128, m)
	for j := 0; j < n; j++ {
		a[j] = x[j] * w[j]
	}

	// b[j] = conj(w[j]) for j in [0,N), mirrored so the cyclic convolution of
	// a and b yields the chirp-z sum.
	b := make([]complex128, m)
	b[0] = complexConj(w[0])
	for j := 1; j < n; j++ {
		c := complexConj(w[j])
		b[j] = c
		b[m-j] = c
	}

	// Convolution via FFT: out = IFFT(FFT(a)·FFT(b)).
	kernels.BitReverse(a)
	kernels.Radix2(a, false)
	kernels.BitReverse(b)
	kernels.Radix2(b, false)
	for i := 0; i < m; i++ {
		a[i] *= b[i]
	}
	kernels.BitReverse(a)
	kernels.Radix2(a, true)
	invM := complex(1/float64(m), 0)
	for i := 0; i < m; i++ {
		a[i] *= invM
	}

	// X[k] = w[k]·conv[k].
	out := make([]complex128, n)
	for k := 0; k < n; k++ {
		out[k] = a[k] * w[k]
	}
	return out
}

// complexConj returns the complex conjugate of z.
func complexConj(z complex128) complex128 {
	return complex(real(z), -imag(z))
}
