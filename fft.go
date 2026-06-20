package fft

// FFT returns the forward discrete Fourier transform of x, using the
// unnormalized convention
//
//	X[k] = sum_{n=0}^{N-1} x[n] * exp(-2πi·k·n/N).
//
// Highly-composite lengths (whose prime factors are all small) use mixed-radix
// Cooley–Tukey; a length with a large prime factor falls back to Bluestein's
// chirp-z algorithm. Twiddle factors are cached per length in an internal plan
// cache, so repeated transforms of the same length recompute no sin/cos. The
// input is not modified. Empty input returns an empty (non-nil) slice; length 1
// returns a copy.
func FFT(x []complex128) []complex128 {
	n := len(x)
	out := make([]complex128, n)
	if n <= 1 {
		copy(out, x)
		return out
	}
	cachedPlan(n).FFT(out, x)
	return out
}

// IFFT returns the inverse discrete Fourier transform of x, normalized by N so
// that IFFT(FFT(x)) ≈ x to floating-point tolerance:
//
//	x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2πi·k·n/N).
//
// The input is not modified.
func IFFT(x []complex128) []complex128 {
	n := len(x)
	out := make([]complex128, n)
	if n <= 1 {
		copy(out, x)
		return out
	}
	cachedPlan(n).IFFT(out, x)
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

// complexConj returns the complex conjugate of z.
func complexConj(z complex128) complex128 {
	return complex(real(z), -imag(z))
}
