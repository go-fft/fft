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
// signal as the full complex spectrum (length len(x)). For an even length it
// runs the real transform (RealPlan.RFFT, a half-length complex FFT) into the
// lower half of the result and fills the upper half with the conjugate mirror
// X[N-k] = conj(X[k]), so the result is exactly Hermitian and about twice as
// fast as transforming the promoted signal; its error is the real transform's
// (see docs/accuracy.md), and the bits differ from FFT of the promoted signal
// in the last place. An odd length cannot be packed and runs the complex FFT of
// the promoted signal in place, as before. FFTRealWith with Options.Exact
// always promotes and transforms, bit for bit FFT of the promoted signal.
func FFTReal(x []float64) []complex128 {
	return fftReal(x, len(x), arithDefault)
}

// fftReal is the full spectrum of x truncated or zero-padded to n points, on
// plans of arithmetic a: promote-then-FFT when a is Exact or n is odd, the real
// transform plus the conjugate mirror otherwise.
func fftReal(x []float64, n int, a arith) []complex128 {
	out := make([]complex128, n)
	if n <= 1 || a == arithExact || n%2 != 0 {
		for i, v := range x[:min(len(x), n)] {
			out[i] = complex(v, 0)
		}
		if n > 1 {
			cachedPlanA(n, a).FFT(out, out)
		}
		return out
	}
	in := x
	if len(x) != n {
		in = make([]float64, n)
		copy(in, x)
	}
	return cachedRealPlanA(n, a).FFTReal(out, in)
}

// complexConj returns the complex conjugate of z.
func complexConj(z complex128) complex128 {
	return complex(real(z), -imag(z))
}
