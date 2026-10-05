package fft

// This file holds the 1-D transforms with numpy.fft's optional arguments (n
// and norm), and the Hermitian pair hfft/ihfft. The semantics follow
// numpy.fft (https://numpy.org/doc/stable/reference/routines.fft.html):
//
//   - n truncates the input to n points or zero-pads it to n points before
//     the transform (numpy.fft.fft, "n: Length of the transformed axis of the
//     output. If n is smaller than the length of the input, the input is
//     cropped. If it is larger, the input is padded with zeros.").
//   - norm scales the pair as described on Norm: "backward" leaves the
//     forward unscaled and divides the inverse by n, "ortho" scales both by
//     1/sqrt(n), "forward" divides the forward by n.

// FFTWith is FFT with numpy.fft.fft's n and norm arguments: the input is
// truncated or zero-padded to o.N points (len(x) when o.N is zero) and the
// forward transform is scaled as o.Norm says. FFTWith(x, Options{}) equals
// FFT(x). The input is not modified. o.Axes must be nil.
func FFTWith(x []complex128, o Options) []complex128 {
	return complexWith("FFTWith", x, o, false)
}

// IFFTWith is IFFT with numpy.fft.ifft's n and norm arguments: the input is
// truncated or zero-padded to o.N points (len(x) when o.N is zero) and the
// inverse transform is scaled as o.Norm says (1/n under the default
// NormBackward). IFFTWith(x, Options{}) equals IFFT(x). The input is not
// modified. o.Axes must be nil.
func IFFTWith(x []complex128, o Options) []complex128 {
	return complexWith("IFFTWith", x, o, true)
}

func complexWith(name string, x []complex128, o Options, inverse bool) []complex128 {
	n := o.check1D(name, len(x))
	f := o.Norm.scale(n, inverse)
	out := make([]complex128, n)
	copy(out, x) // truncates to n, or leaves the zero padding past len(x)
	if n > 1 {
		cachedPlan(n).execute(out, out, inverse)
	}
	scaleComplex(out, f)
	return out
}

// RFFTWith is RFFT with numpy.fft.rfft's n and norm arguments: the real input
// is truncated or zero-padded to o.N points (len(x) when o.N is zero), the
// result holds o.N/2+1 bins, and the transform is scaled as o.Norm says.
// RFFTWith(x, Options{}) equals RFFT(x). The input is not modified. o.Axes
// must be nil.
func RFFTWith(x []float64, o Options) []complex128 {
	n := o.check1D("RFFTWith", len(x))
	return rfftScaled(x, n, o.Norm.scale(n, false))
}

// rfftScaled is the real forward transform of x truncated or zero-padded to n
// points, multiplied by f.
func rfftScaled(x []float64, n int, f float64) []complex128 {
	if n == 0 {
		return []complex128{}
	}
	in := x
	if len(x) != n {
		in = make([]float64, n)
		copy(in, x)
	}
	out := cachedRealPlan(n).RFFT(make([]complex128, n/2+1), in)
	scaleComplex(out, f)
	return out
}

// IRFFTWith is IRFFT with numpy.fft.irfft's n and norm arguments. o.N is the
// length of the real output; zero selects numpy's default 2*(len(spectrum)-1),
// which reconstructs an even-length signal (pass the length explicitly for an
// odd one: n and n-1 keep the same number of bins). The spectrum is read up to
// o.N/2+1 bins and missing bins are zero, as for IRFFT; the imaginary parts of
// the zero-frequency bin and, for an even length, of the Nyquist bin are
// ignored, as numpy does. The result is scaled as o.Norm says (1/n under the
// default NormBackward). A length of zero (a spectrum of fewer than two bins
// and o.N zero) gives an empty result, where numpy raises. The input is not
// modified. o.Axes must be nil.
//
// The output length is the caller's o.N whatever the spectrum's length: bound
// it when it comes from untrusted input (see SECURITY.md).
func IRFFTWith(spectrum []complex128, o Options) []float64 {
	n := o.check1D("IRFFTWith", 2*(len(spectrum)-1))
	f := unitInverseRescale(o.Norm.scale(n, true), n)
	out := IRFFT(spectrum, n)
	scaleReal(out, f)
	return out
}

// HFFT returns the DFT of a signal with Hermitian symmetry, given its first
// half x (x[0] .. x[n/2]; the rest is conj(x[n-k])), as a real slice of n
// values: numpy.fft.hfft(x, n). It is the forward transform, unscaled:
//
//	X[k] = sum_{j=0}^{n-1} y[j] * exp(-2πi·k·j/n),  y[j] = x[j], y[n-j] = conj(x[j]).
//
// numpy defines it as irfft(conj(x), n) multiplied by n; HFFT computes the
// same. n is the length of the output, as for IRFFT (n and n-1 share the same
// half); n <= 0 returns an empty slice. x is read up to n/2+1 values and
// missing ones are zero. The input is not modified.
//
// See https://numpy.org/doc/stable/reference/generated/numpy.fft.hfft.html.
func HFFT(x []complex128, n int) []float64 {
	if n <= 0 {
		return []float64{}
	}
	return HFFTWith(x, Options{N: n})
}

// HFFTWith is HFFT with numpy.fft.hfft's n and norm arguments: o.N is the
// length of the real output (zero selects numpy's default 2*(len(x)-1)) and
// o.Norm scales it as a FORWARD transform — unscaled under the default
// NormBackward, 1/sqrt(n) under NormOrtho, 1/n under NormForward. numpy
// implements this as irfft(conj(x), n, norm=<the opposite direction>), which
// is the same thing. o.Axes must be nil.
func HFFTWith(x []complex128, o Options) []float64 {
	n := o.check1D("HFFTWith", 2*(len(x)-1))
	f := o.Norm.scale(n, false)
	if n == 0 {
		return []float64{}
	}
	c := make([]complex128, min(len(x), n/2+1))
	for i := range c {
		c[i] = complexConj(x[i])
	}
	out := IRFFT(c, n) // normalized by 1/n: undo it into f
	scaleReal(out, unitInverseRescale(f, n))
	return out
}

// IHFFT is the inverse of HFFT: it returns the first n/2+1 values of the
// Hermitian-symmetric signal whose real DFT is x (n = len(x)), normalized by
// 1/n — numpy.fft.ihfft(x), which numpy defines as conj(rfft(x))/n. Empty
// input returns an empty slice. The input is not modified.
//
// See https://numpy.org/doc/stable/reference/generated/numpy.fft.ihfft.html.
func IHFFT(x []float64) []complex128 {
	return IHFFTWith(x, Options{})
}

// IHFFTWith is IHFFT with numpy.fft.ihfft's n and norm arguments: the input is
// truncated or zero-padded to o.N points (len(x) when o.N is zero) and o.Norm
// scales the result as an INVERSE transform — 1/n under the default
// NormBackward, 1/sqrt(n) under NormOrtho, unscaled under NormForward — so
// that HFFTWith(IHFFTWith(x, o), Options{N: len(x), Norm: o.Norm}) ≈ x for
// every o.Norm. o.Axes must be nil.
func IHFFTWith(x []float64, o Options) []complex128 {
	n := o.check1D("IHFFTWith", len(x))
	out := rfftScaled(x, n, o.Norm.scale(n, true))
	for i, v := range out {
		out[i] = complexConj(v)
	}
	return out
}
