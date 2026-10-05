package fft

// This file holds the single-precision transforms with numpy's optional
// arguments: the float32 counterparts of options1d.go and optionsnd.go. They
// take the same Options, validated by the same code (Options.check1D,
// Options.checkND, validateShape), and follow the same semantics; only the
// element types differ. Every Norm factor is computed in float64 and rounded
// once to float32, and applied in one pass: an inverse that the float64 path
// computes as "normalize by 1/n, then rescale" runs unscaled here and is
// scaled once, so no factor is rounded twice.
//
// Precision follows scipy.fft, which keeps single precision: complex64 and
// float32 in, complex64 and float32 out (numpy.fft does too since NumPy 2.0;
// NumPy 1.x upcast to float64).

// FFT32With is FFT32 with numpy.fft.fft's n and norm arguments, as FFTWith:
// the input is truncated or zero-padded to o.N points (len(x) when o.N is
// zero) and the transform is scaled as o.Norm says. The input is not
// modified. o.Axes must be nil.
func FFT32With(x []complex64, o Options) []complex64 {
	return f32ndComplexWith("FFT32With", x, o, false)
}

// IFFT32With is IFFT32 with numpy.fft.ifft's n and norm arguments, as
// IFFTWith (1/n under the default NormBackward). o.Axes must be nil.
func IFFT32With(x []complex64, o Options) []complex64 {
	return f32ndComplexWith("IFFT32With", x, o, true)
}

func f32ndComplexWith(name string, x []complex64, o Options, inverse bool) []complex64 {
	n := o.check1D(name, len(x))
	out := make([]complex64, n)
	copy(out, x) // truncates to n, or leaves the zero padding past len(x)
	p := cachedPlan32(n)
	if inverse {
		return p.IFFTNorm(out, out, o.Norm)
	}
	return p.FFTNorm(out, out, o.Norm)
}

// RFFT32With is RFFT32 with numpy.fft.rfft's n and norm arguments, as
// RFFTWith: the result holds n/2+1 bins of the input truncated or zero-padded
// to n = o.N points (len(x) when o.N is zero). o.Axes must be nil.
func RFFT32With(x []float32, o Options) []complex64 {
	n := o.check1D("RFFT32With", len(x))
	return f32ndRFFT(x, n, o.Norm)
}

// f32ndRFFT is the real forward transform of x truncated or zero-padded to n
// points, scaled as m says for a forward transform.
func f32ndRFFT(x []float32, n int, m Norm) []complex64 {
	if n == 0 {
		return []complex64{}
	}
	in := x
	if len(x) != n {
		in = make([]float32, n)
		copy(in, x)
	}
	return cachedRealPlan32(n).RFFTNorm(make([]complex64, n/2+1), in, m)
}

// IRFFT32With is IRFFT32 with numpy.fft.irfft's n and norm arguments, as
// IRFFTWith: o.N is the length of the real output, zero selecting numpy's
// default 2*(len(spectrum)-1). o.Axes must be nil. The output length is the
// caller's o.N whatever the spectrum's length: bound it when it comes from
// untrusted input (see SECURITY.md).
func IRFFT32With(spectrum []complex64, o Options) []float32 {
	n := o.check1D("IRFFT32With", 2*(len(spectrum)-1))
	if n == 0 {
		return []float32{}
	}
	return cachedRealPlan32(n).IRFFTNorm(make([]float32, n), spectrum, o.Norm)
}

// HFFT32 is HFFT in single precision: the real DFT, of n points, of the
// Hermitian-symmetric signal whose first half is x (numpy.fft.hfft(x, n)).
// n <= 0 returns an empty slice. The input is not modified.
func HFFT32(x []complex64, n int) []float32 {
	if n <= 0 {
		return []float32{}
	}
	return HFFT32With(x, Options{N: n})
}

// HFFT32With is HFFTWith in single precision: o.N is the output length (zero
// selects 2*(len(x)-1)) and o.Norm scales it as a forward transform. o.Axes
// must be nil.
func HFFT32With(x []complex64, o Options) []float32 {
	n := o.check1D("HFFT32With", 2*(len(x)-1))
	if n == 0 {
		return []float32{}
	}
	c := make([]complex64, min(len(x), n/2+1))
	for i := range c {
		c[i] = complex(real(x[i]), -imag(x[i]))
	}
	return cachedRealPlan32(n).IRFFTNorm(make([]float32, n), c, f32ndSwapNorm(o.Norm))
}

// IHFFT32 is IHFFT in single precision: the first n/2+1 values of the
// Hermitian signal whose real DFT is x, normalized by 1/n
// (numpy.fft.ihfft(x)). The input is not modified.
func IHFFT32(x []float32) []complex64 {
	return IHFFT32With(x, Options{})
}

// IHFFT32With is IHFFTWith in single precision: the input is truncated or
// zero-padded to o.N points (len(x) when zero) and o.Norm scales the result
// as an inverse transform. o.Axes must be nil.
func IHFFT32With(x []float32, o Options) []complex64 {
	n := o.check1D("IHFFT32With", len(x))
	out := f32ndRFFT(x, n, f32ndSwapNorm(o.Norm))
	for i, v := range out {
		out[i] = complex(real(v), -imag(v))
	}
	return out
}

// f32ndSwapNorm is the mode whose forward factor is m's inverse factor and
// vice versa: NormBackward and NormForward exchange, NormOrtho stays. numpy
// computes hfft as irfft with the opposite norm, and ihfft as rfft with it.
func f32ndSwapNorm(m Norm) Norm {
	return NormForward - m
}

// FFTN32With is FFTN32 with numpy.fft.fftn's axes and norm arguments, as
// FFTNWith: only the axes in o.Axes are transformed (every axis when nil) and
// the result is scaled as o.Norm says. o.N must be zero.
func FFTN32With(data []complex64, shape []int, o Options) []complex64 {
	return f32ndComplexNWith("FFTN32With", data, shape, o, false)
}

// IFFTN32With is IFFTN32 with numpy.fft.ifftn's axes and norm arguments; see
// FFTN32With.
func IFFTN32With(data []complex64, shape []int, o Options) []complex64 {
	return f32ndComplexNWith("IFFTN32With", data, shape, o, true)
}

// FFT2_32With is FFT2_32 with numpy.fft.fft2's axes and norm arguments:
// FFTN32With on a two-axis shape.
func FFT2_32With(data []complex64, shape [2]int, o Options) []complex64 {
	return f32ndComplexNWith("FFT2_32With", data, shape[:], o, false)
}

// IFFT2_32With is IFFT2_32 with numpy.fft.ifft2's axes and norm arguments:
// IFFTN32With on a two-axis shape.
func IFFT2_32With(data []complex64, shape [2]int, o Options) []complex64 {
	return f32ndComplexNWith("IFFT2_32With", data, shape[:], o, true)
}

func f32ndComplexNWith(name string, data []complex64, shape []int, o Options, inverse bool) []complex64 {
	validateShape(shape, len(data))
	axes := o.checkND(name, len(shape))
	f := o.Norm.scale(axesProduct(shape, axes), inverse)
	out := make([]complex64, len(data))
	f32ndPlanFor(shape, axes).transform(out, data, inverse)
	scale32(out, f)
	return out
}

// RFFTN32With is RFFTN32 with numpy.fft.rfftn's axes and norm arguments, as
// RFFTNWith: the real transform runs along the LAST axis listed. o.Axes must
// name at least one axis; o.N must be zero.
func RFFTN32With(data []float32, shape []int, o Options) []complex64 {
	validateShape(shape, len(data))
	axes := o.checkND("RFFTN32With", len(shape))
	if len(axes) == 0 {
		panic("fft: RFFTN32With: no axis to transform")
	}
	f := o.Norm.scale(axesProduct(shape, axes), false)
	ra := axes[len(axes)-1]
	half := halfShape(shape, ra)
	out := make([]complex64, len(data)/shape[ra]*half[ra])
	f32ndRealForward(out, data, shape, half, ra)
	if len(axes) > 1 {
		f32ndPlanFor(half, axes[:len(axes)-1]).transform(out, out, false)
	}
	scale32(out, f)
	return out
}

// IRFFTN32With is IRFFTN32 with numpy.fft.irfftn's axes and norm arguments, as
// IRFFTNWith: the real inverse runs along the LAST axis in o.Axes, the complex
// inverse along the others. o.Axes must name at least one axis; o.N must be
// zero.
func IRFFTN32With(data []complex64, shape []int, o Options) []float32 {
	total := shapeProduct(shape...)
	axes := o.checkND("IRFFTN32With", len(shape))
	if len(axes) == 0 {
		panic("fft: IRFFTN32With: no axis to transform")
	}
	ra := axes[len(axes)-1]
	f := o.Norm.scale(axesProduct(shape, axes), true)
	half := halfShape(shape, ra)
	spec := make([]complex64, total/shape[ra]*half[ra])
	copy(spec, data) // missing bins read as zero, extra ones are ignored
	if len(axes) > 1 {
		f32ndPlanFor(half, axes[:len(axes)-1]).transform(spec, spec, true)
	}
	out := make([]float32, total)
	f32ndRealInverse(out, spec, shape, half, ra) // unscaled
	scaleReal32(out, f)
	return out
}

// RFFT2_32With is RFFT2_32 with numpy.fft.rfft2's axes and norm arguments.
// With o.Axes nil it runs RFFT2_32's plan; with axes it is RFFTN32With on a
// two-axis shape.
func RFFT2_32With(data []float32, shape [2]int, o Options) []complex64 {
	if o.Axes != nil {
		return RFFTN32With(data, shape[:], o)
	}
	o.checkND("RFFT2_32With", 2)
	if shapeProduct(shape[0], shape[1]) != len(data) {
		panic("fft: shape product does not match len(data)")
	}
	p := cachedRealPlan2_32(shape[0], shape[1])
	return p.RFFTNorm(make([]complex64, p.SpectrumLen()), data, o.Norm)
}

// IRFFT2_32With is IRFFT2_32 with numpy.fft.irfft2's axes and norm arguments.
// With o.Axes nil it runs IRFFT2_32's plan; with axes it is IRFFTN32With on a
// two-axis shape.
func IRFFT2_32With(data []complex64, shape [2]int, o Options) []float32 {
	if o.Axes != nil {
		return IRFFTN32With(data, shape[:], o)
	}
	o.checkND("IRFFT2_32With", 2)
	rows, cols := shape[0], shape[1]
	shapeProduct(rows, cols)
	p := cachedRealPlan2_32(rows, cols)
	spec := data
	if len(spec) != p.SpectrumLen() {
		spec = make([]complex64, p.SpectrumLen())
		copy(spec, data)
	}
	return p.IRFFTNorm(make([]float32, rows*cols), spec, o.Norm)
}
