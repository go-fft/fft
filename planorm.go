package fft

// This file gives the plans numpy's norm argument. A plan's length is fixed
// when it is built, so of numpy's optional arguments (n, norm, axes) only norm
// can vary per call; the plans therefore take a Norm directly instead of an
// Options. Each method computes its factor, and so panics on an unknown Norm,
// before it writes anything. Under NormBackward each one does exactly what the
// method without Norm does, with no extra pass.

// FFTNorm is FFT scaled as m says: unscaled under NormBackward, 1/sqrt(n)
// under NormOrtho, 1/n under NormForward. dst and src follow FFT's rules.
func (p *Plan) FFTNorm(dst, src []complex128, m Norm) []complex128 {
	f := m.scale(p.n, false)
	p.FFT(dst, src)
	scaleComplex(dst[:p.n], f)
	return dst
}

// IFFTNorm is IFFT scaled as m says: 1/n under NormBackward (as IFFT),
// 1/sqrt(n) under NormOrtho, unscaled under NormForward. dst and src follow
// IFFT's rules.
func (p *Plan) IFFTNorm(dst, src []complex128, m Norm) []complex128 {
	f := m.scale(p.n, true)
	p.checkLen(dst, src)
	p.execute(dst, src, true)
	scaleComplex(dst[:p.n], f)
	return dst
}

// RFFTNorm is RFFT scaled as m says (see Plan.FFTNorm). dst and src follow
// RFFT's rules.
func (p *RealPlan) RFFTNorm(dst []complex128, src []float64, m Norm) []complex128 {
	f := m.scale(p.n, false)
	out := p.RFFT(dst, src)
	scaleComplex(out, f)
	return out
}

// IRFFTNorm is IRFFT scaled as m says (see Plan.IFFTNorm). dst and src follow
// IRFFT's rules.
func (p *RealPlan) IRFFTNorm(dst []float64, src []complex128, m Norm) []float64 {
	f := unitInverseRescale(m.scale(p.n, true), p.n)
	out := p.IRFFT(dst, src)
	scaleReal(out, f)
	return out
}

// FFTNorm is FFT scaled as m says, n being Len(), the product of the shape
// (see Plan.FFTNorm).
func (p *PlanN) FFTNorm(dst, src []complex128, m Norm) []complex128 {
	f := m.scale(p.size, false)
	p.transform(dst, src, false)
	scaleComplex(dst, f)
	return dst
}

// IFFTNorm is IFFT scaled as m says, n being Len(), the product of the shape
// (see Plan.IFFTNorm).
func (p *PlanN) IFFTNorm(dst, src []complex128, m Norm) []complex128 {
	f := m.scale(p.size, true)
	p.transform(dst, src, true)
	scaleComplex(dst, f)
	return dst
}

// RFFTNorm is RFFT scaled as m says, n being rows×cols (see Plan.FFTNorm).
func (p *RealPlan2) RFFTNorm(dst []complex128, src []float64, m Norm) []complex128 {
	f := m.scale(p.rows*p.cols, false)
	p.RFFT(dst, src)
	scaleComplex(dst, f)
	return dst
}

// IRFFTNorm is IRFFT scaled as m says, n being rows×cols (see
// Plan.IFFTNorm).
func (p *RealPlan2) IRFFTNorm(dst []float64, src []complex128, m Norm) []float64 {
	f := unitInverseRescale(m.scale(p.rows*p.cols, true), p.rows*p.cols)
	p.IRFFT(dst, src)
	scaleReal(dst, f)
	return dst
}
