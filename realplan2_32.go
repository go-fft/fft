package fft

import "sync"

// This file holds the real multi-dimensional transforms in single precision:
// RealPlan2_32, RFFT2_32/IRFFT2_32 and RFFTN32/IRFFTN32, the float32
// counterparts of RealPlan2, RFFT2/IRFFT2 and RFFTN/IRFFTN (scipy.fft.rfft2,
// irfft2, rfftn and irfftn on float32 input, which return complex64 and
// float32). See plann32.go for the naming.

// A RealPlan2_32 is the single-precision counterpart of RealPlan2: a reusable
// real 2-D transform of a fixed rows×cols shape over float32 samples and
// complex64 bins. It is immutable after construction and safe for concurrent
// use.
type RealPlan2_32 struct {
	rows, cols, rcols int
	row               *RealPlan32 // the real transform along each row
	col               *PlanN32    // shape rows×rcols; only its axis 0 is used
	scratch           sync.Pool   // rows×rcols spectrum for IRFFT, which must not touch src
}

// NewRealPlan2_32 returns a single-precision plan for real rows×cols matrices.
// Both lengths must be positive; NewRealPlan2_32 panics otherwise.
func NewRealPlan2_32(rows, cols int) *RealPlan2_32 {
	shapeProduct(rows, cols)
	p := &RealPlan2_32{rows: rows, cols: cols, rcols: cols/2 + 1}
	p.row = cachedRealPlan32(cols)
	p.col = f32ndNewPlanN([]int{rows, p.rcols}, []int{0})
	size := rows * p.rcols
	p.scratch.New = func() any { b := make([]complex64, size); return &b }
	return p
}

// SpectrumLen is the number of bins RFFT writes and IRFFT reads:
// rows×(cols/2+1).
func (p *RealPlan2_32) SpectrumLen() int { return p.rows * p.rcols }

// RFFT writes the forward 2-D DFT of the real row-major matrix src (rows×cols)
// into dst (rows×(cols/2+1), SpectrumLen bins), unnormalized, and returns dst.
// It matches RFFT2_32. src is not modified.
func (p *RealPlan2_32) RFFT(dst []complex64, src []float32) []complex64 {
	return p.RFFTNorm(dst, src, NormBackward)
}

// IRFFT writes the real matrix (rows×cols) whose spectrum is src (SpectrumLen
// bins, the RFFT layout) into dst, normalized by rows×cols, and returns dst.
// It matches IRFFT2_32. src is not modified.
func (p *RealPlan2_32) IRFFT(dst []float32, src []complex64) []float32 {
	return p.IRFFTNorm(dst, src, NormBackward)
}

// RFFTNorm is RFFT scaled as m says, n being rows×cols (see Plan32.FFTNorm).
func (p *RealPlan2_32) RFFTNorm(dst []complex64, src []float32, m Norm) []complex64 {
	f := m.scale(p.rows*p.cols, false)
	if len(src) != p.rows*p.cols || len(dst) != p.SpectrumLen() {
		panic("fft: RealPlan2_32 slice length does not match the plan's shape")
	}
	if parallelizeLines(p.rows, p.cols) {
		parChunks(p.rows, p.cols, func(lo, hi int) { p.forwardRows(dst, src, lo, hi) })
	} else {
		p.forwardRows(dst, src, 0, p.rows)
	}
	if p.col.axes[0] != nil {
		p.col.transformAxis(dst, dst, 0, false)
	}
	scale32(dst, f)
	return dst
}

// IRFFTNorm is IRFFT scaled as m says, n being rows×cols (see
// Plan32.IFFTNorm). The rows are inverted unscaled and the factor is applied
// once, rounded once to float32.
func (p *RealPlan2_32) IRFFTNorm(dst []float32, src []complex64, m Norm) []float32 {
	f := m.scale(p.rows*p.cols, true)
	if len(dst) != p.rows*p.cols || len(src) != p.SpectrumLen() {
		panic("fft: RealPlan2_32 slice length does not match the plan's shape")
	}
	bp := p.scratch.Get().(*[]complex64)
	defer p.scratch.Put(bp)
	half := *bp
	if p.col.axes[0] != nil {
		p.col.transformAxis(half, src, 0, true)
	} else {
		copy(half, src)
	}
	if parallelizeLines(p.rows, p.cols) {
		parChunks(p.rows, p.cols, func(lo, hi int) { p.inverseRows(dst, half, lo, hi) })
	} else {
		p.inverseRows(dst, half, 0, p.rows)
	}
	scaleReal32(dst, f)
	return dst
}

func (p *RealPlan2_32) forwardRows(dst []complex64, src []float32, lo, hi int) {
	for r := lo; r < hi; r++ {
		p.row.RFFT(dst[r*p.rcols:(r+1)*p.rcols], src[r*p.cols:(r+1)*p.cols])
	}
}

// inverseRows runs the unscaled real inverse of each row.
func (p *RealPlan2_32) inverseRows(dst []float32, half []complex64, lo, hi int) {
	for r := lo; r < hi; r++ {
		p.row.IRFFTNorm(dst[r*p.cols:(r+1)*p.cols], half[r*p.rcols:(r+1)*p.rcols], NormForward)
	}
}

// scaleReal32 multiplies x by s, rounded once to float32; s == 1 is a no-op.
func scaleReal32(x []float32, s float64) {
	if s == 1 {
		return
	}
	f := float32(s)
	for i := range x {
		x[i] *= f
	}
}

// RFFT2_32 is RFFT2 in single precision: the forward 2-D DFT of a real
// row-major float32 matrix of shape shape[0]×shape[1], keeping shape[1]/2+1
// bins per row (scipy.fft.rfft2 on float32 input). The same shape rules as
// RFFT2 apply; the input is not modified.
func RFFT2_32(data []float32, shape [2]int) []complex64 {
	return RFFT2_32With(data, shape, Options{})
}

// IRFFT2_32 inverts RFFT2_32, as IRFFT2 inverts RFFT2: shape is the real
// output's, data is read up to shape[0]*(shape[1]/2+1) bins (missing ones are
// zero) and the result is normalized by shape[0]*shape[1]. The output has
// shape[0]*shape[1] values whatever the spectrum's length: bound the shape when
// it comes from untrusted input (see SECURITY.md).
func IRFFT2_32(data []complex64, shape [2]int) []float32 {
	return IRFFT2_32With(data, shape, Options{})
}

var (
	realPlan2_32Mu    sync.Mutex
	realPlan2_32Cache = map[[2]int]*RealPlan2_32{}
)

// cachedRealPlan2_32 returns the shared plan for rows×cols (already
// validated), building it on first use.
func cachedRealPlan2_32(rows, cols int) *RealPlan2_32 {
	realPlan2_32Mu.Lock()
	defer realPlan2_32Mu.Unlock()
	key := [2]int{rows, cols}
	p, ok := realPlan2_32Cache[key]
	if !ok {
		p = NewRealPlan2_32(rows, cols)
		realPlan2_32Cache[key] = p
	}
	return p
}

// RFFTN32 is RFFTN in single precision: the forward N-dimensional DFT of the
// real row-major float32 array data, the real transform along the last axis
// (which keeps n/2+1 bins) and the complex one along the others
// (scipy.fft.rfftn on float32 input). The same rules as RFFTN apply.
func RFFTN32(data []float32, shape []int) []complex64 {
	return RFFTN32With(data, shape, Options{})
}

// IRFFTN32 inverts RFFTN32, as IRFFTN inverts RFFTN: shape is the REAL
// output's, and the result is normalized by its product. The output has the
// product of shape values whatever the spectrum's length: bound the shape when
// it comes from untrusted input (see SECURITY.md).
func IRFFTN32(data []complex64, shape []int) []float32 {
	return IRFFTN32With(data, shape, Options{})
}

// f32ndRealForward runs the real forward transform (unnormalized) along axis
// ra of src (shape shape) into dst (shape half), line by line, as
// realAxisForward does in float64.
func f32ndRealForward(dst []complex64, src []float32, shape, half []int, ra int) {
	n, h := shape[ra], half[ra]
	rp := cachedRealPlan32(n)
	inSt, outSt := rowMajorStrides(shape), rowMajorStrides(half)
	lines := len(src) / n
	body := func(lo, hi int) {
		if inSt[ra] == 1 {
			for c := lo; c < hi; c++ {
				rp.RFFT(dst[c*h:(c+1)*h], src[c*n:(c+1)*n])
			}
			return
		}
		rb, cb := make([]float32, n), make([]complex64, h)
		for c := lo; c < hi; c++ {
			ib, ob := lineBase(c, shape, inSt, ra), lineBase(c, half, outSt, ra)
			for i := range rb {
				rb[i] = src[ib+i*inSt[ra]]
			}
			rp.RFFT(cb, rb)
			for k, v := range cb {
				dst[ob+k*outSt[ra]] = v
			}
		}
	}
	runLines(lines, n, body)
}

// f32ndRealInverse runs the UNSCALED real inverse transform along axis ra of
// the spectrum src (shape half) into dst (shape shape), line by line.
func f32ndRealInverse(dst []float32, src []complex64, shape, half []int, ra int) {
	n, h := shape[ra], half[ra]
	rp := cachedRealPlan32(n)
	outSt, inSt := rowMajorStrides(shape), rowMajorStrides(half)
	lines := len(dst) / n
	body := func(lo, hi int) {
		if outSt[ra] == 1 {
			for c := lo; c < hi; c++ {
				rp.IRFFTNorm(dst[c*n:(c+1)*n], src[c*h:(c+1)*h], NormForward)
			}
			return
		}
		rb, cb := make([]float32, n), make([]complex64, h)
		for c := lo; c < hi; c++ {
			ib, ob := lineBase(c, half, inSt, ra), lineBase(c, shape, outSt, ra)
			for k := range cb {
				cb[k] = src[ib+k*inSt[ra]]
			}
			rp.IRFFTNorm(rb, cb, NormForward)
			for i, v := range rb {
				dst[ob+i*outSt[ra]] = v
			}
		}
	}
	runLines(lines, n, body)
}
