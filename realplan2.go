package fft

import "sync"

// A RealPlan2 is a reusable real 2-D transform of a fixed rows×cols shape: the
// plan behind RFFT2/IRFFT2, as PlanN is behind FFTN. Its RFFT and IRFFT write
// into caller-supplied slices, so a repeated transform of an even-width matrix
// allocates nothing in steady state (an odd width goes through the full
// complex row transform, which allocates). It is immutable after construction
// and safe for concurrent use.
type RealPlan2 struct {
	rows, cols, rcols int
	row               *RealPlan // the real transform along each row
	col               *PlanN    // shape rows×rcols; only its axis 0 is used
	scratch           sync.Pool // rows×rcols spectrum for IRFFT, which must not touch src
}

// NewRealPlan2 returns a plan for real rows×cols matrices. Both lengths must be
// positive; NewRealPlan2 panics otherwise.
func NewRealPlan2(rows, cols int) *RealPlan2 {
	shapeProduct(rows, cols)
	p := &RealPlan2{rows: rows, cols: cols, rcols: cols/2 + 1}
	p.row = cachedRealPlan(cols)
	p.col = NewPlanN(rows, p.rcols)
	size := rows * p.rcols
	p.scratch.New = func() any { b := make([]complex128, size); return &b }
	return p
}

// SpectrumLen is the number of bins RFFT writes and IRFFT reads:
// rows×(cols/2+1).
func (p *RealPlan2) SpectrumLen() int { return p.rows * p.rcols }

// RFFT writes the forward 2-D DFT of the real row-major matrix src (rows×cols)
// into dst (rows×(cols/2+1), SpectrumLen bins) and returns dst. It matches
// RFFT2. src is not modified.
func (p *RealPlan2) RFFT(dst []complex128, src []float64) []complex128 {
	if len(src) != p.rows*p.cols || len(dst) != p.SpectrumLen() {
		panic("fft: RealPlan2 slice length does not match the plan's shape")
	}
	// Step 1: the real FFT of each row, written straight into its row of dst.
	if parallelizeLines(p.rows, p.cols) {
		parChunks(p.rows, p.cols, func(lo, hi int) { p.forwardRows(dst, src, lo, hi) })
	} else {
		p.forwardRows(dst, src, 0, p.rows)
	}
	// Step 2: the complex FFT down each column, in place.
	if p.col.axes[0] != nil {
		p.col.transformAxis(dst, dst, 0, false)
	}
	return dst
}

// IRFFT writes the real matrix (rows×cols) whose spectrum is src (SpectrumLen
// bins, the RFFT layout) into dst, normalized by rows×cols, and returns dst.
// It matches IRFFT2. src is not modified.
func (p *RealPlan2) IRFFT(dst []float64, src []complex128) []float64 {
	if len(dst) != p.rows*p.cols || len(src) != p.SpectrumLen() {
		panic("fft: RealPlan2 slice length does not match the plan's shape")
	}
	bp := p.scratch.Get().(*[]complex128)
	defer p.scratch.Put(bp)
	half := *bp
	// Step 1: the unnormalized complex inverse down each column, from src
	// into half.
	if p.col.axes[0] != nil {
		p.col.transformAxis(half, src, 0, true)
	} else {
		copy(half, src)
	}
	// Step 2: the real inverse of each row (normalized by cols), then 1/rows.
	if parallelizeLines(p.rows, p.cols) {
		parChunks(p.rows, p.cols, func(lo, hi int) { p.inverseRows(dst, half, lo, hi) })
	} else {
		p.inverseRows(dst, half, 0, p.rows)
	}
	return dst
}

func (p *RealPlan2) forwardRows(dst []complex128, src []float64, lo, hi int) {
	for r := lo; r < hi; r++ {
		p.row.RFFT(dst[r*p.rcols:(r+1)*p.rcols], src[r*p.cols:(r+1)*p.cols])
	}
}

func (p *RealPlan2) inverseRows(dst []float64, half []complex128, lo, hi int) {
	inv := 1 / float64(p.rows)
	for r := lo; r < hi; r++ {
		row := dst[r*p.cols : (r+1)*p.cols]
		p.row.IRFFT(row, half[r*p.rcols:(r+1)*p.rcols])
		for i := range row {
			row[i] *= inv
		}
	}
}
