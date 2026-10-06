package fft

import "github.com/go-fft/fft/internal/kernels"

// The float32 batched passes (Round 25): a non-contiguous axis of a PlanN32
// runs as Stockham passes over strips of neighbouring lines, as PlanN's does
// (stripLines, Round 17), instead of gathering each block of lines into
// scratch. Every value of a batch gets the arithmetic its own 1-D transform
// gives it, so strips and lines agree bit for bit.

// stripAxes32 makes f32ndNewPlanN run a non-contiguous axis as batched
// passes where it can. It is on where the float32 batched kernels run (AVX2
// on amd64, NEON on arm64), and a variable so the tests run both paths.
var stripAxes32 = kernels.StockhamBatchKernels32()

// stripWidth32 is how many neighbouring lines one float32 strip of an axis of
// length n holds: PlanN's rule (stripWidth) unless Round 25's sweep says
// otherwise; a variable so the benchmarks can sweep it.
var stripWidth32 = stripWidth

// f32rStripsFit reports whether a float32 axis plan can run as batched
// passes: a Stockham plan of at least two passes (the first reads the array,
// the last writes it), every one of a radix with a batched pass.
func f32rStripsFit(pl *Plan32) bool {
	if pl.sk == nil || len(pl.sk.stages) < 2 {
		return false
	}
	for _, st := range pl.sk.stages {
		if !batchRadix(st.r) {
			return false
		}
	}
	return true
}

// batchTwiddles returns the stage's twiddles in the order a batched pass
// reads them, forward and conjugate: for i = 1 .. ido-1, the r-1 twiddles of
// point i one after the other. nil when ido == 1.
func (st *skStage32) batchTwiddles() (fwd, conj []complex64) {
	if st.ido == 1 {
		return nil, nil
	}
	m, r := st.ido-1, st.r
	fwd = make([]complex64, m*(r-1))
	conj = make([]complex64, m*(r-1))
	for i := 0; i < m; i++ {
		for j := 0; j < r-1; j++ {
			fwd[i*(r-1)+j] = st.tw[j*m+i]
			conj[i*(r-1)+j] = st.twc[j*m+i]
		}
	}
	return fwd, conj
}

// passBatch runs the stage as a batched pass over w lines, on the SIMD kernel
// when there is one. tw is batchTwiddles' table for the direction.
func (st *skStage32) passBatch(ch, cc, tw []complex64, w, sIn, sOut int, inverse bool) {
	if !kernels.StockhamBatchPass32(st.r, st.ido, st.l1, cc, ch, tw, w, sIn, sOut, inverse) {
		st.passBatchScalar(ch, cc, tw, w, sIn, sOut, inverse)
	}
}

// passBatchScalar is the Go batched pass: each value of the batch through the
// butterfly the Go pass uses (f32Bfly3..f32Bfly8, and a ± b for radix 2),
// then, for i > 0, its twiddle by f32Mul.
func (st *skStage32) passBatchScalar(ch, cc, tw []complex64, w, sIn, sOut int, inverse bool) {
	r, ido, l1 := st.r, st.ido, st.l1
	s := f32DirSign(inverse)
	var x, y [8]complex64
	for k := 0; k < l1; k++ {
		for i := 0; i < ido; i++ {
			for c := 0; c < w; c++ {
				for j := 0; j < r; j++ {
					x[j] = cc[(i+ido*(j+r*k))*sIn+c]
				}
				f32rBflyR(&y, &x, r, s)
				for j := 0; j < r; j++ {
					v := y[j]
					if i > 0 && j > 0 {
						v = f32Mul(v, tw[(i-1)*(r-1)+j-1])
					}
					ch[(i+ido*(k+l1*j))*sOut+c] = v
				}
			}
		}
	}
}

// f32rBflyR is the size-r DFT of x[:r] into y[:r] for a batched radix.
func f32rBflyR(y, x *[8]complex64, r int, s float32) {
	switch r {
	case 2:
		y[0], y[1] = x[0]+x[1], x[0]-x[1]
	case 3:
		y[0], y[1], y[2] = f32Bfly3(x[0], x[1], x[2], s)
	case 4:
		y[0], y[1], y[2], y[3] = f32Bfly4(x[0], x[1], x[2], x[3], s)
	case 5:
		y[0], y[1], y[2], y[3], y[4] = f32Bfly5(x[0], x[1], x[2], x[3], x[4], s)
	default:
		f32Bfly8(y, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], s)
	}
}

// stripLines transforms strips lo..hi-1 of the non-contiguous axis ax as
// batched passes, as PlanN.stripLines does: strip s covers up to
// stripWidth32(n) neighbouring lines of one run along the axes after ax; its
// first pass reads them where they lie in src, the passes between ping-pong
// in private scratch, and its last pass writes them into dst.
func (p *PlanN32) stripLines(dst, src []complex64, ax, lo, hi int, inverse bool) {
	n, st := p.shape[ax], p.stride[ax]
	stages, tws := p.axes[ax].sk.stages, p.strips[ax]
	dir := 0
	if inverse {
		dir = 1
	}
	sw := stripWidth32(n)
	per := (st + sw - 1) / sw
	bp := p.bufs.Get().(*[]complex64)
	a, b := (*bp)[:n*sw], (*bp)[n*sw:2*n*sw]
	last := len(stages) - 1
	for s := lo; s < hi; s++ {
		c0 := s % per * sw
		w := min(sw, st-c0)
		base := s/per*n*st + c0
		in, sIn := src[base:], st
		for k := range stages {
			out, sOut := a, w
			if k == last {
				out, sOut = dst[base:], st
			}
			stages[k].passBatch(out, in, tws[k][dir], w, sIn, sOut, inverse)
			in, sIn = out, sOut
			a, b = b, a
		}
	}
	p.bufs.Put(bp)
}
