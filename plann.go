package fft

import (
	"sync"

	"github.com/go-fft/fft/internal/kernels"
)

// A PlanN is a reusable N-dimensional complex transform of a fixed row-major
// (C-order) shape: the N-D counterpart of Plan, the way FFTW's
// fftw_plan_dft(rank, dims, …) is the counterpart of its 1-D plans. Building it
// resolves one 1-D plan per axis; FFT and IFFT then write into a caller-supplied
// slice and, in steady state, allocate nothing — unlike FFTN/FFT2, which return
// a new slice each call. On a many-core host that allocation is what dominates a
// small repeated transform: a 128×128 FFT2 spent about three quarters of its
// time in the garbage collector on a 128-thread machine (see BENCHMARKS.md).
//
// PlanN is immutable after construction and safe for concurrent use; each call
// borrows private scratch from a pool.
type PlanN struct {
	shape  []int
	stride []int // row-major strides; the last axis is contiguous
	size   int
	axes   []*Plan // per-axis 1-D plans; nil for a length-1 axis
	maxLen int     // longest axis
	// strips[ax] holds, for an axis run as batched passes (see stripLines),
	// each pass's batchTwiddles, forward then conjugate; nil for an axis
	// whose lines are transformed one by one.
	strips [][][2][]complex128
	bufs   sync.Pool
}

// stripAxes makes NewPlanN run a non-contiguous axis as batched Stockham
// passes over strips of stripWidth neighbouring lines (stripLines) where it
// can, instead of gathering lines into scratch. It is on where the batched
// pass kernels run (AVX2), and a variable so the tests run both paths
// everywhere.
var stripAxes = kernels.StockhamBatchKernels()

// stripWidth is how many neighbouring lines one strip of an axis of length n
// holds. Swept over 4/8/16/32 on n×n plans, one core (2026-10-05, line-by-line
// time ÷ strip time): 16 was the best single width up to 512 on Zen 3 and
// Cascade Lake, within 10% of each size's best (128: 1.59 and 1.54), and 32
// above (1024: 1.21 and 1.06, where 16 gave 1.01 and 0.89; 2048: 3.54 and
// 2.71).
func stripWidth(n int) int {
	if n <= 512 {
		return 16
	}
	return 32
}

// lineBlock is the most neighbouring lines a non-contiguous axis gathers at
// once: element i of eight adjacent lines is eight adjacent complex128 values,
// two 64-byte cache lines read whole instead of 16 bytes of each. blockWidth
// narrows the block for long lines so its scratch stays within blockBytes.
//
// blockBytes was swept (16/32/64/128 KB, PlanN at 256²…2048², 2026-10-01):
// 128 KB was best or within 2% at every size on Apple M4 and Neoverse-N1, and
// the best single value on amd64 too (geometric mean over Haswell and Zen 3:
// 1.10 of each size's best, against 1.13–1.17 for the smaller budgets), though
// there the best budget moves with size and CPU.
const (
	lineBlock  = 8
	blockBytes = 128 << 10
	// linePad offsets consecutive lines in the gather scratch by 4 complex128
	// (64 bytes). Unpadded, eight 512-point lines sit 8 KB apart, a multiple of
	// 4 KB, and every gather write buf[b·n+i] lands in one 4K-aliasing set on
	// Intel: the column gather ran 1.3× slower than gathering one column at a
	// time on Haswell until the lines were staggered.
	linePad = 4
)

// blockWidth is how many lines of length n one gather block holds.
func blockWidth(n int) int {
	return max(1, min(lineBlock, blockBytes/(16*n)))
}

// NewPlanN returns a plan for arrays of the given row-major shape. Every
// length must be positive; NewPlanN panics otherwise. An empty shape is the
// single-element (scalar) array.
func NewPlanN(shape ...int) *PlanN {
	p := &PlanN{shape: append([]int(nil), shape...)}
	p.size = shapeProduct(shape...)
	p.stride = make([]int, len(shape))
	acc := 1
	for ax := len(shape) - 1; ax >= 0; ax-- {
		p.stride[ax] = acc
		acc *= shape[ax]
	}
	p.axes = make([]*Plan, len(shape))
	for ax, n := range shape {
		if n > 1 {
			p.axes[ax] = cachedPlan(n)
		}
		p.maxLen = max(p.maxLen, n)
	}
	p.initScratch()
	return p
}

// initScratch sets up, once the axis plans are in place, the strips of the
// axes that run as batched passes and the pool of scratch buffers.
func (p *PlanN) initScratch() {
	bl := lineBlock * (p.maxLen + linePad)
	p.strips = make([][][2][]complex128, len(p.shape))
	for ax, n := range p.shape {
		if ax < len(p.shape)-1 && p.axes[ax] != nil && stripAxes && stripsFit(p.axes[ax]) {
			sts := p.axes[ax].sk.stages
			p.strips[ax] = make([][2][]complex128, len(sts))
			for k := range sts {
				p.strips[ax][k][0], p.strips[ax][k][1] = sts[k].batchTwiddles()
			}
			bl = max(bl, 2*n*stripWidth(n))
		}
	}
	p.bufs.New = func() any { b := make([]complex128, bl); return &b }
}

// Shape returns a copy of the plan's shape.
func (p *PlanN) Shape() []int { return append([]int(nil), p.shape...) }

// Len reports the number of elements the plan transforms (the product of its
// shape).
func (p *PlanN) Len() int { return p.size }

// FFT writes the forward N-dimensional DFT of src into dst and returns dst. dst
// and src must each have length Len(); dst may alias src. It matches FFTN.
func (p *PlanN) FFT(dst, src []complex128) []complex128 {
	p.transform(dst, src, false)
	return dst
}

// IFFT writes the inverse N-dimensional DFT of src into dst, normalized by
// Len(), and returns dst. dst may alias src. It matches IFFTN.
func (p *PlanN) IFFT(dst, src []complex128) []complex128 {
	p.transform(dst, src, true)
	inv := complex(1/float64(p.size), 0)
	for i := range dst {
		dst[i] *= inv
	}
	return dst
}

func (p *PlanN) transform(dst, src []complex128, inverse bool) {
	if len(dst) != p.size || len(src) != p.size {
		panic("fft: PlanN slice length does not match the plan's shape")
	}
	// The first axis transformed reads src: no copy into dst first.
	in := src
	for ax := range p.shape {
		if p.axes[ax] != nil {
			p.transformAxis(dst, in, ax, inverse)
			in = dst
		}
	}
	if !aliases(dst, in) {
		copy(dst, in) // every axis has length 1
	}
}

// stripsFit reports whether an axis plan can run as batched passes: a
// Stockham plan of at least two passes (the first reads the array, the last
// writes it), every one of a radix with a batched pass.
func stripsFit(pl *Plan) bool {
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

// transformAxis writes into dst the unnormalized 1-D transform along every
// line of axis ax of src; src is dst or does not overlap it. The serial path
// calls the line workers directly: a closure handed to parChunks escapes to
// the heap, which would cost a steady-state allocation per axis, so one is
// only built to fan out.
func (p *PlanN) transformAxis(dst, src []complex128, ax int, inverse bool) {
	n := p.shape[ax]
	lines := p.size / n
	par := parallelizeLines(lines, n)
	if p.stride[ax] == 1 {
		// The last axis: each line is contiguous and is transformed where it
		// lies — no gather, no scatter.
		if par {
			parChunks(lines, n, func(lo, hi int) { p.contiguousLines(dst, src, ax, lo, hi, inverse) })
		} else {
			p.contiguousLines(dst, src, ax, 0, lines, inverse)
		}
		return
	}
	if p.strips[ax] != nil {
		st, sw := p.stride[ax], stripWidth(n)
		strips := p.size / (n * st) * ((st + sw - 1) / sw)
		if par {
			parChunks(strips, sw*n, func(lo, hi int) { p.stripLines(dst, src, ax, lo, hi, inverse) })
		} else {
			p.stripLines(dst, src, ax, 0, strips, inverse)
		}
		return
	}
	// Any other axis: lines c and c+1 start one element apart until the last
	// axis wraps, so blocks of up to lineBlock lines are gathered together.
	last, bw := p.shape[len(p.shape)-1], blockWidth(n)
	blocks := lines / last * ((last + bw - 1) / bw)
	if par {
		parChunks(blocks, bw*n, func(lo, hi int) { p.blockedLines(dst, src, ax, lo, hi, inverse) })
	} else {
		p.blockedLines(dst, src, ax, 0, blocks, inverse)
	}
}

// contiguousLines transforms lines lo..hi-1 of the contiguous last axis. A
// Stockham axis takes its scratch buffer once for all of them.
func (p *PlanN) contiguousLines(dst, src []complex128, ax, lo, hi int, inverse bool) {
	n, plan := p.shape[ax], p.axes[ax]
	if sk := plan.sk; sk != nil {
		bp := sk.scratch.Get().(*[]complex128)
		for c := lo; c < hi; c++ {
			sk.run(dst[c*n:(c+1)*n], src[c*n:(c+1)*n], *bp, inverse)
		}
		sk.scratch.Put(bp)
		return
	}
	for c := lo; c < hi; c++ {
		plan.execute(dst[c*n:(c+1)*n], src[c*n:(c+1)*n], inverse)
	}
}

// stripLines transforms strips lo..hi-1 of the non-contiguous axis ax as
// batched Stockham passes. Strip s covers up to stripWidth neighbouring lines
// of one run along the axes after ax (those lines start one element apart):
// its first pass reads them out of src where they lie, the passes between
// ping-pong in private scratch, and its last pass writes them into dst. Each
// line gets exactly the arithmetic of its own 1-D transform.
func (p *PlanN) stripLines(dst, src []complex128, ax, lo, hi int, inverse bool) {
	n, st := p.shape[ax], p.stride[ax]
	stages, tws := p.axes[ax].sk.stages, p.strips[ax]
	dir := 0
	if inverse {
		dir = 1
	}
	sw := stripWidth(n)
	per := (st + sw - 1) / sw
	bp := p.bufs.Get().(*[]complex128)
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

// blockedLines transforms gather blocks lo..hi-1 of a non-contiguous axis
// ax. Block blk covers up to blockWidth(n) neighbouring lines of one run along the
// last axis; they are gathered element by element into private scratch (each
// element i of the block is one contiguous read), transformed, and scattered
// back.
func (p *PlanN) blockedLines(dst, src []complex128, ax, lo, hi int, inverse bool) {
	n, plan, st := p.shape[ax], p.axes[ax], p.stride[ax]
	ld := n + linePad // scratch distance between gathered lines
	last, bw := p.shape[len(p.shape)-1], blockWidth(n)
	perRun := (last + bw - 1) / bw
	bp := p.bufs.Get().(*[]complex128)
	buf := *bp
	for blk := lo; blk < hi; blk++ {
		run, part := blk/perRun, blk%perRun
		c0 := run*last + part*bw
		w := min(bw, last-part*bw)
		base := lineBase(c0, p.shape, p.stride, ax)
		if w == lineBlock {
			gather8(buf, src, base, st, ld, n)
		} else {
			for i := 0; i < n; i++ {
				row := src[base+i*st : base+i*st+w]
				for b, v := range row {
					buf[b*ld+i] = v
				}
			}
		}
		for b := 0; b < w; b++ {
			line := buf[b*ld : b*ld+n]
			plan.execute(line, line, inverse)
		}
		if w == lineBlock {
			scatter8(dst, buf, base, st, ld, n)
		} else {
			for i := 0; i < n; i++ {
				row := dst[base+i*st : base+i*st+w]
				for b := range row {
					row[b] = buf[b*ld+i]
				}
			}
		}
	}
	p.bufs.Put(bp)
}

// gather8 copies a full block of eight neighbouring lines (element i of line b
// is out[base+i·st+b]) into buf (line b at buf[b·ld:]). The eight destination
// lines are resliced to exactly n once, so the loop body is eight plain loads
// and stores with no inner loop and no bounds check: the general w-wide loop
// spent its time on loop overhead, not memory (1.5× slower on Haswell).
func gather8(buf, out []complex128, base, st, ld, n int) {
	l0, l1, l2, l3 := buf[0*ld:][:n], buf[1*ld:][:n], buf[2*ld:][:n], buf[3*ld:][:n]
	l4, l5, l6, l7 := buf[4*ld:][:n], buf[5*ld:][:n], buf[6*ld:][:n], buf[7*ld:][:n]
	for i := range l0 {
		r := out[base+i*st:][:8]
		l0[i], l1[i], l2[i], l3[i] = r[0], r[1], r[2], r[3]
		l4[i], l5[i], l6[i], l7[i] = r[4], r[5], r[6], r[7]
	}
}

// scatter8 is gather8's inverse.
func scatter8(out, buf []complex128, base, st, ld, n int) {
	l0, l1, l2, l3 := buf[0*ld:][:n], buf[1*ld:][:n], buf[2*ld:][:n], buf[3*ld:][:n]
	l4, l5, l6, l7 := buf[4*ld:][:n], buf[5*ld:][:n], buf[6*ld:][:n], buf[7*ld:][:n]
	for i := range l0 {
		r := out[base+i*st:][:8]
		r[0], r[1], r[2], r[3] = l0[i], l1[i], l2[i], l3[i]
		r[4], r[5], r[6], r[7] = l4[i], l5[i], l6[i], l7[i]
	}
}
