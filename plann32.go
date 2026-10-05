package fft

import (
	"strconv"
	"sync"
)

// This file holds the multi-dimensional complex transforms in single
// precision: PlanN32 and FFTN32/IFFTN32/FFT2_32/IFFT2_32, the complex64
// counterparts of PlanN and FFTN/IFFTN/FFT2/IFFT2.
//
// Naming. Every single-precision function or type is the float64 name with
// "32" appended (FFT → FFT32, PlanN → PlanN32, FFTN → FFTN32), as v0.6.0 began
// with Plan32 and FFT32. Where the float64 name already ends in a digit, an
// underscore keeps the two numbers apart: FFT2_32 is the float32 FFT2, where
// "FFT232" would read as a transform of 232 points. The ...With variants put
// "With" after the whole name (FFT32With is "FFT32, with options"), so the
// godoc index lists FFT32 and FFT32With together.
//
// Like the 1-D single-precision engines, this is separate complex64 code and
// not a generic instantiation of PlanN: the float64 path is left exactly as it
// was. Only cold code is shared — shape validation (validateShape,
// shapeProduct, rowMajorStrides, lineBase), axis resolution (Options.checkND)
// and the goroutine fan-out (parChunks), none of which touches an element.

// A PlanN32 is the single-precision counterpart of PlanN: a reusable
// N-dimensional complex64 transform of a fixed row-major (C-order) shape, as
// scipy.fft.fftn computes for complex64 input. It runs a Plan32 along each
// axis; FFT and IFFT write into a caller-supplied slice. PlanN32 is immutable
// after construction and safe for concurrent use.
type PlanN32 struct {
	shape  []int
	stride []int
	size   int
	axes   []*Plan32 // per-axis 1-D plans; nil for an axis not transformed
	maxLen int
	bufs   sync.Pool // gather blocks for the non-contiguous axes
}

// NewPlanN32 returns a single-precision plan for arrays of the given row-major
// shape. Every length must be positive; NewPlanN32 panics otherwise. An empty
// shape is the single-element (scalar) array.
func NewPlanN32(shape ...int) *PlanN32 {
	shapeProduct(shape...)
	return f32ndNewPlanN(shape, resolveAxes("NewPlanN32", nil, len(shape)))
}

// f32ndNewPlanN builds a PlanN32 for shape (already validated) whose
// transform runs along the listed axes only.
func f32ndNewPlanN(shape, axes []int) *PlanN32 {
	p := &PlanN32{shape: append([]int(nil), shape...)}
	p.size = shapeProduct(shape...)
	p.stride = rowMajorStrides(shape)
	p.axes = make([]*Plan32, len(shape))
	for _, ax := range axes {
		n := shape[ax]
		if n > 1 {
			p.axes[ax] = cachedPlan32(n)
		}
		p.maxLen = max(p.maxLen, n)
	}
	bl := lineBlock * (p.maxLen + linePad)
	p.bufs.New = func() any { b := make([]complex64, bl); return &b }
	return p
}

// Shape returns a copy of the plan's shape.
func (p *PlanN32) Shape() []int { return append([]int(nil), p.shape...) }

// Len reports the number of elements the plan transforms (the product of its
// shape).
func (p *PlanN32) Len() int { return p.size }

// FFT writes the forward N-dimensional DFT of src into dst, unnormalized, and
// returns dst. dst and src must each have length Len(); dst may alias src. It
// matches FFTN32.
func (p *PlanN32) FFT(dst, src []complex64) []complex64 {
	return p.FFTNorm(dst, src, NormBackward)
}

// IFFT writes the inverse N-dimensional DFT of src into dst, normalized by
// Len(), and returns dst. dst may alias src. It matches IFFTN32.
func (p *PlanN32) IFFT(dst, src []complex64) []complex64 {
	return p.IFFTNorm(dst, src, NormBackward)
}

// FFTNorm is FFT scaled as m says, n being Len(): unscaled under
// NormBackward, 1/sqrt(n) under NormOrtho, 1/n under NormForward.
func (p *PlanN32) FFTNorm(dst, src []complex64, m Norm) []complex64 {
	f := m.scale(p.size, false)
	p.transform(dst, src, false)
	scale32(dst, f)
	return dst
}

// IFFTNorm is IFFT scaled as m says, n being Len(): 1/n under NormBackward
// (as IFFT), 1/sqrt(n) under NormOrtho, unscaled under NormForward.
func (p *PlanN32) IFFTNorm(dst, src []complex64, m Norm) []complex64 {
	f := m.scale(p.size, true)
	p.transform(dst, src, true)
	scale32(dst, f)
	return dst
}

// transform leaves in dst the unnormalized transform of src along every axis
// that has a plan.
func (p *PlanN32) transform(dst, src []complex64, inverse bool) {
	if len(dst) != p.size || len(src) != p.size {
		panic("fft: PlanN32 slice length does not match the plan's shape")
	}
	in := src
	for ax := range p.shape {
		if p.axes[ax] != nil {
			p.transformAxis(dst, in, ax, inverse)
			in = dst
		}
	}
	if len(dst) > 0 && &in[0] != &dst[0] {
		copy(dst, in) // no axis was transformed
	}
}

// f32ndBlockWidth is how many lines of length n one gather block holds: as
// blockWidth, for complex64's 8 bytes.
func f32ndBlockWidth(n int) int {
	return max(1, min(lineBlock, blockBytes/(8*n)))
}

// transformAxis writes into dst the unnormalized 1-D transform along every
// line of axis ax of src; src is dst or does not overlap it.
func (p *PlanN32) transformAxis(dst, src []complex64, ax int, inverse bool) {
	n := p.shape[ax]
	lines := p.size / n
	par := parallelizeLines(lines, n)
	if p.stride[ax] == 1 {
		// Every line is contiguous and transformed where it lies.
		if par {
			parChunks(lines, n, func(lo, hi int) { p.contiguousLines(dst, src, ax, lo, hi, inverse) })
		} else {
			p.contiguousLines(dst, src, ax, 0, lines, inverse)
		}
		return
	}
	last, bw := p.shape[len(p.shape)-1], f32ndBlockWidth(n)
	blocks := lines / last * ((last + bw - 1) / bw)
	if par {
		parChunks(blocks, bw*n, func(lo, hi int) { p.blockedLines(dst, src, ax, lo, hi, inverse) })
	} else {
		p.blockedLines(dst, src, ax, 0, blocks, inverse)
	}
}

// contiguousLines transforms lines lo..hi-1 of a contiguous axis.
func (p *PlanN32) contiguousLines(dst, src []complex64, ax, lo, hi int, inverse bool) {
	n, plan := p.shape[ax], p.axes[ax]
	for c := lo; c < hi; c++ {
		plan.execute(dst[c*n:(c+1)*n], src[c*n:(c+1)*n], inverse)
	}
}

// blockedLines transforms gather blocks lo..hi-1 of a non-contiguous axis, as
// PlanN.blockedLines does: up to f32ndBlockWidth(n) neighbouring lines (one
// element apart) are gathered into private scratch, transformed, and
// scattered back.
func (p *PlanN32) blockedLines(dst, src []complex64, ax, lo, hi int, inverse bool) {
	n, plan, st := p.shape[ax], p.axes[ax], p.stride[ax]
	ld := n + linePad
	last, bw := p.shape[len(p.shape)-1], f32ndBlockWidth(n)
	perRun := (last + bw - 1) / bw
	bp := p.bufs.Get().(*[]complex64)
	buf := *bp
	for blk := lo; blk < hi; blk++ {
		run, part := blk/perRun, blk%perRun
		c0 := run*last + part*bw
		w := min(bw, last-part*bw)
		base := lineBase(c0, p.shape, p.stride, ax)
		for i := 0; i < n; i++ {
			row := src[base+i*st : base+i*st+w]
			for b, v := range row {
				buf[b*ld+i] = v
			}
		}
		for b := 0; b < w; b++ {
			line := buf[b*ld : b*ld+n]
			plan.execute(line, line, inverse)
		}
		for i := 0; i < n; i++ {
			row := dst[base+i*st : base+i*st+w]
			for b := range row {
				row[b] = buf[b*ld+i]
			}
		}
	}
	p.bufs.Put(bp)
}

// FFTN32 returns the forward N-dimensional DFT of the complex64 row-major array
// data of the given shape, in single precision and unnormalized, like FFTN
// (scipy.fft.fftn on complex64 input). The input is not modified; the same
// shape rules as FFTN apply.
func FFTN32(data []complex64, shape []int) []complex64 {
	validateShape(shape, len(data))
	return cachedPlanN32(shape).FFT(make([]complex64, len(data)), data)
}

// IFFTN32 returns the inverse N-dimensional DFT of data in single precision,
// normalized by the product of the shape, like IFFTN. The input is not
// modified.
func IFFTN32(data []complex64, shape []int) []complex64 {
	validateShape(shape, len(data))
	return cachedPlanN32(shape).IFFT(make([]complex64, len(data)), data)
}

// FFT2_32 is the two-dimensional FFTN32: FFT2 in single precision
// (scipy.fft.fft2 on complex64 input).
func FFT2_32(data []complex64, shape [2]int) []complex64 {
	return FFTN32(data, shape[:])
}

// IFFT2_32 is the two-dimensional IFFTN32, the inverse of FFT2_32, normalized
// by shape[0]*shape[1].
func IFFT2_32(data []complex64, shape [2]int) []complex64 {
	return IFFTN32(data, shape[:])
}

// f32ndKey is the cache key of a plan over shape transforming axes.
func f32ndKey(shape, axes []int) string {
	var kb [96]byte
	key := kb[:0]
	for _, s := range shape {
		key = strconv.AppendInt(key, int64(s), 10)
		key = append(key, ',')
	}
	key = append(key, '|')
	for _, ax := range axes {
		key = strconv.AppendInt(key, int64(ax), 10)
		key = append(key, ',')
	}
	return string(key)
}

var (
	planN32Mu    sync.Mutex
	planN32Cache = map[string]*PlanN32{}
)

// cachedPlanN32 returns the shared plan over every axis of shape (already
// validated).
func cachedPlanN32(shape []int) *PlanN32 {
	return f32ndPlanFor(shape, resolveAxes("FFTN32", nil, len(shape)))
}

// f32ndPlanFor returns the shared PlanN32 over shape (already validated) that
// transforms only the listed axes, building it on first use under the lock
// (f32ndNewPlanN takes only the 1-D cache's own mutex). An unlisted axis gets
// no 1-D plan, so a batch of many rows never builds a plan for its row count.
func f32ndPlanFor(shape, axes []int) *PlanN32 {
	key := f32ndKey(shape, axes)
	planN32Mu.Lock()
	defer planN32Mu.Unlock()
	p, ok := planN32Cache[key]
	if !ok {
		p = f32ndNewPlanN(shape, axes)
		planN32Cache[key] = p
	}
	return p
}
