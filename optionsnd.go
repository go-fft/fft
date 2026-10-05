package fft

import (
	"strconv"
	"sync"
)

// This file holds the multi-dimensional transforms with numpy.fft's optional
// arguments (axes and norm), and the N-dimensional real pair rfftn/irfftn.
// numpy.fft.fftn(a, axes=..., norm=...) transforms only the listed axes of a;
// rfftn runs the real transform along the LAST listed axis and the complex
// one along the others; irfftn inverts that, the complex inverse first and
// the real inverse last
// (https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftn.html).
// norm applies to the transform as a whole, so its factor is computed from
// the product of the transformed axis lengths: 1/sqrt(n0·n1·…) is the product
// of the per-axis 1/sqrt(ni).
//
// Transforming the last axis alone is the batched 1-D transform — numpy's
// fft(x, axis=-1) on a matrix of signals: every row is transformed with one
// plan, across goroutines when the work is large enough.

// FFTNWith is FFTN with numpy.fft.fftn's axes and norm arguments: data is a
// row-major array of the given shape, only the axes in o.Axes are transformed
// (every axis when o.Axes is nil), and the result is scaled as o.Norm says.
// FFTNWith(data, shape, Options{}) equals FFTN(data, shape). The output has the
// input's shape; the input is not modified. o.N must be zero.
//
// With o.Axes = []int{-1} it is the batched 1-D transform: every row of a
// matrix of signals is transformed with one plan.
func FFTNWith(data []complex128, shape []int, o Options) []complex128 {
	return complexNWith("FFTNWith", data, shape, o, false)
}

// IFFTNWith is IFFTN with numpy.fft.ifftn's axes and norm arguments; see
// FFTNWith. Under the default NormBackward the result is divided by the
// product of the transformed axis lengths.
func IFFTNWith(data []complex128, shape []int, o Options) []complex128 {
	return complexNWith("IFFTNWith", data, shape, o, true)
}

// FFT2With is FFT2 with numpy.fft.fft2's axes and norm arguments: FFTNWith on
// a two-axis shape.
func FFT2With(data []complex128, shape [2]int, o Options) []complex128 {
	return complexNWith("FFT2With", data, shape[:], o, false)
}

// IFFT2With is IFFT2 with numpy.fft.ifft2's axes and norm arguments: IFFTNWith
// on a two-axis shape.
func IFFT2With(data []complex128, shape [2]int, o Options) []complex128 {
	return complexNWith("IFFT2With", data, shape[:], o, true)
}

func complexNWith(name string, data []complex128, shape []int, o Options, inverse bool) []complex128 {
	validateShape(shape, len(data))
	axes := o.checkND(name, len(shape))
	f := o.Norm.scale(axesProduct(shape, axes), inverse)
	out := make([]complex128, len(data))
	planNFor(shape, axes).transform(out, data, inverse)
	scaleComplex(out, f)
	return out
}

// RFFTN returns the forward N-dimensional DFT of the real row-major array data
// of the given shape, keeping only the non-redundant half of the last axis: the
// result is a row-major array of the same shape except that the last length n
// becomes n/2+1. The real transform runs along the last axis and the complex
// one along the others, unnormalized. This is numpy.fft.rfftn; RFFT2 is its
// two-dimensional case.
//
// shape must hold at least one axis, every length must be positive and their
// product must equal len(data); RFFTN panics otherwise. The input is not
// modified.
func RFFTN(data []float64, shape []int) []complex128 {
	return RFFTNWith(data, shape, Options{})
}

// RFFTNWith is RFFTN with numpy.fft.rfftn's axes and norm arguments: only the
// axes in o.Axes are transformed (every axis when nil), the real transform runs
// along the LAST axis listed — that axis of the output keeps n/2+1 bins — and
// the result is scaled as o.Norm says. o.Axes must name at least one axis; o.N
// must be zero.
func RFFTNWith(data []float64, shape []int, o Options) []complex128 {
	validateShape(shape, len(data))
	axes := o.checkND("RFFTNWith", len(shape))
	if len(axes) == 0 {
		panic("fft: RFFTNWith: no axis to transform")
	}
	f := o.Norm.scale(axesProduct(shape, axes), false)
	ra := axes[len(axes)-1]
	half := halfShape(shape, ra)
	out := make([]complex128, len(data)/shape[ra]*half[ra])
	realAxisForward(out, data, shape, half, ra)
	if len(axes) > 1 {
		planNFor(half, axes[:len(axes)-1]).transform(out, out, false)
	}
	scaleComplex(out, f)
	return out
}

// IRFFTN inverts RFFTN: shape is the shape of the REAL output (its last length
// n cannot be recovered from the n/2+1 bins it keeps, as for IRFFT), and data
// is the spectrum laid out as RFFTN produces it, that shape with the last
// length replaced by n/2+1. The complex inverse runs along every axis but the
// last, then the real inverse along the last; the result is normalized by the
// product of the shape. This is numpy.fft.irfftn(data, s=shape).
//
// data is read up to the spectrum's size and missing bins are zero, as for
// IRFFT2. shape must hold at least one axis and every length must be positive;
// IRFFTN panics otherwise. The input is not modified. The output has the
// product of shape values whatever the spectrum's length: bound the shape when
// it comes from untrusted input (see SECURITY.md).
func IRFFTN(data []complex128, shape []int) []float64 {
	return IRFFTNWith(data, shape, Options{})
}

// IRFFTNWith is IRFFTN with numpy.fft.irfftn's axes and norm arguments: the
// real inverse runs along the LAST axis in o.Axes (every axis when nil), whose
// spectrum holds shape[axis]/2+1 bins, the complex inverse along the others,
// and the result is scaled as o.Norm says. o.Axes must name at least one axis;
// o.N must be zero.
func IRFFTNWith(data []complex128, shape []int, o Options) []float64 {
	total := shapeProduct(shape...)
	axes := o.checkND("IRFFTNWith", len(shape))
	if len(axes) == 0 {
		panic("fft: IRFFTNWith: no axis to transform")
	}
	ra := axes[len(axes)-1]
	n := shape[ra]
	f := unitInverseRescale(o.Norm.scale(axesProduct(shape, axes), true), n)
	half := halfShape(shape, ra)
	spec := make([]complex128, total/n*half[ra])
	copy(spec, data) // missing bins read as zero, extra ones are ignored
	if len(axes) > 1 {
		planNFor(half, axes[:len(axes)-1]).transform(spec, spec, true)
	}
	out := make([]float64, total)
	realAxisInverse(out, spec, shape, half, ra) // normalized by 1/n
	scaleReal(out, f)
	return out
}

// RFFT2With is RFFT2 with numpy.fft.rfft2's axes and norm arguments. With
// o.Axes nil it runs RFFT2's plan and scales the result; with axes it is
// RFFTNWith on a two-axis shape.
func RFFT2With(data []float64, shape [2]int, o Options) []complex128 {
	if o.Axes != nil {
		return RFFTNWith(data, shape[:], o)
	}
	o.checkND("RFFT2With", 2)
	f := o.Norm.scale(shapeProduct(shape[0], shape[1]), false)
	out := RFFT2(data, shape)
	scaleComplex(out, f)
	return out
}

// IRFFT2With is IRFFT2 with numpy.fft.irfft2's axes and norm arguments. With
// o.Axes nil it runs IRFFT2's plan and rescales the result; with axes it is
// IRFFTNWith on a two-axis shape.
func IRFFT2With(data []complex128, shape [2]int, o Options) []float64 {
	if o.Axes != nil {
		return IRFFTNWith(data, shape[:], o)
	}
	o.checkND("IRFFT2With", 2)
	total := shapeProduct(shape[0], shape[1])
	f := unitInverseRescale(o.Norm.scale(total, true), total)
	out := IRFFT2(data, shape)
	scaleReal(out, f)
	return out
}

// axesProduct is the product of the lengths of the listed axes: the number of
// points the transform covers, which sets its Norm factor. It cannot overflow:
// the whole shape's product has been checked.
func axesProduct(shape, axes []int) int {
	p := 1
	for _, ax := range axes {
		p *= shape[ax]
	}
	return p
}

// halfShape is shape with axis ra replaced by the n/2+1 bins a real transform
// along it keeps.
func halfShape(shape []int, ra int) []int {
	h := append([]int(nil), shape...)
	h[ra] = shape[ra]/2 + 1
	return h
}

// rowMajorStrides returns the row-major strides of shape.
func rowMajorStrides(shape []int) []int {
	st := make([]int, len(shape))
	acc := 1
	for ax := len(shape) - 1; ax >= 0; ax-- {
		st[ax] = acc
		acc *= shape[ax]
	}
	return st
}

// realAxisForward runs the real forward transform along axis ra of src (shape
// shape) into dst (shape half), line by line. A line of the last axis is
// contiguous on both sides and transformed in place; any other axis is
// gathered into scratch and scattered back.
func realAxisForward(dst []complex128, src []float64, shape, half []int, ra int) {
	n, h := shape[ra], half[ra]
	rp := cachedRealPlan(n)
	inSt, outSt := rowMajorStrides(shape), rowMajorStrides(half)
	lines := len(src) / n
	body := func(lo, hi int) {
		if inSt[ra] == 1 {
			for c := lo; c < hi; c++ {
				rp.RFFT(dst[c*h:(c+1)*h], src[c*n:(c+1)*n])
			}
			return
		}
		rb, cb := make([]float64, n), make([]complex128, h)
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

// realAxisInverse runs the real inverse transform (normalized by 1/n) along
// axis ra of the spectrum src (shape half) into dst (shape shape), line by
// line, as realAxisForward does in the other direction.
func realAxisInverse(dst []float64, src []complex128, shape, half []int, ra int) {
	n, h := shape[ra], half[ra]
	rp := cachedRealPlan(n)
	outSt, inSt := rowMajorStrides(shape), rowMajorStrides(half)
	lines := len(dst) / n
	body := func(lo, hi int) {
		if outSt[ra] == 1 {
			for c := lo; c < hi; c++ {
				rp.IRFFT(dst[c*n:(c+1)*n], src[c*h:(c+1)*h])
			}
			return
		}
		rb, cb := make([]float64, n), make([]complex128, h)
		for c := lo; c < hi; c++ {
			ib, ob := lineBase(c, half, inSt, ra), lineBase(c, shape, outSt, ra)
			for k := range cb {
				cb[k] = src[ib+k*inSt[ra]]
			}
			rp.IRFFT(rb, cb)
			for i, v := range rb {
				dst[ob+i*outSt[ra]] = v
			}
		}
	}
	runLines(lines, n, body)
}

// runLines runs body over lines independent transforms of length n, across
// goroutines when the work is large enough (parallel.go's rule), inline
// otherwise.
func runLines(lines, n int, body func(lo, hi int)) {
	if parallelizeLines(lines, n) {
		parChunks(lines, n, body)
		return
	}
	body(0, lines)
}

// planNFor returns a PlanN over shape that transforms only the listed axes.
// Every axis is FFTN's own cached plan; a subset gets a plan of its own,
// cached by shape and axes, whose unlisted axes have no 1-D plan — so a batch
// of a million rows never builds a million-point plan it will not run.
func planNFor(shape, axes []int) *PlanN {
	if len(axes) == len(shape) {
		return cachedPlanN(shape)
	}
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
	planNAxesMu.Lock()
	defer planNAxesMu.Unlock()
	p, ok := planNAxesCache[string(key)]
	if !ok {
		p = newPlanNAxes(shape, axes)
		planNAxesCache[string(key)] = p
	}
	return p
}

var (
	planNAxesMu    sync.Mutex
	planNAxesCache = map[string]*PlanN{}
)

// newPlanNAxes builds a PlanN for shape whose transform runs along the listed
// axes only: PlanN.transform skips an axis without a 1-D plan. Its IFFT
// method's 1/Len() normalization would be wrong for a subset, so such a plan is
// never handed out; callers use transform and scale themselves.
func newPlanNAxes(shape, axes []int) *PlanN {
	p := &PlanN{shape: append([]int(nil), shape...)}
	p.size = shapeProduct(shape...)
	p.stride = rowMajorStrides(shape)
	p.axes = make([]*Plan, len(shape))
	for _, ax := range axes {
		n := shape[ax]
		if n > 1 {
			p.axes[ax] = cachedPlan(n)
		}
		p.maxLen = max(p.maxLen, n)
	}
	bl := lineBlock * (p.maxLen + linePad)
	p.bufs.New = func() any { b := make([]complex128, bl); return &b }
	return p
}
