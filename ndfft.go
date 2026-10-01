package fft

import (
	"strconv"
	"sync"
)

// This file implements the multi-dimensional transforms. They operate on a flat
// row-major (C-order) slice paired with an explicit shape, mirroring the
// semantics of numpy.fft.fft2 / ifft2 / fftn / ifftn / rfft2 / irfft2.
//
// A separable N-dimensional DFT is the composition of 1-D DFTs along each axis:
// transform along axis 0, then axis 1, and so on. Each 1-D transform reuses the
// scalar FFT/IFFT core. The forward transforms are unnormalized; the inverse
// transforms divide by the product of the transformed axis lengths, so that
// IFFTN(FFTN(x)) ≈ x. None of these functions mutate the caller's input.

// FFTN returns the forward N-dimensional discrete Fourier transform of data,
// interpreted as a row-major (C-order) array of the given shape. The result is a
// new slice of the same length; the input is not modified.
//
// The transform is separable and unnormalized: it applies the 1-D FFT along each
// axis in turn, matching numpy.fft.fftn. shape must contain only positive
// lengths whose product equals len(data); FFTN panics otherwise (consistent with
// numpy raising on a mismatched shape). An empty shape transforms the single
// scalar unchanged.
func FFTN(data []complex128, shape []int) []complex128 {
	return transformN(data, shape, false)
}

// IFFTN returns the inverse N-dimensional discrete Fourier transform of data,
// the inverse of FFTN. It applies the 1-D IFFT along each axis, so the result is
// normalized by the product of the transformed axis lengths and
// IFFTN(FFTN(x), shape) ≈ x. The input is not modified; the same shape rules as
// FFTN apply.
func IFFTN(data []complex128, shape []int) []complex128 {
	return transformN(data, shape, true)
}

// FFT2 is the two-dimensional FFTN: it transforms data as a row-major
// shape[0]×shape[1] matrix (rows then columns), matching numpy.fft.fft2.
func FFT2(data []complex128, shape [2]int) []complex128 {
	return FFTN(data, shape[:])
}

// IFFT2 is the two-dimensional IFFTN, the inverse of FFT2, normalized by
// shape[0]*shape[1]. It matches numpy.fft.ifft2.
func IFFT2(data []complex128, shape [2]int) []complex128 {
	return IFFTN(data, shape[:])
}

// validateShape checks that every axis length is positive and that the product
// equals total, returning the product. It panics on any violation, mirroring
// numpy's refusal to run on a shape that does not match the data.
func validateShape(shape []int, total int) int {
	prod := 1
	for _, s := range shape {
		if s <= 0 {
			panic("fft: shape lengths must be positive")
		}
		prod *= s
	}
	if prod != total {
		panic("fft: shape product does not match len(data)")
	}
	return prod
}

// transformN applies a 1-D FFT (or IFFT when inverse) along every axis of a
// row-major array, without mutating the caller's slice: a PlanN writing into a
// new slice.
func transformN(data []complex128, shape []int, inverse bool) []complex128 {
	validateShape(shape, len(data))
	p := cachedPlanN(shape)
	out := make([]complex128, len(data))
	if inverse {
		return p.IFFT(out, data)
	}
	return p.FFT(out, data)
}

// lineBase maps a linear line index c (0 <= c < total/shape[ax]) to the flat
// base offset of that line in a row-major array, with axis ax held at index 0.
// It decodes c as a mixed-radix number over the axes other than ax (least
// significant = last axis) and accumulates each digit times its stride, so any
// line can be located directly without a sequential odometer — the property the
// parallel chunking relies on.
func lineBase(c int, shape, stride []int, ax int) int {
	base := 0
	for a := len(shape) - 1; a >= 0; a-- {
		if a == ax {
			continue
		}
		s := shape[a]
		base += (c % s) * stride[a]
		c /= s
	}
	return base
}

// planNCache memoizes N-D plans by shape, so FFTN/FFT2 called repeatedly on
// one shape reuse its per-axis plans and its warm scratch pool instead of
// rebuilding them every call.
var (
	planNMu    sync.Mutex
	planNCache = map[string]*PlanN{}
)

// cachedPlanN returns the shared plan for shape, building it on first use.
func cachedPlanN(shape []int) *PlanN {
	var kb [64]byte
	key := kb[:0]
	for _, s := range shape {
		key = strconv.AppendInt(key, int64(s), 10)
		key = append(key, ',')
	}
	// The plan is built under the lock (NewPlanN takes only cachedPlan's own,
	// separate mutex), so concurrent first calls on one shape build it once.
	planNMu.Lock()
	defer planNMu.Unlock()
	p, ok := planNCache[string(key)]
	if !ok {
		p = NewPlanN(shape...)
		planNCache[string(key)] = p
	}
	return p
}
