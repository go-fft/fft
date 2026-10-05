package fft

// This file implements numpy.fft.fftshift / ifftshift
// (https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html).
// fftshift rolls each shifted axis of length n by n/2 (integer division) so
// that the zero-frequency bin moves to the centre: out[(j+n/2) mod n] = x[j].
// ifftshift rolls by -(n/2) and undoes it. For even n the two are the same
// permutation; for odd n they differ by one place, and only ifftshift inverts
// fftshift. Both are generic, so they apply to a spectrum ([]complex128), to
// its magnitudes ([]float64) or to the bin frequencies of FFTFreq alike.

// FFTShift returns x with its zero-frequency element moved to the centre:
// x rotated right by len(x)/2, numpy.fft.fftshift on a 1-D array. For
// FFTFreq(n, d) it returns the frequencies in increasing order. The input is
// not modified; the result is a new slice (empty, not nil, for empty input).
func FFTShift[T any](x []T) []T {
	return rotate(x, len(x)-len(x)/2)
}

// IFFTShift is the inverse of FFTShift: x rotated left by len(x)/2,
// numpy.fft.ifftshift on a 1-D array. IFFTShift(FFTShift(x)) == x for every
// length; for an even length the two functions coincide. The input is not
// modified.
func IFFTShift[T any](x []T) []T {
	return rotate(x, len(x)/2)
}

// rotate returns a new slice holding x[k:] followed by x[:k].
func rotate[T any](x []T, k int) []T {
	out := make([]T, 0, len(x))
	out = append(out, x[k:]...)
	return append(out, x[:k]...)
}

// FFTShiftN is numpy.fft.fftshift on a row-major array of the given shape:
// every axis in axes (every axis when axes is nil; negative axes count from
// the end) is rotated by its length/2, which moves the zero-frequency bin of
// an N-D spectrum to the centre. shape must hold positive lengths whose
// product is len(x), and an axis may not be listed twice; FFTShiftN panics
// otherwise. The input is not modified.
func FFTShiftN[T any](x []T, shape []int, axes []int) []T {
	return shiftN("FFTShiftN", x, shape, axes, false)
}

// IFFTShiftN is numpy.fft.ifftshift on a row-major array of the given shape,
// the inverse of FFTShiftN: every axis in axes (every axis when nil) is
// rotated back by its length/2. The same rules as FFTShiftN apply.
func IFFTShiftN[T any](x []T, shape []int, axes []int) []T {
	return shiftN("IFFTShiftN", x, shape, axes, true)
}

// shiftN copies x row by row (a row being a line of the contiguous last axis):
// output coordinate j along axis a reads source coordinate (j+shift[a]) mod
// shape[a], shift being 0 on an axis that is not shifted, so each output row
// is one source row, rotated left by the last axis's shift — two copies.
func shiftN[T any](name string, x []T, shape, axes []int, inverse bool) []T {
	validateShape(shape, len(x))
	axes = resolveAxes(name, axes, len(shape))
	out := make([]T, len(x))
	if len(shape) == 0 {
		copy(out, x)
		return out
	}
	shift := make([]int, len(shape))
	for _, ax := range axes {
		n := shape[ax]
		if inverse {
			shift[ax] = n / 2
		} else {
			shift[ax] = n - n/2
		}
	}
	stride := rowMajorStrides(shape)
	last := len(shape) - 1
	n := shape[last]
	k := shift[last] // the output row is the source row rotated left by k
	coord := make([]int, last)
	for row := 0; row < len(x)/n; row++ {
		base := 0
		for a, j := range coord {
			base += ((j + shift[a]) % shape[a]) * stride[a]
		}
		src, dst := x[base:base+n], out[row*n:(row+1)*n]
		copy(dst, src[k:])
		copy(dst[n-k:], src[:k])
		// Advance the odometer over the leading axes.
		for a := last - 1; a >= 0; a-- {
			coord[a]++
			if coord[a] < shape[a] {
				break
			}
			coord[a] = 0
		}
	}
	return out
}
