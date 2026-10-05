package fft

import "strconv"

// Options carries numpy.fft's optional keyword arguments — n, norm and axes —
// to the ...With variants of the transforms (FFTWith, RFFTNWith, ...). Its
// zero value is numpy's defaults, so Options{} makes every ...With function
// behave exactly like its plain counterpart, and a call names only what it
// changes:
//
//	fft.FFTWith(x, fft.Options{N: 1024, Norm: fft.NormOrtho})
//	fft.FFTNWith(img, []int{rows, cols}, fft.Options{Axes: []int{-1}})
//
// A field that has no meaning for the transform it is passed to is refused
// with a panic rather than silently ignored: N belongs to the 1-D transforms,
// Axes to the multi-dimensional ones.
//
// A struct was chosen over functional options or extra positional arguments
// because it keeps the existing signatures untouched (adding a variadic
// parameter to FFT would break every func-typed use of it), because its zero
// value already is numpy's default for each field (the zero Norm is
// NormBackward), and because it costs nothing at run time: no closures, no
// allocation, one type to learn for every transform.
type Options struct {
	// N is numpy's n: the number of input points a 1-D transform uses. The
	// input is truncated to N points, or zero-padded to N, before the
	// transform. For IRFFTWith and HFFTWith, N is the length of the real
	// output (numpy's default for both being 2*(len(input)-1)). Zero selects
	// numpy's default; a negative N panics. N must be zero for the
	// multi-dimensional transforms.
	N int

	// Norm is numpy's norm: how the forward/inverse pair is scaled. The zero
	// value, NormBackward, is numpy's default and what FFT/IFFT do.
	Norm Norm

	// Axes is numpy's axes: the axes of a row-major array the
	// multi-dimensional transforms run along. Negative values count from the
	// last axis, as in numpy (-1 is the last axis). nil transforms every axis;
	// an empty non-nil slice transforms none. An axis may not appear twice.
	// For RFFTNWith/IRFFTNWith the real transform runs along the LAST axis
	// listed, as numpy.fft.rfftn does. Axes must be nil for the 1-D
	// transforms.
	Axes []int
}

// check1D validates o for a 1-D transform called name and returns the number
// of points it transforms: o.N, or def when o.N is zero. It panics before
// anything is allocated.
func (o Options) check1D(name string, def int) int {
	if o.Axes != nil {
		panic("fft: " + name + ": Options.Axes applies only to the multi-dimensional transforms")
	}
	o.Norm.scale(0, false) // panics on an unknown Norm
	if o.N < 0 {
		panic("fft: " + name + ": Options.N must not be negative, got " + strconv.Itoa(o.N))
	}
	if o.N == 0 {
		return max(def, 0)
	}
	return o.N
}

// checkND validates o for a multi-dimensional transform called name over an
// array of ndim axes and returns the axes to transform, normalized to
// 0..ndim-1 in the order given. It panics before anything is allocated.
func (o Options) checkND(name string, ndim int) []int {
	if o.N != 0 {
		panic("fft: " + name + ": Options.N applies only to the 1-D transforms")
	}
	o.Norm.scale(0, false) // panics on an unknown Norm
	return resolveAxes(name, o.Axes, ndim)
}

// resolveAxes normalizes axes against an array of ndim axes: nil means every
// axis, a negative axis counts from the end (numpy's convention), and an axis
// out of range or listed twice panics. numpy itself accepts a repeated axis in
// fftn and transforms it twice; a repeat is almost always a mistake, and for
// rfftn it is not well defined, so it is refused here.
func resolveAxes(name string, axes []int, ndim int) []int {
	if axes == nil {
		all := make([]int, ndim)
		for i := range all {
			all[i] = i
		}
		return all
	}
	seen := make([]bool, ndim)
	out := make([]int, len(axes))
	for i, ax := range axes {
		a := ax
		if a < 0 {
			a += ndim
		}
		if a < 0 || a >= ndim {
			panic("fft: " + name + ": axis " + strconv.Itoa(ax) + " out of range for " + strconv.Itoa(ndim) + " dimensions")
		}
		if seen[a] {
			panic("fft: " + name + ": axis " + strconv.Itoa(ax) + " listed twice")
		}
		seen[a] = true
		out[i] = a
	}
	return out
}

// unitInverseRescale returns the factor that turns the output of an inverse
// transform already normalized by 1/n (IFFT, IRFFT, the plans' inverses) into
// one scaled by want. It is exactly 1 when want is that 1/n, so the default
// NormBackward costs no extra pass and no rounding.
func unitInverseRescale(want float64, n int) float64 {
	if n <= 0 || want == 1/float64(n) {
		return 1
	}
	return want * float64(n)
}

// scaleComplex multiplies every element of x by f, skipping the pass when f
// is 1.
func scaleComplex(x []complex128, f float64) {
	if f == 1 {
		return
	}
	c := complex(f, 0)
	for i := range x {
		x[i] *= c
	}
}

// scaleReal multiplies every element of x by f, skipping the pass when f is 1.
func scaleReal(x []float64, f float64) {
	if f == 1 {
		return
	}
	for i := range x {
		x[i] *= f
	}
}
