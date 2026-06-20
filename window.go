package fft

import "math"

// This file implements the classic symmetric window functions, matching the
// definitions and edge-case behavior of numpy (numpy.hanning, numpy.hamming,
// numpy.blackman, numpy.bartlett) and scipy's Blackman–Harris window. Each
// returns a freshly allocated []float64 of length n.
//
// For all windows: n <= 0 returns an empty (non-nil) slice, and n == 1 returns
// []float64{1}. For n >= 2 the window is sampled over n points so that the
// argument runs from 0 to 2π across the full window (the "symmetric" form, the
// numpy default), using the divisor n-1.

// Hann returns the length-n Hann (Hanning) window:
//
//	w[k] = 0.5 - 0.5·cos(2π·k/(n-1)),  k = 0 .. n-1.
//
// This matches numpy.hanning(n).
func Hann(n int) []float64 {
	return cosineWindow(n, []float64{0.5, 0.5})
}

// Hamming returns the length-n Hamming window:
//
//	w[k] = 0.54 - 0.46·cos(2π·k/(n-1)),  k = 0 .. n-1.
//
// This matches numpy.hamming(n).
func Hamming(n int) []float64 {
	return cosineWindow(n, []float64{0.54, 0.46})
}

// Blackman returns the length-n Blackman window:
//
//	w[k] = 0.42 - 0.5·cos(2π·k/(n-1)) + 0.08·cos(4π·k/(n-1)).
//
// This matches numpy.blackman(n).
func Blackman(n int) []float64 {
	return cosineWindow(n, []float64{0.42, 0.5, 0.08})
}

// BlackmanHarris returns the length-n (4-term) Blackman–Harris window with the
// standard coefficients a0..a3 = 0.35875, 0.48829, 0.14128, 0.01168:
//
//	w[k] = a0 - a1·cos(2π·k/(n-1)) + a2·cos(4π·k/(n-1)) - a3·cos(6π·k/(n-1)).
//
// This matches scipy.signal.windows.blackmanharris(n, sym=True).
func BlackmanHarris(n int) []float64 {
	return cosineWindow(n, []float64{0.35875, 0.48829, 0.14128, 0.01168})
}

// cosineWindow evaluates a generalized cosine window with the given non-negative
// coefficients a:
//
//	w[k] = sum_{j} (-1)^j · a[j] · cos(2π·j·k/(n-1)).
//
// a[0] is the constant term. This is the shared engine for Hann, Hamming,
// Blackman, and Blackman–Harris.
func cosineWindow(n int, a []float64) []float64 {
	if n <= 0 {
		return []float64{}
	}
	if n == 1 {
		return []float64{1}
	}
	w := make([]float64, n)
	denom := float64(n - 1)
	for k := 0; k < n; k++ {
		s := a[0]
		sign := -1.0
		for j := 1; j < len(a); j++ {
			s += sign * a[j] * math.Cos(2*math.Pi*float64(j)*float64(k)/denom)
			sign = -sign
		}
		w[k] = s
	}
	return w
}

// Bartlett returns the length-n Bartlett (triangular) window:
//
//	w[k] = (2/(n-1)) · ((n-1)/2 - |k - (n-1)/2|),  k = 0 .. n-1,
//
// which rises linearly from 0 to 1 at the center and back to 0. This matches
// numpy.bartlett(n).
func Bartlett(n int) []float64 {
	if n <= 0 {
		return []float64{}
	}
	if n == 1 {
		return []float64{1}
	}
	w := make([]float64, n)
	denom := float64(n - 1)
	half := denom / 2
	for k := 0; k < n; k++ {
		w[k] = (2 / denom) * (half - math.Abs(float64(k)-half))
	}
	return w
}
