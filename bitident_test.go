package fft

import "math"

// simdSignals returns the inputs the bit-identity test runs: a generic signal,
// then signed zeros and infinities. A generic signal cannot tell a multiply by
// one from no multiply, but -0 can: (-0)·1 - (-0)·0 = +0. So the zero and
// infinite signals are what hold the kernels to the scalar i = 0 handling.
func simdSignals(n int) [][]complex128 {
	neg := math.Copysign(0, -1)
	zeros := make([]complex128, n)
	mixed := make([]complex128, n)
	inf := cmplxSignal(n)
	for i := range zeros {
		zeros[i] = complex(neg, neg)
		re, im := neg, 0.0
		if i%3 == 0 {
			re = 0
		}
		if i%2 == 0 {
			im = neg
		}
		if i%5 == 0 {
			re = -1
		}
		mixed[i] = complex(re, im)
	}
	inf[n/2] = complex(math.Inf(1), 0)
	return [][]complex128{cmplxSignal(n), zeros, mixed, inf}
}

// sameBits compares bit patterns; two NaNs count as equal whatever their
// payload, since IEEE 754 leaves the payload of an invalid operation open.
func sameBits(a, b complex128) bool {
	eq := func(x, y float64) bool {
		return math.Float64bits(x) == math.Float64bits(y) || (math.IsNaN(x) && math.IsNaN(y))
	}
	return eq(real(a), real(b)) && eq(imag(a), imag(b))
}
