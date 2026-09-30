//go:build amd64 && !amd64.v3

package fft

import (
	"math"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// TestStockhamPassMatchesScalar runs every smooth length up to 2100, and a few
// larger ones, through the Stockham engine twice — AVX2 pass kernels on, then
// off — and requires the two results to be bit-identical, forward and inverse,
// in place and not. The range covers radix 2/3/4/5/8 passes at ido == 2, at
// odd ido (the 128-bit tail) and at long ido, with many and few blocks. It is
// restricted to GOAMD64 < v3, where the scalar oracle does not fuse multiply-
// adds; the kernels never do.
func TestStockhamPassMatchesScalar(t *testing.T) {
	if !kernels.UseStockhamAVX2 {
		t.Skip("no AVX2 on this CPU: the scalar passes run and there is nothing to compare")
	}
	defer func(v bool) { kernels.UseStockhamAVX2 = v }(kernels.UseStockhamAVX2)
	sizes := []int{8192, 20160, 45000, 65536 * 3}
	for n := 2; n <= 2100; n++ {
		sizes = append(sizes, n)
	}
	for _, n := range sizes {
		if !factorsAreSmall(n) {
			continue
		}
		p := newSKPlan(n)
		for s, x := range simdSignals(n) {
			for _, inverse := range []bool{false, true} {
				kernels.UseStockhamAVX2 = true
				simd := make([]complex128, n)
				p.transform(simd, x, inverse)
				alias := append([]complex128(nil), x...)
				p.transform(alias, alias, inverse)
				kernels.UseStockhamAVX2 = false
				scalar := make([]complex128, n)
				p.transform(scalar, x, inverse)
				for i := range scalar {
					if !sameBits(simd[i], scalar[i]) || !sameBits(alias[i], scalar[i]) {
						t.Fatalf("n=%d signal %d inverse=%v index %d: AVX2 %v (in place %v) vs scalar %v (factors %v)",
							n, s, inverse, i, simd[i], alias[i], scalar[i], skFactorize(n))
					}
				}
			}
		}
	}
}

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
