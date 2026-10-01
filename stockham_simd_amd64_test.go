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
	defer func(a, b bool) { kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = a, b }(kernels.UseStockhamAVX2, kernels.UseStockhamAVX512)
	modes := []struct {
		name         string
		avx2, avx512 bool
	}{{"AVX2", true, false}}
	if kernels.UseStockhamAVX512 {
		modes = append(modes, struct {
			name         string
			avx2, avx512 bool
		}{"AVX-512", true, true})
	} else {
		t.Log("no AVX-512 on this CPU: only the AVX2 kernels are compared")
	}
	sizes := []int{4096, 8192, 16384, 20160, 45000, 65536, 65536 * 3}
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
				kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = false, false
				scalar := make([]complex128, n)
				p.transform(scalar, x, inverse)
				for _, m := range modes {
					kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = m.avx2, m.avx512
					simd := make([]complex128, n)
					p.transform(simd, x, inverse)
					alias := append([]complex128(nil), x...)
					p.transform(alias, alias, inverse)
					for i := range scalar {
						if !sameBits(simd[i], scalar[i]) || !sameBits(alias[i], scalar[i]) {
							t.Fatalf("%s n=%d signal %d inverse=%v index %d: %v (in place %v) vs scalar %v (factors %v)",
								m.name, n, s, inverse, i, simd[i], alias[i], scalar[i], skFactorize(n))
						}
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

// TestUntangleMatchesScalar holds the AVX2 real-FFT untangle to the Go loop,
// bit for bit, at every half-length m up to 700 and a few large ones, on the
// same generic, signed-zero and infinite signals as the pass kernels.
func TestUntangleMatchesScalar(t *testing.T) {
	if !kernels.UseUntangleAVX2 {
		t.Skip("no AVX2 on this CPU: the Go loop runs and there is nothing to compare")
	}
	defer func(v bool) { kernels.UseUntangleAVX2 = v }(kernels.UseUntangleAVX2)
	sizes := []int{2048, 4097, 32768}
	for m := 2; m <= 700; m++ {
		sizes = append(sizes, m)
	}
	for _, m := range sizes {
		tw := NewRealPlan(2 * m).tw
		for s, z := range simdSignals(m) {
			kernels.UseUntangleAVX2 = true
			simd := make([]complex128, m+1)
			rfftUntangle(simd, z, tw, m)
			kernels.UseUntangleAVX2 = false
			scalar := make([]complex128, m+1)
			rfftUntangle(scalar, z, tw, m)
			for k := range scalar {
				if !sameBits(simd[k], scalar[k]) {
					t.Fatalf("m=%d signal %d bin %d: AVX2 %v vs Go %v", m, s, k, simd[k], scalar[k])
				}
			}
		}
	}
}

// TestRetangleMatchesScalar holds the AVX2 inverse untangle to the Go loop,
// bit for bit, the same way.
func TestRetangleMatchesScalar(t *testing.T) {
	if !kernels.UseUntangleAVX2 {
		t.Skip("no AVX2 on this CPU: the Go loop runs and there is nothing to compare")
	}
	defer func(v bool) { kernels.UseUntangleAVX2 = v }(kernels.UseUntangleAVX2)
	sizes := []int{2048, 4097, 32768}
	for m := 2; m <= 700; m++ {
		sizes = append(sizes, m)
	}
	for _, m := range sizes {
		tw := NewRealPlan(2 * m).tw
		for s, x := range simdSignals(m + 1) {
			for _, h := range []float64{0.5, 0.5 / float64(m)} {
				kernels.UseUntangleAVX2 = true
				simd := make([]complex128, m)
				irfftRetangle(simd, x, tw, m, h)
				kernels.UseUntangleAVX2 = false
				scalar := make([]complex128, m)
				irfftRetangle(scalar, x, tw, m, h)
				for k := range scalar {
					if !sameBits(simd[k], scalar[k]) {
						t.Fatalf("m=%d signal %d h=%v bin %d: AVX2 %v vs Go %v", m, s, h, k, simd[k], scalar[k])
					}
				}
			}
		}
	}
}
