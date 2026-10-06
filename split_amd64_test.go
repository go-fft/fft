//go:build amd64 && !amd64.v3

package fft

import (
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// splitToBlocks returns x in the amd64 split layout: four points 4q .. 4q+3
// as re(4q), re(4q+2), re(4q+1), re(4q+3), then the imaginary parts in the
// same order, in the 64 bytes the four points occupy (len(x) a multiple of 4).
func splitToBlocks(x []complex128) []complex128 {
	y := make([]complex128, len(x))
	for q := 0; q < len(x); q += 4 {
		a, b, c, d := x[q], x[q+1], x[q+2], x[q+3]
		y[q] = complex(real(a), real(c))
		y[q+1] = complex(real(b), real(d))
		y[q+2] = complex(imag(a), imag(c))
		y[q+3] = complex(imag(b), imag(d))
	}
	return y
}

// splitFromBlocks is splitToBlocks' inverse.
func splitFromBlocks(y []complex128) []complex128 {
	x := make([]complex128, len(y))
	for q := 0; q < len(y); q += 4 {
		x[q] = complex(real(y[q]), real(y[q+2]))
		x[q+1] = complex(real(y[q+1]), real(y[q+3]))
		x[q+2] = complex(imag(y[q]), imag(y[q+2]))
		x[q+3] = complex(imag(y[q+1]), imag(y[q+3]))
	}
	return x
}

// splitOn builds plans with the split layout on (or off) until the returned
// function restores the setting.
func splitOn(on bool) func() {
	old := kernels.UseStockhamSplit
	kernels.UseStockhamSplit = on
	return func() { kernels.UseStockhamSplit = old }
}

// TestSplitEachPassMatchesScalar compares every split pass kernel alone with
// the Go pass, output for output, bit for bit, in every layout mode: the
// input converted to the split layout where the kernel reads it so, the
// output converted back where it writes it so. A whole transform absorbs most
// of a zero's sign that one pass gets wrong (Round 18), so this is the test
// that sees one. It covers l1 from 1 to 9 and ido from 4 to 36 (one group,
// several, the first group only), on radix16Signals' seven signals.
func TestSplitEachPassMatchesScalar(t *testing.T) {
	if !kernels.UseStockhamAVX2 {
		t.Skip("no AVX2 on this CPU: the Go passes run and there is nothing to compare")
	}
	defer splitOn(true)()
	for _, r := range []int{4, 8} {
		for _, l1 := range []int{1, 2, 3, 4, 5, 9} {
			for _, ido := range []int{4, 8, 12, 16, 20, 36} {
				var f []int
				stage := 0
				if l1 > 1 {
					f, stage = append(f, l1), 1
				}
				f = append(f, r, ido)
				n := r * ido * l1
				p := newSKPlanFactors(n, f)
				st := p.stages[stage]
				if st.r != r || st.l1 != l1 || st.ido != ido {
					t.Fatalf("stage %+v, want r=%d l1=%d ido=%d", st, r, l1, ido)
				}
				if st.split == 0 {
					t.Fatalf("stage %+v: not split, so its twiddles are not in the split order", st)
				}
				for mode := uint8(1); mode <= 4; mode++ {
					st.split = mode
					for s, x := range radix16Signals(n) {
						for _, inverse := range []bool{false, true} {
							scalar := make([]complex128, n)
							st.passScalar(scalar, x, inverse)
							in := x
							if mode == 2 || mode == 3 {
								in = splitToBlocks(x)
							}
							simd := make([]complex128, n)
							st.pass(simd, in, inverse)
							if mode == 1 || mode == 2 {
								simd = splitFromBlocks(simd)
							}
							for i := range scalar {
								if !sameBits(simd[i], scalar[i]) {
									t.Fatalf("r=%d l1=%d ido=%d mode %d signal %d inverse=%v index %d: %v vs Go %v",
										r, l1, ido, mode, s, inverse, i, simd[i], scalar[i])
								}
							}
						}
					}
				}
			}
		}
	}
}

// splitFactorLists are factorizations whose passes take the split layout in
// runs of one, two and more, next to radix-8, radix-16 and odd passes, and
// with the run first, in the middle and before a final pass of every radix.
var splitFactorLists = [][]int{
	{4, 4}, {4, 8}, {4, 16}, {4, 4, 4}, {4, 4, 8}, {4, 4, 16}, {4, 4, 4, 4},
	{4, 4, 4, 4, 4}, {4, 4, 4, 16}, {8, 4, 4, 8}, {8, 4, 4, 4}, {4, 8, 4, 4},
	{3, 4, 4, 4}, {5, 4, 4, 8}, {4, 4, 3}, {4, 4, 5}, {4, 4, 2}, {4, 4, 7},
	{16, 4, 4}, {4, 4, 4, 4, 4, 4, 4, 4},
	{8, 8}, {8, 8, 8}, {4, 8, 8, 8}, {8, 8, 16}, {8, 4, 8}, {3, 8, 8}, {8, 8, 5},
	{8, 8, 8, 8, 8, 8},
}

// TestSplitTransformMatchesScalar runs whole transforms with split runs,
// kernels on and off, in place and not, bit for bit.
func TestSplitTransformMatchesScalar(t *testing.T) {
	defer func(a, b bool) { kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = a, b }(kernels.UseStockhamAVX2, kernels.UseStockhamAVX512)
	modes := radix16Modes(t)
	defer splitOn(true)()
	for _, f := range splitFactorLists {
		n := 1
		for _, r := range f {
			n *= r
		}
		p := newSKPlanFactors(n, f)
		split := false
		for _, st := range p.stages {
			split = split || st.split != 0
		}
		if !split {
			t.Fatalf("factors %v: no split pass", f)
		}
		for s, x := range radix16Signals(n) {
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
							t.Fatalf("%s factors %v signal %d inverse=%v index %d: %v (in place %v) vs Go %v",
								m.name, f, s, inverse, i, simd[i], alias[i], scalar[i])
						}
					}
				}
			}
		}
	}
}
