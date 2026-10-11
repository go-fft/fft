//go:build amd64 && !amd64.v3

package fft

import (
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// TestR32EachPassMatchesScalar compares every radix-32 pass the kernels run
// with the Go pass on its own, output against output, bit for bit, on
// Round 19's seven signals: ido from 1 (the final pass) to 33, so every
// remainder of ido mod 4 with and without four-point groups, and l1 from 1
// to 9 (odd l1: the final pass's 128-bit tail).
func TestR32EachPassMatchesScalar(t *testing.T) {
	defer func(a, b bool) { kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = a, b }(kernels.UseStockhamAVX2, kernels.UseStockhamAVX512)
	modes := radix16Modes(t)
	for _, l1 := range []int{1, 2, 3, 4, 5, 9} {
		for _, ido := range []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 33} {
			var f []int
			stage := 0
			if l1 > 1 {
				f, stage = append(f, l1), 1
			}
			f = append(f, 32)
			if ido > 1 {
				f = append(f, ido)
			}
			n := 32 * ido * l1
			st := newSKPlanFactors(n, f).stages[stage]
			if st.r != 32 || st.l1 != l1 || st.ido != ido {
				t.Fatalf("stage %+v, want r=32 l1=%d ido=%d", st, l1, ido)
			}
			st.wide = true // no 512-bit radix-32 kernel: the AVX2 one runs
			for s, x := range radix16Signals(n) {
				for _, inverse := range []bool{false, true} {
					scalar := make([]complex128, n)
					st.passScalar(scalar, x, inverse)
					for _, m := range modes {
						kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = m.avx2, m.avx512
						simd := make([]complex128, n)
						tw := st.twX
						if inverse {
							tw = st.twXc
						}
						// The kernel itself, so that a pass falling back to Go
						// cannot pass this test.
						if !kernels.StockhamPass(32, ido, l1, x, simd, tw, inverse, st.wide) {
							t.Fatalf("%s: no kernel ran the radix-32 pass", m.name)
						}
						for i := range scalar {
							if !sameBits(simd[i], scalar[i]) {
								t.Fatalf("%s l1=%d ido=%d signal %d inverse=%v index %d: %v vs Go %v",
									m.name, l1, ido, s, inverse, i, simd[i], scalar[i])
							}
						}
					}
				}
			}
		}
	}
}

// TestR32TransformMatchesScalar runs whole transforms with radix-32 passes,
// kernels on and off, in place and not.
func TestR32TransformMatchesScalar(t *testing.T) {
	defer func(a, b bool) { kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = a, b }(kernels.UseStockhamAVX2, kernels.UseStockhamAVX512)
	modes := radix16Modes(t)
	for _, f := range r32FactorLists {
		n := 1
		for _, r := range f {
			n *= r
		}
		p := newSKPlanFactors(n, f)
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
