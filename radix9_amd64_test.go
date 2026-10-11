//go:build amd64 && !amd64.v3

package fft

import (
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// TestCs9EachPassMatchesScalar compares every radix-9 pass a kernel runs with
// the Go pass on its own, output against output, bit for bit, as
// TestComp12EachPassMatchesScalar does for radix 12: ido from 1 (the final
// pass) to 33, every remainder of ido mod 4 with and without four-point
// groups, l1 from 1 to 9, on Round 19's signals.
func TestCs9EachPassMatchesScalar(t *testing.T) {
	defer func(a, b bool) { kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = a, b }(kernels.UseStockhamAVX2, kernels.UseStockhamAVX512)
	modes := radix16Modes(t)
	for _, l1 := range []int{1, 2, 3, 4, 5, 9} {
		for _, ido := range []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 33} {
			var f []int
			stage := 0
			if l1 > 1 {
				f, stage = append(f, l1), 1
			}
			f = append(f, 9)
			if ido > 1 {
				f = append(f, ido)
			}
			n := 9 * ido * l1
			// wide is set as a power of two's would be: radix 9 has no
			// 512-bit kernel, so the AVX2 one must run anyway.
			st := newSKPlanFactors(n, f).stages[stage]
			if st.r != 9 || st.l1 != l1 || st.ido != ido {
				t.Fatalf("stage %+v, want r=9 l1=%d ido=%d", st, l1, ido)
			}
			st.wide = true
			for s, x := range radix16Signals(n) {
				for _, inverse := range []bool{false, true} {
					scalar := make([]complex128, n)
					st.passScalar(scalar, x, inverse)
					for _, m := range modes {
						kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = m.avx2, m.avx512
						simd := make([]complex128, n)
						st.pass(simd, x, inverse)
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

// TestCs9TransformMatchesScalar runs whole transforms with radix-9 passes,
// kernels on and off, in place and not.
func TestCs9TransformMatchesScalar(t *testing.T) {
	defer func(a, b bool) { kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = a, b }(kernels.UseStockhamAVX2, kernels.UseStockhamAVX512)
	modes := radix16Modes(t)
	for _, f := range cs9FactorLists {
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

// TestCs9Columns: with the radix-9 rule on, an N-D plan's strided axes take
// the factorization without radix 9 (no batched radix-9 kernel: compTakes16)
// and its last axis the rule's, and the result is still the DFT.
func TestCs9Columns(t *testing.T) {
	defer func(v, w, x, y bool) { compRadix16On, comp2On, hswCompOn, csRadix9On = v, w, x, y }(compRadix16On, comp2On, hswCompOn, csRadix9On)
	compRadix16On, comp2On, hswCompOn, csRadix9On = true, false, true, true
	forget := func() { planMu.Lock(); delete(planCache, 18); delete(planNo16Cache, 18); planMu.Unlock() }
	forget() // plans cached under another rule
	defer forget()
	p := NewPlanN(18, 18)
	for _, st := range p.axes[0].sk.stages {
		if st.r == 9 {
			t.Fatalf("column axis factored %v", p.axes[0].sk.stages)
		}
	}
	if p.axes[1].sk.stages[0].r != 9 {
		t.Fatalf("row axis factored %v, want a radix-9 pass", p.axes[1].sk.stages)
	}
	x := cmplxSignal(18 * 18)
	closeVec(t, p.FFT(make([]complex128, len(x)), x), naiveDFTN(x, []int{18, 18}))
}
