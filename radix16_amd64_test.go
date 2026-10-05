//go:build amd64 && !amd64.v3

package fft

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// radix16Signals is simdSignals plus the signals a pass-level comparison
// needs to see a zero's sign or a rounding move: pseudo-random draws from
// {+0, −0} and from {+0, −0, ±1}, and odd multiples of the smallest
// subnormal, where the products by the butterfly's constants round.
func radix16Signals(n int) [][]complex128 {
	sig := simdSignals(n)
	rng := rand.New(rand.NewPCG(16, uint64(n)))
	neg := math.Copysign(0, -1)
	pick := func(vals []float64) []complex128 {
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(vals[rng.IntN(len(vals))], vals[rng.IntN(len(vals))])
		}
		return x
	}
	sub := make([]complex128, n)
	for i := range sub {
		sub[i] = complex(float64(2*(i%7)+1)*5e-324, -float64(2*(i%5)+1)*5e-324)
	}
	return append(sig, pick([]float64{0, neg}), pick([]float64{0, neg, 1, -1}), sub)
}

// radix16Modes are the kernel families this CPU can run.
func radix16Modes(t *testing.T) []struct {
	name         string
	avx2, avx512 bool
} {
	if !kernels.UseStockhamAVX2 {
		t.Skip("no AVX2 on this CPU: the Go passes run and there is nothing to compare")
	}
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
	return modes
}

// TestRadix16EachPassMatchesScalar compares every radix-16 pass a kernel runs
// with the Go pass on its own, output against output, bit for bit: through a
// whole transform a zero's sign that differs inside one pass is mostly
// absorbed by later sums (Round 18). It covers ido from 1 (the final pass) to
// 33, so every remainder of ido mod 4 (the 256- and 128-bit tails) with and
// without four-point groups, and l1 from 1 to 9.
func TestRadix16EachPassMatchesScalar(t *testing.T) {
	defer func(a, b bool) { kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = a, b }(kernels.UseStockhamAVX2, kernels.UseStockhamAVX512)
	modes := radix16Modes(t)
	for _, l1 := range []int{1, 2, 3, 4, 5, 9} {
		for _, ido := range []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 33} {
			var f []int
			stage := 0
			if l1 > 1 {
				f, stage = append(f, l1), 1
			}
			f = append(f, 16)
			if ido > 1 {
				f = append(f, ido)
			}
			n := 16 * ido * l1
			// The stage under test is built as part of a power-of-two plan
			// when n is one, so wide (AVX-512) is allowed where it would be.
			st := newSKPlanFactors(n, f).stages[stage]
			if st.r != 16 || st.l1 != l1 || st.ido != ido {
				t.Fatalf("stage %+v, want r=16 l1=%d ido=%d", st, l1, ido)
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

// TestRadix16TransformMatchesScalar runs whole transforms with radix-16
// passes, kernels on and off, in place and not.
func TestRadix16TransformMatchesScalar(t *testing.T) {
	defer func(a, b bool) { kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = a, b }(kernels.UseStockhamAVX2, kernels.UseStockhamAVX512)
	modes := radix16Modes(t)
	for _, f := range radix16FactorLists {
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
