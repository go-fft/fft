package fft

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// r42R32Factors are factorizations with a radix-32 pass: first, last,
// between, beside every radix with a kernel, with ido from 1 to 64 and l1
// from 1 to 32, so that the split kernel runs with one and several groups
// and blocks, both widths' group sizes, and the final kernel with odd and
// even l1.
var r42R32Factors = [][]int{
	{32}, {32, 32}, {32, 8}, {8, 32}, {32, 4}, {4, 32}, {32, 2}, {2, 32}, {32, 16}, {16, 32},
	{4, 32, 32}, {32, 4, 32}, {32, 32, 4}, {8, 8, 32}, {2, 32, 32}, {32, 32, 2}, {4, 32, 8},
	{3, 32}, {32, 3}, {5, 32}, {32, 5}, {3, 32, 3}, {32, 9}, {9, 32}, {32, 12}, {32, 32, 3},
	{8, 4, 32}, {32, 8, 4}, {4, 8, 32}, {16, 32, 4},
}

// TestR42R32MatchesReference: every radix-32 FMA plan, under every layout the
// machine runs, gives the bits of the same plan run on the Go passes
// (passScalarFMA with bfly32FMA), in place and not, both directions, on the
// generic, signed-zero and infinite signals.
func TestR42R32MatchesReference(t *testing.T) {
	fmaKernelsHere(t)
	defer func(a bool) { kernels.UseStockhamAVX2 = a }(kernels.UseStockhamAVX2)
	for _, l := range r41Layouts() {
		withLayout(l.avx512, l.split, l.sp512, func() {
			split32 := 0
			for _, f := range r42R32Factors {
				n := factorProduct(f)
				p := newSKPlanFactorsOpt(n, f, true)
				for _, st := range p.stages {
					if st.r == 32 && st.split != 0 {
						split32++
					}
				}
				for s, x := range simdSignals(n) {
					for _, inverse := range []bool{false, true} {
						kernels.UseStockhamAVX2 = false
						ref := make([]complex128, n)
						p.transform(ref, x, inverse)
						kernels.UseStockhamAVX2 = true
						got := make([]complex128, n)
						p.transform(got, x, inverse)
						alias := slices.Clone(x)
						p.transform(alias, alias, inverse)
						for i := range ref {
							if !sameBits(got[i], ref[i]) || !sameBits(alias[i], ref[i]) {
								t.Fatalf("%s factors %v signal %d inverse=%v index %d: %v (in place %v), reference %v",
									l.name, f, s, inverse, i, got[i], alias[i], ref[i])
							}
						}
					}
				}
			}
			if l.split || l.sp512 {
				if split32 == 0 {
					t.Fatalf("%s: no split radix-32 pass was built", l.name)
				}
			}
		})
	}
}

// TestR42R32AgainstNaive holds the radix-32 FMA plans (kernels on) and
// bfly32FMA to the O(n²) DFT.
func TestR42R32AgainstNaive(t *testing.T) {
	for _, f := range r42R32Factors {
		n := factorProduct(f)
		if n > 4096 {
			continue
		}
		p := newSKPlanFactorsOpt(n, f, true)
		x := cmplxSignal(n)
		want := r42DFT(x)
		got := make([]complex128, n)
		p.transform(got, x, false)
		var num, den float64
		for i := range got {
			d := cmplx.Abs(got[i] - want[i])
			num += d * d
			den += cmplx.Abs(want[i]) * cmplx.Abs(want[i])
		}
		if num > 1e-26*den {
			t.Errorf("factors %v: relative error² %.3g", f, num/den)
		}
	}
}

// r42DFT is the O(n²) forward DFT with its angles reduced exactly (k·j mod
// n), which naiveDFT's are not.
func r42DFT(x []complex128) []complex128 {
	n := len(x)
	w := make([]complex128, n)
	for k := range w {
		w[k] = cmplx.Rect(1, -2*math.Pi*float64(k)/float64(n))
	}
	out := make([]complex128, n)
	for k := range out {
		var sum complex128
		for j := range x {
			sum += x[j] * w[k*j%n]
		}
		out[k] = sum
	}
	return out
}
