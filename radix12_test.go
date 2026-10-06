package fft

import (
	"math/cmplx"
	"testing"
)

// comp12FactorLists are factorizations with radix-12 passes first (l1 = 1),
// last (ido = 1), between, next to the other radices, and with ido values
// that leave every remainder mod 4 (the kernels' tails).
var comp12FactorLists = [][]int{
	{12}, {12, 12}, {3, 12}, {12, 3}, {3, 3, 12, 12}, {12, 4}, {5, 12, 2},
	{12, 5}, {12, 7}, {2, 12}, {12, 2, 3}, {3, 12, 16}, {12, 12, 3},
}

// TestComp12AgainstNaive holds the radix-12 passes to the O(N²) DFT, forward
// and inverse, whichever of the Go pass or a kernel runs them.
func TestComp12AgainstNaive(t *testing.T) {
	for _, f := range comp12FactorLists {
		n := 1
		for _, r := range f {
			n *= r
		}
		p := newSKPlanFactors(n, f)
		x := cmplxSignal(n)
		got := make([]complex128, n)
		p.transform(got, x, false)
		closeVec(t, got, naiveDFT(x))
		cx := make([]complex128, n)
		for i, v := range x {
			cx[i] = cmplx.Conj(v)
		}
		want := naiveDFT(cx)
		for i := range want {
			want[i] = cmplx.Conj(want[i])
		}
		p.transform(got, x, true)
		closeVec(t, got, want)
	}
}
