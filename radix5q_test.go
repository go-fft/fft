package fft

import (
	"math/cmplx"
	"testing"
)

// comp2FactorLists are factorizations with radix-10, -15 and -20 passes
// first (l1 = 1), last (ido = 1), between, next to the other radices, and
// with ido values that leave every remainder mod 4 (the kernels' tails).
var comp2FactorLists = [][]int{
	{10}, {15}, {20}, {10, 10}, {20, 20}, {15, 15}, {10, 10, 10}, {5, 20, 10},
	{3, 3, 15, 8}, {3, 3, 10, 12}, {20, 3}, {10, 7}, {15, 2}, {2, 15}, {20, 4, 3},
	{4, 20}, {8, 15}, {3, 20, 16}, {5, 5, 5, 8}, {20, 5, 10}, {15, 8, 3, 3},
}

// TestComp2Maps checks the index maps: every input and every output index
// once, and n·k ≡ q·n1·k1 + 5·n2·k2 (mod r).
func TestComp2Maps(t *testing.T) {
	for _, r := range []int{10, 15, 20} {
		m := comp2Maps[r]
		q := m.q
		seenIn, seenOut := map[int]bool{}, map[int]bool{}
		for n2 := range q {
			for n1 := range 5 {
				seenIn[m.in[n2][n1]] = true
				for k1 := range 5 {
					for k2 := range q {
						n, k := m.in[n2][n1], m.out[k1][k2]
						if (n*k-q*n1*k1-5*n2*k2)%r != 0 {
							t.Fatalf("r=%d: n=%d k=%d", r, n, k)
						}
					}
				}
			}
		}
		for k1 := range 5 {
			for k2 := range q {
				seenOut[m.out[k1][k2]] = true
			}
		}
		if len(seenIn) != r || len(seenOut) != r {
			t.Fatalf("r=%d: %d inputs, %d outputs", r, len(seenIn), len(seenOut))
		}
	}
}

// TestComp2AgainstNaive holds the radix-10/15/20 passes to the O(N²) DFT,
// forward and inverse, whichever of the Go pass or a kernel runs them.
func TestComp2AgainstNaive(t *testing.T) {
	for _, f := range comp2FactorLists {
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
