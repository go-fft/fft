package fft

import (
	"math"
	"math/cmplx"
	"testing"
)

// cs9FactorLists are factorizations with radix-9 passes first (l1 = 1), last
// (ido = 1), between, next to the other radices, and with ido values that
// leave every remainder mod 4 (the kernels' tails).
var cs9FactorLists = [][]int{
	{9}, {9, 9}, {3, 9}, {9, 3}, {9, 12, 12}, {12, 12, 9}, {16, 9, 9}, {9, 4},
	{5, 9, 2}, {9, 5}, {9, 7}, {2, 9}, {9, 2, 3}, {3, 9, 16}, {9, 12, 3}, {9, 8, 5},
}

// TestCs9AgainstNaive holds the radix-9 passes to the O(N²) DFT, forward and
// inverse, whichever of the Go pass or a kernel runs them.
func TestCs9AgainstNaive(t *testing.T) {
	for _, f := range cs9FactorLists {
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

// TestCs9Constants: the radix-9 constants are cos and sin of 2π·e/9 to
// within 4e-17 relative of math.Cos and math.Sin, which take the angle
// rounded to float64 and are themselves a few units in the last place off
// (e = 2: 2 ulp). The literals are the values to 50 digits.
func TestCs9Constants(t *testing.T) {
	// 8π/9 is taken as π − π/9: near π the rounded angle would cost sin its
	// relative accuracy.
	for _, c := range []struct {
		e            float64
		c, s         float64
		wantC, wantS float64
	}{
		{1, r9C1, r9S1, math.Cos(2 * math.Pi / 9), math.Sin(2 * math.Pi / 9)},
		{2, r9C2, r9S2, math.Cos(4 * math.Pi / 9), math.Sin(4 * math.Pi / 9)},
		{4, r9C4, r9S4, -math.Cos(math.Pi / 9), math.Sin(math.Pi / 9)},
	} {
		for _, p := range [][2]float64{{c.c, c.wantC}, {c.s, c.wantS}} {
			if d := math.Abs(p[0] - p[1]); d > 4e-16*math.Abs(p[1]) {
				t.Errorf("e=%v: %v, math gives %v", c.e, p[0], p[1])
			}
		}
	}
}
