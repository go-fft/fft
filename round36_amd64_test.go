package fft

import (
	"math/cmplx"
	"slices"
	"testing"
)

// TestZmPow2Tail pins zmPow2Tail's factorizations and the lengths it leaves
// alone, that skFactorize takes it where comp2On says, that every factorization it gives multiplies back to its
// length, and that the transform those passes run agrees with the
// factorization it replaces.
func TestZmPow2Tail(t *testing.T) {
	for _, c := range []struct {
		n    int
		on   bool
		want []int
	}{
		{6144, true, []int{3, 4, 4, 4, 8, 4}},
		{10240, true, []int{5, 4, 4, 4, 8, 4}},
		{18432, true, []int{3, 3, 4, 4, 4, 8, 4}},
		{30720, true, []int{3, 5, 4, 4, 4, 8, 4}},
		{12288, true, []int{3, 4, 4, 4, 4, 4, 4}},
		{20480, true, []int{5, 4, 4, 4, 4, 4, 4}},
		{6144, false, nil},
		{2048, true, nil},  // a power of two
		{4096, true, nil},  // a power of two
		{3072, true, nil},  // e = 10
		{24576, true, nil}, // e = 13
		{36864, true, nil}, // above 32768
		{14336, true, nil}, // a 7
		{22528, true, nil}, // an 11
	} {
		if got := zmPow2Tail(c.n, c.on); !slices.Equal(got, c.want) {
			t.Errorf("zmPow2Tail(%d, %v) = %v, want %v", c.n, c.on, got, c.want)
		}
	}
	// skFactorize takes it on AMD: comp2On, and Intel's rule off.
	defer func(c, h bool) { comp2On, hswCompOn = c, h }(comp2On, hswCompOn)
	comp2On, hswCompOn = true, false
	for _, n := range []int{6144, 12288, 30720} {
		if got, want := skFactorize(n), zmPow2Tail(n, true); !slices.Equal(got, want) {
			t.Errorf("skFactorize(%d) = %v on AMD, want %v", n, got, want)
		}
	}
	for n := 2; n <= 1<<16; n++ {
		if f := zmPow2Tail(n, true); f != nil && product(f) != n {
			t.Fatalf("%d: %v multiplies to %d", n, f, product(f))
		}
	}
	for _, n := range []int{6144, 10240, 12288, 18432, 20480, 30720} {
		x := cmplxSignal(n)
		got := make([]complex128, n)
		want := make([]complex128, n)
		newSKPlanFactors(n, zmPow2Tail(n, true)).transform(got, x, false)
		newSKPlanFactors(n, skFactorizeOrder(n, true)).transform(want, x, false)
		var e, m float64
		for i := range got {
			e = max(e, cmplx.Abs(got[i]-want[i]))
			m = max(m, cmplx.Abs(want[i]))
		}
		if e > 1e-13*m {
			t.Errorf("%d: largest difference %.3g of %.3g", n, e, m)
		}
	}
}
