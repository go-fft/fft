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
	defer func(c, h, e bool) { comp2On, hswCompOn, hswE11On = c, h, e }(comp2On, hswCompOn, hswE11On)
	comp2On, hswCompOn, hswE11On = true, false, false
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

// TestZmCompTable: every entry multiplies back to its length, a composite
// 2^e·3^a·5^b of at most 16384 points, with radices that have AVX2 kernels;
// skFactorize takes it where comp2On says and skFactorizeOrder's without
// comp2On or Intel's rule; it differs from comp2Factors; and each
// transform agrees with the factorization it replaces.
func TestZmCompTable(t *testing.T) {
	defer func(c, h, r, e bool) { comp2On, hswCompOn, compRadix16On, hswE11On = c, h, r, e }(comp2On, hswCompOn, compRadix16On, hswE11On)
	compRadix16On, hswE11On = false, false // nor the Intel rules
	for n, f := range zmCompTable {
		e, a, b := 0, 0, 0
		m := n
		for ; m%2 == 0; m /= 2 {
			e++
		}
		for ; m%3 == 0; m /= 3 {
			a++
		}
		for ; m%5 == 0; m /= 5 {
			b++
		}
		if product(f) != n || m != 1 || e == 0 || a+b == 0 || n > 1<<14 {
			t.Fatalf("%d: %v", n, f)
		}
		for _, r := range f {
			if !slices.Contains([]int{2, 3, 4, 5, 8, 10, 12, 15, 16, 20}, r) {
				t.Fatalf("%d: radix %d has no AVX2 kernel", n, r)
			}
		}
		comp2On, hswCompOn = true, false
		if got := skFactorize(n); !slices.Equal(got, f) {
			t.Errorf("skFactorize(%d) = %v on AMD, want the table's %v", n, got, f)
		}
		if slices.Equal(f, comp2Factors(n, true)) {
			t.Errorf("%d: the table repeats comp2Factors' %v", n, f)
		}
		comp2On = false
		if got, want := skFactorize(n), skFactorizeOrder(n, compOddFirst); !slices.Equal(got, want) {
			t.Errorf("skFactorize(%d) = %v with comp2On and hswCompOn off, want skFactorizeOrder's %v", n, got, want)
		}
		comp2On = true
		x := cmplxSignal(n)
		got := make([]complex128, n)
		want := make([]complex128, n)
		newSKPlanFactors(n, f).transform(got, x, false)
		newSKPlanFactors(n, comp2Factors(n, true)).transform(want, x, false)
		var d, mx float64
		for i := range got {
			d = max(d, cmplx.Abs(got[i]-want[i]))
			mx = max(mx, cmplx.Abs(want[i]))
		}
		if d > 1e-13*mx {
			t.Errorf("%d: largest difference %.3g of %.3g", n, d, mx)
		}
	}
}
