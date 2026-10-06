package fft

import (
	"slices"
	"testing"
)

// TestCompRadix16For pins the composites that end with radix-16 passes
// (Round 24) and the lengths that keep skFactorizeOrder's factorization.
func TestCompRadix16For(t *testing.T) {
	for _, c := range []struct {
		n    int
		on   bool
		want []int
	}{
		{1296, false, nil},
		{1296, true, []int{3, 3, 3, 3, 16}},
		{2000, true, []int{5, 5, 5, 16}},
		{6000, true, []int{3, 5, 5, 5, 16}},
		{1920, true, []int{3, 5, 8, 16}},
		{3072, true, []int{3, 8, 8, 16}},
		{15360, true, []int{3, 5, 8, 8, 16}},
		{768, true, []int{3, 16, 16}},
		{6400, true, []int{5, 5, 16, 16}},
		{480, true, nil},    // e = 5
		{960, true, nil},    // e = 6
		{1536, true, nil},   // e = 9
		{1000, true, nil},   // e = 3
		{4096, true, nil},   // a power of two: radix16Table's
		{1008, true, nil},   // a factor 7
		{24576, true, nil},  // above 16384
		{3 * 13, true, nil}, // a factor 13, no 2
	} {
		if got := compRadix16For(c.n, c.on); !slices.Equal(got, c.want) {
			t.Errorf("compRadix16For(%d, %v) = %v, want %v", c.n, c.on, got, c.want)
		}
	}
	// Every factorization it gives multiplies back to n.
	for n := 2; n <= 1<<14; n++ {
		f := compRadix16For(n, true)
		if f == nil {
			continue
		}
		p := 1
		for _, r := range f {
			p *= r
		}
		if p != n {
			t.Fatalf("compRadix16For(%d) = %v, product %d", n, f, p)
		}
	}
}

// TestCompFactorize takes skFactorize through both routes of compRadix16On.
func TestCompFactorize(t *testing.T) {
	defer func(v bool) { compRadix16On = v }(compRadix16On)
	compRadix16On = true
	if got := skFactorize(1296); !slices.Equal(got, []int{3, 3, 3, 3, 16}) {
		t.Errorf("skFactorize(1296) with radix-16 tails = %v", got)
	}
	compRadix16On = false
	if got, want := skFactorize(1296), skFactorizeOrder(1296, compOddFirst); !slices.Equal(got, want) {
		t.Errorf("skFactorize(1296) without = %v, want %v", got, want)
	}
}
