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
	defer func(v, w bool) { compRadix16On, comp2On = v, w }(compRadix16On, comp2On)
	compRadix16On, comp2On = true, false
	for n, want := range map[int][]int{1296: {3, 3, 12, 12}, 2000: {5, 5, 5, 16}, 1000: skFactorizeOrder(1000, compOddFirst)} {
		if got := skFactorize(n); !slices.Equal(got, want) {
			t.Errorf("skFactorize(%d) with the composite routes = %v, want %v", n, got, want)
		}
	}
	compRadix16On = false
	if got, want := skFactorize(1296), skFactorizeOrder(1296, compOddFirst); !slices.Equal(got, want) {
		t.Errorf("skFactorize(1296) without = %v, want %v", got, want)
	}
	comp2On = true
	for n, want := range map[int][]int{1296: {3, 3, 12, 12}, 2000: {5, 20, 20}, 1000: {5, 10, 20}, 1008: skFactorizeOrder(1008, compOddFirst)} {
		if got := skFactorize(n); !slices.Equal(got, want) {
			t.Errorf("skFactorize(%d) with comp2On = %v, want %v", n, got, want)
		}
	}
}

// TestComp2Factors pins the AMD rule (Round 26) on the lengths the parity
// report and the A/B runs use, its limits, and that every factorization it
// gives multiplies back to n with a kernel radix in every pass.
func TestComp2Factors(t *testing.T) {
	for _, c := range []struct {
		n    int
		on   bool
		want []int
	}{
		{1000, false, nil},
		{1000, true, []int{5, 10, 20}},
		{1080, true, []int{3, 3, 10, 12}},
		{1296, true, []int{3, 3, 12, 12}},
		{1920, true, []int{8, 12, 20}},
		{2000, true, []int{5, 20, 20}},
		{6000, true, []int{5, 5, 12, 20}},
		{500, true, []int{5, 5, 20}},
		{540, true, []int{3, 15, 12}},
		{960, true, []int{4, 12, 20}},
		{120, true, []int{10, 12}},
		{16000, true, []int{5, 8, 20, 20}}, // a lone 2 left: back with a 20
		{72, true, []int{3, 3, 8}},         // e = 3 and no 5: the 12 gives its 4 back
		{1536, true, []int{4, 4, 8, 12}},   // e = 9: radix 12, no radix 20
		{2560, true, []int{5, 8, 8, 8}},    // e = 9, no 3
		{3072, true, []int{3, 8, 8, 16}},   // e = 10
		{6144, true, nil},                  // e = 11
		{15, true, nil},                    // odd
		{1008, true, nil},                  // a factor 7
		{24576, true, nil},                 // above 16384
	} {
		if got := comp2Factors(c.n, c.on); !slices.Equal(got, c.want) {
			t.Errorf("comp2Factors(%d, %v) = %v, want %v", c.n, c.on, got, c.want)
		}
	}
	for n := 2; n <= 1<<14; n++ {
		f := comp2Factors(n, true)
		if f == nil {
			continue
		}
		p := 1
		for _, r := range f {
			p *= r
			switch r {
			case 2, 3, 4, 5, 8, 10, 12, 15, 16, 20:
			default:
				t.Fatalf("comp2Factors(%d) = %v: radix %d", n, f, r)
			}
		}
		if p != n {
			t.Fatalf("comp2Factors(%d) = %v, product %d", n, f, p)
		}
	}
}

// TestCompRadix12For pins the composites that take radix-12 passes (Round 24)
// and the lengths that keep their other factorization.
func TestCompRadix12For(t *testing.T) {
	for _, c := range []struct {
		n    int
		on   bool
		want []int
	}{
		{1296, false, nil},
		{1296, true, []int{3, 3, 12, 12}},
		{12, true, []int{12}},
		{720, true, []int{5, 12, 12}},
		{1920, true, []int{5, 4, 8, 12}},
		{6000, true, []int{5, 5, 5, 4, 12}},
		{288, true, []int{2, 12, 12}},
		{13824, true, []int{8, 12, 12, 12}},
		{1080, true, nil},  // e = 3, t = 1
		{1536, true, nil},  // e = 9, t = 1
		{9216, true, nil},  // e = 10
		{2000, true, nil},  // no 3
		{6, true, nil},     // e = 1
		{1008, true, nil},  // a factor 7
		{20736, true, nil}, // above 16384
	} {
		if got := compRadix12For(c.n, c.on); !slices.Equal(got, c.want) {
			t.Errorf("compRadix12For(%d, %v) = %v, want %v", c.n, c.on, got, c.want)
		}
	}
	for n := 2; n <= 1<<14; n++ {
		f := compRadix12For(n, true)
		p := 1
		for _, r := range f {
			p *= r
		}
		if f != nil && p != n {
			t.Fatalf("compRadix12For(%d) = %v, product %d", n, f, p)
		}
	}
}
