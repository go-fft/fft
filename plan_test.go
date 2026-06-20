package fft

import (
	"math"
	"math/cmplx"
	"testing"
)

// mixedRadixSizes exercises every butterfly: radix 2/3/4/5 specializations, the
// general radix-7/11/13 path, deeply composite lengths, and the Bluestein
// fallback (a large prime factor, e.g. 17, 23, 49=7², 121=11², 169=13²). The
// prime 23 (> maxRadix) forces Bluestein; 289=17² also forces it.
var mixedRadixSizes = []int{
	2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 20, 21, 24, 25, 27,
	30, 32, 33, 35, 36, 45, 48, 49, 50, 60, 64, 72, 81, 100, 105, 121, 125,
	128, 169, 1000, 1296,
	// Bluestein-forced lengths (a prime factor above maxRadix):
	17, 19, 23, 29, 31, 34, 46, 289, 100, 101,
}

func cmplxSignal(n int) []complex128 {
	x := make([]complex128, n)
	for i := range x {
		x[i] = complex(math.Sin(float64(i)*0.7)+float64(i%3), math.Cos(float64(i)*0.3)-float64(i%5))
	}
	return x
}

// TestMixedRadixAgainstNaive validates every radix path and the Bluestein
// fallback differentially against the O(N²) DFT oracle.
func TestMixedRadixAgainstNaive(t *testing.T) {
	for _, n := range mixedRadixSizes {
		x := cmplxSignal(n)
		closeVec(t, FFT(x), naiveDFT(x))
	}
}

// TestMixedRadixRoundTrip checks IFFT(FFT(x)) ≈ x across every path.
func TestMixedRadixRoundTrip(t *testing.T) {
	for _, n := range mixedRadixSizes {
		x := cmplxSignal(n)
		closeVec(t, IFFT(FFT(x)), x)
	}
}

// TestPlanReuse checks that a reused Plan gives identical results to the
// one-shot FFT across calls, and that the cache returns a working plan.
func TestPlanReuse(t *testing.T) {
	for _, n := range []int{8, 9, 12, 1000, 1296, 23} {
		p := NewPlan(n)
		if p.Len() != n {
			t.Fatalf("Len()=%d want %d", p.Len(), n)
		}
		x := cmplxSignal(n)
		dst := make([]complex128, n)
		p.FFT(dst, x)
		closeVec(t, dst, naiveDFT(x))
		// Reuse for the inverse, expecting the round trip.
		back := make([]complex128, n)
		p.IFFT(back, dst)
		closeVec(t, back, x)
		// A second forward call must reproduce the first (no state leak).
		dst2 := make([]complex128, n)
		p.FFT(dst2, x)
		closeVec(t, dst2, dst)
	}
}

// TestPlanAlias confirms dst may alias src.
func TestPlanAlias(t *testing.T) {
	for _, n := range []int{8, 12, 1296, 23} {
		x := cmplxSignal(n)
		want := naiveDFT(x)
		p := NewPlan(n)
		p.FFT(x, x) // in place
		closeVec(t, x, want)
	}
}

// TestPlanTrivial covers the n==0 and n==1 plans.
func TestPlanTrivial(t *testing.T) {
	p0 := NewPlan(0)
	if got := p0.FFT([]complex128{}, []complex128{}); len(got) != 0 {
		t.Fatalf("n=0 FFT: got %v", got)
	}
	if got := p0.IFFT([]complex128{}, []complex128{}); len(got) != 0 {
		t.Fatalf("n=0 IFFT: got %v", got)
	}
	one := []complex128{complex(7, -3)}
	dst := make([]complex128, 1)
	p1 := NewPlan(1)
	if got := p1.FFT(dst, one); got[0] != one[0] {
		t.Fatalf("n=1 FFT: got %v", got)
	}
	if got := p1.IFFT(dst, one); got[0] != one[0] {
		t.Fatalf("n=1 IFFT: got %v", got)
	}
}

// TestFactorize checks the factorization helper preferences (4 over 2·2) and
// the small-factor predicate boundary.
func TestFactorize(t *testing.T) {
	cases := map[int][]int{
		16:   {4, 4},
		8:    {4, 2},
		12:   {4, 3},
		1296: {4, 4, 3, 3, 3, 3},
		1000: {4, 2, 5, 5, 5},
		13:   {13},
	}
	for n, want := range cases {
		got := factorize(n)
		if len(got) != len(want) {
			t.Fatalf("factorize(%d)=%v want %v", n, got, want)
		}
		prod := 1
		for i, f := range got {
			if f != want[i] {
				t.Fatalf("factorize(%d)=%v want %v", n, got, want)
			}
			prod *= f
		}
		if prod != n {
			t.Fatalf("factorize(%d) product=%d", n, prod)
		}
	}
	// factorsAreSmall boundary: 13 is the largest direct radix; 17 is not.
	if !factorsAreSmall(13 * 13) {
		t.Fatal("169 should be all-small")
	}
	if factorsAreSmall(17) {
		t.Fatal("17 should not be all-small")
	}
	if factorsAreSmall(2 * 17) {
		t.Fatal("34 should not be all-small")
	}
}

// TestImpulseAllPaths checks a unit impulse → flat spectrum on radix and
// Bluestein lengths alike.
func TestImpulseAllPaths(t *testing.T) {
	for _, n := range []int{6, 9, 15, 25, 23, 49} {
		x := make([]complex128, n)
		x[0] = 1
		X := FFT(x)
		for k := range X {
			if cmplx.Abs(X[k]-1) > tol {
				t.Fatalf("n=%d bin %d: %v", n, k, X[k])
			}
		}
	}
}
