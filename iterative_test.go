package fft

import (
	"math/cmplx"
	"testing"
)

// TestIterativeAgainstNaive validates the iterative cache-blocked engine
// differentially against the O(N²) DFT oracle at every small power-of-two
// length, exercising both the even (radix-4 only) and odd (leading radix-2)
// log2(n) schedules and the block boundary.
func TestIterativeAgainstNaive(t *testing.T) {
	for _, n := range srSizes {
		x := cmplxSignal(n)
		it := newITPlan(n)
		dst := make([]complex128, n)
		it.transform(dst, x, false)
		closeVec(t, dst, naiveDFT(x))
	}
}

// TestIterativeVsSplitRadix cross-checks the iterative engine against the
// independent split-radix engine on the mid-range power-of-two sizes (where the
// naive oracle is too slow), for both directions.
func TestIterativeVsSplitRadix(t *testing.T) {
	sizes := append(append([]int{}, srSizes...), srLargeSizes...)
	sizes = append(sizes, 2048, 16384, 32768)
	for _, n := range sizes {
		x := cmplxSignal(n)
		it := newITPlan(n)
		sr := newSRPlan(n)
		for _, inverse := range []bool{false, true} {
			a := make([]complex128, n)
			b := make([]complex128, n)
			it.transform(a, x, inverse)
			sr.transform(b, x, inverse)
			tolN := 1e-12 * float64(n)
			for i := range a {
				if d := cmplx.Abs(a[i] - b[i]); d > tolN {
					t.Fatalf("n=%d inverse=%v index %d: iterative %v vs split-radix %v (|diff|=%g)",
						n, inverse, i, a[i], b[i], d)
				}
			}
		}
	}
}

// TestIterativeRoundTrip checks IFFT(FFT(x)) ≈ x through the iterative engine,
// including the transformScratch entry point.
func TestIterativeRoundTrip(t *testing.T) {
	sizes := append(append([]int{}, srSizes...), srLargeSizes...)
	sizes = append(sizes, 2048)
	for _, n := range sizes {
		x := cmplxSignal(n)
		it := newITPlan(n)
		fwd := make([]complex128, n)
		back := make([]complex128, n)
		scr := make([]complex128, n)
		copy(scr, x)
		it.transformScratch(fwd, scr, false)
		it.transform(back, fwd, true)
		inv := complex(1/float64(n), 0)
		for i := range back {
			back[i] *= inv
		}
		tolN := 1e-12 * float64(n)
		for i := range back {
			if d := cmplx.Abs(back[i] - x[i]); d > tolN {
				t.Fatalf("n=%d index %d: round-trip %v vs %v (|diff|=%g)", n, i, back[i], x[i], d)
			}
		}
	}
}
