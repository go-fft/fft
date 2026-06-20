package fft

import (
	"math/cmplx"
	"testing"
)

// srSizes are the power-of-two lengths the split-radix engine handles. Small
// ones are validated against the O(N²) naive oracle directly; the larger
// mid-range sizes (where the naive oracle would be too slow) are cross-checked
// against the independent mixed-radix engine and by round-trip.
var srSizes = []int{2, 4, 8, 16, 32, 64, 128, 256, 512}

// srLargeSizes are the mid-range power-of-two lengths that motivated this engine
// (the documented FFTW gap). They are validated against the mixed-radix engine
// (itself naive-validated at small N) and by round-trip, avoiding a multi-second
// O(N²) oracle at N up to 65536.
var srLargeSizes = []int{1024, 2048, 4096, 8192, 65536}

// TestSplitRadixAgainstNaive validates the split-radix engine differentially
// against the O(N²) DFT oracle at every small power-of-two length, exercising
// the length-2 and radix-4 leaves and the L-shaped recombination butterfly.
func TestSplitRadixAgainstNaive(t *testing.T) {
	for _, n := range srSizes {
		x := cmplxSignal(n)
		sr := &srPlan{n: n, tw: twiddleTable(n)}
		sr.twConj = make([]complex128, n)
		for i, w := range sr.tw {
			sr.twConj[i] = complex(real(w), -imag(w))
		}
		dst := make([]complex128, n)
		sr.transform(dst, x, false)
		closeVec(t, dst, naiveDFT(x))
	}
}

// TestSplitRadixVsMixedRadix cross-checks the split-radix engine against the
// independent mixed-radix engine on the mid-range power-of-two sizes, where the
// naive oracle is too slow. Building a ctPlan directly bypasses the pow2 router
// (which now sends these lengths to split-radix).
func TestSplitRadixVsMixedRadix(t *testing.T) {
	for _, n := range srLargeSizes {
		x := cmplxSignal(n)
		sr := &Plan{n: n, sr: newSRPlan(n)}
		ct := &Plan{n: n, ct: newCTPlan(n)}
		a := make([]complex128, n)
		b := make([]complex128, n)
		sr.FFT(a, x)
		ct.FFT(b, x)
		// Two correct FFTs of the same data differ only by rounding; allow an
		// N-scaled tolerance as the Rader/Bluestein tests do.
		tolN := 1e-12 * float64(n)
		for i := range a {
			if d := cmplx.Abs(a[i] - b[i]); d > tolN {
				t.Fatalf("n=%d index %d: split-radix %v vs mixed-radix %v (|diff|=%g)", n, i, a[i], b[i], d)
			}
		}
	}
}

// TestSplitRadixRoundTrip checks IFFT(FFT(x)) ≈ x through the split-radix engine
// across the full and mid-range size sets.
func TestSplitRadixRoundTrip(t *testing.T) {
	for _, n := range append(append([]int{}, srSizes...), srLargeSizes...) {
		x := cmplxSignal(n)
		p := NewPlan(n)
		fwd := make([]complex128, n)
		back := make([]complex128, n)
		p.FFT(fwd, x)
		p.IFFT(back, fwd)
		tolN := 1e-12 * float64(n)
		for i := range back {
			if d := cmplx.Abs(back[i] - x[i]); d > tolN {
				t.Fatalf("n=%d index %d: round-trip %v vs %v (|diff|=%g)", n, i, back[i], x[i], d)
			}
		}
	}
}
