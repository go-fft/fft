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
// naive oracle is too slow. Both engines are built directly: the pow2 router in
// NewPlan now sends these lengths to the iterative kernel, so split-radix and
// mixed-radix are exercised here as independent reference implementations.
func TestSplitRadixVsMixedRadix(t *testing.T) {
	for _, n := range srLargeSizes {
		x := cmplxSignal(n)
		sr := newSRPlan(n)
		ct := &Plan{n: n, ct: newCTPlan(n)}
		a := make([]complex128, n)
		b := make([]complex128, n)
		sr.transform(a, x, false)
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
// directly across the full and mid-range size sets. It builds the srPlan itself
// (NewPlan now routes power-of-two lengths to the iterative kernel) so the
// split-radix reference engine — including its conjugate-roots inverse path and
// the transformScratch entry point — stays exercised, and it normalizes the
// inverse by N as the public IFFT does.
func TestSplitRadixRoundTrip(t *testing.T) {
	for _, n := range append(append([]int{}, srSizes...), srLargeSizes...) {
		x := cmplxSignal(n)
		sr := newSRPlan(n)
		fwd := make([]complex128, n)
		back := make([]complex128, n)
		// Forward via the scratch entry point (caller-owned read-only buffer).
		scr := make([]complex128, n)
		copy(scr, x)
		sr.transformScratch(fwd, scr, false)
		// Inverse uses the conjugate roots; normalize by N.
		sr.transform(back, fwd, true)
		inv := complex(1/float64(n), 0)
		tolN := 1e-12 * float64(n)
		for i := range back {
			back[i] *= inv
			if d := cmplx.Abs(back[i] - x[i]); d > tolN {
				t.Fatalf("n=%d index %d: round-trip %v vs %v (|diff|=%g)", n, i, back[i], x[i], d)
			}
		}
	}
}
