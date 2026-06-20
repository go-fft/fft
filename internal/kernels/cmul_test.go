package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// referenceCMul is an independent, obviously-correct scalar oracle written
// directly from the textbook complex-product formula, used to validate BOTH
// CMulScalar and the dispatched (possibly SIMD) CMul.
func referenceCMul(a, b []complex128) []complex128 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	out := make([]complex128, n)
	for i := 0; i < n; i++ {
		ar, ai := real(a[i]), imag(a[i])
		br, bi := real(b[i]), imag(b[i])
		out[i] = complex(ar*br-ai*bi, ar*bi+ai*br)
	}
	return out
}

// bitEqual reports whether two complex128 values are identical bit-for-bit on
// both the real and imaginary parts (so we can require the SIMD kernel to match
// the scalar oracle exactly, not merely within a tolerance).
func bitEqual(x, y complex128) bool {
	return math.Float64bits(real(x)) == math.Float64bits(real(y)) &&
		math.Float64bits(imag(x)) == math.Float64bits(imag(y))
}

func structuredInputs() (a, b []complex128) {
	a = []complex128{
		0, 1, 1i, 1 + 1i, -1 - 1i,
		complex(2, 3), complex(-4, 5), complex(0.5, -0.25),
		complex(1e10, -1e-10), complex(-3.5, 7.25),
		complex(math.Pi, math.E), complex(-math.Sqrt2, 1.0/3.0),
	}
	b = []complex128{
		1, 1i, 1, complex(2, -3), complex(0.5, 0.5),
		complex(-1, 1), complex(3, 3), complex(-8, 16),
		complex(2, 2), complex(1, -1),
		complex(-math.E, math.Pi), complex(1.0/7.0, -9.0),
	}
	return a, b
}

// TestCMulScalarMatchesReference pins the scalar oracle to the textbook formula.
func TestCMulScalarMatchesReference(t *testing.T) {
	a, b := structuredInputs()
	want := referenceCMul(a, b)
	got := append([]complex128(nil), a...)
	CMulScalar(got, b)
	for i := range want {
		if !bitEqual(got[i], want[i]) {
			t.Fatalf("CMulScalar index %d: got %v want %v", i, got[i], want[i])
		}
	}
}

// TestCMulMatchesScalar is the core SIMD-vs-scalar assertion: the dispatched
// CMul (the SIMD kernel where one is selected for this CPU, else the scalar
// fallback) must equal the scalar oracle bit-for-bit on structured and random
// inputs, across every length so the loop count, including the empty and odd
// tails, is exercised.
func TestCMulMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0xF17))
	randCx := func() complex128 {
		return complex(rng.NormFloat64()*1e3, rng.NormFloat64()*1e-2)
	}

	// Structured inputs first.
	a, b := structuredInputs()
	checkCMulEqualsScalar(t, a, b)

	// Random inputs over a sweep of lengths (0..257) to exercise the loop tail.
	for _, n := range []int{0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 33, 64, 100, 257} {
		ra := make([]complex128, n)
		rb := make([]complex128, n)
		for i := 0; i < n; i++ {
			ra[i] = randCx()
			rb[i] = randCx()
		}
		checkCMulEqualsScalar(t, ra, rb)
	}
}

// TestCMulMismatchedLengths confirms CMul honours the min(len(a), len(b))
// contract (only the overlapping prefix is written) for both orderings.
func TestCMulMismatchedLengths(t *testing.T) {
	a := []complex128{complex(1, 2), complex(3, 4), complex(5, 6)}
	b := []complex128{complex(7, 8)}
	checkCMulEqualsScalar(t, a, b)
	checkCMulEqualsScalar(t, b, a)
}

// checkCMulEqualsScalar asserts that both the dispatched CMul and the per-arch
// cmulSIMD are bit-for-bit identical to the scalar oracle on the given input.
// On amd64 cmulSIMD is the generated SSE2 kernel; on the other targets it
// aliases the scalar path (so the assertion trivially holds there). This is the
// per-arch SIMD==scalar proof the CI execution jobs rely on.
func checkCMulEqualsScalar(t *testing.T, a, b []complex128) {
	t.Helper()
	want := append([]complex128(nil), a...)
	CMulScalar(want, b)

	gotDispatch := append([]complex128(nil), a...)
	CMul(gotDispatch, b)
	assertBitEqual(t, "CMul", a, b, gotDispatch, want)

	gotSIMD := append([]complex128(nil), a...)
	cmulSIMD(gotSIMD, b)
	assertBitEqual(t, "cmulSIMD", a, b, gotSIMD, want)
}

func assertBitEqual(t *testing.T, who string, a, b, got, want []complex128) {
	t.Helper()
	for i := range got {
		if !bitEqual(got[i], want[i]) {
			t.Fatalf("%s: len(a)=%d len(b)=%d index %d: got=%v scalar=%v",
				who, len(a), len(b), i, got[i], want[i])
		}
	}
}
