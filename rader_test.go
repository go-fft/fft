package fft

import (
	"math/cmplx"
	"testing"
)

// smallRaderPrimes are primes for which the O(N²) naive oracle is cheap, used to
// validate the Rader engine itself (constructed directly, bypassing the
// router). They exercise small/medium primitive-root orders.
var smallRaderPrimes = []int{17, 23, 31, 97, 101, 257, 521, 1009}

// largeRaderPrimes are larger primes whose N-1 the mixed-radix engine handles
// directly (all 7-smooth, the ones the router sends to Rader, except 6007 and
// 8009 whose N-1 carries 11 and 13). They are validated by round-trip and by
// agreement with the (already-naive-validated) Bluestein engine, avoiding a
// multi-second O(N²) oracle at N≈10⁴.
var largeRaderPrimes = []int{769, 1009, 2017, 3001, 6007, 7681, 8009, 10753, 12289}

// scaledTol grows the differential tolerance with N: an O(N) reduction over
// magnitudes ~N accumulates rounding a fixed 1e-9 floor would wrongly flag. The
// bound tracks the same growth Bluestein exhibits at equal N.
func scaledTol(n int) float64 { return 1e-13 * float64(n) * float64(n) }

func maxDiff(a, b []complex128) float64 {
	var m float64
	for i := range a {
		if d := cmplx.Abs(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}

// TestRaderEngineAgainstNaive validates the Rader engine directly (built for a
// prime regardless of the router threshold) against the O(N²) DFT oracle, both
// directions.
func TestRaderEngineAgainstNaive(t *testing.T) {
	for _, n := range smallRaderPrimes {
		if !isPrime(n) {
			t.Fatalf("%d is not prime", n)
		}
		rp := newRaderPlan(n)
		x := cmplxSignal(n)

		fwd := make([]complex128, n)
		rp.transform(fwd, x, false)
		if d := maxDiff(fwd, naiveDFT(x)); d > scaledTol(n) {
			t.Errorf("n=%d forward maxdiff=%g tol=%g", n, d, scaledTol(n))
		}

		// Inverse engine (unnormalized) vs conjugate-naive: IFFT*N.
		inv := make([]complex128, n)
		rp.transform(inv, x, true)
		ref := naiveIDFTUnnormalized(x)
		if d := maxDiff(inv, ref); d > scaledTol(n) {
			t.Errorf("n=%d inverse maxdiff=%g tol=%g", n, d, scaledTol(n))
		}
	}
}

// naiveIDFTUnnormalized is the conjugate-root DFT without the 1/N factor, the
// oracle for the Rader engine's inverse transform (the plan applies 1/N).
func naiveIDFTUnnormalized(x []complex128) []complex128 {
	n := len(x)
	out := make([]complex128, n)
	for k := 0; k < n; k++ {
		var s complex128
		for j := 0; j < n; j++ {
			ang := 2 * 3.141592653589793 * float64(k*j) / float64(n)
			s += x[j] * cmplx.Exp(complex(0, ang))
		}
		out[k] = s
	}
	return out
}

// TestRaderMatchesBluestein cross-checks the Rader engine against the Bluestein
// engine (independently naive-validated) for the larger primes — the same
// transform by two different reductions must agree.
func TestRaderMatchesBluestein(t *testing.T) {
	for _, n := range largeRaderPrimes {
		if !isPrime(n) {
			t.Fatalf("%d is not prime", n)
		}
		x := cmplxSignal(n)
		rp := newRaderPlan(n)
		bp := newBluesteinPlan(n)
		gotR := make([]complex128, n)
		gotB := make([]complex128, n)
		rp.transform(gotR, x, false)
		bp.transform(gotB, x, false)
		if d := maxDiff(gotR, gotB); d > scaledTol(n) {
			t.Errorf("n=%d Rader vs Bluestein maxdiff=%g tol=%g", n, d, scaledTol(n))
		}
	}
}

// TestRaderRoundTrip checks IFFT(FFT(x)) ≈ x through the router, which selects
// Rader exactly for the primes whose N-1 is 7-smooth (Bluestein for the rest).
func TestRaderRoundTrip(t *testing.T) {
	for _, n := range largeRaderPrimes {
		p := NewPlan(n)
		if _, smooth := convCost(n - 1); (p.rader != nil) != smooth {
			t.Fatalf("n=%d: routed to Rader = %v, want %v", n, p.rader != nil, smooth)
		}
		x := cmplxSignal(n)
		fwd := make([]complex128, n)
		p.FFT(fwd, x)
		back := make([]complex128, n)
		p.IFFT(back, fwd)
		if d := maxDiff(back, x); d > 1e-9 {
			t.Errorf("n=%d roundtrip maxdiff=%g", n, d)
		}
	}
}

// TestRaderAlias confirms dst may alias src in the Rader engine.
func TestRaderAlias(t *testing.T) {
	n := 257
	x := cmplxSignal(n)
	want := naiveDFT(x)
	newRaderPlan(n).transform(x, x, false) // in place
	if d := maxDiff(x, want); d > scaledTol(n) {
		t.Errorf("n=%d alias maxdiff=%g", n, d)
	}
}

// TestPrimitiveRoot checks the generator against known values and verifies the
// returned g actually generates Z/pZ*.
func TestPrimitiveRoot(t *testing.T) {
	known := map[int]int{2: 1, 3: 2, 5: 2, 7: 3, 11: 2, 13: 2, 17: 3, 23: 5}
	for p, want := range known {
		if g := primitiveRoot(p); g != want {
			t.Errorf("primitiveRoot(%d)=%d want %d", p, g, want)
		}
	}
	for _, p := range []int{17, 31, 101, 257} {
		g := primitiveRoot(p)
		seen := make([]bool, p)
		x := 1
		for k := 0; k < p-1; k++ {
			if seen[x] {
				t.Fatalf("p=%d g=%d not a generator (repeat at %d)", p, g, x)
			}
			seen[x] = true
			x = x * g % p
		}
	}
}

// TestIsPrime spot-checks the primality predicate over its branches.
func TestIsPrime(t *testing.T) {
	cases := map[int]bool{
		0: false, 1: false, 2: true, 3: true, 4: false, 9: false,
		17: true, 25: false, 10007: true, 10006: false, 9973: true,
	}
	for n, want := range cases {
		if got := isPrime(n); got != want {
			t.Errorf("isPrime(%d)=%v want %v", n, got, want)
		}
	}
}

// TestPrimeRouting confirms the Rader/Bluestein split: a prime goes to Rader
// exactly when N-1 is 7-smooth, and to Bluestein otherwise (N-1 with a factor
// 11 or 13, or not smooth at all).
func TestPrimeRouting(t *testing.T) {
	for _, n := range []int{17, 641, 769, 2017, 7681} { // 16, 640, 768, 2016, 7680
		if pl := NewPlan(n); pl.rader == nil || pl.bluestein != nil {
			t.Errorf("n=%d (n-1 7-smooth) should route to Rader", n)
		}
	}
	// 23: 22 = 2·11; 6007: 6006 = 2·3·7·11·13; 47: 46 = 2·23; 9973: 9972 =
	// 2²·3²·277; 10007: 10006 = 2·5003.
	for _, n := range []int{23, 47, 6007, 9973, 10007} {
		if pl := NewPlan(n); pl.bluestein == nil || pl.rader != nil {
			t.Errorf("n=%d (n-1 not 7-smooth) should route to Bluestein", n)
		}
	}
}
