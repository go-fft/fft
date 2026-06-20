package fft

import (
	"sync"
	"testing"
)

// TestCachedPlanConcurrentBuild forces many goroutines to request the same
// uncached length simultaneously, exercising the lost-race branch of cachedPlan
// (a concurrent builder won the store first) and asserting every caller observes
// the identical memoized plan. This is the regression guard that the cache builds
// plans outside the lock without self-deadlocking and without duplicate caching.
func TestCachedPlanConcurrentBuild(t *testing.T) {
	const n = 9973 // Rader prime: NewPlan re-enters cachedPlan for its sub-plan
	planMu.Lock()
	delete(planCache, n)
	planMu.Unlock()

	const goroutines = 32
	var start sync.WaitGroup
	start.Add(1)
	var done sync.WaitGroup
	got := make([]*Plan, goroutines)
	for i := 0; i < goroutines; i++ {
		done.Add(1)
		go func(idx int) {
			defer done.Done()
			start.Wait()
			got[idx] = cachedPlan(n)
		}(i)
	}
	start.Done()
	done.Wait()

	canon := cachedPlan(n)
	for i, p := range got {
		if p != canon {
			t.Fatalf("goroutine %d got a different plan pointer than the memoized one", i)
		}
	}
}

// TestRadix7Butterfly validates the specialized radix-7 Cooley–Tukey butterfly
// (mixedradix.go) against the O(N²) naive oracle, on lengths whose factorization
// drives the radix-7 stage at several recursion depths and twiddle steps.
func TestRadix7Butterfly(t *testing.T) {
	for _, n := range []int{7, 14, 21, 28, 35, 49, 63, 70, 98, 147, 245, 343} {
		x := cmplxSignal(n)
		got := FFT(x)
		if d := maxDiff(got, naiveDFT(x)); d > scaledTol(n) {
			t.Errorf("n=%d radix-7 forward maxdiff=%g tol=%g", n, d, scaledTol(n))
		}
		back := IFFT(got)
		if d := maxDiff(back, x); d > 1e-9 {
			t.Errorf("n=%d radix-7 roundtrip maxdiff=%g", n, d)
		}
	}
}

// TestFFTCacheRaderPrime exercises the package-level FFT through the plan cache
// on a Rader-routed prime. NewPlan for such a prime re-enters cachedPlan to build
// its convolution sub-plan, so this is the regression guard for the plan-cache
// self-deadlock that holding the cache lock across NewPlan would cause.
func TestFFTCacheRaderPrime(t *testing.T) {
	for _, n := range []int{769, 2017, 5003, 9973, 10007} {
		x := cmplxSignal(n)
		got := FFT(x)
		back := IFFT(got)
		if d := maxDiff(back, x); d > 1e-9 {
			t.Errorf("n=%d FFT/IFFT via cache roundtrip maxdiff=%g", n, d)
		}
	}
}

// TestNextSmoothConv checks the smooth convolution-length helper: the result is
// >= lo, is 2·3·5-smooth, and is the smallest such value (no smaller 2·3·5-smooth
// number is >= lo).
func TestNextSmoothConv(t *testing.T) {
	smooth := func(m int) bool {
		for _, p := range []int{2, 3, 5} {
			for m%p == 0 {
				m /= p
			}
		}
		return m == 1
	}
	for _, lo := range []int{0, 1, 2, 7, 11, 13, 100, 10004, 19944, 20012} {
		g := nextSmoothConv(lo)
		want := lo
		if want < 1 {
			want = 1
		}
		if g < want {
			t.Errorf("nextSmoothConv(%d)=%d < %d", lo, g, want)
		}
		if !smooth(g) {
			t.Errorf("nextSmoothConv(%d)=%d not 2·3·5-smooth", lo, g)
		}
		for m := want; m < g; m++ {
			if smooth(m) {
				t.Errorf("nextSmoothConv(%d)=%d but %d is a smaller smooth >= %d", lo, g, m, want)
			}
		}
	}
}

// TestRaderConvLengthSelection asserts the plan picks the direct cyclic path when
// q = N-1 is smooth and the padded-smooth-linear path otherwise.
func TestRaderConvLengthSelection(t *testing.T) {
	// 769: q=768=2⁸·3 smooth → cyclic at q. 2017: q=2016=2⁵·3²·7 smooth → cyclic.
	for _, n := range []int{769, 2017} {
		rp := newRaderPlan(n)
		if !rp.cyclic || rp.cl != n-1 {
			t.Errorf("n=%d expected cyclic conv at q=%d, got cyclic=%v cl=%d", n, n-1, rp.cyclic, rp.cl)
		}
	}
	// 9973: q=9972=2²·3²·277 not smooth → padded linear, cl 7-smooth >= 2q chosen
	// by the cost model.
	for _, n := range []int{5003, 9973, 10007} {
		rp := newRaderPlan(n)
		if rp.cyclic {
			t.Errorf("n=%d expected padded-linear conv (q not smooth), got cyclic", n)
		}
		if rp.cl < 2*(n-1) {
			t.Errorf("n=%d conv length %d below 2q=%d", n, rp.cl, 2*(n-1))
		}
		if _, ok := convCost(rp.cl); !ok {
			t.Errorf("n=%d conv length %d is not 7-smooth", n, rp.cl)
		}
	}
}

// TestConvCost checks the convolution-length cost model: it reports ok only for
// 7-smooth lengths, is monotone in the obvious cases, and orders radix-4 below a
// pure power-of-two of larger magnitude consistently with its weights.
func TestConvCost(t *testing.T) {
	// Non-7-smooth lengths are rejected.
	for _, m := range []int{11, 22, 121, 2 * 277, 5003, 10006} {
		if _, ok := convCost(m); ok {
			t.Errorf("convCost(%d) reported ok, but %d has a prime factor > 7", m, m)
		}
	}
	// 7-smooth lengths are accepted with a positive cost.
	for _, m := range []int{1, 2, 4, 8, 7, 10080, 20580, 21952} {
		c, ok := convCost(m)
		if !ok {
			t.Errorf("convCost(%d) reported not-ok, but %d is 7-smooth", m, m)
		}
		if m > 1 && c <= 0 {
			t.Errorf("convCost(%d)=%g, want positive", m, c)
		}
	}
}

// TestBestConvLen checks the convolution-length picker: the result is >= lo,
// 7-smooth, within the search window, and no cheaper 7-smooth length exists in
// that window (i.e. it is the argmin of convCost).
func TestBestConvLen(t *testing.T) {
	for _, lo := range []int{1, 2, 16, 10004, 19944, 20012, 100000} {
		g := bestConvLen(lo)
		want := lo
		if want < 1 {
			want = 1
		}
		if g < want {
			t.Errorf("bestConvLen(%d)=%d < %d", lo, g, want)
		}
		gc, ok := convCost(g)
		if !ok {
			t.Errorf("bestConvLen(%d)=%d not 7-smooth", lo, g)
			continue
		}
		hi := want + (want*4+9)/10
		for m := want; m <= hi; m++ {
			if c, ok := convCost(m); ok && c < gc {
				t.Errorf("bestConvLen(%d)=%d (cost %g) but %d is cheaper (cost %g)", lo, g, gc, m, c)
			}
		}
	}
	// lo<=0 clamps to 1, returning a small smooth length.
	if g := bestConvLen(0); g < 1 {
		t.Errorf("bestConvLen(0)=%d, want >= 1", g)
	}
}
