package fft

import (
	"math/cmplx"
	"strings"
	"testing"
)

// ndStripWrong returns the lengths n in lens for which the leading (strip)
// axis of an [n, 2] transform differs from the 1-D FFT of a column.
func ndStripWrong(lens []int) []int {
	var bad []int
	for _, n := range lens {
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(float64(i%5), float64(i%3))
		}
		want := FFT(x)
		d := make([]complex128, 2*n)
		for i := range x {
			d[2*i], d[2*i+1] = x[i], x[i]
		}
		y := FFTN(d, []int{n, 2})
		for i := range x {
			// Both columns are x: the length-2 axis gives 2·X and 0.
			if cmplx.Abs(y[2*i]-2*want[i]) > 1e-8*float64(n) || cmplx.Abs(y[2*i+1]) > 1e-8*float64(n) {
				bad = append(bad, n)
				break
			}
		}
	}
	return bad
}

// TestNDStripAxisEveryLength: every length up to 2000 as an N-D strip axis
// matches the 1-D transform, on whatever route this CPU takes. On Cascade
// Lake, v0.25.0–v0.26.2 got 120 of them wrong (153, 171, 207, …: 9·p with a
// prime p no pass handles); see TestNDStripAxisRadix9Route for the forced route.
func TestNDStripAxisEveryLength(t *testing.T) {
	lens := make([]int, 0, 1999)
	for n := 2; n <= 2000; n++ {
		lens = append(lens, n)
	}
	if bad := ndStripWrong(lens); len(bad) > 0 {
		t.Fatalf("%d strip-axis lengths transform wrongly: %v", len(bad), bad)
	}
}

// ndForgetPlans empties the 1-D, no-16 and N-D plan caches, so plans are
// rebuilt on the route the test has just set.
func ndForgetPlans() {
	planMu.Lock()
	planCache = map[int]*Plan{}
	planNo16Cache = map[int]*Plan{}
	planMu.Unlock()
	planNMu.Lock()
	planNCache = map[string]*PlanN{}
	planNMu.Unlock()
}

// TestCachedPlanNo16RefusesAPartialFactorization: a no-16 plan whose passes do
// not multiply to n is refused, not built.
func TestCachedPlanNo16RefusesAPartialFactorization(t *testing.T) {
	defer ndForgetPlans()
	defer func() {
		r, _ := recover().(string)
		if !strings.Contains(r, "no-16 factorization of 171") {
			t.Fatalf("cachedPlanNo16(171) did not refuse: %q", r)
		}
	}()
	cachedPlanNo16(171)
}
