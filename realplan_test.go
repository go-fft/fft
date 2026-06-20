package fft

import "testing"

// TestRealPlanReuse validates the RealPlan API directly across even, odd,
// composite, and Bluestein lengths, including the trivial n==0/1 cases.
func TestRealPlanReuse(t *testing.T) {
	for _, n := range []int{2, 3, 4, 5, 8, 9, 16, 17, 1000, 1296} {
		p := NewRealPlan(n)
		if p.Len() != n {
			t.Fatalf("Len()=%d want %d", p.Len(), n)
		}
		x := realSignal(n)
		dst := make([]complex128, n/2+1)
		got := p.RFFT(dst, x)
		full := FFTReal(x)
		closeVec(t, got, full[:n/2+1])

		// Inverse round-trip through the plan.
		out := make([]float64, n)
		rec := p.IRFFT(out, got)
		for i := range rec {
			if d := rec[i] - x[i]; d > tol || d < -tol {
				t.Fatalf("n=%d IRFFT[%d]=%g want %g", n, i, rec[i], x[i])
			}
		}
	}
}

// TestRealPlanTrivial covers the n==0 and n==1 RealPlan paths.
func TestRealPlanTrivial(t *testing.T) {
	p0 := NewRealPlan(0)
	if got := p0.RFFT(nil, nil); len(got) != 0 {
		t.Fatalf("n=0 RFFT: %v", got)
	}
	if got := p0.IRFFT(nil, nil); len(got) != 0 {
		t.Fatalf("n=0 IRFFT: %v", got)
	}
	p1 := NewRealPlan(1)
	dst := make([]complex128, 1)
	if got := p1.RFFT(dst, []float64{5}); len(got) != 1 || got[0] != complex(5, 0) {
		t.Fatalf("n=1 RFFT: %v", got)
	}
	out := make([]float64, 1)
	if got := p1.IRFFT(out, []complex128{complex(5, 0)}); len(got) != 1 || got[0] != 5 {
		t.Fatalf("n=1 IRFFT: %v", got)
	}
}
