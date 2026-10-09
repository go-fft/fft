package fft

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// znForget drops length n from the plan cache, so that the next RealPlan of
// length 2n builds its half plan under the frame limits in force.
func znForget(n int) {
	planMu.Lock()
	delete(planCache, n)
	planMu.Unlock()
}

// znSetMax sets both frame limits and returns a func restoring them.
func znSetMax(two, three int) func() {
	t2, t3 := znTwoPassMax, znThreePassMax
	znTwoPassMax, znThreePassMax = two, three
	return func() { znTwoPassMax, znThreePassMax = t2, t3 }
}

// TestZnFrameMatchesPool: every plan of two Stockham passes up to 256 points
// and of three up to 512, built with the frame kernels on, gives the bits the
// same plan built with them off gives (its passes on the pool's buffer),
// forward and inverse, in place and not, and so do the real transforms whose
// halves are those plans. A plan takes a frame kernel exactly where znPlanFor
// accepts it; where the kernels refuse (no kernel for a radix, another
// architecture), or where they turn out not to run, transform uses the pool.
func TestZnFrameMatchesPool(t *testing.T) {
	defer znSetMax(0, 0)()
	rng := rand.New(rand.NewPCG(32, 2))
	same := func(a, b []complex128) bool {
		for i := range a {
			if math.Float64bits(real(a[i])) != math.Float64bits(real(b[i])) ||
				math.Float64bits(imag(a[i])) != math.Float64bits(imag(b[i])) {
				return false
			}
		}
		return true
	}
	plans := map[int]int{}
	taken := map[int]int{}
	for n := 2; n <= 512; n++ {
		znSetMax(0, 0)
		p0 := NewPlan(n)
		znForget(n)
		r0 := NewRealPlan(2 * n)
		znSetMax(256, 512)
		p := NewPlan(n)
		znForget(n)
		r := NewRealPlan(2 * n)
		znForget(n)
		if p.sk == nil {
			continue
		}
		s := len(p.sk.stages)
		if want := znPlanFor(n, p.sk.stages) != nil; (p.sk.zn != nil) != want || p0.sk.zn != nil {
			t.Fatalf("n=%d: zn %v, off %v, znPlanFor %v", n, p.sk.zn != nil, p0.sk.zn != nil, want)
		}
		if s != 2 && s != 3 || s == 2 && n > 256 {
			continue
		}
		if (r.half.sk.zn != nil) != (p.sk.zn != nil) || r0.half.sk.zn != nil {
			t.Fatalf("RFFT %d: half plans' zn %v and %v", 2*n, r.half.sk.zn != nil, r0.half.sk.zn != nil)
		}
		plans[s]++
		if p.sk.zn != nil {
			taken[s]++
		}
		src := make([]complex128, n)
		for i := range src {
			src[i] = complex(rng.NormFloat64(), rng.NormFloat64())
		}
		for _, inverse := range []bool{false, true} {
			want := make([]complex128, n)
			p0.sk.transform(want, src, inverse)
			got := make([]complex128, n)
			p.sk.transform(got, src, inverse)
			in := append([]complex128(nil), src...)
			p.sk.transform(in, in, inverse)
			// The kernel's own answer, where the plan has one.
			direct := want
			if p.sk.zn != nil {
				direct = make([]complex128, n)
				if !p.sk.zn.Run(src, direct, inverse) {
					t.Fatalf("n=%d: Run refused", n)
				}
			}
			if !same(got, want) || !same(in, want) || !same(direct, want) {
				t.Fatalf("n=%d (%d passes) inverse=%v: differs from the pool's passes", n, s, inverse)
			}
		}
		x := make([]float64, 2*n)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		wantR := r0.RFFT(make([]complex128, n+1), x)
		wantI := r0.IRFFT(make([]float64, 2*n), wantR)
		gotR := r.RFFT(make([]complex128, n+1), x)
		gotI := r.IRFFT(make([]float64, 2*n), wantR)
		if !same(gotR, wantR) {
			t.Fatalf("RFFT %d: differs from the pool's passes", 2*n)
		}
		for i := range gotI {
			if math.Float64bits(gotI[i]) != math.Float64bits(wantI[i]) {
				t.Fatalf("IRFFT %d: [%d] differs from the pool's passes", 2*n, i)
			}
		}
	}
	if plans[2] < 20 || plans[3] < 20 {
		t.Fatalf("only %v plans of two and three passes", plans)
	}
	if kernels.ZnTwoPassMax == 0 {
		// No frame kernel here: the stub runs nothing.
		var none *kernels.ZnPlan
		if none.Run(nil, nil, false) {
			t.Error("the stub ran")
		}
	} else if p64 := NewPlan(64); p64.sk != nil && p64.sk.zn != nil && (taken[2] < 20 || taken[3] < 20) {
		t.Fatalf("only %v of %v plans took a frame kernel", taken, plans)
	}
}
