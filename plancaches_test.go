package fft

import "testing"

// forgetAllPlans empties EVERY plan cache: the default, FMA, no-radix-16, row
// and axes plans of the complex transforms, the real and 2-D real plans, the
// N-D plans, the float32 plans and the DCT/DST kernels. A test that overrides
// a routing table or a kernel switch (splitTable, intelStripOrder, cascadeMin,
// stripAxes, kernels.Use..., ...) and then calls a function that caches its
// plan (FFT, FFTN, RFFT2, DCT, ...) leaves behind a plan built on the
// overridden route; any later test, in any order under -shuffle, would run on
// it. isolatePlans is the one way such a test protects the others.
func forgetAllPlans() {
	planMu.Lock()
	planCache, planCacheFMA = map[int]*Plan{}, map[int]*Plan{}
	planNo16Cache, planNo16CacheFMA = map[int]*Plan{}, map[int]*Plan{}
	clRowCache, clRowCacheFMA = map[int]*Plan{}, map[int]*Plan{}
	planMu.Unlock()
	realPlanMu.Lock()
	realPlanCache, realPlanCacheFMA = map[int]*RealPlan{}, map[int]*RealPlan{}
	realPlanMu.Unlock()
	realPlan2Mu.Lock()
	realPlan2Cache, realPlan2CacheFMA = map[[2]int]*RealPlan2{}, map[[2]int]*RealPlan2{}
	realPlan2Mu.Unlock()
	planNMu.Lock()
	planNCache, planNCacheFMA = map[string]*PlanN{}, map[string]*PlanN{}
	planNMu.Unlock()
	planNAxesMu.Lock()
	planNAxesCache = map[string]*PlanN{}
	planNAxesMu.Unlock()
	plan32Mu.Lock()
	plan32Cache = map[int]*Plan32{}
	plan32Mu.Unlock()
	planN32Mu.Lock()
	planN32Cache = map[string]*PlanN32{}
	planN32Mu.Unlock()
	realPlan32Mu.Lock()
	realPlan32Cache = map[int]*RealPlan32{}
	realPlan32Mu.Unlock()
	realPlan2_32Mu.Lock()
	realPlan2_32Cache = map[[2]int]*RealPlan2_32{}
	realPlan2_32Mu.Unlock()
	r2rMu.Lock()
	r2rCache = map[r2rKey]any{}
	r2rMu.Unlock()
	r2r32Mu.Lock()
	r2r32Cache = map[r2rKey]any{}
	r2r32Mu.Unlock()
}

// isolatePlans empties every plan cache now and again when the test ends, so
// that plans built under the test's overrides neither reach it from an earlier
// test nor outlive it. Call it first, before the overrides' deferred
// restorations are registered, so that it runs after them.
func isolatePlans(tb testing.TB) {
	tb.Helper()
	forgetAllPlans()
	tb.Cleanup(forgetAllPlans)
}

// TestForgetAllPlans: every cache forgetAllPlans lists is emptied, and a
// function that caches repopulates it.
func TestForgetAllPlans(t *testing.T) {
	_ = FFT(make([]complex128, 12))
	_ = FFTN(make([]complex128, 16), []int{4, 4})
	_ = RFFT(make([]float64, 8))
	_ = RFFT2(make([]float64, 16), [2]int{4, 4})
	_ = FFT32(make([]complex64, 12))
	_ = DCT(make([]float64, 8), 2, NormBackward)
	_ = DCT32(make([]float32, 8), 2, NormBackward)
	planMu.Lock()
	planNMu.Lock()
	filled := len(planCache) > 0 && len(planNCache) > 0
	planNMu.Unlock()
	planMu.Unlock()
	if !filled {
		t.Fatal("the transforms left no cached plan")
	}
	forgetAllPlans()
	planMu.Lock()
	n := len(planCache) + len(planNo16Cache) + len(clRowCache) + len(planCacheFMA)
	planMu.Unlock()
	planNMu.Lock()
	n += len(planNCache) + len(planNCacheFMA)
	planNMu.Unlock()
	realPlanMu.Lock()
	n += len(realPlanCache) + len(realPlanCacheFMA)
	realPlanMu.Unlock()
	realPlan2Mu.Lock()
	n += len(realPlan2Cache) + len(realPlan2CacheFMA)
	realPlan2Mu.Unlock()
	plan32Mu.Lock()
	n += len(plan32Cache)
	plan32Mu.Unlock()
	r2rMu.Lock()
	n += len(r2rCache)
	r2rMu.Unlock()
	r2r32Mu.Lock()
	n += len(r2r32Cache)
	r2r32Mu.Unlock()
	if n != 0 {
		t.Fatalf("%d plans survive forgetAllPlans", n)
	}
}
