package fft

import (
	"strings"
	"testing"
)

// planCounts is how many plans each arithmetic's caches hold, summed over the
// plan kinds (complex, no-radix-16, row, real, 2-D real, N-D, and the axes
// subsets, whose cache tells the arithmetic by the last byte of its key).
func planCounts() (def, fma, exact int) {
	planMu.Lock()
	def += len(planCache) + len(planNo16Cache) + len(clRowCache)
	fma += len(planCacheFMA) + len(planNo16CacheFMA) + len(clRowCacheFMA)
	exact += len(planCacheExact) + len(planNo16CacheExact) + len(clRowCacheExact)
	planMu.Unlock()
	realPlanMu.Lock()
	def, fma, exact = def+len(realPlanCache), fma+len(realPlanCacheFMA), exact+len(realPlanCacheExact)
	realPlanMu.Unlock()
	realPlan2Mu.Lock()
	def, fma, exact = def+len(realPlan2Cache), fma+len(realPlan2CacheFMA), exact+len(realPlan2CacheExact)
	realPlan2Mu.Unlock()
	planNMu.Lock()
	def, fma, exact = def+len(planNCache), fma+len(planNCacheFMA), exact+len(planNCacheExact)
	planNMu.Unlock()
	planNAxesMu.Lock()
	for k := range planNAxesCache {
		switch {
		case strings.HasSuffix(k, "f"):
			fma++
		case strings.HasSuffix(k, "x"):
			exact++
		case strings.HasSuffix(k, "p"):
			// a NoFMA plan: counted by plainPlans
		default:
			def++
		}
	}
	planNAxesMu.Unlock()
	return
}

// plainPlans is how many NoFMA plans the caches hold (planCounts' fourth
// arithmetic).
func plainPlans() int {
	n := 0
	planMu.Lock()
	n += len(planCachePlain) + len(planNo16CachePlain) + len(clRowCachePlain)
	planMu.Unlock()
	realPlanMu.Lock()
	n += len(realPlanCachePlain)
	realPlanMu.Unlock()
	realPlan2Mu.Lock()
	n += len(realPlan2CachePlain)
	realPlan2Mu.Unlock()
	planNMu.Lock()
	n += len(planNCachePlain)
	planNMu.Unlock()
	planNAxesMu.Lock()
	for k := range planNAxesCache {
		if strings.HasSuffix(k, "p") {
			n++
		}
	}
	planNAxesMu.Unlock()
	return n
}

// TestNoFMACallsTouchOnlyPlainCaches: the same partition for Options.NoFMA: with
// every cache empty, a NoFMA call of the case set leaves plans only in the
// NoFMA caches, and, today, gives Exact's bits (the default without fusing is
// the reference arithmetic; the day the default reorders its kernels this
// equality is the thing to revisit).
func TestNoFMACallsTouchOnlyPlainCaches(t *testing.T) {
	isolatePlans(t)
	defer func(m func(Options) Options) { exactMode = m }(exactMode)
	ref := exactCases()
	want := make([]string, len(ref))
	for i, c := range ref {
		want[i] = digestAnyZ(c.exact()) // up to the sign of a zero (scaleComplexA)
	}
	// The cases call exact(o) when they run, so NoFMA stays in place for the loop.
	exactMode = func(o Options) Options { o.NoFMA = true; return o }
	cases := exactCases()
	saw := 0
	for i, c := range cases {
		if strings.Contains(c.name, "32") || strings.HasPrefix(c.name, "FFTRealWith") {
			continue // float32: one arithmetic; FFTRealWith's reference is Exact by definition
		}
		forgetAllPlans()
		got := c.exact()
		d, f, x := planCounts()
		if d != 0 || f != 0 || x != 0 {
			t.Errorf("%s: a NoFMA call left %d default, %d FMA and %d Exact plans in their caches", c.name, d, f, x)
		}
		if plainPlans() > 0 {
			saw++
		}
		if digestAnyZ(got) != want[i] {
			t.Errorf("%s: NoFMA is not Exact's bits", c.name)
		}
	}
	if saw < len(cases)/3 {
		t.Errorf("only %d of %d cases used a NoFMA cache", saw, len(cases))
	}
}

// TestExactCallsTouchOnlyExactCaches: with every cache empty, no Exact call of
// the case set (every entry point that takes Options, on every engine) leaves
// a plan in the default or the FMA caches, which is to say none of them, at
// any depth, takes a plan of another arithmetic: the convolution of a Rader or
// Bluestein length, the half plan of a real transform, the axes of an N-D plan
// are Exact too. And the reverse: a default call leaves the Exact caches empty.
func TestExactCallsTouchOnlyExactCaches(t *testing.T) {
	isolatePlans(t)
	cases := exactCases() // built first: its setup calls the default functions
	sawExact, sawDefault := 0, 0
	for _, c := range cases {
		if strings.Contains(c.name, "32") {
			// A float32 transform takes no arithmetic: its plans are float32 ones,
			// and what it borrows from the float64 caches (an odd axis, a prime) is
			// the default's.
			continue
		}
		forgetAllPlans()
		c.exact()
		d, f, x := planCounts()
		if d != 0 || f != 0 || plainPlans() != 0 {
			t.Errorf("%s: an Exact call left %d default, %d FMA and %d NoFMA plans in their caches", c.name, d, f, plainPlans())
		}
		if x > 0 {
			sawExact++
		}
		if strings.HasPrefix(c.name, "FFTRealWith") {
			continue // its "default" call is the Exact reference too (exact_cases_test.go)
		}
		forgetAllPlans()
		c.def()
		d, f, x = planCounts()
		if x != 0 || f != 0 || plainPlans() != 0 {
			t.Errorf("%s: a default call left %d Exact, %d FMA and %d NoFMA plans in their caches", c.name, x, f, plainPlans())
		}
		if d > 0 {
			sawDefault++
		}
	}
	// The float32 and the single-plan cases use caches with one arithmetic; the
	// float64 ones are the rest.
	if sawExact < len(cases)/3 || sawDefault < len(cases)/3 {
		t.Errorf("only %d Exact and %d default cases used a partitioned cache of %d: the check sees too little", sawExact, sawDefault, len(cases))
	}
}

// TestArithPlansAreDistinct: for the same size every cache returns three plans,
// one per arithmetic, each of its own kind, and the same one on a second
// request; Exact plans never fuse, and the engines under a plan keep its
// arithmetic (a Rader or Bluestein plan's convolution, a real plan's half, an
// N-D plan's axes).
func TestArithPlansAreDistinct(t *testing.T) {
	isolatePlans(t)
	forceFMA(t) // FMA plans are built whatever the CPU
	for _, n := range []int{64, 1000, 1009, 4099, 4096} {
		d, f, x := cachedPlanA(n, arithDefault), cachedPlanA(n, arithFMA), cachedPlanA(n, arithExact)
		if d == f || d == x || f == x {
			t.Fatalf("n=%d: the arithmetics share a plan", n)
		}
		if cachedPlanA(n, arithExact) != x || cachedPlanA(n, arithDefault) != d || cachedPlanA(n, arithFMA) != f {
			t.Fatalf("n=%d: a second request returned another plan", n)
		}
		if x.sk != nil && x.sk.fma || x.rader != nil && (x.rader.fma || x.rader.ar != arithExact) || x.bluestein != nil && (x.bluestein.fma || x.bluestein.ar != arithExact) {
			t.Fatalf("n=%d: an Exact plan is not Exact", n)
		}
		if f.rader != nil && f.rader.ar != arithFMA || f.bluestein != nil && f.bluestein.ar != arithFMA || d.rader != nil && d.rader.ar != arithDefault {
			t.Fatalf("n=%d: a convolution plan lost its arithmetic", n)
		}
		rd, rf, rx := cachedRealPlanA(n, arithDefault), cachedRealPlanA(n, arithFMA), cachedRealPlanA(n, arithExact)
		if rd == rf || rd == rx || rf == rx || rx.ar != arithExact || rx.fma {
			t.Fatalf("n=%d: real plans", n)
		}
		if rx.half != nil && rx.half != cachedPlanA(n/2, arithExact) || rx.full != nil && rx.full != cachedPlanA(n, arithExact) {
			t.Fatalf("n=%d: an Exact real plan runs on a plan of another arithmetic", n)
		}
	}
	for _, sh := range [][]int{{12, 20}, {17, 9}, {64, 64}} {
		d, f, x := cachedPlanNA(sh, arithDefault), cachedPlanNA(sh, arithFMA), cachedPlanNA(sh, arithExact)
		if d == f || d == x || f == x || x.ar != arithExact || x.fma {
			t.Fatalf("shape %v: N-D plans", sh)
		}
		for ax, n := range sh {
			if p := x.axes[ax]; n > 1 && p != cachedPlanA(n, arithExact) && p != cachedPlanNo16A(n, arithExact) && p != clRowPlanA(n, x.size, arithExact) {
				t.Fatalf("shape %v axis %d: an Exact N-D plan's axis is a plan of another arithmetic", sh, ax)
			}
		}
		pd, pf, px := planNForA(sh, []int{0}, arithDefault), planNForA(sh, []int{0}, arithFMA), planNForA(sh, []int{0}, arithExact)
		if pd == pf || pd == px || pf == px {
			t.Fatalf("shape %v: axes-subset plans share an entry", sh)
		}
		r2d, r2f, r2x := cachedRealPlan2A(sh[0], sh[1], arithDefault), cachedRealPlan2A(sh[0], sh[1], arithFMA), cachedRealPlan2A(sh[0], sh[1], arithExact)
		if r2d == r2f || r2d == r2x || r2f == r2x || r2x.col.ar != arithExact || r2x.row != cachedRealPlanA(sh[1], arithExact) {
			t.Fatalf("shape %v: 2-D real plans", sh)
		}
	}
	if a, b, c, d := cachedPlanNo16A(48, arithDefault), cachedPlanNo16A(48, arithFMA), cachedPlanNo16A(48, arithExact), cachedPlanNo16A(48, arithPlain); a == b || a == c || b == c || d == a || d == b || d == c || cachedPlanNo16A(48, arithPlain) != d {
		t.Fatal("no-radix-16 plans share an entry")
	}
	if (Options{Exact: true}).arith() != arithExact || (Options{FMA: true}).arith() != arithFMA || (Options{}).arith() != arithDefault {
		t.Fatal("Options.arith")
	}
}

// TestArithRowPlans: the row plans of an N-D plan's last axis (clRowOrder, the
// lines Round 37 factorizes by table) are distinct per arithmetic too. The
// table is empty on some architectures, so the test sets one.
func TestArithRowPlans(t *testing.T) {
	isolatePlans(t)
	defer func(o map[int][]int, m int) { clRowOrder, clRowMin = o, m }(clRowOrder, clRowMin)
	clRowOrder, clRowMin = map[int][]int{64: {8, 8}}, 0
	d, f, x := clRowPlanA(64, 64*64, arithDefault), clRowPlanA(64, 64*64, arithFMA), clRowPlanA(64, 64*64, arithExact)
	if d == nil || f == nil || x == nil || d == f || d == x || f == x {
		t.Fatalf("row plans %p %p %p", d, f, x)
	}
	if pl := clRowPlanA(64, 64*64, arithPlain); pl == nil || pl == d || pl == f || pl == x || pl.sk.fma || clRowPlanA(64, 64*64, arithPlain) != pl {
		t.Fatal("NoFMA row plan")
	}
	if x.sk.fma || !f.sk.fma || clRowPlanA(64, 64*64, arithExact) != x || clRowPlanA(64, 100, arithExact) != nil && clRowMin > 100 {
		t.Fatal("row plan arithmetic or caching")
	}
}
