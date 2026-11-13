package fft

import (
	"math/cmplx"
	"slices"
	"testing"
)

// TestSplitPow2Factors: the split factorization of 2^e multiplies to 2^e,
// ends with a radix-4 pass, and puts its radix-4 passes before its radix-8
// ones, which number as many as leave an even remainder.
func TestSplitPow2Factors(t *testing.T) {
	for e := 4; e <= 30; e++ {
		f := splitPow2Factors(e)
		p, n8 := 1, 0
		for k, r := range f {
			p *= r
			if r == 8 {
				n8++
			}
			if k > 0 && r == 4 && f[k-1] == 8 && k != len(f)-1 {
				t.Errorf("e=%d: %v has a radix-4 pass after a radix-8 one", e, f)
			}
		}
		if p != 1<<e || f[len(f)-1] != 4 {
			t.Errorf("e=%d: %v", e, f)
		}
		if rest := e - 2; 3*(n8+1) <= rest && (rest-3*(n8+1))%2 == 0 {
			t.Errorf("e=%d: %v could take one more radix-8 pass", e, f)
		}
	}
}

// TestSplitTable checks that skFactorize takes a power of two's split
// factorization from splitTable, as a copy, before radix16Table.
func TestSplitTable(t *testing.T) {
	isolatePlans(t)
	defer func(m, r map[int][]int) { splitTable, radix16Table = m, r }(splitTable, radix16Table)
	splitTable = map[int][]int{256: {8, 8, 4}}
	radix16Table = map[int][]int{256: {16, 16}}
	f := skFactorize(256)
	if len(f) != 3 || f[0] != 8 || f[2] != 4 {
		t.Fatalf("skFactorize(256) = %v, want [8 8 4]", f)
	}
	f[0] = 2
	if splitTable[256][0] != 8 {
		t.Fatal("skFactorize returned the table's own slice")
	}
}

// TestIntelStripOrder: a length in intelStripOrder runs its N-D strip axes
// with skFactorizeOrder's factorization, and the plan stays correct.
func TestIntelStripOrder(t *testing.T) {
	isolatePlans(t)
	defer func(m map[int]bool, s map[int][]int) { intelStripOrder, splitTable = m, s }(intelStripOrder, splitTable)
	const n = 64
	splitTable = map[int][]int{n: {2, 4, 8}}
	intelStripOrder = map[int]bool{n: true}
	// Without the cached plans, the strip axis would take splitTable's.
	planMu.Lock()
	delete(planCache, n)
	delete(planNo16Cache, n)
	planMu.Unlock()
	defer func() {
		planMu.Lock()
		delete(planCache, n)
		delete(planNo16Cache, n)
		planMu.Unlock()
	}()
	p := NewPlanN(n, n)
	var col []int
	for _, st := range p.axes[0].sk.stages {
		col = append(col, st.r)
	}
	if want := skFactorizeOrder(n, compOddFirst); !slices.Equal(col, want) {
		t.Errorf("strip axis factors %v, want %v", col, want)
	}
	x := make([]complex128, n*n)
	for i := range x {
		x[i] = complex(float64(i%7), -float64(i%5))
	}
	got := p.FFT(make([]complex128, n*n), x)
	want := FFTN(x, []int{n, n})
	for i := range got {
		if cmplx.Abs(got[i]-want[i]) > 1e-9*float64(n*n) {
			t.Fatalf("index %d: %v, want %v", i, got[i], want[i])
		}
	}
}

// TestHswSplitKept: without a floor every plan keeps its split modes; with
// Round 29's, only the powers of two from the floor up.
func TestHswSplitKept(t *testing.T) {
	isolatePlans(t)
	defer func(v, m int) { hswSplitFloor, zen3RouteMax = v, m }(hswSplitFloor, zen3RouteMax)
	hswSplitFloor, zen3RouteMax = 0, 0 // a Zen 4's ceiling has its own test
	for _, n := range []int{64, 1000, 4096} {
		if !hswSplitKept(n) {
			t.Errorf("no floor: %d not kept", n)
		}
	}
	hswSplitFloor = 512
	for n, want := range map[int]bool{64: false, 256: false, 512: true, 4096: true, 1 << 20: true, 1000: false, 1536: false, 15360: false} {
		if got := hswSplitKept(n); got != want {
			t.Errorf("floor 512: hswSplitKept(%d) = %v, want %v", n, got, want)
		}
	}
}

// TestZen3RouteMax: with a ceiling, the lengths above it keep neither the
// split modes nor the AVX2 route, the lengths up to it do (Round 49).
func TestZen3RouteMax(t *testing.T) {
	isolatePlans(t)
	defer func(f, m int, s, z map[int]bool) {
		hswSplitFloor, zen3RouteMax, zen3RouteSkip, zen4Split512 = f, m, s, z
	}(hswSplitFloor, zen3RouteMax, zen3RouteSkip, zen4Split512)
	hswSplitFloor, zen3RouteMax, zen3RouteSkip, zen4Split512 = 0, 1<<16, map[int]bool{2048: true}, nil
	for n, want := range map[int]bool{64: true, 1000: true, 2048: false, 4096: true, 1 << 16: true, 1<<16 + 1: false, 1 << 20: false} {
		if got := hswSplitKept(n); got != want {
			t.Errorf("ceiling 65536: hswSplitKept(%d) = %v, want %v", n, got, want)
		}
		if got := zen3Route(n); got != want {
			t.Errorf("ceiling 65536: zen3Route(%d) = %v, want %v", n, got, want)
		}
	}
	for n, want := range map[int]bool{128: false, 255: false, 256: false, 2048: true, 4096: false, 1 << 17: true, 1 << 20: true, 1000: false} {
		if got := wide512(n); got != want {
			t.Errorf("ceiling 65536: wide512(%d) = %v, want %v", n, got, want)
		}
	}
	zen3RouteMax = 0
	if zen3Route(1) || !hswSplitKept(1<<30) {
		t.Error("no ceiling: the AVX2 route must be off and every length kept")
	}
}

// TestClRowPlan: the last axis of an N-D plan of at least clRowMin
// elements takes clRowOrder's factorization, a smaller one its 1-D plan's;
// the large plan reuses one cached row plan and stays correct.
func TestClRowPlan(t *testing.T) {
	isolatePlans(t)
	defer func(m map[int][]int) { clRowOrder = m }(clRowOrder)
	defer func(v int) { clRowMin = v }(clRowMin)
	clRowMin = 1 << 19
	const n = 64
	want := []int{2, 4, 8}
	clRowOrder = map[int][]int{n: want}
	planMu.Lock()
	delete(clRowCache, n)
	planMu.Unlock()
	defer func() {
		planMu.Lock()
		delete(clRowCache, n)
		planMu.Unlock()
	}()
	factors := func(p *PlanN) []int {
		var f []int
		sk := p.axes[len(p.axes)-1].sk
		if sk == nil {
			return nil // a 1-D plan on another engine (s390x, loong64: the pow2 kernel)
		}
		for _, st := range sk.stages {
			f = append(f, st.r)
		}
		return f
	}
	m := clRowMin / n
	small := NewPlanN(m/2, n)
	if got := factors(small); slices.Equal(got, want) || small.axes[1] != cachedPlan(n) {
		t.Errorf("a plan below clRowMin: row factors %v, want the 1-D plan's", got)
	}
	if other := NewPlanN(2*m, n/2); other.axes[1] != cachedPlan(n/2) {
		t.Error("a large plan's rows of a length not in clRowOrder: not the 1-D plan")
	}
	big := NewPlanN(m, n)
	if got := factors(big); !slices.Equal(got, want) {
		t.Errorf("row factors %v, want %v", got, want)
	}
	if again := NewPlanN(2, m/2, n); again.axes[2] != big.axes[1] {
		t.Error("the row plan was built twice")
	}
	x := make([]complex128, m*n)
	for i := range x {
		x[i] = complex(float64(i%7), -float64(i%5))
	}
	got := big.FFT(make([]complex128, m*n), x)
	clRowOrder = nil
	ref := NewPlanN(m, n).FFT(make([]complex128, m*n), x)
	for i := range got {
		if cmplx.Abs(got[i]-ref[i]) > 1e-9*float64(m*n) {
			t.Fatalf("index %d: %v, want %v", i, got[i], ref[i])
		}
	}
}
