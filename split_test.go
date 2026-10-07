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
	defer func(m map[int]bool, s map[int][]int) { intelStripOrder, splitTable = m, s }(intelStripOrder, splitTable)
	const n = 64
	splitTable = map[int][]int{n: {2, 4, 8}}
	intelStripOrder = map[int]bool{n: true}
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
