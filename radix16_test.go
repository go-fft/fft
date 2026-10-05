package fft

import (
	"math/cmplx"
	"testing"
)

// radix16FactorLists are factorizations with radix-16 passes at every
// position: first (l1 = 1), last (ido = 1), between, next to odd radices, and
// with ido values that leave every remainder mod 4 (the kernels' tails).
var radix16FactorLists = [][]int{
	{16}, {16, 16}, {2, 16}, {16, 2}, {3, 16}, {16, 3}, {16, 5}, {16, 7},
	{4, 16, 16}, {16, 4, 16}, {16, 16, 4}, {8, 16}, {16, 8}, {5, 16, 3},
	{16, 3, 3}, {16, 2, 3}, {16, 16, 2}, {16, 16, 3},
}

// TestRadix16AgainstNaive holds the radix-16 passes to the O(N²) DFT, forward
// and inverse, whichever of the Go pass or a kernel runs them.
func TestRadix16AgainstNaive(t *testing.T) {
	for _, f := range radix16FactorLists {
		n := 1
		for _, r := range f {
			n *= r
		}
		p := newSKPlanFactors(n, f)
		x := cmplxSignal(n)
		got := make([]complex128, n)
		p.transform(got, x, false)
		closeVec(t, got, naiveDFT(x))
		// The inverse DFT is conj(DFT(conj(x))).
		cx := make([]complex128, n)
		for i, v := range x {
			cx[i] = cmplx.Conj(v)
		}
		want := naiveDFT(cx)
		for i := range want {
			want[i] = cmplx.Conj(want[i])
		}
		p.transform(got, x, true)
		closeVec(t, got, want)
	}
}

// TestRadix16GoPass runs the Go pass itself against the DFT, on every
// architecture: on amd64 the kernels would otherwise take every pass.
func TestRadix16GoPass(t *testing.T) {
	for _, f := range radix16FactorLists {
		n := 1
		for _, r := range f {
			n *= r
		}
		p := newSKPlanFactors(n, f)
		x := cmplxSignal(n)
		for _, inverse := range []bool{false, true} {
			a, b := append([]complex128(nil), x...), make([]complex128, n)
			for k := range p.stages {
				p.stages[k].passScalar(b, a, inverse)
				a, b = b, a
			}
			want := make([]complex128, n)
			p.transform(want, x, inverse)
			closeVec(t, a, want)
		}
	}
}

// TestRadix16Table checks that skFactorize takes a power of two's radix-16
// factorization from radix16Table, as a copy, wherever the table has one.
func TestRadix16Table(t *testing.T) {
	defer func(m map[int][]int) { radix16Table = m }(radix16Table)
	radix16Table = map[int][]int{256: {16, 16}}
	f := skFactorize(256)
	if len(f) != 2 || f[0] != 16 || f[1] != 16 {
		t.Fatalf("skFactorize(256) = %v, want [16 16]", f)
	}
	f[0] = 2
	if radix16Table[256][0] != 16 {
		t.Fatal("skFactorize returned the table's own slice")
	}
	if g := skFactorize(512); len(g) == 0 || g[0] == 16 {
		t.Fatalf("skFactorize(512) = %v: 512 is not in the table", g)
	}
}

// TestRadix16Columns checks that an N-D plan's strided axes take the
// factorization without radix 16 and its last axis the table's, and that the
// result is still the DFT.
func TestRadix16Columns(t *testing.T) {
	defer func(m map[int][]int) { radix16Table = m }(radix16Table)
	// 48 rather than a power of two: on amd64 without AVX2 powers of two
	// take the pow2 kernel, and only a Stockham plan has passes to inspect.
	radix16Table = map[int][]int{48: {16, 3}}
	forget := func() { planMu.Lock(); delete(planCache, 48); planMu.Unlock() }
	forget() // a plan cached before the table changed
	defer forget()
	p := NewPlanN(48, 48)
	for _, st := range p.axes[0].sk.stages {
		if st.r == 16 {
			t.Fatalf("column axis factored %v", p.axes[0].sk.stages)
		}
	}
	if p.axes[1].sk.stages[0].r != 16 {
		t.Fatal("row axis lost its radix-16 pass")
	}
	if cachedPlanNo16(48) != p.axes[0] {
		t.Fatal("cachedPlanNo16 did not cache")
	}
	x := cmplxSignal(48 * 48)
	closeVec(t, p.FFT(make([]complex128, len(x)), x), naiveDFTN(x, []int{48, 48}))
}
