package fft

import (
	"math"
	"math/cmplx"
	"testing"
)

// r32FactorLists are factorizations with radix-32 passes at every position:
// first (l1 = 1), last (ido = 1), between, next to odd radices, and with ido
// values that leave every remainder mod 4 (the kernels' tails).
var r32FactorLists = [][]int{
	{32}, {32, 32}, {8, 32}, {32, 8}, {4, 32}, {32, 4}, {2, 32}, {32, 2},
	{3, 32}, {32, 3}, {32, 5}, {32, 7}, {4, 32, 32}, {32, 4, 32}, {32, 32, 4},
	{5, 32, 3}, {32, 3, 3}, {16, 32}, {32, 16}, {2, 32, 2}, {32, 9},
}

// TestR32Trig holds r32Trig to math.Cos and math.Sin (whose own error at
// 31π/16 is 4e-16, π being rounded), and to
// cos² + sin² = 1, for every e it serves.
func TestR32Trig(t *testing.T) {
	for e := 0; e < 32; e++ {
		c, s := r32Trig(e)
		th := math.Pi * float64(e) / 16
		if math.Abs(c-math.Cos(th)) > 1e-15 || math.Abs(s-math.Sin(th)) > 1e-15 {
			t.Errorf("r32Trig(%d) = %v, %v; want %v, %v", e, c, s, math.Cos(th), math.Sin(th))
		}
	}
	// The cos(kπ/16) table against the identities its values satisfy.
	for k := 0; k <= 8; k++ {
		c, s := r32CosK[k], r32CosK[8-k]
		if math.Abs(c*c+s*s-1) > 4e-16 {
			t.Errorf("cos(%dπ/16)² + sin² = %v", k, c*c+s*s)
		}
	}
}

// TestR32Butterfly holds bfly32 to the 32-point DFT on unit impulses and a
// random signal, both directions.
func TestR32Butterfly(t *testing.T) {
	for _, s := range []float64{-1, 1} {
		var x, y [32]complex128
		sig := cmplxSignal(32)
		copy(x[:], sig)
		bfly32(&y, &x, s)
		want := make([]complex128, 32)
		for k := range want {
			for n, v := range sig {
				want[k] += v * cmplx.Exp(complex(0, s*2*math.Pi*float64(n*k)/32))
			}
		}
		closeVec(t, y[:], want)
	}
}

// TestR32AgainstNaive holds the radix-32 passes to the O(N²) DFT, forward and
// inverse, whichever of the Go pass or a kernel runs them.
func TestR32AgainstNaive(t *testing.T) {
	for _, f := range r32FactorLists {
		n := 1
		for _, r := range f {
			n *= r
		}
		p := newSKPlanFactors(n, f)
		x := cmplxSignal(n)
		for _, inverse := range []bool{false, true} {
			got := make([]complex128, n)
			p.transform(got, x, inverse)
			closeVec(t, got, naiveDFTDir(x, inverse))
		}
	}
}

// TestR32GoPass runs the Go pass itself against the transform, on every
// architecture: on amd64 the kernels would otherwise take every pass.
func TestR32GoPass(t *testing.T) {
	for _, f := range r32FactorLists {
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
			closeVec(t, a, naiveDFTDir(x, inverse))
		}
	}
}

// naiveDFTDir is the O(N²) DFT, or the unnormalized inverse DFT when
// inverse, with each angle reduced exactly (k·j mod n) before it is rounded:
// naiveDFT's unreduced angles are off by about 1e-8 at 4096 points.
func naiveDFTDir(x []complex128, inverse bool) []complex128 {
	n := len(x)
	s := -1.0
	if inverse {
		s = 1
	}
	y := make([]complex128, n)
	for k := range y {
		for j, v := range x {
			y[k] += v * cmplx.Rect(1, s*2*math.Pi*float64(k*j%n)/float64(n))
		}
	}
	return y
}

// TestR32Table checks that skFactorize takes a length's factorization from
// r32Table, as a copy, before splitTable and radix16Table.
func TestR32Table(t *testing.T) {
	defer func(a, b, c map[int][]int) { r32Table, splitTable, radix16Table = a, b, c }(r32Table, splitTable, radix16Table)
	r32Table = map[int][]int{1024: {32, 32}}
	splitTable = map[int][]int{1024: {4, 4, 4, 16}}
	radix16Table = map[int][]int{1024: {8, 8, 16}}
	f := skFactorize(1024)
	if len(f) != 2 || f[0] != 32 || f[1] != 32 {
		t.Fatalf("skFactorize(1024) = %v, want [32 32]", f)
	}
	f[0] = 2
	if r32Table[1024][0] != 32 {
		t.Fatal("skFactorize returned the table's own slice")
	}
	if g := skFactorize(2048); len(g) == 0 || g[0] == 32 {
		t.Fatalf("skFactorize(2048) = %v: 2048 is not in the table", g)
	}
}

// TestR32Columns checks that an N-D plan's strided axes take the
// factorization without radix 32 (there is no batched radix-32 kernel) and
// its last axis the table's, and that the result is still the DFT.
func TestR32Columns(t *testing.T) {
	defer func(m map[int][]int) { r32Table = m }(r32Table)
	// 96 rather than a power of two: on amd64 without AVX2 powers of two
	// take the pow2 kernel, and only a Stockham plan has passes to inspect.
	r32Table = map[int][]int{96: {32, 3}}
	forget := func() { planMu.Lock(); delete(planCache, 96); delete(planNo16Cache, 96); planMu.Unlock() }
	forget()
	defer forget()
	if !compTakes16(96) {
		t.Fatal("compTakes16(96) = false with a radix-32 pass")
	}
	p := NewPlanN(96, 96)
	for _, st := range p.axes[0].sk.stages {
		if st.r == 32 {
			t.Fatalf("column axis factored %v", p.axes[0].sk.stages)
		}
	}
	if p.axes[1].sk.stages[0].r != 32 {
		t.Fatal("row axis lost its radix-32 pass")
	}
	x := cmplxSignal(96 * 96)
	closeVec(t, p.FFT(make([]complex128, len(x)), x), naiveDFTN(x, []int{96, 96}))
}
