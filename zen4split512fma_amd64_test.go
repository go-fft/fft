package fft

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// TestZen4Split512FMALengths: a Zen 4 sends the FMA plans of 1024 and 2048 points to the
// 512-bit split layout; any other CPU none.
func TestZen4Split512FMALengths(t *testing.T) {
	if zen4Split512FMALengths(false) != nil {
		t.Error("a CPU that is not a Zen 4 has lengths")
	}
	got := zen4Split512FMALengths(true)
	if len(got) != 2 || !got[1024] || !got[2048] {
		t.Errorf("zen4Split512FMALengths(true) = %v", got)
	}
	if kernels.AMDZen4 != (zen4Split512FMA != nil) {
		t.Errorf("zen4Split512FMA = %v on Zen 4 = %v", zen4Split512FMA, kernels.AMDZen4)
	}
}

// TestZen4Split512FMAPlans: under the hook, an FMA plan of a listed length is wide with
// split passes (the 512-bit modes), and the plan of the other arithmetic keeps its
// route and its factorization; the transforms agree to the accuracy bound.
func TestZen4Split512FMAPlans(t *testing.T) {
	isolatePlans(t)
	defer func(m map[int]bool, z map[int]bool, f map[int]fmaChoice) {
		zen4Split512FMA, zen4Split512, fmaTable = m, z, f
	}(zen4Split512FMA, zen4Split512, fmaTable)
	if !kernels.UseStockhamAVX2 || !kernels.UseStockhamAVX512 {
		t.Skip("the forced route runs the 512-bit split kernels, which need AVX-512 (absent here, e.g. Zen 3 or Rosetta)")
	}
	zen4Split512FMA = map[int]bool{1024: true, 2048: true}
	if zen4Modes == nil {
		zen4Modes = kernels.StockhamSplitModes512
		defer func() { zen4Modes = nil }()
	}
	fmaTable = map[int]fmaChoice{1024: {factors: []int{32, 32}, fuse: fmaAll}, 2048: {factors: []int{32, 8, 8}, fuse: fmaAll}}
	for _, n := range []int{1024, 2048} {
		p := newSKPlanOpt(n, true)
		for k, st := range p.stages {
			if !st.wide {
				t.Errorf("n=%d stage %d is not wide", n, k)
			}
		}
		if p.stages[0].split == 0 {
			t.Errorf("n=%d: the first pass of an FMA plan is not split", n)
		}
		q := newSKPlanOpt(n, false)
		if wantWide := wide512(n); q.stages[0].wide != wantWide {
			t.Errorf("n=%d: the other arithmetic's route moved (wide %v)", n, q.stages[0].wide)
		}
		x := cmplxSignal(n)
		got, want := make([]complex128, n), naiveDFT(x)
		(&Plan{n: n, sk: p}).execute(got, x, false)
		var worst float64
		for i := range got {
			worst = max(worst, cmplx.Abs(got[i]-want[i]))
		}
		if worst > 1e-11*float64(n) {
			t.Errorf("n=%d: error %g", n, worst)
		}
	}
}

// TestNoScalarRadix32: no default or FMA plan of the running CPU has a radix-32 pass
// that no kernel runs (it would fall back to passScalarFMA, 4.8 times slower), at any
// power of two, with the table as it is and with a table that asks for 32·8·8 where the
// route has no kernel for it (the planner then refuses it and plans skFactorize's).
func TestNoScalarRadix32(t *testing.T) {
	isolatePlans(t)
	defer func(f map[int]fmaChoice) { fmaTable = f }(fmaTable)
	bad := map[int]fmaChoice{}
	for k, v := range fmaTable {
		bad[k] = v
	}
	for e := 5; e <= 16; e++ {
		bad[1<<e] = fmaChoice{factors: append([]int{32}, skFactorize(1<<e/32)...), fuse: fmaAll}
	}
	for name, tab := range map[string]map[int]fmaChoice{"table": fmaTable, "asking for a first radix-32 pass everywhere": bad} {
		fmaTable = tab
		for e := 5; e <= 16; e++ {
			n := 1 << e
			for _, fma := range []bool{false, true} {
				p := newSKPlanOpt(n, fma)
				if hasScalarRadix32(p) {
					t.Errorf("%s: n=%d fma=%v has a radix-32 pass with no kernel", name, n, fma)
				}
			}
		}
	}
}

// TestZen4Split512FMABound: the FMA plans of 1024 and 2048 points (the Zen 4's 512-bit
// split route when run on one) hold the c2c bound of docs/accuracy.md against the
// double-double oracle: err/(eps·log2 n) at most 1.5, forward and inverse.
func TestZen4Split512FMABound(t *testing.T) {
	for _, n := range []int{1024, 2048} {
		for _, inverse := range []bool{false, true} {
			for seed := uint64(1); seed <= 2; seed++ {
				x := accComplexF64(n, seed)
				want := oFFTN(cddFromC(x), []int{n}, []int{0}, inverse)
				p := NewPlanWith(n, Options{FMA: true})
				got := make([]complex128, n)
				if inverse {
					p.IFFT(got, x) // scaled by 1/n, as the oracle is
				} else {
					p.FFT(got, x)
				}
				r := errC(got, want) / (eps64 * math.Log2(float64(n)))
				if r > 1.5 {
					t.Errorf("n=%d inverse=%v seed=%d: err/(eps log2 n) = %.3f > 1.5", n, inverse, seed, r)
				}
				t.Logf("n=%d inverse=%v seed=%d: %.3f", n, inverse, seed, r)
			}
		}
	}
}
