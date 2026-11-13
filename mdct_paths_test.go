package fft

import (
	"math"
	"testing"
)

// forcePath sets both windows: the fused Go loops (kernel false) or the kernels at
// every size (kernel true); its callers call isolatePlans (the plans keep no state
// of these variables, but the audit of routing state asks it).
func forcePath(kernel bool) {
	if kernel {
		mdctFoldMin, mdctFoldMax, mdctUnfoldMin, mdctUnfoldMax = 0, 1<<30, 0, 1<<30
	} else {
		mdctFoldMin, mdctFoldMax, mdctUnfoldMin, mdctUnfoldMax = 1<<30, 1<<30, 1<<30, 1<<30
	}
}

// TestMDCTBothPaths: the MDCT and IMDCT have two paths for the fold and the unfold,
// the fused Go loops (small blocks) and the go-simd/floats kernels (from
// mdctFoldMin / mdctUnfoldMin). Both are forced at sizes on each side of the
// thresholds and held to the accuracy bound against the double-double definition,
// and the Mixed methods must stay the float32 of the float64 ones on each path.
func TestMDCTBothPaths(t *testing.T) {
	isolatePlans(t)
	sf, sfx, su, sux := mdctFoldMin, mdctFoldMax, mdctUnfoldMin, mdctUnfoldMax
	defer func() { mdctFoldMin, mdctFoldMax, mdctUnfoldMin, mdctUnfoldMax = sf, sfx, su, sux }()
	for _, path := range []struct {
		name   string
		kernel bool
	}{{"loop", false}, {"kernel", true}} {
		forcePath(path.kernel)
		for _, n := range []int{4, 8, 16, 20, 64, 100, 128, 252, 256, 512, 1024, 2048} {
			m := n / 2
			if m%2 != 0 {
				continue // the odd route has no fold or unfold
			}
			lg := math.Max(1, math.Log2(float64(n)))
			bound := mdctC64 * 0x1p-52 * lg
			p := NewMDCTPlan(n)
			x, c := accRealF64(n, 91), accRealF64(m, 92)
			eF := errR(p.MDCT(make([]float64, m), x, 1), oMDCT(cddFromR(x), m, false))
			eI := errR(p.IMDCT(make([]float64, n), c, 1), oMDCT(cddFromR(c), m, true))
			if eF > bound || eI > bound {
				t.Errorf("%s N=%d: MDCT %.3g, IMDCT %.3g exceed %.3g", path.name, n, eF, eI, bound)
			}
			x32, c32 := accRealF32(n, 93), accRealF32(m, 94)
			sameBits32(t, path.name+" MDCTMixed", p.MDCTMixed(make([]float32, m), x32, 0.5), p.MDCT(make([]float64, m), widenF32(x32), 0.5))
			sameBits32(t, path.name+" IMDCTMixed", p.IMDCTMixed(make([]float32, n), c32, 0.5), p.IMDCT(make([]float64, n), widenF32(c32), 0.5))
			// dst over src, the in-place forms the kernels must not see aliased
			want := p.IMDCT(make([]float64, n), c, 0.5)
			buf := append(append([]float64(nil), c...), make([]float64, n-m)...)
			if got := p.IMDCT(buf, buf, 0.5); !equalF(got, want) {
				t.Errorf("%s N=%d: IMDCT over its src differs", path.name, n)
			}
			wantM := p.MDCT(make([]float64, m), x, 0.5)
			bx := append([]float64(nil), x...)
			if got := p.MDCT(bx[:m], bx, 0.5); !equalF(got, wantM) {
				t.Errorf("%s N=%d: MDCT into the start of its src differs", path.name, n)
			}
		}
	}
}
