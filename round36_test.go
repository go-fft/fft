package fft

import (
	"math"
	"math/cmplx"
	"runtime"
	"slices"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// zmRowsByRun is the contiguous axis ax of PlanN.transform as it ran before
// Round 36: every line through skPlan.run, here on a fresh buffer each.
func zmRowsByRun(p *PlanN, dst, src []complex128, ax int, inverse bool) {
	n, sk := p.shape[ax], p.axes[ax].sk
	for c := 0; c < p.size/n; c++ {
		if sk == nil {
			p.axes[ax].execute(dst[c*n:(c+1)*n], src[c*n:(c+1)*n], inverse)
			continue
		}
		buf := make([]complex128, 2*n+setSpan)
		sk.run(dst[c*n:(c+1)*n], src[c*n:(c+1)*n], buf, inverse)
	}
}

// TestZmRowsMatchRun holds the N-D transforms whose last axis runs on the
// frame kernels (Round 36) to the same transform with that axis run line by
// line through skPlan.run, bit for bit: both directions, in place and not,
// square, oblong and 3-D shapes (those arm64 runs as rotating passes are
// skipped, but not all of them). Where the frame kernels exist
// (znTwoPassMax > 0) it first asserts that some of these axes take them.
func TestZmRowsMatchRun(t *testing.T) {
	// A 1-D shape, or one whose other axes have length 1, runs its last
	// axis out of place: src to dst. Every other shape runs it in place.
	shapes := [][]int{{64}, {256}, {1, 128}, {1, 3, 1, 200}, {3, 64}, {5, 256}, {2, 3, 128}, {4, 512}, {7, 100}, {2, 1000}, {6, 13}}
	for n := 2; n <= 64; n++ {
		shapes = append(shapes, []int{n, n})
	}
	shapes = append(shapes, []int{128, 128}, []int{256, 256})
	onFrame, axisByAxis := 0, 0
	for _, sh := range shapes {
		p := NewPlanN(sh...)
		if p.rot != nil {
			continue // arm64's rotating passes (armreal_nd.go): no axis by axis
		}
		axisByAxis++
		last := len(sh) - 1
		if sk := p.axes[last].sk; sk != nil && sk.zn != nil {
			onFrame++
		}
		for s, x := range smallSignals(p.size)[:4] {
			for _, inverse := range []bool{false, true} {
				want := make([]complex128, p.size)
				in := x
				for ax := range last {
					if p.axes[ax] != nil {
						p.transformAxis(want, in, ax, inverse)
						in = want
					}
				}
				zmRowsByRun(p, want, in, last, inverse)
				got := make([]complex128, p.size)
				p.transform(got, x, inverse)
				inPlace := slices.Clone(x)
				p.transform(inPlace, inPlace, inverse)
				for i := range want {
					if !sameBits(got[i], want[i]) || !sameBits(inPlace[i], want[i]) {
						t.Fatalf("shape %v signal %d inverse %v point %d: %v, in place %v, want %v", sh, s, inverse, i, got[i], inPlace[i], want[i])
					}
				}
			}
		}
	}
	if znTwoPassMax > 0 && onFrame == 0 {
		t.Fatal("no last axis ran on the frame kernels")
	}
	if axisByAxis == 0 {
		t.Fatal("no shape ran axis by axis")
	}
	t.Logf("%d of %d shapes ran axis by axis, %d with their last axis on the frame kernels", axisByAxis, len(shapes), onFrame)
}

// zmBlueLoops is bluesteinPlan.transform as it ran before Round 36: the
// pointwise products as Go loops and kernels.CMul.
func zmBlueLoops(p *bluesteinPlan, dst, src []complex128, inverse bool) {
	n, w, bSpec := p.n, p.wF, p.bF
	if inverse {
		w, bSpec = p.wI, p.bI
	}
	a := make([]complex128, p.m)
	for j := 0; j < n; j++ {
		a[j] = src[j] * w[j]
	}
	plan := cachedPlan(p.m)
	plan.execute(a, a, false)
	kernels.CMul(a, bSpec)
	plan.execute(a, a, true)
	for k := 0; k < n; k++ {
		dst[k] = a[k] * w[k]
	}
}

// TestZmBluesteinMatchesLoops holds Bluestein with its products on
// kernels.CMulTo (Round 36) to the Go loops it replaced, both directions, in
// place and not, on the pooled buffer's second use too (its pad must be
// cleared again): bit for bit on amd64, where neither fuses a product into
// an FMA; elsewhere within 1e-12 of the largest output, because the
// compiler may fuse the same expression differently in another function
// (it did on macOS arm64 under -race and -coverpkg, by an ULP).
func TestZmBluesteinMatchesLoops(t *testing.T) {
	exact := runtime.GOARCH == "amd64"
	for _, n := range []int{23, 46, 47, 59, 94, 4099, 10007} {
		p := NewPlan(n)
		if p.bluestein == nil {
			t.Fatalf("%d does not run Bluestein", n)
		}
		for s, x := range smallSignals(n) {
			for _, inverse := range []bool{false, true} {
				want := make([]complex128, n)
				zmBlueLoops(p.bluestein, want, x, inverse)
				scale := 0.0
				for _, v := range want {
					if a := cmplx.Abs(v); !math.IsInf(a, 0) && !math.IsNaN(a) {
						scale = max(scale, a)
					}
				}
				same := func(a, b complex128) bool {
					if exact {
						return sameBits(a, b)
					}
					c := func(u, v float64) bool {
						return u == v || math.IsNaN(u) && math.IsNaN(v) || math.Abs(u-v) <= 1e-12*scale
					}
					return c(real(a), real(b)) && c(imag(a), imag(b))
				}
				for rep := range 2 {
					got := make([]complex128, n)
					p.bluestein.transform(got, x, inverse)
					inPlace := slices.Clone(x)
					p.bluestein.transform(inPlace, inPlace, inverse)
					for i := range want {
						if !same(got[i], want[i]) || !same(inPlace[i], want[i]) {
							t.Fatalf("n=%d signal %d inverse %v call %d point %d: %v, in place %v, want %v", n, s, inverse, rep, i, got[i], inPlace[i], want[i])
						}
					}
				}
			}
		}
	}
}
