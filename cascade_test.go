package fft

import (
	"math"
	"math/cmplx"
	"runtime"
	"testing"
)

// TestCascadeShape pins the blocked schedule's shape for rule A (radix 8 as
// far as it goes) and the cases that run breadth first.
func TestCascadeShape(t *testing.T) {
	for _, c := range []struct{ n, min, t, b int }{
		{1 << 16, 1 << 16, 2, 8}, // 8·8, then groups of 8 blocks of 1024
		{1 << 17, 1 << 16, 2, 4},
		{1 << 18, 1 << 16, 3, 16},
		{1 << 19, 1 << 16, 3, 8},
		{1 << 20, 1 << 16, 3, 4},
		{1 << 14, 1, 1, 4},
		{1 << 15, 1 << 16, 0, 0}, // below the threshold
		{1 << 16, 0, 0, 0},       // no threshold: never
		{1 << 12, 1, 0, 0},       // a single group: breadth first
	} {
		stages := newSKPlanFactors(c.n, radix8Maximal(log2Exact(c.n))).stages
		if gt, gb := cascadeShape(c.n, stages, c.min); gt != c.t || gb != c.b {
			t.Errorf("cascadeShape(%d, min %d) = %d, %d, want %d, %d", c.n, c.min, gt, gb, c.t, c.b)
		}
	}
	n := 3 << 14 // not a power of two
	if gt, gb := cascadeShape(n, newSKPlan(n).stages, 1); gt != 0 || gb != 0 {
		t.Errorf("cascadeShape(%d) = %d, %d, want breadth first", n, gt, gb)
	}
}

// TestCascadePairs pins when the first two passes run as one sweep.
func TestCascadePairs(t *testing.T) {
	for _, c := range []struct {
		n, t    int
		factors []int
		want    bool
	}{
		{1 << 16, 2, radix8Maximal(16), true},
		{1 << 17, 3, pow2Radices(17, false), true},   // radix 4, 4
		{1 << 14, 1, radix8Maximal(14), false},       // one pass over the array
		{1 << 16, 2, []int{2, 8, 8, 8, 8, 8}, false}, // no strided radix-2 kernel
		{1 << 16, 2, []int{8, 2, 8, 8, 8, 8}, false},
		{1 << 12, 2, []int{8, 8, 8, 8}, false}, // pass 1's ido (64) is less than a chunk
		{1 << 16, 2, []int{8, 8, 8, 8, 4, 4}, true},
	} {
		if got := cascadePairs(c.t, newSKPlanFactors(c.n, c.factors).stages); got != c.want {
			t.Errorf("cascadePairs(%d, %v) = %v, want %v", c.t, c.factors, got, c.want)
		}
	}
}

// log2Exact is log2 of a power of two.
func log2Exact(n int) int {
	e := 0
	for 1<<e < n {
		e++
	}
	return e
}

// cascadePlans builds the same plan twice, blocked and breadth first.
func cascadePlans(n int, factors []int) (blocked, plain *skPlan) {
	save := cascadeMin
	defer func() { cascadeMin = save }()
	cascadeMin = 1
	blocked = newSKPlanFactors(n, factors)
	cascadeMin = 0
	plain = newSKPlanFactors(n, factors)
	return blocked, plain
}

// cascadeSignals returns the inputs the identity tests run: a generic signal
// and one cycling through signed zeros, infinities, subnormals and NaN.
func cascadeSignals(n int) [][]complex128 {
	special := []float64{0, math.Copysign(0, -1), math.Inf(1), math.Inf(-1), 5e-324, -2.2250738585072014e-308, 1, -3.5, math.NaN()}
	sp := make([]complex128, n)
	for i := range sp {
		sp[i] = complex(special[i%len(special)], special[(i*7+3)%len(special)])
	}
	return [][]complex128{benchComplex(n), sp}
}

// cascadeSame is sameBits on amd64. Where gc fuses multiply-adds, the Go
// strided pass (bflyR, then the twiddle) and the Go or NEON passes may fuse
// different products, as the batched passes do (Round 17), so there the
// values may differ by a few ulps; the blocked schedule runs only on amd64.
// The tolerance is a few ulps of the transform's scale, which grows with n.
func cascadeSame(a, b complex128, n int) bool {
	if sameBits(a, b) || runtime.GOARCH == "amd64" {
		return sameBits(a, b)
	}
	return cmplx.Abs(a-b) <= 1e-15*float64(n)
}

// checkCascade runs the blocked plan against the breadth-first one, forward
// and inverse, out of place, in place and through run, and fails on the first
// value whose bits differ.
func checkCascade(t *testing.T, mode string) {
	t.Helper()
	for _, c := range []struct {
		n       int
		factors []int
	}{
		{1 << 14, radix8Maximal(14)},       // t = 1, b = 4
		{1 << 15, radix8Maximal(15)},       // t = 2 (in place: the copy), b = 16
		{1 << 16, radix8Maximal(16)},       // t = 2, b = 8, final radix 4
		{1 << 17, pow2Radices(17, false)},  // rule B: final radix 2
		{1 << 18, radix8Maximal(18)},       // t = 3, b = 16, final radix 8
		{1 << 15, pow2Radices(15, true)},   // rule C
		{1 << 16, []int{2, 8, 8, 8, 8, 8}}, // radix 2 first
		{1 << 15, []int{2, 8, 8, 8, 8, 4}}, // t = 2 with no pair: the in-place copy
	} {
		prod := 1
		for _, r := range c.factors {
			prod *= r
		}
		if prod != c.n {
			t.Fatalf("factors %v multiply to %d, not %d", c.factors, prod, c.n)
		}
		bl, pl := cascadePlans(c.n, c.factors)
		if bl.cascT == 0 {
			t.Fatalf("%s n=%d %v: not blocked", mode, c.n, c.factors)
		}
		for si, src := range cascadeSignals(c.n) {
			for _, inverse := range []bool{false, true} {
				want := make([]complex128, c.n)
				pl.transform(want, src, inverse)
				got := make([]complex128, c.n)
				bl.transform(got, src, inverse)
				inPlace := append([]complex128(nil), src...)
				bl.transform(inPlace, inPlace, inverse)
				viaRun := make([]complex128, c.n)
				bp := bl.scratch.Get().(*[]complex128)
				bl.run(viaRun, src, *bp, inverse)
				bl.scratch.Put(bp)
				for i := range want {
					for name, v := range map[string]complex128{"out of place": got[i], "in place": inPlace[i], "run": viaRun[i]} {
						if !cascadeSame(v, want[i], c.n) {
							t.Fatalf("%s n=%d %v signal %d inverse=%v %s: [%d] = %v, breadth first %v", mode, c.n, c.factors, si, inverse, name, i, v, want[i])
						}
					}
				}
			}
		}
	}
}

// TestCascadeIsBitIdentical: the blocked schedule changes where values are
// computed, never how, on the kernels this machine runs by default.
func TestCascadeIsBitIdentical(t *testing.T) {
	if testing.Short() {
		t.Skip("large transforms")
	}
	checkCascade(t, "default")
}

// TestCascadeThroughThePlan runs a Plan on the blocked schedule, with this
// machine's radix rule, against the DFT at a few bins and through the inverse.
func TestCascadeThroughThePlan(t *testing.T) {
	save := cascadeMin
	defer func() { cascadeMin = save }()
	cascadeMin = 1 << 14
	n := 1 << 14
	p := &Plan{n: n, sk: newSKPlan(n)} // whatever this machine routes powers of two to
	if p.sk.cascT == 0 {
		t.Fatalf("n=%d does not take the blocked schedule", n)
	}
	src := benchComplex(n)
	dst := p.FFT(make([]complex128, n), src)
	for _, k := range []int{0, 1, 77, n / 2, n - 1} {
		var want complex128
		for j, x := range src {
			s, c := math.Sincos(-2 * math.Pi * float64((j*k)%n) / float64(n))
			want += x * complex(c, s)
		}
		if d := dst[k] - want; math.Hypot(real(d), imag(d)) > 1e-9*float64(n) {
			t.Errorf("X[%d] = %v, want %v", k, dst[k], want)
		}
	}
	back := p.IFFT(make([]complex128, n), dst)
	for i := range back {
		if d := back[i] - src[i]; math.Hypot(real(d), imag(d)) > 1e-12 {
			t.Fatalf("round trip [%d] = %v, want %v", i, back[i], src[i])
		}
	}
}
