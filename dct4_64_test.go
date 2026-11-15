package fft

import (
	"math"
	"testing"
)

// The fused DCT-IV(64) is held to the #121 contract: each result within
// c·ε·log2 n of a double-double oracle (c = 2, the r2r family's constant in
// accBound), the Mixed one plus half a float32 ulp, and every route of the
// plan (DCT, IDCT, DCTMixed, IDCTMixed, in place) to the same arithmetic.

// fusedPlans returns the plan the package builds and one forced onto the fused
// kernels, which off arm64 means the scalar Go ones: every platform runs every
// line of the fused path.
func fusedPlans() []*DCTPlan {
	forced := *NewDCTPlan(64, 4)
	forced.fused = true
	return []*DCTPlan{NewDCTPlan(64, 4), &forced}
}

func TestDCT4N64Oracle(t *testing.T) {
	for _, p := range fusedPlans() {
		testDCT4N64Oracle(t, p)
	}
}

func TestDCT4N64Pair(t *testing.T) {
	for _, p := range fusedPlans() {
		testDCT4N64Pair(t, p)
	}
}

func TestDCT4N64PairShortPanics(t *testing.T) {
	for _, p := range fusedPlans() {
		testDCT4N64PairShortPanics(t, p)
	}
}

func testDCT4N64Oracle(t *testing.T, p *DCTPlan) {
	b64 := 2 * eps64 * math.Log2(64)
	b32 := b64 + 0x1p-24
	for seed := uint64(1); seed <= 4; seed++ {
		x := accRealF64(64, seed)
		x32 := accRealF32(64, seed)
		for _, norm := range accNorms {
			for _, inverse := range []bool{false, true} {
				want := oR2R(cddFromR(x), true, 4, norm, inverse)
				want32 := oR2R(cddFromR(x32), true, 4, norm, inverse)
				var got []float64
				var got32 []float32
				if inverse {
					got = p.IDCT(make([]float64, 64), x, norm)
					got32 = p.IDCTMixed(make([]float32, 64), x32, norm)
				} else {
					got = p.DCT(make([]float64, 64), x, norm)
					got32 = p.DCTMixed(make([]float32, 64), x32, norm)
				}
				if e := errR(got, want); e > b64 {
					t.Errorf("seed %d %v inverse=%v: DCT error %.3g exceeds %.3g", seed, norm, inverse, e, b64)
				}
				if e := errR(got32, want32); e > b32 {
					t.Errorf("seed %d %v inverse=%v: DCTMixed error %.3g exceeds %.3g", seed, norm, inverse, e, b32)
				}
				// Mixed is float32 of the float64 plan on the widened input.
				w64 := widenF32(x32)
				if inverse {
					sameBits32(t, "IDCTMixed", got32, p.IDCT(make([]float64, 64), w64, norm))
				} else {
					sameBits32(t, "DCTMixed", got32, p.DCT(make([]float64, 64), w64, norm))
				}
			}
		}
	}
}

// TestDCT4N64AgainstPooled: the fused kernel and the pooled path (the plan
// before this change, fused off) agree to a few ulps and neither reads past an
// in-place alias.
func TestDCT4N64AgainstPooled(t *testing.T) {
	fused := NewDCTPlan(64, 4)
	pooled := *fused
	pooled.fused = false
	for seed := uint64(1); seed <= 8; seed++ {
		x := accRealF64(64, seed)
		for _, norm := range accNorms {
			a := fused.DCT(make([]float64, 64), x, norm)
			b := pooled.DCT(make([]float64, 64), x, norm)
			var num, den float64
			for i := range a {
				num += (a[i] - b[i]) * (a[i] - b[i])
				den += b[i] * b[i]
			}
			if e := math.Sqrt(num / den); e > 8*eps64 {
				t.Errorf("seed %d %v: fused and pooled differ by %.3g", seed, norm, e)
			}
			buf := append([]float64(nil), x...)
			c := fused.DCT(buf, buf, norm)
			for i := range a {
				if c[i] != a[i] {
					t.Fatalf("seed %d %v: in-place result differs at %d", seed, norm, i)
				}
			}
		}
	}
}

// testDCT4N64Pair: the pair forms give each stream what its own call gives, to
// the oracle bound, in place, and with the cross-overlap fallback.
func testDCT4N64Pair(t *testing.T, p *DCTPlan) {
	b64 := 2 * eps64 * math.Log2(64)
	b32 := b64 + 0x1p-24
	for seed := uint64(1); seed <= 4; seed++ {
		x0, x1 := accRealF64(64, seed), accRealF64(64, seed+100)
		y0, y1 := accRealF32(64, seed), accRealF32(64, seed+100)
		for _, norm := range accNorms {
			for _, inverse := range []bool{false, true} {
				var a, b []float64
				var c, d []float32
				if inverse {
					a, b = p.IDCTPair(make([]float64, 64), make([]float64, 64), x0, x1, norm)
					c, d = p.IDCTMixedPair(make([]float32, 64), make([]float32, 64), y0, y1, norm)
				} else {
					a, b = p.DCTPair(make([]float64, 64), make([]float64, 64), x0, x1, norm)
					c, d = p.DCTMixedPair(make([]float32, 64), make([]float32, 64), y0, y1, norm)
				}
				for i, got := range [][]float64{a, b} {
					x := [][]float64{x0, x1}[i]
					if e := errR(got, oR2R(cddFromR(x), true, 4, norm, inverse)); e > b64 {
						t.Errorf("seed %d %v inverse=%v: pair stream %d error %.3g exceeds %.3g", seed, norm, inverse, i, e, b64)
					}
				}
				for i, got := range [][]float32{c, d} {
					x := [][]float32{y0, y1}[i]
					if e := errR(got, oR2R(cddFromR(x), true, 4, norm, inverse)); e > b32 {
						t.Errorf("seed %d %v inverse=%v: mixed pair stream %d error %.3g exceeds %.3g", seed, norm, inverse, i, e, b32)
					}
				}
				// Mixed pair is float32 of the float64 pair on the widened inputs.
				var w0, w1 []float64
				if inverse {
					w0, w1 = p.IDCTPair(make([]float64, 64), make([]float64, 64), widenF32(y0), widenF32(y1), norm)
				} else {
					w0, w1 = p.DCTPair(make([]float64, 64), make([]float64, 64), widenF32(y0), widenF32(y1), norm)
				}
				sameBits32(t, "mixed pair 0", c, w0)
				sameBits32(t, "mixed pair 1", d, w1)
				// In place.
				b0, b1 := append([]float64(nil), x0...), append([]float64(nil), x1...)
				var e0, e1 []float64
				if inverse {
					e0, e1 = p.IDCTPair(b0, b1, b0, b1, norm)
				} else {
					e0, e1 = p.DCTPair(b0, b1, b0, b1, norm)
				}
				for i := range a {
					if e0[i] != a[i] || e1[i] != b[i] {
						t.Fatalf("seed %d %v: in-place pair differs at %d", seed, norm, i)
					}
				}
			}
		}
		// dst0 over src1: the two calls run in order, as written, so stream 1
		// reads what stream 0 wrote.
		s := append([]float64(nil), x1...)
		want0 := p.DCT(make([]float64, 64), x0, NormOrtho)
		want1 := p.DCT(make([]float64, 64), want0, NormOrtho)
		g0, g1 := p.DCTPair(s, make([]float64, 64), x0, s, NormOrtho)
		for i := range want0 {
			if g0[i] != want0[i] || g1[i] != want1[i] {
				t.Fatalf("seed %d: dst0 over src1 differs from the two calls in order at %d", seed, i)
			}
		}
	}
}

// testDCT4N64PairShortPanics: the fused forms panic like DCT on a short slice, and
// a plan that is not DCT-IV of length 64 runs the two calls.
func testDCT4N64PairShortPanics(t *testing.T, p *DCTPlan) {
	ok := make([]float64, 64)
	ok32 := make([]float32, 64)
	mustPanicWith(t, "short dst", "fft: DCTPlan slice", func() { p.DCT(make([]float64, 63), ok, NormBackward) })
	mustPanicWith(t, "short src", "fft: DCTPlan slice", func() { p.IDCT(ok, make([]float64, 63), NormBackward) })
	mustPanicWith(t, "short mixed dst", "fft: DCTPlan slice", func() { p.DCTMixed(make([]float32, 63), ok32, NormBackward) })
	mustPanicWith(t, "short mixed src", "fft: DCTPlan slice", func() { p.IDCTMixed(ok32, make([]float32, 63), NormBackward) })
	mustPanicWith(t, "short mixed pair", "fft: DCTPlan slice", func() { p.DCTMixedPair(ok32, ok32, make([]float32, 63), ok32, NormBackward) })
	// Mixed pair with dst0 over src1 runs the two calls in order.
	s := append([]float32(nil), ok32...)
	for i := range s {
		s[i] = float32(i)
	}
	want0 := p.DCTMixed(make([]float32, 64), s, NormOrtho)
	want1 := p.DCTMixed(make([]float32, 64), want0, NormOrtho)
	s1 := append([]float32(nil), s...)
	g0, g1 := p.DCTMixedPair(s1, make([]float32, 64), s, s1, NormOrtho)
	for i := range want0 {
		if g0[i] != want0[i] || g1[i] != want1[i] {
			t.Fatalf("mixed pair, dst0 over src1, differs from the two calls in order at %d", i)
		}
	}
	mustPanicWith(t, "short dst1", "fft: DCTPlan slice", func() { p.DCTPair(ok, make([]float64, 63), ok, ok, NormBackward) })
	mustPanicWith(t, "short src0", "fft: DCTPlan slice", func() { p.DCTPair(ok, ok, make([]float64, 63), ok, NormBackward) })
	mustPanicWith(t, "norm", "fft:", func() { p.DCTPair(ok, ok, ok, ok, Norm(99)) })
	for _, n := range []int{32, 128} {
		q := NewDCTPlan(n, 4)
		a, b := accRealF64(n, 1), accRealF64(n, 2)
		g0, g1 := q.DCTPair(make([]float64, n), make([]float64, n), a, b, NormOrtho)
		w0, w1 := q.DCT(make([]float64, n), a, NormOrtho), q.DCT(make([]float64, n), b, NormOrtho)
		for i := range w0 {
			if g0[i] != w0[i] || g1[i] != w1[i] {
				t.Fatalf("n=%d: pair differs from two calls at %d", n, i)
			}
		}
	}
}

// impulseTol is the relative RMS error an impulse may have against the oracle:
// an impulse's output is a few products of constants, so a constant wrong in its
// last few bits shows here (it is lost in the noise of a dense random input).
// Measured worst on the M4 Max: see TestDCT4N64Impulses -v.
const impulseTol = 3 * eps64 / 2

// TestDCT4N64Impulses: every impulse e_j through every kernel, against the
// double-double oracle. Each twiddle constant of the kernel lies on the path of
// some impulse, and an error of 1e-14 in one of them moves that impulse's
// result well past rounding noise.
func TestDCT4N64Impulses(t *testing.T) {
	for pi, p := range fusedPlans() {
		var worst, worst32 float64
		for _, norm := range []Norm{NormBackward, NormOrtho} {
			for j := 0; j < 64; j++ {
				k := (j + 17) % 64
				e0, e1 := make([]float64, 64), make([]float64, 64)
				e0[j], e1[k] = 1, 1
				w0 := oR2R(cddFromR(e0), true, 4, norm, false)
				w1 := oR2R(cddFromR(e1), true, 4, norm, false)
				single := p.DCT(make([]float64, 64), e0, norm)
				a, b := p.DCTPair(make([]float64, 64), make([]float64, 64), e0, e1, norm)
				f0, f1 := make([]float32, 64), make([]float32, 64)
				f0[j], f1[k] = 1, 1
				m0 := p.DCTMixed(make([]float32, 64), f0, norm)
				c, d := p.DCTMixedPair(make([]float32, 64), make([]float32, 64), f0, f1, norm)
				for _, e := range []float64{errR(single, w0), errR(a, w0), errR(b, w1)} {
					worst = math.Max(worst, e)
				}
				for _, e := range []float64{errR(m0, w0), errR(c, w0), errR(d, w1)} {
					worst32 = math.Max(worst32, e)
				}
			}
		}
		t.Logf("plan %d: worst impulse error float64 %.3g, Mixed %.3g (bound %.3g, float32 half ulp %.3g)", pi, worst, worst32, impulseTol, 0x1p-24)
		if worst > impulseTol {
			t.Errorf("plan %d: impulse error %.3g exceeds %.3g", pi, worst, impulseTol)
		}
		if worst32 > 0x1p-24 {
			t.Errorf("plan %d: Mixed impulse error %.3g exceeds half a float32 ulp", pi, worst32)
		}
	}
}

// TestDCT4N64MixedIsFloat32OfFloat64: on 10 000 random vectors, in each norm and
// direction, the Mixed result is bit for bit float32 of the float64 result on
// the widened input, for the single and the pair forms. The same instruction
// list runs both, so any difference is a defect in one kernel.
func TestDCT4N64MixedIsFloat32OfFloat64(t *testing.T) {
	for _, p := range fusedPlans() {
		for seed := uint64(1); seed <= 10000; seed++ {
			x, y := accRealF32(64, seed), accRealF32(64, seed+1<<20)
			wx, wy := widenF32(x), widenF32(y)
			norm, inverse := accNorms[seed%3], seed%2 == 1
			var m, m0, m1 []float32
			var w, w0, w1 []float64
			if inverse {
				m, w = p.IDCTMixed(make([]float32, 64), x, norm), p.IDCT(make([]float64, 64), wx, norm)
				m0, m1 = p.IDCTMixedPair(make([]float32, 64), make([]float32, 64), x, y, norm)
				w0, w1 = p.IDCTPair(make([]float64, 64), make([]float64, 64), wx, wy, norm)
			} else {
				m, w = p.DCTMixed(make([]float32, 64), x, norm), p.DCT(make([]float64, 64), wx, norm)
				m0, m1 = p.DCTMixedPair(make([]float32, 64), make([]float32, 64), x, y, norm)
				w0, w1 = p.DCTPair(make([]float64, 64), make([]float64, 64), wx, wy, norm)
			}
			sameBits32(t, "Mixed", m, w)
			sameBits32(t, "MixedPair 0", m0, w0)
			sameBits32(t, "MixedPair 1", m1, w1)
		}
	}
}

// TestDCT4N64PairValidatesBeforeWriting: a pair call with a short slice, an
// unknown norm or a plan of another type panics with every destination
// untouched.
func TestDCT4N64PairValidatesBeforeWriting(t *testing.T) {
	for _, p := range fusedPlans() {
		x := accRealF64(64, 3)
		x32 := accRealF32(64, 3)
		fill := func() ([]float64, []float64, []float32, []float32) {
			a, b := make([]float64, 64), make([]float64, 64)
			c, d := make([]float32, 64), make([]float32, 64)
			for i := range a {
				a[i], b[i], c[i], d[i] = -1, -1, -1, -1
			}
			return a, b, c, d
		}
		untouched := func(what string, a, b []float64, c, d []float32) {
			t.Helper()
			for i := range a {
				if a[i] != -1 || b[i] != -1 || c[i] != -1 || d[i] != -1 {
					t.Fatalf("%s wrote a destination before panicking (index %d)", what, i)
				}
			}
		}
		a, b, c, d := fill()
		mustPanicWith(t, "short dst1", "fft: DCTPlan slice", func() { p.DCTPair(a, b[:63], x, x, NormBackward) })
		untouched("short dst1", a, b, c, d)
		mustPanicWith(t, "short src1", "fft: DCTPlan slice", func() { p.IDCTPair(a, b, x, x[:63], NormBackward) })
		untouched("short src1", a, b, c, d)
		mustPanicWith(t, "short mixed src1", "fft: DCTPlan slice", func() { p.DCTMixedPair(c, d, x32, x32[:63], NormBackward) })
		untouched("short mixed src1", a, b, c, d)
		mustPanicWith(t, "mixed dst1", "fft: DCTPlan slice", func() { p.IDCTMixedPair(c, d[:63], x32, x32, NormBackward) })
		untouched("mixed dst1", a, b, c, d)
		mustPanicWith(t, "norm", "fft:", func() { p.DCTPair(a, b, x, x, Norm(99)) })
		untouched("norm", a, b, c, d)
		mustPanicWith(t, "mixed norm", "fft:", func() { p.DCTMixedPair(c, d, x32, x32, Norm(99)) })
		untouched("mixed norm", a, b, c, d)
	}
	mustPanicWith(t, "mixed pair, DCT-II plan", "fft: DCTPlan.DCTMixed needs a DCT-IV", func() {
		NewDCTPlan(64, 2).DCTMixedPair(make([]float32, 64), make([]float32, 64), make([]float32, 64), make([]float32, 64), NormBackward)
	})
}
