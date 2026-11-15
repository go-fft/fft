package fft

import (
	"math"
	"testing"
)

// The fused DCT-IV kernels (lengths 64 and 128) are held to the #121 contract:
// each result within c·ε·log2 n of a double-double oracle (c = 2, the r2r family's
// constant in accBound), the Mixed one plus half a float32 ulp, and every route of
// the plan (DCT, IDCT, DCTMixed, IDCTMixed, the pair forms, in place) to the same
// arithmetic.

var fusedLens = []int{64, 128}

// fusedPlans returns the plan the package builds and one forced onto the fused
// kernels, which off arm64 means the scalar Go ones: every platform runs every
// line of the fused path.
func fusedPlans(n int) []*DCTPlan {
	forced := *NewDCTPlan(n, 4)
	forced.fused, forced.pair = true, true
	return []*DCTPlan{NewDCTPlan(n, 4), &forced}
}

// withoutPairAsm runs f with the amd64 pair kernels switched off.
func withoutPairAsm(f func()) {
	old := dct4PairAsmSwitch(false)
	defer dct4PairAsmSwitch(old)
	f()
}

// forEachFused runs f on every fused length and plan.
func forEachFused(t *testing.T, f func(t *testing.T, n int, p *DCTPlan)) {
	for _, n := range fusedLens {
		for _, p := range fusedPlans(n) {
			f(t, n, p)
		}
	}
}

func TestDCT4FusedOracle(t *testing.T) { forEachFused(t, testDCT4FusedOracle) }

func testDCT4FusedOracle(t *testing.T, n int, p *DCTPlan) {
	b64 := 2 * eps64 * math.Log2(float64(n))
	b32 := b64 + 0x1p-24
	for seed := uint64(1); seed <= 4; seed++ {
		x := accRealF64(n, seed)
		x32 := accRealF32(n, seed)
		for _, norm := range accNorms {
			for _, inverse := range []bool{false, true} {
				want := oR2R(cddFromR(x), true, 4, norm, inverse)
				want32 := oR2R(cddFromR(x32), true, 4, norm, inverse)
				var got []float64
				var got32 []float32
				if inverse {
					got = p.IDCT(make([]float64, n), x, norm)
					got32 = p.IDCTMixed(make([]float32, n), x32, norm)
				} else {
					got = p.DCT(make([]float64, n), x, norm)
					got32 = p.DCTMixed(make([]float32, n), x32, norm)
				}
				if e := errR(got, want); e > b64 {
					t.Errorf("n=%d seed %d %v inverse=%v: DCT error %.3g exceeds %.3g", n, seed, norm, inverse, e, b64)
				}
				if e := errR(got32, want32); e > b32 {
					t.Errorf("n=%d seed %d %v inverse=%v: DCTMixed error %.3g exceeds %.3g", n, seed, norm, inverse, e, b32)
				}
				// Mixed is float32 of the float64 plan on the widened input.
				w64 := widenF32(x32)
				if inverse {
					sameBits32(t, "IDCTMixed", got32, p.IDCT(make([]float64, n), w64, norm))
				} else {
					sameBits32(t, "DCTMixed", got32, p.DCT(make([]float64, n), w64, norm))
				}
			}
		}
	}
}

// TestDCT4FusedAgainstPooled: the fused kernel and the pooled path (the plan
// before the fused kernels, fused off) agree to a few ulps, and an in-place call
// gives the same bits.
func TestDCT4FusedAgainstPooled(t *testing.T) {
	for _, n := range fusedLens {
		fused := NewDCTPlan(n, 4)
		fused.fused = true // on amd64 the plan does not route single calls to the kernel
		pooled := *fused
		pooled.fused = false
		for seed := uint64(1); seed <= 8; seed++ {
			x := accRealF64(n, seed)
			for _, norm := range accNorms {
				a := fused.DCT(make([]float64, n), x, norm)
				b := pooled.DCT(make([]float64, n), x, norm)
				var num, den float64
				for i := range a {
					num += (a[i] - b[i]) * (a[i] - b[i])
					den += b[i] * b[i]
				}
				if e := math.Sqrt(num / den); e > 8*eps64 {
					t.Errorf("n=%d seed %d %v: fused and pooled differ by %.3g", n, seed, norm, e)
				}
				buf := append([]float64(nil), x...)
				c := fused.DCT(buf, buf, norm)
				for i := range a {
					if c[i] != a[i] {
						t.Fatalf("n=%d seed %d %v: in-place result differs at %d", n, seed, norm, i)
					}
				}
			}
		}
	}
}

func TestDCT4FusedPair(t *testing.T) { forEachFused(t, testDCT4FusedPair) }

// testDCT4FusedPair: the pair forms give each stream what its own call gives, to
// the oracle bound, in place, and with the cross-overlap fallback.
func testDCT4FusedPair(t *testing.T, n int, p *DCTPlan) {
	b64 := 2 * eps64 * math.Log2(float64(n))
	b32 := b64 + 0x1p-24
	for seed := uint64(1); seed <= 4; seed++ {
		x0, x1 := accRealF64(n, seed), accRealF64(n, seed+100)
		y0, y1 := accRealF32(n, seed), accRealF32(n, seed+100)
		for _, norm := range accNorms {
			for _, inverse := range []bool{false, true} {
				var a, b []float64
				var c, d []float32
				if inverse {
					a, b = p.IDCTPair(make([]float64, n), make([]float64, n), x0, x1, norm)
					c, d = p.IDCTMixedPair(make([]float32, n), make([]float32, n), y0, y1, norm)
				} else {
					a, b = p.DCTPair(make([]float64, n), make([]float64, n), x0, x1, norm)
					c, d = p.DCTMixedPair(make([]float32, n), make([]float32, n), y0, y1, norm)
				}
				for i, got := range [][]float64{a, b} {
					x := [][]float64{x0, x1}[i]
					if e := errR(got, oR2R(cddFromR(x), true, 4, norm, inverse)); e > b64 {
						t.Errorf("n=%d seed %d %v inverse=%v: pair stream %d error %.3g exceeds %.3g", n, seed, norm, inverse, i, e, b64)
					}
				}
				for i, got := range [][]float32{c, d} {
					x := [][]float32{y0, y1}[i]
					if e := errR(got, oR2R(cddFromR(x), true, 4, norm, inverse)); e > b32 {
						t.Errorf("n=%d seed %d %v inverse=%v: mixed pair stream %d error %.3g exceeds %.3g", n, seed, norm, inverse, i, e, b32)
					}
				}
				// Mixed pair is float32 of the float64 pair on the widened inputs.
				var w0, w1 []float64
				if inverse {
					w0, w1 = p.IDCTPair(make([]float64, n), make([]float64, n), widenF32(y0), widenF32(y1), norm)
				} else {
					w0, w1 = p.DCTPair(make([]float64, n), make([]float64, n), widenF32(y0), widenF32(y1), norm)
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
						t.Fatalf("n=%d seed %d %v: in-place pair differs at %d", n, seed, norm, i)
					}
				}
			}
		}
		// dst0 over src1: the two calls run in order, as written, so stream 1
		// reads what stream 0 wrote.
		s := append([]float64(nil), x1...)
		want0 := p.DCT(make([]float64, n), x0, NormOrtho)
		want1 := p.DCT(make([]float64, n), want0, NormOrtho)
		g0, g1 := p.DCTPair(s, make([]float64, n), x0, s, NormOrtho)
		for i := range want0 {
			if g0[i] != want0[i] || g1[i] != want1[i] {
				t.Fatalf("n=%d seed %d: dst0 over src1 differs from the two calls in order at %d", n, seed, i)
			}
		}
	}
}

func TestDCT4FusedShortPanics(t *testing.T) { forEachFused(t, testDCT4FusedShortPanics) }

// testDCT4FusedShortPanics: the fused forms panic like DCT on a short slice, and a
// plan that is not a fused length runs the two calls.
func testDCT4FusedShortPanics(t *testing.T, n int, p *DCTPlan) {
	ok := make([]float64, n)
	ok32 := make([]float32, n)
	mustPanicWith(t, "short dst", "fft: DCTPlan slice", func() { p.DCT(make([]float64, n-1), ok, NormBackward) })
	mustPanicWith(t, "short src", "fft: DCTPlan slice", func() { p.IDCT(ok, make([]float64, n-1), NormBackward) })
	mustPanicWith(t, "short mixed dst", "fft: DCTPlan slice", func() { p.DCTMixed(make([]float32, n-1), ok32, NormBackward) })
	mustPanicWith(t, "short mixed src", "fft: DCTPlan slice", func() { p.IDCTMixed(ok32, make([]float32, n-1), NormBackward) })
	mustPanicWith(t, "short mixed pair", "fft: DCTPlan slice", func() { p.DCTMixedPair(ok32, ok32, make([]float32, n-1), ok32, NormBackward) })
	// Mixed pair with dst0 over src1 runs the two calls in order.
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(i)
	}
	want0 := p.DCTMixed(make([]float32, n), s, NormOrtho)
	want1 := p.DCTMixed(make([]float32, n), want0, NormOrtho)
	s1 := append([]float32(nil), s...)
	g0, g1 := p.DCTMixedPair(s1, make([]float32, n), s, s1, NormOrtho)
	for i := range want0 {
		if g0[i] != want0[i] || g1[i] != want1[i] {
			t.Fatalf("n=%d: mixed pair, dst0 over src1, differs from the two calls in order at %d", n, i)
		}
	}
	mustPanicWith(t, "short dst1", "fft: DCTPlan slice", func() { p.DCTPair(ok, make([]float64, n-1), ok, ok, NormBackward) })
	mustPanicWith(t, "short src0", "fft: DCTPlan slice", func() { p.DCTPair(ok, ok, make([]float64, n-1), ok, NormBackward) })
	mustPanicWith(t, "norm", "fft:", func() { p.DCTPair(ok, ok, ok, ok, Norm(99)) })
	for _, m := range []int{32, 96, 256} {
		q := NewDCTPlan(m, 4)
		a, b := accRealF64(m, 1), accRealF64(m, 2)
		g0, g1 := q.DCTPair(make([]float64, m), make([]float64, m), a, b, NormOrtho)
		w0, w1 := q.DCT(make([]float64, m), a, NormOrtho), q.DCT(make([]float64, m), b, NormOrtho)
		for i := range w0 {
			if g0[i] != w0[i] || g1[i] != w1[i] {
				t.Fatalf("n=%d: pair differs from two calls at %d", m, i)
			}
		}
	}
}

// impulseTol is the relative RMS error an impulse may have against the oracle: an
// impulse's output is a few products of constants, so a constant wrong in its last
// few bits shows here (it is lost in the noise of a dense random input). Measured
// worst at 64: 2.2e-16 with fused multiply-adds (arm64), 3.5e-16 without (amd64 CI,
// riscv64); 5e-16 leaves 1.4x over the latter. The length 128 scales it by
// log2(n)/6 (5.8e-16). A 1.5e-14 constant error gives 1.4e-15 here at 128 (a
// post-twiddle constant of the scalar kernel, 2.4x the bound) and 1.5e-14 (a
// NEON row), so this check is the weaker one; impulseAbsTol below is the sharp
// one.
func impulseTol(n int) float64 { return 5e-16 * math.Log2(float64(n)) / 6 }

// impulseAbsTol bounds the largest absolute error of any output of any impulse (the
// outputs are cosines of magnitude up to 2). Measured worst: 7.2e-16 and 8.5e-16
// (arm64, lengths 64 and 128), 1.4e-15 and 1.7e-15 on the amd64 path without fused
// multiply-adds; 3e-15 leaves 1.8x over that. A relative error of 1.5e-14 in one
// constant at length 128 gave 2.2e-14 (a post-twiddle constant of the scalar
// kernel, 7x the bound) and 2.9e-14 (a NEON table row, 10x), where the relative
// RMS bound below sees them at 2.4x and 25x.
const impulseAbsTol = 3e-15

// TestDCT4FusedImpulses: every impulse e_j through every kernel, against the
// double-double oracle. Each twiddle constant of the kernel lies on the path of
// some impulse, and an error of 1e-14 in one of them moves that impulse's result
// well past rounding noise.
func TestDCT4FusedImpulses(t *testing.T) {
	for _, n := range fusedLens {
		for pi, p := range fusedPlans(n) {
			var worst, worst32, worstAbs float64
			for _, norm := range []Norm{NormBackward, NormOrtho} {
				for j := 0; j < n; j++ {
					k := (j + 17) % n
					e0, e1 := make([]float64, n), make([]float64, n)
					e0[j], e1[k] = 1, 1
					w0 := oR2R(cddFromR(e0), true, 4, norm, false)
					w1 := oR2R(cddFromR(e1), true, 4, norm, false)
					single := p.DCT(make([]float64, n), e0, norm)
					a, b := p.DCTPair(make([]float64, n), make([]float64, n), e0, e1, norm)
					f0, f1 := make([]float32, n), make([]float32, n)
					f0[j], f1[k] = 1, 1
					m0 := p.DCTMixed(make([]float32, n), f0, norm)
					c, d := p.DCTMixedPair(make([]float32, n), make([]float32, n), f0, f1, norm)
					for _, e := range []float64{errR(single, w0), errR(a, w0), errR(b, w1)} {
						worst = math.Max(worst, e)
					}
					for _, e := range []float64{maxAbsErr(single, w0), maxAbsErr(a, w0), maxAbsErr(b, w1)} {
						worstAbs = math.Max(worstAbs, e)
					}
					for _, e := range []float64{errR(m0, w0), errR(c, w0), errR(d, w1)} {
						worst32 = math.Max(worst32, e)
					}
				}
			}
			t.Logf("n=%d plan %d: worst impulse error float64 %.3g, Mixed %.3g (bound %.3g, float32 half ulp %.3g)", n, pi, worst, worst32, impulseTol(n), 0x1p-24)
			t.Logf("n=%d plan %d: worst absolute error %.3g (bound %.3g)", n, pi, worstAbs, impulseAbsTol)
			if worstAbs > impulseAbsTol {
				t.Errorf("n=%d plan %d: impulse absolute error %.3g exceeds %.3g", n, pi, worstAbs, impulseAbsTol)
			}
			if worst > impulseTol(n) {
				t.Errorf("n=%d plan %d: impulse error %.3g exceeds %.3g", n, pi, worst, impulseTol(n))
			}
			if worst32 > 0x1p-24 {
				t.Errorf("n=%d plan %d: Mixed impulse error %.3g exceeds half a float32 ulp", n, pi, worst32)
			}
		}
	}
}

// TestDCT4FusedMixedIsFloat32OfFloat64: on 10 000 random vectors, in each norm and
// direction, the Mixed result is bit for bit float32 of the float64 result on the
// widened input, for the single and the pair forms. The same instruction list runs
// both, so any difference is a defect in one kernel.
func TestDCT4FusedMixedIsFloat32OfFloat64(t *testing.T) {
	for _, n := range fusedLens {
		for _, p := range fusedPlans(n) {
			for seed := uint64(1); seed <= 10000; seed++ {
				x, y := accRealF32(n, seed), accRealF32(n, seed+1<<20)
				wx, wy := widenF32(x), widenF32(y)
				norm, inverse := accNorms[seed%3], seed%2 == 1
				var m, m0, m1 []float32
				var w, w0, w1 []float64
				if inverse {
					m, w = p.IDCTMixed(make([]float32, n), x, norm), p.IDCT(make([]float64, n), wx, norm)
					m0, m1 = p.IDCTMixedPair(make([]float32, n), make([]float32, n), x, y, norm)
					w0, w1 = p.IDCTPair(make([]float64, n), make([]float64, n), wx, wy, norm)
				} else {
					m, w = p.DCTMixed(make([]float32, n), x, norm), p.DCT(make([]float64, n), wx, norm)
					m0, m1 = p.DCTMixedPair(make([]float32, n), make([]float32, n), x, y, norm)
					w0, w1 = p.DCTPair(make([]float64, n), make([]float64, n), wx, wy, norm)
				}
				sameBits32(t, "Mixed", m, w)
				sameBits32(t, "MixedPair 0", m0, w0)
				sameBits32(t, "MixedPair 1", m1, w1)
			}
		}
	}
}

// TestDCT4FusedPairValidatesBeforeWriting: a pair call with a short slice, an
// unknown norm or a plan of another type panics with every destination untouched.
func TestDCT4FusedPairValidatesBeforeWriting(t *testing.T) {
	for _, n := range fusedLens {
		for _, p := range fusedPlans(n) {
			x := accRealF64(n, 3)
			x32 := accRealF32(n, 3)
			fill := func() ([]float64, []float64, []float32, []float32) {
				a, b := make([]float64, n), make([]float64, n)
				c, d := make([]float32, n), make([]float32, n)
				for i := range a {
					a[i], b[i], c[i], d[i] = -1, -1, -1, -1
				}
				return a, b, c, d
			}
			untouched := func(what string, a, b []float64, c, d []float32) {
				t.Helper()
				for i := range a {
					if a[i] != -1 || b[i] != -1 || c[i] != -1 || d[i] != -1 {
						t.Fatalf("n=%d %s wrote a destination before panicking (index %d)", n, what, i)
					}
				}
			}
			a, b, c, d := fill()
			mustPanicWith(t, "short dst1", "fft: DCTPlan slice", func() { p.DCTPair(a, b[:n-1], x, x, NormBackward) })
			untouched("short dst1", a, b, c, d)
			mustPanicWith(t, "short src1", "fft: DCTPlan slice", func() { p.IDCTPair(a, b, x, x[:n-1], NormBackward) })
			untouched("short src1", a, b, c, d)
			mustPanicWith(t, "short mixed src1", "fft: DCTPlan slice", func() { p.DCTMixedPair(c, d, x32, x32[:n-1], NormBackward) })
			untouched("short mixed src1", a, b, c, d)
			mustPanicWith(t, "mixed dst1", "fft: DCTPlan slice", func() { p.IDCTMixedPair(c, d[:n-1], x32, x32, NormBackward) })
			untouched("mixed dst1", a, b, c, d)
			mustPanicWith(t, "norm", "fft:", func() { p.DCTPair(a, b, x, x, Norm(99)) })
			untouched("norm", a, b, c, d)
			mustPanicWith(t, "mixed norm", "fft:", func() { p.DCTMixedPair(c, d, x32, x32, Norm(99)) })
			untouched("mixed norm", a, b, c, d)
		}
	}
	mustPanicWith(t, "mixed pair, DCT-II plan", "fft: DCTPlan.DCTMixed needs a DCT-IV", func() {
		NewDCTPlan(64, 2).DCTMixedPair(make([]float32, 64), make([]float32, 64), make([]float32, 64), make([]float32, 64), NormBackward)
	})
}

// TestMDCTFused: the MDCT and IMDCT of N = 128 and 256 on the fused DCT-IV kernel
// (forced, so every platform runs it) agree with the unfused plan to a few ulps,
// and the Mixed forms are bit for bit float32 of the float64 ones, on 10 000
// vectors per length.
func TestMDCTFused(t *testing.T) {
	for _, n := range []int{128, 256} {
		fused, plain := newMDCTPlan(n, true), newMDCTPlan(n, false)
		if !fused.fused || plain.fused || fused.k != nil || plain.k == nil {
			t.Fatalf("N=%d: plans not built as asked: fused %v/%v, kernels %v/%v", n, fused.fused, plain.fused, fused.k != nil, plain.k != nil)
		}
		m := n / 2
		for seed := uint64(1); seed <= 10000; seed++ {
			x, c := accRealF32(n, seed), accRealF32(m, seed+1<<20)
			wx, wc := widenF32(x), widenF32(c)
			scale := []float64{1, 0.5, 2 / float64(m)}[seed%3]
			a := fused.MDCT(make([]float64, m), wx, scale)
			b := plain.MDCT(make([]float64, m), wx, scale)
			ia := fused.IMDCT(make([]float64, n), wc, scale)
			ib := plain.IMDCT(make([]float64, n), wc, scale)
			for _, pr := range [][2][]float64{{a, b}, {ia, ib}} {
				var num, den float64
				for i := range pr[0] {
					num += (pr[0][i] - pr[1][i]) * (pr[0][i] - pr[1][i])
					den += pr[1][i] * pr[1][i]
				}
				if e := math.Sqrt(num / den); e > 8*eps64 {
					t.Fatalf("N=%d seed %d: fused and unfused differ by %.3g", n, seed, e)
				}
			}
			sameBits32(t, "MDCTMixed", fused.MDCTMixed(make([]float32, m), x, scale), a)
			sameBits32(t, "IMDCTMixed", fused.IMDCTMixed(make([]float32, n), c, scale), ia)
		}
		// Every impulse against the oracle: a wrong constant shows here.
		for j := 0; j < m; j++ {
			e := make([]float64, m)
			e[j] = 1
			got := fused.IMDCT(make([]float64, n), e, 1)
			want := plain.IMDCT(make([]float64, n), e, 1)
			var num, den float64
			for i := range got {
				num += (got[i] - want[i]) * (got[i] - want[i])
				den += want[i] * want[i]
			}
			if r := math.Sqrt(num / den); r > 4*eps64 {
				t.Fatalf("N=%d impulse %d: fused IMDCT differs from unfused by %.3g", n, j, r)
			}
		}
		// The in-place forms: dst over src, as the fold and unfold are read before
		// the write.
		x, c := accRealF64(n, 91), accRealF64(m, 92)
		wantM := fused.MDCT(make([]float64, m), x, 0.5)
		bx := append([]float64(nil), x...)
		if got := fused.MDCT(bx[:m], bx, 0.5); !equalF(got, wantM) {
			t.Errorf("N=%d: fused MDCT into the start of its src differs", n)
		}
		wantI := fused.IMDCT(make([]float64, n), c, 0.5)
		bc := append(append([]float64(nil), c...), make([]float64, n-m)...)
		if got := fused.IMDCT(bc, bc, 0.5); !equalF(got, wantI) {
			t.Errorf("N=%d: fused IMDCT over its src differs", n)
		}
		mustPanicWith(t, "short dst", "fft: MDCTPlan slice", func() { fused.MDCT(make([]float64, m-1), make([]float64, n), 1) })
		mustPanicWith(t, "short idst", "fft: MDCTPlan slice", func() { fused.IMDCT(make([]float64, n-1), make([]float64, m), 1) })
	}
}

// maxAbsErr is the largest absolute difference between got and the oracle's real
// parts: unlike the relative RMS it is not diluted by the outputs a wrong constant
// does not touch.
func maxAbsErr(got []float64, want []cdd) float64 {
	var m float64
	for i, g := range got {
		m = math.Max(m, math.Abs((g-want[i].re.hi)-want[i].re.lo))
	}
	return m
}
