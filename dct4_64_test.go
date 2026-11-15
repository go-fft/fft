package fft

import (
	"math"
	"testing"
)

// The fused DCT-IV(64) is held to the #121 contract: each result within
// c·ε·log2 n of a double-double oracle (c = 2, the r2r family's constant in
// accBound), the Mixed one plus half a float32 ulp, and every route of the
// plan (DCT, IDCT, DCTMixed, IDCTMixed, in place) to the same arithmetic.

func TestDCT4N64Oracle(t *testing.T) {
	p := NewDCTPlan(64, 4)
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

// TestDCT4N64Pair: the pair forms give each stream what its own call gives, to
// the oracle bound, in place, and with the cross-overlap fallback.
func TestDCT4N64Pair(t *testing.T) {
	p := NewDCTPlan(64, 4)
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

// TestDCT4N64PairShortPanics: the pair forms panic like DCT on a short slice, and
// a plan that is not DCT-IV of length 64 runs the two calls.
func TestDCT4N64PairShortPanics(t *testing.T) {
	p := NewDCTPlan(64, 4)
	ok := make([]float64, 64)
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
