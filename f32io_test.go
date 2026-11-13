package fft

import (
	"math"
	"testing"
)

func widenF32(x []float32) []float64 {
	y := make([]float64, len(x))
	for i, v := range x {
		y[i] = float64(v)
	}
	return y
}

func sameBits32(t *testing.T, what string, got []float32, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", what, len(got), len(want))
	}
	for i := range got {
		if w := float32(want[i]); math.Float32bits(got[i]) != math.Float32bits(w) {
			t.Fatalf("%s: [%d] = %g, want float32(%g) = %g", what, i, got[i], want[i], w)
		}
	}
}

// TestMixedIsFloat32OfFloat64: the mixed methods run the float64 arithmetic, so
// their output is exactly the float64 methods' output rounded once to float32,
// not merely close, on both MDCT routes and on even and odd DCT-IV lengths, for
// every Norm.
func TestMixedIsFloat32OfFloat64(t *testing.T) {
	for _, n := range []int{2, 4, 6, 8, 10, 12, 16, 30, 64, 100, 128, 258, 256} {
		m := n / 2
		p := NewMDCTPlan(n)
		x := accRealF32(n, 31)
		c := accRealF32(m, 32)
		for _, scale := range []float64{1, 2 / float64(m), 0.37} {
			sameBits32(t, "MDCTMixed", p.MDCTMixed(make([]float32, m), x, scale), p.MDCT(make([]float64, m), widenF32(x), scale))
			sameBits32(t, "IMDCTMixed", p.IMDCTMixed(make([]float32, n), c, scale), p.IMDCT(make([]float64, n), widenF32(c), scale))
		}
	}
	for _, n := range []int{1, 2, 3, 4, 5, 9, 16, 63, 64, 128, 1024} {
		p := NewDCTPlan(n, 4)
		x := accRealF32(n, 33)
		for _, norm := range []Norm{NormBackward, NormOrtho, NormForward} {
			sameBits32(t, "DCTMixed", p.DCTMixed(make([]float32, n), x, norm), p.DCT(make([]float64, n), widenF32(x), norm))
			sameBits32(t, "IDCTMixed", p.IDCTMixed(make([]float32, n), x, norm), p.IDCT(make([]float64, n), widenF32(x), norm))
		}
	}
}

// TestMixedAccuracy: against the double-double definition the error is the
// float64 bound plus one rounding to float32, c64·ε64·log2 N + ε32/2 in relative
// RMS (docs/accuracy.md), and the float32 plan, run on the same inputs, is
// reported beside it. The mixed error is also at most the bound with ε32/2 alone
// for the rounding, so it is never worse than the output's own precision allows.
func TestMixedAccuracy(t *testing.T) {
	const eps64, halfEps32 = 0x1p-52, 0x1p-24
	var worstM, worstD, worst32 float64
	for _, n := range mdctLens() {
		m := n / 2
		lg := math.Max(1, math.Log2(float64(n)))
		bound := mdctC64*eps64*lg + halfEps32
		p, p32 := NewMDCTPlan(n), NewMDCTPlan32(n)
		x, c := accRealF32(n, 41), accRealF32(m, 42)
		wF := oMDCT(cddFromR(x), m, false)
		wI := oMDCT(cddFromR(c), m, true)
		eF := errR(p.MDCTMixed(make([]float32, m), x, 1), wF)
		eI := errR(p.IMDCTMixed(make([]float32, n), c, 1), wI)
		worst32 = math.Max(worst32, math.Max(errR(NewMDCTPlan32(n).MDCT(make([]float32, m), x, 1), wF), errR(p32.IMDCT(make([]float32, n), c, 1), wI)))
		worstM = math.Max(worstM, math.Max(eF, eI)/bound)
		if eF > bound || eI > bound {
			t.Errorf("N=%d: MDCTMixed %.3g, IMDCTMixed %.3g exceed %.3g", n, eF, eI, bound)
		}
	}
	for _, n := range accR2RLengths() {
		if n > 1100 {
			continue
		}
		lg := math.Max(1, math.Log2(float64(n)))
		bound := 2.0*eps64*lg + halfEps32 // r2r c = 2.0 in docs/accuracy.md
		p := NewDCTPlan(n, 4)
		x := accRealF32(n, 43)
		for _, norm := range []Norm{NormBackward, NormOrtho, NormForward} {
			for _, inv := range []bool{false, true} {
				want := oR2R(cddFromR(x), true, 4, norm, inv)
				var got []float32
				if inv {
					got = p.IDCTMixed(make([]float32, n), x, norm)
				} else {
					got = p.DCTMixed(make([]float32, n), x, norm)
				}
				e := errR(got, want)
				worstD = math.Max(worstD, e/bound)
				if e > bound {
					t.Errorf("DCT-IV n=%d norm=%v inverse=%v: error %.3g exceeds %.3g", n, norm, inv, e, bound)
				}
			}
		}
	}
	t.Logf("worst error / bound: MDCT %.3f, DCT-IV %.3f; the float32 MDCT plan's worst error is %.3g (the float32 rounding alone is %.3g)", worstM, worstD, worst32, halfEps32)
}

func TestMixedPanicsAndAliasing(t *testing.T) {
	mustPanicWith(t, "DCTMixed type 2", "fft: DCTPlan.DCTMixed needs a DCT-IV", func() {
		NewDCTPlan(8, 2).DCTMixed(make([]float32, 8), make([]float32, 8), NormBackward)
	})
	p := NewDCTPlan(8, 4)
	mustPanicWith(t, "DCTMixed short dst", "fft: DCTPlan slice", func() { p.DCTMixed(make([]float32, 7), make([]float32, 8), NormBackward) })
	mustPanicWith(t, "IDCTMixed short src", "fft: DCTPlan slice", func() { p.IDCTMixed(make([]float32, 8), make([]float32, 7), NormBackward) })
	mustPanicWith(t, "DCTMixed norm", "fft:", func() { p.DCTMixed(make([]float32, 8), make([]float32, 8), Norm(99)) })
	mp := NewMDCTPlan(8)
	mustPanicWith(t, "MDCTMixed short", "fft: MDCTPlan", func() { mp.MDCTMixed(make([]float32, 3), make([]float32, 8), 1) })
	mustPanicWith(t, "IMDCTMixed short", "fft: MDCTPlan", func() { mp.IMDCTMixed(make([]float32, 8), make([]float32, 3), 1) })

	// dst may overlap src
	for _, n := range []int{8, 9} {
		d := NewDCTPlan(n, 4)
		x := accRealF32(n, 51)
		want := d.DCTMixed(make([]float32, n), x, NormOrtho)
		buf := append([]float32(nil), x...)
		got := d.DCTMixed(buf, buf, NormOrtho)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("n=%d: aliased DCTMixed differs at %d", n, i)
			}
		}
	}
	for _, n := range []int{8, 12} {
		m := n / 2
		q := NewMDCTPlan(n)
		c := accRealF32(m, 52)
		want := q.IMDCTMixed(make([]float32, n), c, 1)
		buf := append(append([]float32(nil), c...), make([]float32, n-m)...)
		got := q.IMDCTMixed(buf, buf, 1)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("N=%d: aliased IMDCTMixed differs at %d", n, i)
			}
		}
	}
}

func TestMixedAllocs(t *testing.T) {
	if raceEnabled {
		t.Skip("under -race sync.Pool drops items on purpose, so allocation counts are not meaningful")
	}
	for _, n := range []int{2048, 2050} {
		mp := NewMDCTPlan(n)
		x, c, y := make([]float32, n), make([]float32, n/2), make([]float32, n)
		mp.MDCTMixed(c, x, 1)
		mp.IMDCTMixed(y, c, 1)
		for name, f := range map[string]func(){
			"MDCTMixed":  func() { mp.MDCTMixed(c, x, 1) },
			"IMDCTMixed": func() { mp.IMDCTMixed(y, c, 1) },
		} {
			if got := testing.AllocsPerRun(50, f); got != 0 {
				t.Errorf("N=%d %s: %v allocations per call, want 0", n, name, got)
			}
		}
	}
	for _, n := range []int{1024, 1023} {
		d := NewDCTPlan(n, 4)
		x, y := make([]float32, n), make([]float32, n)
		d.DCTMixed(y, x, NormOrtho)
		for name, f := range map[string]func(){
			"DCTMixed":  func() { d.DCTMixed(y, x, NormOrtho) },
			"IDCTMixed": func() { d.IDCTMixed(y, x, NormOrtho) },
		} {
			if got := testing.AllocsPerRun(50, f); got != 0 {
				t.Errorf("n=%d %s: %v allocations per call, want 0", n, name, got)
			}
		}
	}
}

// BenchmarkMixed: the mixed method against the float64 plan with two manual
// conversion passes, and against the float32 plan (float32 arithmetic).
func BenchmarkMixed(b *testing.B) {
	for _, m := range []int{128, 1024} {
		name := map[int]string{128: "128", 1024: "1024"}[m]
		x, y := make([]float32, m), make([]float32, m)
		for i := range x {
			x[i] = float32(i%37 - 18)
		}
		x64, y64 := make([]float64, m), make([]float64, m)
		d, d32 := NewDCTPlan(m, 4), NewDCTPlan32(m, 4)
		b.Run("DCT4/mixed/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				d.DCTMixed(y, x, NormBackward)
			}
		})
		b.Run("DCT4/f64+convert/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				for j, v := range x {
					x64[j] = float64(v)
				}
				d.DCT(y64, x64, NormBackward)
				for j, v := range y64 {
					y[j] = float32(v)
				}
			}
		})
		b.Run("DCT4/f32plan/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				d32.DCT(y, x, NormBackward)
			}
		})
	}
	for _, n := range []int{256, 2048} { // N = 2M for M = 128, 1024
		name := map[int]string{256: "256", 2048: "2048"}[n]
		x, c, y := make([]float32, n), make([]float32, n/2), make([]float32, n)
		for i := range x {
			x[i] = float32(i%37 - 18)
		}
		for i := range c {
			c[i] = x[i]
		}
		x64, c64, y64 := make([]float64, n), make([]float64, n/2), make([]float64, n)
		p, p32 := NewMDCTPlan(n), NewMDCTPlan32(n)
		sc := 2 / float64(n/2)
		b.Run("IMDCT/mixed/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.IMDCTMixed(y, c, sc)
			}
		})
		b.Run("IMDCT/f64+convert/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				for j, v := range c {
					c64[j] = float64(v)
				}
				p.IMDCT(y64, c64, sc)
				for j, v := range y64 {
					y[j] = float32(v)
				}
			}
		})
		b.Run("IMDCT/f32plan/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p32.IMDCT(y, c, float32(sc))
			}
		})
		b.Run("MDCT/mixed/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.MDCTMixed(c, x, 1)
			}
		})
		b.Run("MDCT/f64+convert/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				for j, v := range x {
					x64[j] = float64(v)
				}
				p.MDCT(c64, x64, 1)
				for j, v := range c64 {
					c[j] = float32(v)
				}
			}
		})
		b.Run("MDCT/f32plan/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p32.MDCT(c, x, 1)
			}
		})
	}
}
