package fft

import (
	"math"
	"math/rand"
	"testing"
)

// oMDCT is the O(N·M) definition of the MDCT (inverse false: N = 2m samples to m
// coefficients) or of the IMDCT (inverse true), in double-double. The cosine
// argument is 2π r/(8m) with r = (2n+1+m)(2k+1) reduced modulo 8m in integers, so
// the argument is exact and the cosine comes from the oracle's 200-bit table.
func oMDCT(x []cdd, m int, inverse bool) []cdd {
	tw := ddRoots(8 * m)
	nOut := m
	if inverse {
		nOut = 2 * m
	}
	out := make([]cdd, nOut)
	for o := range out {
		var acc dd
		for i, v := range x {
			n, k := i, o
			if inverse {
				n, k = o, i
			}
			r := ((2*n + 1 + m) * (2*k + 1)) % (8 * m)
			acc = ddAdd(acc, ddMul(v.re, tw[r].re))
		}
		out[o].re = acc
	}
	return out
}

// mdctLens are block lengths N: m = N/2 even takes the DCT-IV route, odd the
// N-point route.
func mdctLens() []int {
	l := []int{2, 4, 6, 8, 10, 12, 14, 16, 18, 30, 64, 100, 126, 128, 256, 258, 512, 1010, 1024, 2048}
	if !testing.Short() && !raceEnabled {
		l = append(l, 4096)
	}
	return l
}

// mdctBound is the contract of docs/accuracy.md, family "mdct": the relative RMS
// error against the oracle is at most c·ε·log2 N.
const (
	mdctC64 = 1.75 // measured worst 0.83 (odd-M route), twice, to the next 0.25
	mdctC32 = 0.75 // measured worst 0.30
)

// TestMDCTAccuracy holds both directions, both precisions, both routes and
// several scales to the bound, against the double-double definition. The float32
// transforms are compared with the oracle of the rounded inputs and scale.
func TestMDCTAccuracy(t *testing.T) {
	worst := map[string]float64{}
	note := func(key string, r float64) {
		if r > worst[key] {
			worst[key] = r
		}
	}
	for _, n := range mdctLens() {
		m := n / 2
		route := "dct4"
		if m%2 == 1 {
			route = "nfft"
		}
		lg := math.Max(1, math.Log2(float64(n)))
		p, p32 := NewMDCTPlan(n), NewMDCTPlan32(n)
		for seed := uint64(1); seed <= 2; seed++ {
			for _, scale := range []float64{1, 2 / float64(m)} {
				x := accRealF64(n, seed)
				c := accRealF64(m, seed+10)
				x32 := accRealF32(n, seed)
				c32 := accRealF32(m, seed+10)
				// float64
				wF := oMDCT(cddFromR(x), m, false)
				wI := oMDCT(cddFromR(c), m, true)
				scaleAll(wF, dd{scale, 0})
				scaleAll(wI, dd{scale, 0})
				eF := errR(p.MDCT(make([]float64, m), x, scale), wF)
				eI := errR(p.IMDCT(make([]float64, n), c, scale), wI)
				// float32: the oracle sees the float32 inputs and the float32 scale
				s32 := float32(scale)
				wF = oMDCT(cddFromR(x32), m, false)
				wI = oMDCT(cddFromR(c32), m, true)
				scaleAll(wF, dd{float64(s32), 0})
				scaleAll(wI, dd{float64(s32), 0})
				eF32 := errR(p32.MDCT(make([]float32, m), x32, s32), wF)
				eI32 := errR(p32.IMDCT(make([]float32, n), c32, s32), wI)
				note(route+"/64", math.Max(eF, eI)/(0x1p-52*lg))
				note(route+"/32", math.Max(eF32, eI32)/(0x1p-23*lg))
				if r := math.Max(eF, eI) / (0x1p-52 * lg); r > mdctC64 {
					t.Errorf("N=%d scale=%g float64: MDCT %.3g IMDCT %.3g exceeds %g·ε·log2N", n, scale, eF/(0x1p-52*lg), eI/(0x1p-52*lg), mdctC64)
				}
				if r := math.Max(eF32, eI32) / (0x1p-23 * lg); r > mdctC32 {
					t.Errorf("N=%d scale=%g float32: MDCT %.3g IMDCT %.3g exceeds %g·ε·log2N", n, scale, eF32/(0x1p-23*lg), eI32/(0x1p-23*lg), mdctC32)
				}
			}
		}
	}
	for k, v := range worst {
		t.Logf("worst err/(ε·log2 N) %s: %.3f", k, v)
	}
}

// TestMDCTOracle checks the oracle on a hand-computable case: N = 4 (m = 2),
// x = (1, 0, 0, 0): X[k] = cos(π/2 · (0+1/2+1)(k+1/2)·…) in the definition.
func TestMDCTOracle(t *testing.T) {
	x := []cdd{{re: dd{1, 0}}, {}, {}, {}}
	got := oMDCT(x, 2, false)
	for k := range got {
		want := math.Cos(math.Pi / 2 * (0.5 + 1) * (float64(k) + 0.5))
		if d := math.Abs(got[k].re.hi - want); d > 1e-15 {
			t.Errorf("X[%d] = %g, want %g", k, got[k].re.hi, want)
		}
	}
}

// TestMDCTPerfectReconstruction: windowed blocks advancing by N/2, an MDCT of
// scale 1, an IMDCT of scale 2/M and a second window, overlap-added, return the
// signal (time-domain aliasing cancellation) for the sine and KBD windows, even
// and odd M, to the bound.
func TestMDCTPerfectReconstruction(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	for _, n := range []int{2, 4, 6, 16, 64, 100, 128, 2048} {
		m := n / 2
		for name, w := range map[string][]float64{"sine": SineWindow(n), "kbd4": KBDWindow(n, 4)} {
			blocks := 6
			x := make([]float64, (blocks+1)*m)
			for i := range x {
				x[i] = rng.NormFloat64()
			}
			p, p32 := NewMDCTPlan(n), NewMDCTPlan32(n)
			y, y32 := make([]float64, len(x)), make([]float32, len(x))
			buf, coef, out := make([]float64, n), make([]float64, m), make([]float64, n)
			buf32, coef32, out32 := make([]float32, n), make([]float32, m), make([]float32, n)
			for b := 0; b < blocks; b++ {
				for i := range buf {
					buf[i] = x[b*m+i] * w[i]
					buf32[i] = float32(buf[i])
				}
				p.MDCT(coef, buf, 1)
				p.IMDCT(out, coef, 2/float64(m))
				p32.MDCT(coef32, buf32, 1)
				p32.IMDCT(out32, coef32, float32(2/float64(m)))
				for i := range out {
					y[b*m+i] += out[i] * w[i]
					y32[b*m+i] += out32[i] * float32(w[i])
				}
			}
			var e64, e32, nrm float64
			for i := m; i < blocks*m; i++ { // the first and last M samples have one block only
				e64 += (y[i] - x[i]) * (y[i] - x[i])
				e32 += (float64(y32[i]) - x[i]) * (float64(y32[i]) - x[i])
				nrm += x[i] * x[i]
			}
			lg := math.Max(1, math.Log2(float64(n)))
			// the round trip is two transforms and a window pass: twice the bound
			if r := math.Sqrt(e64/nrm) / (0x1p-52 * lg); r > 2*mdctC64 {
				t.Errorf("N=%d %s float64: round-trip error %.3g·ε·log2N", n, name, r)
			}
			if r := math.Sqrt(e32/nrm) / (0x1p-23 * lg); r > 2*mdctC32 {
				t.Errorf("N=%d %s float32: round-trip error %.3g·ε·log2N", n, name, r)
			}
		}
	}
}

func TestMDCTWindows(t *testing.T) {
	for _, n := range []int{2, 8, 10, 128, 2048} {
		for name, w := range map[string][]float64{"sine": SineWindow(n), "kbd4": KBDWindow(n, 4), "kbd6": KBDWindow(n, 6)} {
			if len(w) != n {
				t.Fatalf("%s(%d) has length %d", name, n, len(w))
			}
			for i := 0; i < n/2; i++ {
				if s := w[i]*w[i] + w[i+n/2]*w[i+n/2]; math.Abs(s-1) > 1e-14 {
					t.Fatalf("%s(%d) violates Princen-Bradley at %d: %g", name, n, i, s)
				}
				if w[i] != w[n-1-i] && math.Abs(w[i]-w[n-1-i]) > 1e-15 {
					t.Fatalf("%s(%d) is not symmetric at %d", name, n, i)
				}
			}
		}
	}
	if len(SineWindow(0)) != 0 || len(SineWindow(-3)) != 0 || len(KBDWindow(0, 4)) != 0 || len(KBDWindow(-2, 4)) != 0 {
		t.Error("an empty window was expected for n <= 0")
	}
	// KBD alpha=0 is the sine-free limit: Kaiser(0) is rectangular, so the
	// cumulative sum is linear: w[k]² = (k+1)/(n/2+1).
	w := KBDWindow(8, 0)
	for k := 0; k < 4; k++ {
		if want := math.Sqrt(float64(k+1) / 5); math.Abs(w[k]-want) > 1e-15 {
			t.Errorf("KBD(8, 0)[%d] = %g, want %g", k, w[k], want)
		}
	}
	// I0 against its known values.
	for x, want := range map[float64]float64{0: 1, 1: 1.2660658777520082, 5: 27.239871823604442} {
		if got := besselI0(x); math.Abs(got-want) > 1e-14*want {
			t.Errorf("I0(%g) = %.17g, want %.17g", x, got, want)
		}
	}
	mustPanicWith(t, "KBDWindow", "fft: KBDWindow", func() { KBDWindow(7, 4) })
}

// TestMDCTAlias: dst may overlap src, for both routes and precisions.
func TestMDCTAlias(t *testing.T) {
	for _, n := range []int{8, 10} {
		m := n / 2
		x := accRealF64(n, 3)
		c := accRealF64(m, 4)
		p, p32 := NewMDCTPlan(n), NewMDCTPlan32(n)
		wantF := p.MDCT(make([]float64, m), x, 1)
		wantI := p.IMDCT(make([]float64, n), c, 1)
		buf := append([]float64(nil), x...)
		if got := p.MDCT(buf, buf, 1); !equalF(got, wantF) {
			t.Errorf("N=%d: aliased MDCT differs", n)
		}
		buf = append(append([]float64(nil), c...), make([]float64, n-m)...)
		if got := p.IMDCT(buf, buf, 1); !equalF(got, wantI) {
			t.Errorf("N=%d: aliased IMDCT differs", n)
		}
		x32, c32 := accRealF32(n, 3), accRealF32(m, 4)
		wf := p32.MDCT(make([]float32, m), x32, 1)
		wi := p32.IMDCT(make([]float32, n), c32, 1)
		b32 := append([]float32(nil), x32...)
		got := p32.MDCT(b32, b32, 1)
		for i := range got {
			if got[i] != wf[i] {
				t.Errorf("N=%d: aliased MDCT32 differs", n)
				break
			}
		}
		b32 = append(append([]float32(nil), c32...), make([]float32, n-m)...)
		got = p32.IMDCT(b32, b32, 1)
		for i := range got {
			if got[i] != wi[i] {
				t.Errorf("N=%d: aliased IMDCT32 differs", n)
				break
			}
		}
	}
}

func equalF(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestMDCTPanics(t *testing.T) {
	const pre = "fft: MDCT needs"
	for _, n := range []int{-2, 0, 1, 3, 7} {
		mustPanicWith(t, "NewMDCTPlan", pre, func() { NewMDCTPlan(n) })
		mustPanicWith(t, "NewMDCTPlan32", pre, func() { NewMDCTPlan32(n) })
	}
	const short = "fft: MDCTPlan"
	for _, n := range []int{16, 20} { // both routes
		p, p32 := NewMDCTPlan(n), NewMDCTPlan32(n)
		if p.Len() != n || p32.Len() != n {
			t.Fatalf("Len = %d, %d, want %d", p.Len(), p32.Len(), n)
		}
		m := n / 2
		mustPanicWith(t, "MDCT dst", short, func() { p.MDCT(make([]float64, m-1), make([]float64, n), 1) })
		mustPanicWith(t, "MDCT src", short, func() { p.MDCT(make([]float64, m), make([]float64, n-1), 1) })
		mustPanicWith(t, "IMDCT dst", short, func() { p.IMDCT(make([]float64, n-1), make([]float64, m), 1) })
		mustPanicWith(t, "IMDCT src", short, func() { p.IMDCT(make([]float64, n), make([]float64, m-1), 1) })
		mustPanicWith(t, "MDCT32 dst", short, func() { p32.MDCT(make([]float32, m-1), make([]float32, n), 1) })
		mustPanicWith(t, "MDCT32 src", short, func() { p32.MDCT(make([]float32, m), make([]float32, n-1), 1) })
		mustPanicWith(t, "IMDCT32 dst", short, func() { p32.IMDCT(make([]float32, n-1), make([]float32, m), 1) })
		mustPanicWith(t, "IMDCT32 src", short, func() { p32.IMDCT(make([]float32, n), make([]float32, m-1), 1) })
	}
}

// TestMDCTAllocs: no allocation per call in steady state, on both routes.
func TestMDCTAllocs(t *testing.T) {
	if raceEnabled {
		t.Skip("under -race sync.Pool drops items on purpose, so allocation counts are not meaningful")
	}
	for _, n := range []int{2048, 2050} {
		p, p32 := NewMDCTPlan(n), NewMDCTPlan32(n)
		x, c, y := make([]float64, n), make([]float64, n/2), make([]float64, n)
		x32, c32, y32 := make([]float32, n), make([]float32, n/2), make([]float32, n)
		p.MDCT(c, x, 1)
		p.IMDCT(y, c, 1)
		p32.MDCT(c32, x32, 1)
		p32.IMDCT(y32, c32, 1)
		for name, f := range map[string]func(){
			"MDCT":    func() { p.MDCT(c, x, 1) },
			"IMDCT":   func() { p.IMDCT(y, c, 1) },
			"MDCT32":  func() { p32.MDCT(c32, x32, 1) },
			"IMDCT32": func() { p32.IMDCT(y32, c32, 1) },
		} {
			if got := testing.AllocsPerRun(50, f); got != 0 {
				t.Errorf("N=%d %s: %v allocations per call, want 0", n, name, got)
			}
		}
	}
}

// FuzzMDCT: for any even block length 2..120 and any input bytes, both
// precisions return finite values within the absolute-sum bound of the
// definition computed in float64.
func FuzzMDCT(f *testing.F) {
	f.Add(1, []byte{1, 2, 3, 250})
	f.Add(2, []byte("\x00\xff\x10"))
	f.Add(3, []byte{128})
	f.Add(30, []byte("fuzz the odd route"))
	f.Add(59, []byte{})
	f.Fuzz(func(t *testing.T, k int, data []byte) {
		n := 2 * (int(uint(k)%60) + 1)
		m := n / 2
		x := make([]float64, n)
		c := make([]float64, m)
		var sx, sc float64
		for i := range x {
			if len(data) > 0 {
				x[i] = float64(data[i%len(data)])/128 - 1
			}
			sx += math.Abs(x[i])
		}
		for i := range c {
			if len(data) > 0 {
				c[i] = float64(data[(i*7+1)%len(data)])/128 - 1
			}
			sc += math.Abs(c[i])
		}
		p, p32 := NewMDCTPlan(n), NewMDCTPlan32(n)
		x32, c32 := make([]float32, n), make([]float32, m)
		for i, v := range x {
			x32[i] = float32(v)
		}
		for i, v := range c {
			c32[i] = float32(v)
		}
		gF := p.MDCT(make([]float64, m), x, 1)
		gI := p.IMDCT(make([]float64, n), c, 1)
		g32F := p32.MDCT(make([]float32, m), x32, 1)
		g32I := p32.IMDCT(make([]float32, n), c32, 1)
		wF, wI := oMDCT(cddFromR(x), m, false), oMDCT(cddFromR(c), m, true)
		for i, w := range wF {
			if math.Abs(gF[i]-w.re.hi) > 1e-13*(sx+1) || math.Abs(float64(g32F[i])-w.re.hi) > 1e-4*(sx+1) {
				t.Fatalf("N=%d MDCT[%d]: got %g / %g, want %g", n, i, gF[i], g32F[i], w.re.hi)
			}
		}
		for i, w := range wI {
			if math.Abs(gI[i]-w.re.hi) > 1e-13*(sc+1) || math.Abs(float64(g32I[i])-w.re.hi) > 1e-4*(sc+1) {
				t.Fatalf("N=%d IMDCT[%d]: got %g / %g, want %g", n, i, gI[i], g32I[i], w.re.hi)
			}
		}
	})
}

func BenchmarkMDCT(b *testing.B) {
	for _, n := range []int{128, 256, 2048} {
		x, c, y := make([]float64, n), make([]float64, n/2), make([]float64, n)
		x32, c32, y32 := make([]float32, n), make([]float32, n/2), make([]float32, n)
		for i := range x {
			x[i] = float64(i%37 - 18)
			x32[i] = float32(x[i])
		}
		p, p32 := NewMDCTPlan(n), NewMDCTPlan32(n)
		name := map[int]string{128: "128", 256: "256", 2048: "2048"}[n]
		b.Run("MDCT/"+name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				p.MDCT(c, x, 1)
			}
		})
		b.Run("IMDCT/"+name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				p.IMDCT(y, c, 1)
			}
		})
		b.Run("MDCT32/"+name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				p32.MDCT(c32, x32, 1)
			}
		})
		b.Run("IMDCT32/"+name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				p32.IMDCT(y32, c32, 1)
			}
		})
	}
}
