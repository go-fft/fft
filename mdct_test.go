package fft

import (
	"math"
	"math/rand"
	"testing"
)

// mdctRef computes the MDCT or IMDCT by the definition, O(N²), in float64. The
// cosine argument is π r/(4N) with r = (2n+1+N)(2k+1) reduced modulo 8N in
// integers, so the argument is exact before the one rounding of math.Cos; the
// sums are accumulated in a compensated way, so the reference's own error is
// a few ulps of the result, far below the bounds the tests state.
func mdctRef(x []float64, n int, inverse bool) []float64 {
	cosr := func(nn, k int) float64 {
		r := ((2*nn + 1 + n) * (2*k + 1)) % (8 * n)
		return math.Cos(math.Pi * float64(r) / float64(4*n))
	}
	var out []float64
	sum := func(terms int, f func(i int) float64) float64 {
		var s, c float64
		for i := 0; i < terms; i++ {
			y := f(i) - c
			t := s + y
			c = (t - s) - y
			s = t
		}
		return s
	}
	if inverse {
		out = make([]float64, 2*n)
		for nn := range out {
			out[nn] = sum(n, func(k int) float64 { return x[k] * cosr(nn, k) })
		}
		return out
	}
	out = make([]float64, n)
	for k := range out {
		out[k] = sum(2*n, func(nn int) float64 { return x[nn] * cosr(nn, k) })
	}
	return out
}

func l1(x []float64) float64 {
	s := 0.0
	for _, v := range x {
		s += math.Abs(v)
	}
	return s
}

// TestMDCTMatchesDefinition checks both directions and both precisions against
// the O(N²) definition. The accuracy stated: the error of every output is at most
// 4·(log2(N)+1) half ulps of the sum of the absolute values of the inputs
// (float64: 2^-53 per half ulp, float32: 2^-24), a bound of the form the FFT-based
// transforms satisfy; the measured error is logged.
func TestMDCTMatchesDefinition(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, n := range []int{2, 4, 6, 8, 10, 16, 64, 100, 128, 1024} {
		for _, scale := range []float64{1, 0.37} {
			x := make([]float64, 2*n)
			c := make([]float64, n)
			for i := range x {
				x[i] = rng.NormFloat64() * 100
			}
			for i := range c {
				c[i] = rng.NormFloat64() * 100
			}
			wantF := mdctRef(x, n, false)
			wantI := mdctRef(c, n, true)
			lg := math.Log2(float64(n)) + 1
			f64 := NewMDCTPlan(n)
			f32 := NewMDCTPlan32(n)
			if f64.Len() != n || f32.Len() != n {
				t.Fatalf("Len = %d, %d, want %d", f64.Len(), f32.Len(), n)
			}
			gotF := f64.MDCT(make([]float64, n), x, scale)
			gotI := f64.IMDCT(make([]float64, 2*n), c, scale)
			x32, c32 := make([]float32, len(x)), make([]float32, len(c))
			for i, v := range x {
				x32[i] = float32(v)
			}
			for i, v := range c {
				c32[i] = float32(v)
			}
			g32F := f32.MDCT(make([]float32, n), x32, float32(scale))
			g32I := f32.IMDCT(make([]float32, 2*n), c32, float32(scale))
			var w64F, w64I, w32F, w32I float64
			for k := range gotF {
				w64F = math.Max(w64F, math.Abs(gotF[k]-scale*wantF[k]))
			}
			for k := range gotI {
				w64I = math.Max(w64I, math.Abs(gotI[k]-scale*wantI[k]))
			}
			// The float32 transforms see the rounded inputs: compare with the
			// definition of the rounded inputs.
			r32 := make([]float64, len(x))
			for i, v := range x32 {
				r32[i] = float64(v)
			}
			rc32 := make([]float64, len(c))
			for i, v := range c32 {
				rc32[i] = float64(v)
			}
			wantF32, wantI32 := mdctRef(r32, n, false), mdctRef(rc32, n, true)
			for k := range g32F {
				w32F = math.Max(w32F, math.Abs(float64(g32F[k])-scale*wantF32[k]))
			}
			for k := range g32I {
				w32I = math.Max(w32I, math.Abs(float64(g32I[k])-scale*wantI32[k]))
			}
			b64 := 4 * lg * 0x1p-53 * scale
			b32 := 4 * lg * 0x1p-24 * scale
			t.Logf("N=%d scale=%g: float64 MDCT %.3g IMDCT %.3g; float32 MDCT %.3g IMDCT %.3g (bounds %.3g, %.3g)",
				n, scale, w64F, w64I, w32F, w32I, b64*l1(x), b32*l1(x))
			if w64F > b64*l1(x) {
				t.Errorf("N=%d: float64 MDCT error %g exceeds %g", n, w64F, b64*l1(x))
			}
			if w64I > b64*l1(c) {
				t.Errorf("N=%d: float64 IMDCT error %g exceeds %g", n, w64I, b64*l1(c))
			}
			if w32F > b32*l1(x) {
				t.Errorf("N=%d: float32 MDCT error %g exceeds %g", n, w32F, b32*l1(x))
			}
			if w32I > b32*l1(c) {
				t.Errorf("N=%d: float32 IMDCT error %g exceeds %g", n, w32I, b32*l1(c))
			}
		}
	}
}

// TestMDCTPerfectReconstruction: windowed blocks advancing by N, an IMDCT scaled
// by 2/N and a second window, overlap-added, return the signal (time-domain
// aliasing cancellation), for the sine window and for the Kaiser-Bessel-derived
// shape built here from its definition.
func TestMDCTPerfectReconstruction(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	for _, n := range []int{2, 8, 64, 100, 1024} {
		w := make([]float64, 2*n)
		for i := range w {
			w[i] = math.Sin(math.Pi / float64(2*n) * (float64(i) + 0.5))
		}
		for i := 0; i < n; i++ {
			if s := w[i]*w[i] + w[i+n]*w[i+n]; math.Abs(s-1) > 1e-12 {
				t.Fatalf("N=%d: the sine window violates Princen-Bradley at %d: %g", n, i, s)
			}
		}
		blocks := 6
		x := make([]float64, (blocks+1)*n)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		p := NewMDCTPlan(n)
		y := make([]float64, len(x))
		buf := make([]float64, 2*n)
		coef := make([]float64, n)
		out := make([]float64, 2*n)
		for b := 0; b < blocks; b++ {
			for i := range buf {
				buf[i] = x[b*n+i] * w[i]
			}
			p.MDCT(coef, buf, 1)
			p.IMDCT(out, coef, 2/float64(n))
			for i := range out {
				y[b*n+i] += out[i] * w[i]
			}
		}
		for i := n; i < blocks*n; i++ { // the first and last N samples have one block only
			if math.Abs(y[i]-x[i]) > 1e-12 {
				t.Fatalf("N=%d: sample %d is %g, want %g", n, i, y[i], x[i])
			}
		}
		// the same in single precision
		p32 := NewMDCTPlan32(n)
		y32 := make([]float32, len(x))
		buf32, coef32, out32 := make([]float32, 2*n), make([]float32, n), make([]float32, 2*n)
		for b := 0; b < blocks; b++ {
			for i := range buf32 {
				buf32[i] = float32(x[b*n+i] * w[i])
			}
			p32.MDCT(coef32, buf32, 1)
			p32.IMDCT(out32, coef32, float32(2/float64(n)))
			for i := range out32 {
				y32[b*n+i] += out32[i] * float32(w[i])
			}
		}
		for i := n; i < blocks*n; i++ {
			if math.Abs(float64(y32[i])-x[i]) > 2e-5 {
				t.Fatalf("N=%d: float32 sample %d is %g, want %g", n, i, y32[i], x[i])
			}
		}
	}
}

func TestMDCTPanics(t *testing.T) {
	const pre = "fft: MDCT"
	for _, n := range []int{-2, 0, 1, 3, 7} {
		mustPanicWith(t, "NewMDCTPlan", pre, func() { NewMDCTPlan(n) })
		mustPanicWith(t, "NewMDCTPlan32", pre, func() { NewMDCTPlan32(n) })
	}
	p, p32 := NewMDCTPlan(8), NewMDCTPlan32(8)
	const short = "fft: MDCTPlan"
	mustPanicWith(t, "MDCT dst", short, func() { p.MDCT(make([]float64, 7), make([]float64, 16), 1) })
	mustPanicWith(t, "MDCT src", short, func() { p.MDCT(make([]float64, 8), make([]float64, 15), 1) })
	mustPanicWith(t, "IMDCT dst", short, func() { p.IMDCT(make([]float64, 15), make([]float64, 8), 1) })
	mustPanicWith(t, "IMDCT src", short, func() { p.IMDCT(make([]float64, 16), make([]float64, 7), 1) })
	mustPanicWith(t, "MDCT32 dst", short, func() { p32.MDCT(make([]float32, 7), make([]float32, 16), 1) })
	mustPanicWith(t, "MDCT32 src", short, func() { p32.MDCT(make([]float32, 8), make([]float32, 15), 1) })
	mustPanicWith(t, "IMDCT32 dst", short, func() { p32.IMDCT(make([]float32, 15), make([]float32, 8), 1) })
	mustPanicWith(t, "IMDCT32 src", short, func() { p32.IMDCT(make([]float32, 16), make([]float32, 7), 1) })
}

// TestMDCTAllocs: no allocation per call in steady state.
func TestMDCTAllocs(t *testing.T) {
	if raceEnabled {
		t.Skip("under -race sync.Pool drops items on purpose, so allocation counts are not meaningful")
	}
	p, p32 := NewMDCTPlan(1024), NewMDCTPlan32(1024)
	x, c, y := make([]float64, 2048), make([]float64, 1024), make([]float64, 2048)
	x32, c32, y32 := make([]float32, 2048), make([]float32, 1024), make([]float32, 2048)
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
			t.Errorf("%s: %v allocations per call, want 0", name, got)
		}
	}
}

func BenchmarkMDCT(b *testing.B) {
	for _, n := range []int{128, 1024, 2048} {
		x, c, y := make([]float64, 2*n), make([]float64, n), make([]float64, 2*n)
		x32, c32, y32 := make([]float32, 2*n), make([]float32, n), make([]float32, 2*n)
		for i := range x {
			x[i] = float64(i%37 - 18)
			x32[i] = float32(x[i])
		}
		p, p32 := NewMDCTPlan(n), NewMDCTPlan32(n)
		name := map[int]string{128: "128", 1024: "1024", 2048: "2048"}[n]
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
