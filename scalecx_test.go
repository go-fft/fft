package fft

import (
	"math"
	"strconv"
	"testing"
)

func itoa(n int) string { return strconv.Itoa(n) }

// TestScaleComplexA: the kernel scale is bit-identical to the complex-multiply
// loop on finite values with no zero part (every length around the vector
// widths, odd and even factors); an Exact plan keeps the loop, so its signed
// zeros and non-finite results are the loop's, and the default plans differ
// from it only on those (TestScaleComplexAIEEE pins how).
func TestScaleComplexA(t *testing.T) {
	for _, a := range []arith{arithDefault, arithFMA, arithExact, arithPlain} {
		for n := 0; n <= 70; n++ {
			for _, f := range []float64{1.0 / 3, 0.125, 7, 1e-300} {
				x := accComplexF64(n, uint64(n)+1)
				want := append([]complex128(nil), x...)
				for i := range want {
					want[i] *= complex(f, 0)
				}
				got := append([]complex128(nil), x...)
				scaleComplexA(got, f, a)
				for i := range got {
					if math.Float64bits(real(got[i])) != math.Float64bits(real(want[i])) || math.Float64bits(imag(got[i])) != math.Float64bits(imag(want[i])) {
						t.Fatalf("arith %d n=%d f=%g: element %d is %v, the loop gives %v", a, n, f, i, got[i], want[i])
					}
				}
			}
		}
	}
	// A dst longer than the scaled window keeps the rest.
	x := accComplexF64(40, 3)
	y := append([]complex128(nil), x...)
	scaleComplexA(y[:30], 0.5, arithDefault)
	for i := 30; i < 40; i++ {
		if y[i] != x[i] {
			t.Fatalf("element %d beyond the window changed", i)
		}
	}
}

// TestScaleComplexAIEEE pins the bits that changed for the default plans: a
// negative-zero part keeps its sign (the loop's (1-0i)*0.25 had imag +0), and an
// infinite part no longer turns the other part into NaN (the loop's inf*0). The
// Exact arithmetic keeps the loop's results for the same inputs.
func TestScaleComplexAIEEE(t *testing.T) {
	negz, inf := math.Copysign(0, -1), math.Inf(1)
	x := []complex128{complex(1, negz), complex(inf, 2), complex(negz, -1), complex(math.NaN(), 1)}
	bits := func(c complex128) [2]uint64 { return [2]uint64{math.Float64bits(real(c)), math.Float64bits(imag(c))} }
	wantKernel := []complex128{complex(0.25, negz), complex(inf, 0.5), complex(negz, -0.25), complex(math.NaN(), 0.25)}
	wantLoop := []complex128{complex(0.25, 0), complex(inf, math.NaN()), complex(0, -0.25), complex(math.NaN(), math.NaN())}
	for _, a := range []arith{arithDefault, arithFMA, arithPlain, arithExact} {
		got := append([]complex128(nil), x...)
		scaleComplexA(got, 0.25, a)
		want := wantKernel
		if a == arithExact {
			want = wantLoop
		}
		for i := range got {
			g, w := bits(got[i]), bits(want[i])
			if math.IsNaN(real(want[i])) || math.IsNaN(imag(want[i])) { // any NaN payload
				if math.IsNaN(real(got[i])) != math.IsNaN(real(want[i])) || math.IsNaN(imag(got[i])) != math.IsNaN(imag(want[i])) {
					t.Errorf("arith %d element %d: %v, want %v", a, i, got[i], want[i])
				}
				if !math.IsNaN(real(want[i])) && g[0] != w[0] || !math.IsNaN(imag(want[i])) && g[1] != w[1] {
					t.Errorf("arith %d element %d: %v, want %v", a, i, got[i], want[i])
				}
				continue
			}
			if g != w {
				t.Errorf("arith %d element %d: %v, want %v", a, i, got[i], want[i])
			}
		}
	}
}

// TestScaleComplexAExactZerosAndInf: the Exact plans' scaling keeps the loop's
// results for -0 and infinities (v0.28.16's bits).
func TestScaleComplexAExactZerosAndInf(t *testing.T) {
	x := []complex128{complex(math.Copysign(0, -1), -1), complex(math.Inf(1), math.Inf(1))}
	want := append([]complex128(nil), x...)
	for i := range want {
		want[i] *= complex(0.25, 0)
	}
	got := append([]complex128(nil), x...)
	scaleComplexA(got, 0.25, arithExact)
	for i := range got {
		if math.Float64bits(real(got[i])) != math.Float64bits(real(want[i])) || math.Float64bits(imag(got[i])) != math.Float64bits(imag(want[i])) {
			t.Fatalf("element %d: %v, the loop gives %v", i, got[i], want[i])
		}
	}
}

// TestIFFTScaleMatchesLoop: Plan.IFFT, PlanN.IFFT and the Norm forms equal the
// unnormalized inverse times 1/n, bit for bit, for every arithmetic.
func TestIFFTScaleMatchesLoop(t *testing.T) {
	for _, o := range []Options{{}, {FMA: true}, {Exact: true}, {NoFMA: true}} {
		for _, n := range []int{2, 12, 64, 256, 1000} {
			p := NewPlanWith(n, o)
			x := accComplexF64(n, 11)
			raw := p.IFFTNorm(make([]complex128, n), x, NormForward)
			want := make([]complex128, n)
			for i := range raw {
				want[i] = raw[i] * complex(1/float64(n), 0)
			}
			got := p.IFFT(make([]complex128, n), x)
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("%+v n=%d: IFFT[%d] = %v, scaled raw gives %v", o, n, i, got[i], want[i])
				}
			}
		}
	}
}

func BenchmarkIFFTScale(b *testing.B) {
	for _, n := range []int{64, 256, 1024} {
		p := NewPlan(n)
		src := accComplexF64(n, 1)
		dst := make([]complex128, n)
		b.Run("IFFT_"+itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.IFFT(dst, src)
			}
		})
		b.Run("IFFTraw_"+itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.IFFTNorm(dst, src, NormForward)
			}
		})
	}
}

// BenchmarkScaleFamilies: one row per family of scaling sites (the N-D plan's
// inverse, the real plan's Norm form, the real 2-D inverse's 1/rows pass, the
// one-shot With form).
func BenchmarkScaleFamilies(b *testing.B) {
	pn := NewPlanN(64, 64)
	cx := accComplexF64(64*64, 1)
	cd := make([]complex128, 64*64)
	b.Run("PlanN_IFFT_64x64", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			pn.IFFT(cd, cx)
		}
	})
	rp := NewRealPlan(256)
	rx := accRealF64(256, 1)
	rd := make([]complex128, 129)
	b.Run("RealPlan_RFFTNorm_256", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			rp.RFFTNorm(rd, rx, NormOrtho)
		}
	})
	r2 := NewRealPlan2(64, 64)
	r2x := accRealF64(64*64, 1)
	spec := r2.RFFT(make([]complex128, r2.SpectrumLen()), r2x)
	r2d := make([]float64, 64*64)
	b.Run("RealPlan2_IRFFT_64x64", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			r2.IRFFT(r2d, spec)
		}
	})
	w := accComplexF64(256, 1)
	b.Run("IFFTWith_256", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			IFFTWith(w, Options{})
		}
	})
}
