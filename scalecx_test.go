package fft

import (
	"math"
	"strconv"
	"testing"
)

func itoa(n int) string { return strconv.Itoa(n) }

// TestScaleComplexA: the kernel scale is bit-identical to the complex-multiply
// loop on finite values (every length around the vector widths, odd and even
// factors); an Exact plan keeps the loop, so its signed zeros and non-finite
// results are the loop's, and the default plans differ from it only there.
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
