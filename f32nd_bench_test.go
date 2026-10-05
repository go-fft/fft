package fft

import (
	"fmt"
	"testing"
)

// BenchmarkF32ND times the single-precision N-D, real 2-D and DCT/DST paths
// against their float64 counterparts on the same shapes, one core: run with
// -bench F32ND -cpu 1.
func BenchmarkF32ND(b *testing.B) {
	for _, s := range [][]int{{64, 64}, {256, 256}, {32, 32, 32}, {1024, 1024}} {
		total := f32ndProduct(s)
		x, x64 := signal32(total, 1)
		p32, p64 := NewPlanN32(s...), NewPlanN(s...)
		d32, d64 := make([]complex64, total), make([]complex128, total)
		name := fmt.Sprint(s)
		b.Run("PlanN-"+name+"-f32", func(b *testing.B) {
			for b.Loop() {
				p32.FFT(d32, x)
			}
		})
		b.Run("PlanN-"+name+"-f64", func(b *testing.B) {
			for b.Loop() {
				p64.FFT(d64, x64)
			}
		})
	}
	for _, s := range [][2]int{{256, 256}, {1024, 1024}} {
		r, r64 := real32(s[0]*s[1], 2)
		p32, p64 := NewRealPlan2_32(s[0], s[1]), NewRealPlan2(s[0], s[1])
		d32, d64 := make([]complex64, p32.SpectrumLen()), make([]complex128, p64.SpectrumLen())
		name := fmt.Sprint(s[0], "x", s[1])
		b.Run("RealPlan2-"+name+"-f32", func(b *testing.B) {
			for b.Loop() {
				p32.RFFT(d32, r)
			}
		})
		b.Run("RealPlan2-"+name+"-f64", func(b *testing.B) {
			for b.Loop() {
				p64.RFFT(d64, r64)
			}
		})
	}
	for _, n := range []int{1024, 4096} {
		for typ := 1; typ <= 4; typ++ {
			r, r64 := real32(n, 3)
			p32, p64 := NewDCTPlan32(n, typ), NewDCTPlan(n, typ)
			d32, d64 := make([]float32, n), make([]float64, n)
			name := fmt.Sprint("DCT", typ, "-", n)
			b.Run(name+"-f32", func(b *testing.B) {
				for b.Loop() {
					p32.DCT(d32, r, NormBackward)
				}
			})
			b.Run(name+"-f64", func(b *testing.B) {
				for b.Loop() {
					p64.DCT(d64, r64, NormBackward)
				}
			})
		}
	}
	r, r64 := real32(64*64, 4)
	b.Run("DCTN2-64x64-f32", func(b *testing.B) {
		for b.Loop() {
			DCTN32(r, []int{64, 64}, 2, NormOrtho)
		}
	})
	b.Run("DCTN2-64x64-f64", func(b *testing.B) {
		for b.Loop() {
			DCTN(r64, []int{64, 64}, 2, NormOrtho)
		}
	})
}
