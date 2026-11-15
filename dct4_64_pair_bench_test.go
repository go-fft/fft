package fft

import "testing"

// BenchmarkDCT4N64Pair times the pair forms against two single calls on the same
// data (ns/op is per pair in both).
func BenchmarkDCT4N64Pair(b *testing.B) {
	p := NewDCTPlan(64, 4)
	x0, x1 := benchReal(64), benchReal(64)
	d0, d1 := make([]float64, 64), make([]float64, 64)
	f0, f1 := make([]float32, 64), make([]float32, 64)
	g0, g1 := make([]float32, 64), make([]float32, 64)
	for i := range f0 {
		f0[i], f1[i] = float32(x0[i]), float32(x1[i])
	}
	b.Run("f64-2calls", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			p.DCT(d0, x0, NormBackward)
			p.DCT(d1, x1, NormBackward)
		}
	})
	b.Run("f64-pair", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			p.DCTPair(d0, d1, x0, x1, NormBackward)
		}
	})
	b.Run("mixed-2calls", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			p.DCTMixed(g0, f0, NormBackward)
			p.DCTMixed(g1, f1, NormBackward)
		}
	})
	b.Run("mixed-pair", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			p.DCTMixedPair(g0, g1, f0, f1, NormBackward)
		}
	})
	b.Run("mixed-ortho-pair", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			p.DCTMixedPair(g0, g1, f0, f1, NormOrtho)
		}
	})
}
