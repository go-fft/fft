package fft

import "testing"

// BenchmarkDCT4N64F32 times the float32 forms of the DCT-IV(64) against the float64
// pair kernel's Mixed form and the generic float32 plan. ns/op is per call: a pair
// is two transforms, a quad four (the two-pair rows run two DCTMixedPair calls).
func BenchmarkDCT4N64F32(b *testing.B) {
	p := NewDCTPlan(64, 4)
	q := NewDCTPlan32(64, 4)
	var x, y [4][]float32
	for i := range x {
		x[i], y[i] = make([]float32, 64), make([]float32, 64)
		copy(x[i], accRealF32(64, uint64(i+1)))
	}
	b.Run("plan32-2calls", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			q.DCT(y[0], x[0], NormBackward)
			q.DCT(y[1], x[1], NormBackward)
		}
	})
	b.Run("mixed-pair", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			p.DCTMixedPair(y[0], y[1], x[0], x[1], NormBackward)
		}
	})
	b.Run("mixed-2pairs", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			p.DCTMixedPair(y[0], y[1], x[0], x[1], NormBackward)
			p.DCTMixedPair(y[2], y[3], x[2], x[3], NormBackward)
		}
	})
	b.Run("f32-pair", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			q.DCTPair(y[0], y[1], x[0], x[1], NormBackward)
		}
	})
	b.Run("f32-2pairs", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			q.DCTPair(y[0], y[1], x[0], x[1], NormBackward)
			q.DCTPair(y[2], y[3], x[2], x[3], NormBackward)
		}
	})
	b.Run("f32-quad", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			q.DCTQuad(y, x, NormBackward)
		}
	})
}
