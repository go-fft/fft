package fft

import "testing"

// BenchmarkDCT4N64 times the DCT-IV of length 64 that HE-AAC's QMF runs six
// times per slot (go-avkit/decode#77, fft#132): float64 and the Mixed path
// (float32 I/O, float64 arithmetic), through a reused plan.
func BenchmarkDCT4N64(b *testing.B) {
	p := NewDCTPlan(64, 4)
	x := benchReal(64)
	d := make([]float64, 64)
	x32 := make([]float32, 64)
	d32 := make([]float32, 64)
	for i, v := range x {
		x32[i] = float32(v)
	}
	b.Run("f64", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			p.DCT(d, x, NormBackward)
		}
	})
	b.Run("mixed", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			p.DCTMixed(d32, x32, NormBackward)
		}
	})
	b.Run("mixed-ortho", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			p.DCTMixed(d32, x32, NormOrtho)
		}
	})
}
