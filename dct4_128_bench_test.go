package fft

import "testing"

// BenchmarkDCT4N128 times the DCT-IV of length 128, the AAC short-block MDCT's
// kernel (256 samples): float64, and the Mixed path (float32 I/O, float64
// arithmetic), through a reused plan.
func BenchmarkDCT4N128(b *testing.B) {
	p := NewDCTPlan(128, 4)
	x := benchReal(128)
	d := make([]float64, 128)
	x32 := make([]float32, 128)
	d32 := make([]float32, 128)
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
}

// BenchmarkMDCTMixed256 times MDCTMixed/IMDCTMixed at N=256 (AAC short blocks).
func BenchmarkMDCTMixed256(b *testing.B) {
	const n = 256
	p := NewMDCTPlan(n)
	x, c, y := make([]float32, n), make([]float32, n/2), make([]float32, n)
	for i := range x {
		x[i] = float32(i%37 - 18)
	}
	b.Run("MDCTMixed", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			p.MDCTMixed(c, x, 1)
		}
	})
	b.Run("IMDCTMixed", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			p.IMDCTMixed(y, c, 1)
		}
	})
}
