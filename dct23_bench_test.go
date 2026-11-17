package fft

import (
	"fmt"
	"testing"
)

// BenchmarkDCT23 times the DCT-II (forward) and the DCT-III (the inverse of
// the type-2 plan) of 32, 64 and 128 points, NormBackward, through a reused
// plan: the rows dct2/N and dct3/N of the FFTW sweep (fft#162).
func BenchmarkDCT23(b *testing.B) {
	for _, n := range []int{32, 64, 128} {
		p := NewDCTPlan(n, 2)
		x := benchReal(n)
		d := make([]float64, n)
		b.Run(fmt.Sprintf("dct2/%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				p.DCT(d, x, NormBackward)
			}
		})
		b.Run(fmt.Sprintf("dct3/%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				p.IDCT(d, x, NormBackward)
			}
		})
	}
}
