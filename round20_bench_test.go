package fft

// The row set behind BENCHMARKS.md's Round 20 A/B runs: the large powers of
// two the blocked schedule changes, and controls it does not reach.

import (
	"strconv"
	"testing"
)

// BenchmarkLarge times plans writing into a reused slice: complex, real
// forward and inverse at large powers of two, plus controls.
func BenchmarkLarge(b *testing.B) {
	for _, n := range []int{4096, 16384, 32768, 65536, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1000} {
		p := NewPlan(n)
		src := benchComplex(n)
		dst := make([]complex128, n)
		b.Run("C/"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, src)
			}
		})
	}
	for _, n := range []int{65536, 1 << 18, 1 << 20} {
		p := NewRealPlan(n)
		src := benchReal(n)
		dst := make([]complex128, n/2+1)
		b.Run("R/"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.RFFT(dst, src)
			}
		})
		spec := p.RFFT(make([]complex128, n/2+1), src)
		out := make([]float64, n)
		b.Run("IR/"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.IRFFT(out, spec)
			}
		})
	}
	p := NewPlanN(1024, 1024)
	src := benchComplex(1 << 20)
	dst := make([]complex128, 1<<20)
	b.Run("2D/1024", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			p.FFT(dst, src)
		}
	})
}
