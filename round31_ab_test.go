package fft

import (
	"strconv"
	"strings"
	"testing"
)

// BenchmarkR31AB is Round 31's A/B row set between binaries: the rows its
// changes reach on Cascade Lake first (N-D plans whose last axis is 1024
// points with at least 2^19 elements; composites; RFFT and IRFFT whose half
// length is a power of two from 256 to 16384, and RFFT 1000), then rows they
// do not reach, in the same process.
func BenchmarkR31AB(b *testing.B) {
	nd := func(shape ...int) {
		size := shapeProduct(shape...)
		p := NewPlanN(shape...)
		src := benchComplex(size)
		dst := make([]complex128, size)
		var name []string
		for _, n := range shape {
			name = append(name, strconv.Itoa(n))
		}
		b.Run("N/"+strings.Join(name, "x"), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, src)
			}
		})
	}
	c := func(n int) {
		p := NewPlan(n)
		src := benchComplex(n)
		dst := make([]complex128, n)
		b.Run("C/"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, src)
			}
		})
	}
	r := func(n int) {
		p := NewRealPlan(n)
		src := benchReal(n)
		dst := make([]complex128, n/2+1)
		back := make([]float64, n)
		b.Run("R/"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.RFFT(dst, src)
			}
		})
		b.Run("IR/"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.IRFFT(back, dst)
			}
		})
	}
	// Reached.
	nd(1024, 1024)
	nd(512, 1024)
	for _, n := range []int{1000, 1080, 1296, 1920, 2000, 6000, 10000} {
		c(n)
	}
	for _, n := range []int{512, 1024, 4096, 8192, 32768, 1000} {
		r(n)
	}
	// Not reached.
	nd(256, 1024)
	nd(128, 128)
	for _, n := range []int{256, 1024, 4096, 65536, 1009, 10007} {
		c(n)
	}
	for _, n := range []int{256, 65536} {
		r(n)
	}
}
