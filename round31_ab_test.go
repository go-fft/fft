package fft

import (
	"strconv"
	"strings"
	"testing"
)

// BenchmarkR31AB is Round 31's A/B row set: the N-D shapes whose rows the
// change reaches (a last axis of 1024 points, at least 2^19 elements) first,
// then shapes and lengths it does not reach, in the same process.
func BenchmarkR31AB(b *testing.B) {
	for _, shape := range [][]int{
		{1024, 1024}, {512, 1024}, {4, 256, 1024}, // reached
		{256, 1024}, {512, 512}, {128, 128}, {2048, 512}, // not reached
	} {
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
	for _, n := range []int{1024, 8192} {
		p := NewPlan(n)
		src := benchComplex(n)
		dst := make([]complex128, n)
		b.Run("C/"+strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, src)
			}
		})
	}
}
