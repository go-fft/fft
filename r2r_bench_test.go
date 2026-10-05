package fft

import (
	"strconv"
	"testing"
)

// BenchmarkR2R times each DCT/DST type through a reused plan writing into a
// preallocated dst, next to a RealPlan RFFT of the same length run the same
// way (sub-benchmark "rfft"), so the ratio is the DCT/DST's overhead over the
// real FFT it is built on.
func BenchmarkR2R(b *testing.B) {
	for _, n := range []int{256, 1000, 4096} {
		x := benchReal(n)
		dst := make([]float64, n)
		b.Run("rfft/"+strconv.Itoa(n), func(b *testing.B) {
			p := NewRealPlan(n)
			out := make([]complex128, n/2+1)
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				p.RFFT(out, x)
			}
		})
		for typ := 1; typ <= 4; typ++ {
			p := NewDCTPlan(n, typ)
			b.Run("dct"+strconv.Itoa(typ)+"/"+strconv.Itoa(n), func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					p.DCT(dst, x, NormBackward)
				}
			})
		}
		for typ := 1; typ <= 4; typ++ {
			p := NewDSTPlan(n, typ)
			b.Run("dst"+strconv.Itoa(typ)+"/"+strconv.Itoa(n), func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					p.DST(dst, x, NormBackward)
				}
			})
		}
	}
}

// BenchmarkR2RTypeI times DCT-I and DST-I at the lengths their extensions
// favour: DCT-I runs a real FFT of length 2(N-1) and DST-I one of length
// 2(N+1), so N = 2^k+1 and N = 2^k-1 put a power of two under them, where
// N = 2^k may land on a Bluestein length.
func BenchmarkR2RTypeI(b *testing.B) {
	for _, k := range []int{8, 10, 12} {
		for _, c := range []struct {
			name   string
			cosine bool
			n      int
		}{{"dct1", true, 1<<k + 1}, {"dst1", false, 1<<k - 1}} {
			x := benchReal(c.n)
			dst := make([]float64, c.n)
			b.Run(c.name+"/"+strconv.Itoa(c.n), func(b *testing.B) {
				b.ReportAllocs()
				if c.cosine {
					p := NewDCTPlan(c.n, 1)
					for i := 0; i < b.N; i++ {
						p.DCT(dst, x, NormBackward)
					}
					return
				}
				p := NewDSTPlan(c.n, 1)
				for i := 0; i < b.N; i++ {
					p.DST(dst, x, NormBackward)
				}
			})
		}
	}
}
