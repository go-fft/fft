package fft

// The benchmarks behind BENCHMARKS.md's Round 21 (arm64): the fan-out floor
// and the parts of a 2-D transform.

import (
	"strconv"
	"testing"
)

// BenchmarkR21Fanout times n×n plans on all cores for several parMinChunk
// values; "serial" is a floor no plan reaches.
func BenchmarkR21Fanout(b *testing.B) {
	defer func(v int) { parMinChunk = v }(parMinChunk)
	for _, n := range []int{64, 128, 256, 512, 1024} {
		p := NewPlanN(n, n)
		src := benchComplex(n * n)
		dst := make([]complex128, n*n)
		for _, m := range []int{8192, 16384, 32768, 65536, 131072, 1 << 40} {
			parMinChunk = m
			name := "m" + strconv.Itoa(m)
			if m == 1<<40 {
				name = "serial"
			}
			b.Run(strconv.Itoa(n)+"/"+name, func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					p.FFT(dst, src)
				}
			})
		}
	}
}

// BenchmarkR21Parts2D times an n×n plan's parts on one goroutine: the rows,
// the columns gathered and transformed line by line, the columns as strips of
// batched passes, and the whole transform each way.
func BenchmarkR21Parts2D(b *testing.B) {
	defer func(v bool) { stripAxes = v }(stripAxes)
	for _, n := range []int{64, 128, 256, 512, 1024} {
		stripAxes = false
		pl := NewPlanN(n, n)
		stripAxes = true
		ps := NewPlanN(n, n)
		src := benchComplex(n * n)
		dst := make([]complex128, n*n)
		copy(dst, src)
		name := strconv.Itoa(n)
		b.Run(name+"/whole_lines", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				pl.FFT(dst, src)
			}
		})
		if ps.strips[0] != nil {
			b.Run(name+"/whole_strips", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					ps.FFT(dst, src)
				}
			})
		}
		b.Run(name+"/rows", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				pl.contiguousLines(dst, dst, 1, 0, n, false)
			}
		})
		blocks := n / blockWidth(n)
		b.Run(name+"/cols", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				pl.blockedLines(dst, dst, 0, 0, blocks, false)
			}
		})
		if ps.strips[0] != nil {
			strips := (n + stripWidth(n) - 1) / stripWidth(n)
			b.Run(name+"/strips", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					ps.stripLines(dst, dst, 0, 0, strips, false)
				}
			})
		}
	}
}

// BenchmarkR21Threshold times n×n plans, complex128 and complex64, on all
// cores with parThreshold at 16384 (the other architectures') and 65536
// (arm64's).
func BenchmarkR21Threshold(b *testing.B) {
	defer func(v int) { parThreshold = v }(parThreshold)
	for _, n := range []int{128, 192, 256, 512} {
		p := NewPlanN(n, n)
		src := benchComplex(n * n)
		dst := make([]complex128, n*n)
		p32 := NewPlanN32(n, n)
		src32 := make([]complex64, n*n)
		for i, v := range src {
			src32[i] = complex64(v)
		}
		dst32 := make([]complex64, n*n)
		for _, t := range []int{1 << 14, 1 << 16} {
			parThreshold = t
			b.Run(strconv.Itoa(n)+"/c128_t"+strconv.Itoa(t), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					p.FFT(dst, src)
				}
			})
			b.Run(strconv.Itoa(n)+"/c64_t"+strconv.Itoa(t), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					p32.FFT(dst32, src32)
				}
			})
		}
	}
}
