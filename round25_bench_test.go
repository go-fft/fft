package fft

// The benchmarks behind BENCHMARKS.md's Round 25: the float32 untangle alone
// (Go loop against kernel), and the parts of a float32 2-D transform next to
// the float64 ones.

import (
	"strconv"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// BenchmarkR25Untangle32 times f32Untangle and f32rRetangle with the kernels
// off ("go") and on ("simd"), where the platform has them.
func BenchmarkR25Untangle32(b *testing.B) {
	defer func(v bool) { kernels.UseUntangle32 = v }(kernels.UseUntangle32)
	simd := kernels.UseUntangle32
	for _, n := range []int{256, 1024, 4096, 65536, 1000, 1920} {
		m := n / 2
		tw := NewRealPlan32(n).tw
		z := make([]complex64, m+1)
		for i, v := range cmplxSignal(m + 1) {
			z[i] = complex64(v)
		}
		dst := make([]complex64, m+1)
		for _, on := range []bool{false, true} {
			if on && !simd {
				continue
			}
			mode := map[bool]string{false: "go", true: "simd"}[on]
			b.Run(strconv.Itoa(n)+"/untangle-"+mode, func(b *testing.B) {
				kernels.UseUntangle32 = on
				for i := 0; i < b.N; i++ {
					f32Untangle(dst, z[:m], tw, m)
				}
			})
			b.Run(strconv.Itoa(n)+"/retangle-"+mode, func(b *testing.B) {
				kernels.UseUntangle32 = on
				for i := 0; i < b.N; i++ {
					f32rRetangle(dst, z, tw, m, 0.5)
				}
			})
		}
	}
}

// BenchmarkR25Parts2D32 times an n×n float32 plan's rows and its gathered
// columns on one goroutine, next to the float64 plan's rows, gathered
// columns and batched strips (where they run).
func BenchmarkR25Parts2D32(b *testing.B) {
	for _, n := range []int{64, 128, 256, 512, 1024} {
		p32, p64 := withStrips32(false, n, n), NewPlanN(n, n)
		p32.strips = withStrips32(true, n, n).strips
		p32.bufs.New = withStrips32(true, n, n).bufs.New
		d32, d64 := make([]complex64, n*n), make([]complex128, n*n)
		for i, v := range benchComplex(n * n) {
			d32[i], d64[i] = complex64(v), v
		}
		name := strconv.Itoa(n)
		b.Run(name+"/f32-rows", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p32.contiguousLines(d32, d32, 1, 0, n, false)
			}
		})
		blocks32 := n / f32ndBlockWidth(n)
		b.Run(name+"/f32-cols", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p32.blockedLines(d32, d32, 0, 0, blocks32, false)
			}
		})
		if p32.strips[0] != nil {
			strips := (n + stripWidth32(n) - 1) / stripWidth32(n)
			b.Run(name+"/f32-strips", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					p32.stripLines(d32, d32, 0, 0, strips, false)
				}
			})
		}
		b.Run(name+"/f64-rows", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p64.contiguousLines(d64, d64, 1, 0, n, false)
			}
		})
		blocks64 := n / blockWidth(n)
		b.Run(name+"/f64-cols", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p64.blockedLines(d64, d64, 0, 0, blocks64, false)
			}
		})
		if p64.strips[0] != nil {
			strips := (n + stripWidth(n) - 1) / stripWidth(n)
			b.Run(name+"/f64-strips", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					p64.stripLines(d64, d64, 0, 0, strips, false)
				}
			})
		}
	}
}

// BenchmarkR25StripWidth32 times the columns of an n×n float32 plan as strips
// of w lines, for several w, on one goroutine.
func BenchmarkR25StripWidth32(b *testing.B) {
	defer func(f func(int) int) { stripWidth32 = f }(stripWidth32)
	for _, n := range []int{64, 128, 256, 512, 1024} {
		d := make([]complex64, n*n)
		for i, v := range benchComplex(n * n) {
			d[i] = complex64(v)
		}
		for _, w := range []int{4, 8, 16, 32, 64} {
			stripWidth32 = func(int) int { return w }
			p := withStrips32(true, n, n)
			strips := (n + w - 1) / w
			b.Run(strconv.Itoa(n)+"/w"+strconv.Itoa(w), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					p.stripLines(d, d, 0, 0, strips, false)
				}
			})
		}
	}
}

// BenchmarkR25Fanout32 times n×n float32 plans (strips on where they run)
// with the shared fan-out rule ("rule": parallelizeLines, parMinChunk) and
// on one goroutine ("serial"), on all GOMAXPROCS cores.
func BenchmarkR25Fanout32(b *testing.B) {
	defer func(v int) { parMinChunk = v }(parMinChunk)
	def := parMinChunk
	for _, n := range []int{64, 96, 128, 160, 192, 224, 256, 512} {
		p := NewPlanN32(n, n)
		src := make([]complex64, n*n)
		for i, v := range benchComplex(n * n) {
			src[i] = complex64(v)
		}
		dst := make([]complex64, n*n)
		for _, m := range []int{def, 1 << 40} {
			name := "rule"
			if m == 1<<40 {
				name = "serial"
			}
			b.Run(strconv.Itoa(n)+"/"+name, func(b *testing.B) {
				parMinChunk = m
				for i := 0; i < b.N; i++ {
					p.FFT(dst, src)
				}
			})
		}
	}
}
