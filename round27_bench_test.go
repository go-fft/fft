package fft

// The benchmarks behind BENCHMARKS.md's Round 27: the share of the float64
// real-FFT untangle in RFFT and IRFFT, and the parts of a small 2-D
// transform, on arm64.

import (
	"os"
	"strconv"
	"testing"
)

// BenchmarkR27Real times RealPlan's RFFT and IRFFT whole, and their untangle
// (rfftUntangle) and rebuild (irfftRetangle) loops alone on the same buffers.
func BenchmarkR27Real(b *testing.B) {
	for _, n := range []int{256, 1024, 1080, 4096} {
		p := NewRealPlan(n)
		m := n / 2
		x := benchReal(n)
		spec := make([]complex128, m+1)
		p.RFFT(spec, x)
		out := make([]float64, n)
		z := cmplxSignal(m)
		dst := make([]complex128, m+1)
		name := "n" + strconv.Itoa(n)
		b.Run(name+"/rfft", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.RFFT(spec, x)
			}
		})
		b.Run(name+"/irfft", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.IRFFT(out, spec)
			}
		})
		b.Run(name+"/untangle", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				rfftUntangle(dst, z, p.tw, m)
			}
		})
		b.Run(name+"/retangle", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				irfftRetangle(z, spec, p.tw, m, 0.5/float64(m))
			}
		})
	}
}

// BenchmarkR27Parts2D times an n×n complex128 PlanN whole on one goroutine,
// then its rows and its columns (as strips, and gathered) alone.
func BenchmarkR27Parts2D(b *testing.B) {
	for _, n := range []int{64, 128} {
		p := NewPlanN(n, n)
		x := benchComplex(n * n)
		d := make([]complex128, n*n)
		name := "n" + strconv.Itoa(n)
		b.Run(name+"/whole", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(d, x)
			}
		})
		b.Run(name+"/rows", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.contiguousLines(d, x, 1, 0, n, false)
			}
		})
		if p.strips[0] != nil {
			strips := (n + stripWidth(n) - 1) / stripWidth(n)
			b.Run(name+"/strips", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					p.stripLines(d, d, 0, 0, strips, false)
				}
			})
		}
		blocks := n / blockWidth(n)
		b.Run(name+"/gathered", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.blockedLines(d, d, 0, 0, blocks, false)
			}
		})
	}
}

// BenchmarkR27ND times complex128 PlanN forward transforms; R27ROT=0 in the
// environment builds the plans without the rotating passes.
func BenchmarkR27ND(b *testing.B) {
	if os.Getenv("R27ROT") == "0" {
		defer func(v bool) { armrealRotateND = v }(armrealRotateND)
		armrealRotateND = false
	}
	for _, s := range [][]int{{16, 16}, {32, 32}, {64, 64}, {128, 128}, {64, 128}, {100, 100}, {120, 120}, {240, 240}, {16, 16, 16}, {32, 32, 32}, {1000, 3}, {256, 256}, {512, 512}, {1024, 1024}, {64, 64, 64}} {
		p := NewPlanN(s...)
		x := benchComplex(p.Len())
		d := make([]complex128, p.Len())
		name := ""
		for i, n := range s {
			if i > 0 {
				name += "x"
			}
			name += strconv.Itoa(n)
		}
		b.Run("s"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(d, x)
			}
		})
	}
}
