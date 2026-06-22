package fft

// Head-to-head benchmark matrix kept in-package for a quick
// `go test -run=^$ -bench=H2H -count=3 .` check. The full, standardized
// parity sweep against FFTW / numpy / scipy / gonum (same host, same inputs,
// GFLOP/s, correctness-gated) lives in the self-contained benchmarks/ module
// and is reported in BENCHMARKS.md; run it with benchmarks/run.sh.

import (
	"strconv"
	"testing"
)

var h2hSizes = []int{64, 256, 1024, 4096, 65536, 1000, 1296, 10007, 9973, 5003, 2017}

func h2hComplex(n int) []complex128 {
	x := make([]complex128, n)
	for i := range x {
		x[i] = complex(float64((i*7+1)%13)*0.1, float64((i*3+2)%11)*0.1)
	}
	return x
}

func h2hReal(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = float64((i*7+1)%13) * 0.1
	}
	return x
}

func BenchmarkH2HComplex(b *testing.B) {
	for _, n := range h2hSizes {
		x := h2hComplex(n)
		p := NewPlan(n)
		dst := make([]complex128, n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, x)
			}
		})
	}
}

func BenchmarkH2HReal(b *testing.B) {
	for _, n := range []int{64, 256, 1024, 4096, 65536, 1000, 1296} {
		x := h2hReal(n)
		p := NewRealPlan(n)
		dst := make([]complex128, n/2+1)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.RFFT(dst, x)
			}
		})
	}
}

var h2hShapes = [][2]int{{64, 64}, {128, 128}, {256, 256}, {512, 512}, {1024, 1024}}

func BenchmarkH2HFFT2(b *testing.B) {
	for _, s := range h2hShapes {
		x := h2hComplex(s[0] * s[1])
		b.Run(strconv.Itoa(s[0])+"x"+strconv.Itoa(s[1]), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = FFT2(x, s)
			}
		})
	}
}
