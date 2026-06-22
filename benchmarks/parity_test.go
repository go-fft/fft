// Package benchmarks is the head-to-head parity harness for the standardized
// performance report (BENCHMARKS.md). It times go-fft against gonum (the
// fairest pure-Go, CGO=0 peer) on identical inputs and the same size taxonomy
// the Python reference harness (fft_reference.py) uses for FFTW/numpy/scipy, so
// every implementation is compared on the same transforms and the same host.
//
// go-fft is measured through its cached Plan API (NewPlan(n).FFT,
// NewRealPlan(n).RFFT), the apples-to-apples match for gonum's reused plan
// object and FFTW's reused plan: twiddle factors are built once and amortized,
// so each ns/op is the steady-state transform cost. A separate set of
// Benchmark*_Plan benchmarks isolates the plan/setup cost.
//
// Inputs are bit-identical to fft_reference.py's cmplx/realv generators.
//
//	go test -run=^$ -bench=. -benchmem .
package benchmarks

import (
	"strconv"
	"testing"

	gofft "github.com/go-fft/fft"
	"gonum.org/v1/gonum/dsp/fourier"
)

// Size taxonomy — mirrors fft_reference.py.
// Powers of two up to 1M, mixed-radix / highly-composite, and a prime.
var complexSizes = []int{256, 1024, 4096, 65536, 1048576,
	1000, 1080, 1920, 1009, 1296, 10007}

var realSizes = []int{256, 1024, 4096, 65536, 1048576, 1000, 1080, 1920}

var shapes = [][2]int{{64, 64}, {128, 128}, {256, 256}, {512, 512}, {1024, 1024}}

func cmplx(n int) []complex128 {
	x := make([]complex128, n)
	for i := range x {
		x[i] = complex(float64((i*7+1)%13)*0.1, float64((i*3+2)%11)*0.1)
	}
	return x
}

func realv(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = float64((i*7+1)%13) * 0.1
	}
	return x
}

// --- Complex 1-D: steady-state, plan reused ---

func BenchmarkComplex_GoFFT(b *testing.B) {
	for _, n := range complexSizes {
		x := cmplx(n)
		p := gofft.NewPlan(n)
		dst := make([]complex128, n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, x)
			}
		})
	}
}

func BenchmarkComplex_Gonum(b *testing.B) {
	for _, n := range complexSizes {
		x := cmplx(n)
		f := fourier.NewCmplxFFT(n)
		dst := make([]complex128, n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				f.Coefficients(dst, x)
			}
		})
	}
}

// --- Real 1-D: steady-state, plan reused ---

func BenchmarkReal_GoFFT(b *testing.B) {
	for _, n := range realSizes {
		x := realv(n)
		p := gofft.NewRealPlan(n)
		dst := make([]complex128, n/2+1)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.RFFT(dst, x)
			}
		})
	}
}

func BenchmarkReal_Gonum(b *testing.B) {
	for _, n := range realSizes {
		x := realv(n)
		f := fourier.NewFFT(n)
		dst := make([]complex128, n/2+1)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				f.Coefficients(dst, x)
			}
		})
	}
}

// --- 2-D complex (go-fft multicore path) ---

func BenchmarkFFT2_GoFFT(b *testing.B) {
	for _, s := range shapes {
		x := cmplx(s[0] * s[1])
		b.Run(strconv.Itoa(s[0])+"x"+strconv.Itoa(s[1]), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = gofft.FFT2(x, s)
			}
		})
	}
}

// --- Plan / setup cost (reported separately from steady-state) ---

func BenchmarkComplexPlan_GoFFT(b *testing.B) {
	for _, n := range complexSizes {
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = gofft.NewPlan(n)
			}
		})
	}
}

func BenchmarkComplexPlan_Gonum(b *testing.B) {
	for _, n := range complexSizes {
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = fourier.NewCmplxFFT(n)
			}
		})
	}
}
