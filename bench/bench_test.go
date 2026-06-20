// Package bench holds the head-to-head benchmark harness comparing go-fft
// against gonum.org/v1/gonum/dsp/fourier — the fairest pure-Go (CGO=0) peer.
//
// Both libraries are exercised on identical inputs across the size taxonomy in
// the perf plan: powers of two (64..65536), highly-composite N (1000, 1296), a
// large prime (worst case), and the real-input path. Reported ns/op are the
// yardstick recorded in docs/perf.md.
//
// gonum reuses a pre-allocated plan object (NewCmplxFFT / NewFFT) whose twiddle
// tables are built once in the constructor and amortized across calls, so its
// Coefficients call measures the steady-state transform cost only. go-fft is
// measured both via its allocating one-shot API (FFT/RFFT) and via its cached
// Plan API (NewPlan / Plan.FFT) so the comparison is apples-to-apples for the
// reuse case as well as honest about the convenience-API allocation cost.
package bench

import (
	"math"
	"strconv"
	"testing"

	gofft "github.com/go-fft/fft"
	"gonum.org/v1/gonum/dsp/fourier"
)

// sizes is the complex-FFT size taxonomy: powers of two, highly-composite N,
// and a large prime (worst case for the algorithm).
var sizes = []int{64, 256, 1024, 4096, 65536, 1000, 1296, 10007}

func benchComplex(n int) []complex128 {
	x := make([]complex128, n)
	for i := range x {
		x[i] = complex(math.Sin(float64(i)*0.7), math.Cos(float64(i)*0.3))
	}
	return x
}

func benchReal(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(float64(i)*0.7) + float64(i%4)
	}
	return x
}

// BenchmarkComplex_GoFFT measures go-fft's allocating one-shot FFT API.
func BenchmarkComplex_GoFFT(b *testing.B) {
	for _, n := range sizes {
		x := benchComplex(n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = gofft.FFT(x)
			}
		})
	}
}

// BenchmarkComplex_GoFFTPlan measures go-fft's cached Plan API (twiddle tables
// built once), the apples-to-apples match for gonum's reused plan object.
func BenchmarkComplex_GoFFTPlan(b *testing.B) {
	for _, n := range sizes {
		x := benchComplex(n)
		p := gofft.NewPlan(n)
		dst := make([]complex128, n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, x)
			}
		})
	}
}

// BenchmarkComplex_Gonum measures gonum's reused CmplxFFT plan.
func BenchmarkComplex_Gonum(b *testing.B) {
	for _, n := range sizes {
		x := benchComplex(n)
		fft := fourier.NewCmplxFFT(n)
		dst := make([]complex128, n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fft.Coefficients(dst, x)
			}
		})
	}
}

// realSizes excludes the large prime (gonum's real FFT is also slow there and
// the point is made by the complex prime case).
var realSizes = []int{64, 256, 1024, 4096, 65536, 1000, 1296}

// BenchmarkReal_GoFFT measures go-fft's allocating RFFT API.
func BenchmarkReal_GoFFT(b *testing.B) {
	for _, n := range realSizes {
		x := benchReal(n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = gofft.RFFT(x)
			}
		})
	}
}

// BenchmarkReal_GoFFTPlan measures go-fft's cached real-FFT plan.
func BenchmarkReal_GoFFTPlan(b *testing.B) {
	for _, n := range realSizes {
		x := benchReal(n)
		p := gofft.NewRealPlan(n)
		dst := make([]complex128, n/2+1)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.RFFT(dst, x)
			}
		})
	}
}

// BenchmarkReal_Gonum measures gonum's reused real FFT plan.
func BenchmarkReal_Gonum(b *testing.B) {
	for _, n := range realSizes {
		x := benchReal(n)
		fft := fourier.NewFFT(n)
		dst := make([]complex128, n/2+1)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fft.Coefficients(dst, x)
			}
		})
	}
}
