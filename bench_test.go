package fft

import (
	"math"
	"strconv"
	"testing"
)

// In-package benchmarks compare the O(N log N) transforms against the naive
// O(N²) DFT baseline and exercise the real-input path. The head-to-head
// comparison against gonum and FFTW/pocketfft lives in the separate bench/
// module (so gonum never enters the library go.mod); see docs/perf.md.

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

func BenchmarkFFT(b *testing.B) {
	for _, n := range []int{64, 256, 1024, 4096} {
		x := benchComplex(n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = FFT(x)
			}
		})
	}
}

// BenchmarkNaiveDFT is the O(N²) baseline the FFT must beat; only small sizes,
// since it grows quadratically.
func BenchmarkNaiveDFT(b *testing.B) {
	for _, n := range []int{64, 256, 1024} {
		x := benchComplex(n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = naiveDFT(x)
			}
		})
	}
}

func BenchmarkRFFT(b *testing.B) {
	for _, n := range []int{64, 256, 1024, 4096} {
		x := benchReal(n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = RFFT(x)
			}
		})
	}
}

// BenchmarkFFTReal contrasts the half-length-packed RFFT against promoting the
// real signal to complex and running a full FFT, quantifying the ~2× win.
func BenchmarkFFTReal(b *testing.B) {
	for _, n := range []int{64, 256, 1024, 4096} {
		x := benchReal(n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = FFTReal(x)
			}
		})
	}
}

func BenchmarkIRFFT(b *testing.B) {
	for _, n := range []int{64, 256, 1024, 4096} {
		spec := RFFT(benchReal(n))
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = IRFFT(spec, n)
			}
		})
	}
}
