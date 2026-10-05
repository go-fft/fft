package benchmarks

import (
	"strconv"
	"testing"

	gofft "github.com/go-fft/fft"
)

// Single-precision counterparts of BenchmarkComplex_GoFFT, BenchmarkReal_GoFFT
// and BenchmarkCReal_GoFFT, on the same sizes and the same inputs rounded to
// float32, through the reused Plan32/RealPlan32.

func BenchmarkComplex32_GoFFT(b *testing.B) {
	for _, n := range complexSizes {
		x := make([]complex64, n)
		for i, v := range cmplx(n) {
			x[i] = complex64(v)
		}
		p := gofft.NewPlan32(n)
		dst := make([]complex64, n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, x)
			}
		})
	}
}

func BenchmarkReal32_GoFFT(b *testing.B) {
	for _, n := range realSizes {
		x := make([]float32, n)
		for i, v := range realv(n) {
			x[i] = float32(v)
		}
		p := gofft.NewRealPlan32(n)
		dst := make([]complex64, n/2+1)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.RFFT(dst, x)
			}
		})
	}
}

func BenchmarkCReal32_GoFFT(b *testing.B) {
	for _, n := range realSizes {
		spec := make([]complex64, n/2+1)
		for i, v := range halfSpectrum(n) {
			spec[i] = complex64(v)
		}
		p := gofft.NewRealPlan32(n)
		dst := make([]float32, n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.IRFFT(dst, spec)
			}
		})
	}
}
