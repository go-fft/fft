package benchmarks

import (
	"strconv"
	"testing"

	gofft "github.com/go-fft/fft"
)

// BenchmarkPow2_GoFFT times every power of two from 2^6 to 2^22, plan reused
// (the same steady-state method as BenchmarkComplex_GoFFT). It exists to find
// where two routes cross: the parity sizes skip most exponents.
func BenchmarkPow2_GoFFT(b *testing.B) {
	for e := 6; e <= 22; e++ {
		n := 1 << e
		x := cmplx(n)
		p := gofft.NewPlanWith(n, fmaOpt)
		dst := make([]complex128, n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, x)
			}
		})
	}
}
