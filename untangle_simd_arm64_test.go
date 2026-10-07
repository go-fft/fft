package fft

import (
	"math"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// The float64 real-FFT untangle kernels on NEON (kernels.Untangle, Retangle)
// against the Go loops rfftUntangle and irfftRetangle, bit for bit outside
// -race (neonMatch), at every half-length m up to 700 and a few large ones,
// on neonSignals plus subnormals of mixed parity.

func armrealSizes() []int {
	sizes := []int{2048, 4097, 32768}
	for m := 1; m <= 700; m++ {
		sizes = append(sizes, m)
	}
	if testing.Short() {
		sizes = sizes[:3+200]
	}
	return sizes
}

// armrealSignals are neonSignals plus subnormals of both parities: the sum of
// two odd multiples of the smallest subnormal is even and halves exactly, so
// only a mixed signal tells a halving rounded on its own from one fused into
// the next addition.
func armrealSignals(n int) [][]complex128 {
	mixed := make([]complex128, n)
	for i := range mixed {
		mixed[i] = complex(float64(i%7+1)*5e-324, -float64(i%4+1)*5e-324)
	}
	return append(neonSignals(n), mixed)
}

func TestUntangleMatchesScalarNEON(t *testing.T) {
	defer func(v bool) { kernels.UseUntangleNEON = v }(kernels.UseUntangleNEON)
	for _, m := range armrealSizes() {
		tw := NewRealPlan(2 * m).tw
		for s, z := range armrealSignals(m) {
			kernels.UseUntangleNEON = false
			scalar := make([]complex128, m+1)
			rfftUntangle(scalar, z, tw, m)
			kernels.UseUntangleNEON = true
			simd := make([]complex128, m+1)
			rfftUntangle(simd, z, tw, m)
			for i := range scalar {
				if !neonMatch(simd[i], scalar[i]) {
					t.Fatalf("m=%d signal %d bin %d: kernel %v, Go %v", m, s, i, simd[i], scalar[i])
				}
			}
		}
	}
}

func TestRetangleMatchesScalarNEON(t *testing.T) {
	defer func(v bool) { kernels.UseUntangleNEON = v }(kernels.UseUntangleNEON)
	for _, m := range armrealSizes() {
		tw := NewRealPlan(2 * m).tw
		// h: the IRFFT's own (0.5/m, and 0.5 when the inverse normalizes on
		// unpack), 1/sqrt(n), and a factor that sends products subnormal.
		hs := []float64{0.5 / float64(m), 0.5, 1 / math.Sqrt(float64(2*m)), 0x1p-1030}
		for s, x := range armrealSignals(m + 1) {
			for _, h := range hs {
				kernels.UseUntangleNEON = false
				scalar := make([]complex128, m)
				irfftRetangle(scalar, x, tw, m, h)
				kernels.UseUntangleNEON = true
				simd := make([]complex128, m)
				irfftRetangle(simd, x, tw, m, h)
				for i := 1; i < m; i++ {
					if !neonMatch(simd[i], scalar[i]) {
						t.Fatalf("m=%d signal %d h=%g bin %d: kernel %v, Go %v", m, s, h, i, simd[i], scalar[i])
					}
				}
			}
		}
	}
}

// TestRealPlanUntangleNEONEndToEnd runs RealPlan with the untangle kernels on
// and off: RFFT and IRFFT (complete spectrum: kernel; one bin short: the
// per-bin path) agree, bit for bit outside -race.
func TestRealPlanUntangleNEONEndToEnd(t *testing.T) {
	defer func(v bool) { kernels.UseUntangleNEON = v }(kernels.UseUntangleNEON)
	for _, n := range []int{10, 12, 18, 20, 64, 100, 256, 1000, 1024, 1080, 1920, 4096, 65536} {
		p := NewRealPlan(n)
		for s, c := range armrealSignals(n) {
			x := make([]float64, n)
			for i, v := range c {
				x[i] = real(v)
			}
			var spec [2][]complex128
			var back, short [2][]float64
			for k, on := range []bool{false, true} {
				kernels.UseUntangleNEON = on
				spec[k] = p.RFFT(make([]complex128, n/2+1), x)
				back[k] = p.IRFFT(make([]float64, n), spec[0])
				short[k] = p.IRFFT(make([]float64, n), spec[0][:n/2])
			}
			for i := range spec[0] {
				if !neonMatch(spec[1][i], spec[0][i]) {
					t.Fatalf("n=%d signal %d RFFT bin %d: kernel %v, Go %v", n, s, i, spec[1][i], spec[0][i])
				}
			}
			for i := range back[0] {
				if !neonMatch(complex(back[1][i], short[1][i]), complex(back[0][i], short[0][i])) {
					t.Fatalf("n=%d signal %d IRFFT sample %d: kernel %v/%v, Go %v/%v", n, s, i, back[1][i], short[1][i], back[0][i], short[0][i])
				}
			}
		}
	}
}
