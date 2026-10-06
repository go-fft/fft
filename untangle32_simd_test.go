//go:build amd64 || arm64

package fft

import (
	"math"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// The float32 real-FFT untangle kernels (kernels.Untangle32, Retangle32)
// against the Go loops f32Untangle and f32Retangle, bit for bit (to rounding
// under -race on arm64, see f32simdMatch), at every half-length m up to 700
// and a few large ones, on the float32 SIMD signals: generic, signed zeros,
// ±0/±1 draws, an infinity, a constant and float32 subnormals.

func f32rSizes() []int {
	sizes := []int{2048, 4097, 32768}
	for m := 1; m <= 700; m++ {
		sizes = append(sizes, m)
	}
	if testing.Short() {
		sizes = sizes[:3+200]
	}
	return sizes
}

// f32rSignals are f32simdSignals plus subnormals of both parities: a sum of
// two odd multiples of the smallest subnormal is even, so f32simdSignals'
// subnormal signal halves exactly, and only a mixed one tells a halving
// rounded on its own from one fused into the next addition.
func f32rSignals(n int) [][]complex64 {
	sub := float32(math.SmallestNonzeroFloat32)
	mixed := make([]complex64, n)
	for i := range mixed {
		mixed[i] = complex(float32(i%7+1)*sub, -float32(i%4+1)*sub)
	}
	return append(f32simdSignals(n), mixed)
}

func TestUntangle32MatchesScalar(t *testing.T) {
	if !kernels.UseUntangle32 {
		t.Skip("no float32 untangle kernel on this CPU")
	}
	defer func(v bool) { kernels.UseUntangle32 = v }(kernels.UseUntangle32)
	for _, m := range f32rSizes() {
		tw := NewRealPlan32(2 * m).tw
		for s, z := range f32rSignals(m) {
			kernels.UseUntangle32 = false
			scalar := make([]complex64, m+1)
			f32Untangle(scalar, z, tw, m)
			kernels.UseUntangle32 = true
			simd := make([]complex64, m+1)
			f32Untangle(simd, z, tw, m)
			tol := f32simdTol(scalar)
			for i := range scalar {
				if !f32simdMatch(simd[i], scalar[i], tol) {
					t.Fatalf("m=%d signal %d bin %d: kernel %v, Go %v", m, s, i, simd[i], scalar[i])
				}
			}
		}
	}
}

func TestRetangle32MatchesScalar(t *testing.T) {
	if !kernels.UseUntangle32 {
		t.Skip("no float32 untangle kernel on this CPU")
	}
	defer func(v bool) { kernels.UseUntangle32 = v }(kernels.UseUntangle32)
	for _, m := range f32rSizes() {
		tw := NewRealPlan32(2 * m).tw
		// h: the IRFFT's own (s = 1/n under NormBackward), 1/sqrt(n), 1,
		// and a factor that sends the generic signal's products subnormal.
		hs := []float32{float32(1 / float64(2*m)), float32(1 / math.Sqrt(float64(2*m))), 1, 0x1p-120}
		for s, x := range f32rSignals(m + 1) {
			for _, h := range hs {
				kernels.UseUntangle32 = false
				scalar := make([]complex64, m)
				f32rRetangle(scalar, x, tw, m, h)
				kernels.UseUntangle32 = true
				simd := make([]complex64, m)
				f32rRetangle(simd, x, tw, m, h)
				tol := f32simdTol(scalar)
				for i := 1; i < m; i++ {
					if !f32simdMatch(simd[i], scalar[i], tol) {
						t.Fatalf("m=%d signal %d h=%g bin %d: kernel %v, Go %v", m, s, h, i, simd[i], scalar[i])
					}
				}
			}
		}
	}
}

// TestRealPlan32KernelsEndToEnd runs RealPlan32 with the untangle kernels on
// and off: RFFT and IRFFT (complete spectrum, kernel; short spectrum, the
// per-bin path) agree bit for bit, and a short spectrum gives what the
// complete one zero-filled gives.
func TestRealPlan32KernelsEndToEnd(t *testing.T) {
	if !kernels.UseUntangle32 {
		t.Skip("no float32 untangle kernel on this CPU")
	}
	defer func(v bool) { kernels.UseUntangle32 = v }(kernels.UseUntangle32)
	for _, n := range []int{18, 20, 64, 100, 256, 1000, 1024, 1080, 1920, 4096} {
		p := NewRealPlan32(n)
		for s, c := range f32simdSignals(n) {
			x := make([]float32, n)
			for i, v := range c {
				x[i] = real(v)
			}
			var spec [2][]complex64
			var back [2][]float32
			for k, on := range []bool{false, true} {
				kernels.UseUntangle32 = on
				spec[k] = p.RFFT(make([]complex64, n/2+1), x)
				back[k] = p.IRFFT(make([]float32, n), spec[0])
			}
			tol := f32simdTol(spec[0])
			for i := range spec[0] {
				if !f32simdMatch(spec[1][i], spec[0][i], tol) {
					t.Fatalf("n=%d signal %d RFFT bin %d: kernel %v, Go %v", n, s, i, spec[1][i], spec[0][i])
				}
			}
			for i := range back[0] {
				if !f32simdMatch(complex(back[1][i], 0), complex(back[0][i], 0), 1e-6*math.Log2(float64(n))) {
					t.Fatalf("n=%d signal %d IRFFT sample %d: kernel %v, Go %v", n, s, i, back[1][i], back[0][i])
				}
			}
			// A spectrum one bin short (the Nyquist bin) runs the per-bin
			// path; zero-filled, the complete one runs the kernel.
			short := spec[0][:n/2]
			full := append(append([]complex64(nil), short...), 0)
			a := p.IRFFT(make([]float32, n), short)
			b := p.IRFFT(make([]float32, n), full)
			for i := range a {
				if !f32simdMatch(complex(a[i], 0), complex(b[i], 0), 1e-6*math.Log2(float64(n))) {
					t.Fatalf("n=%d signal %d sample %d: short spectrum %v, zero-filled %v", n, s, i, a[i], b[i])
				}
			}
		}
	}
}
