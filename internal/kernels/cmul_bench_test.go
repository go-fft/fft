package kernels

import "testing"

// benchmarkCMul times a single in-place pointwise complex multiply of length n.
func benchmarkCMul(b *testing.B, n int, f func(a, b []complex128)) {
	x := make([]complex128, n)
	y := make([]complex128, n)
	for i := 0; i < n; i++ {
		x[i] = complex(float64(i)*0.5, float64(i)*-0.25)
		y[i] = complex(float64(i)*0.125, float64(i)*0.0625)
	}
	b.SetBytes(int64(n) * 16)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f(x, y)
	}
}

// On arm64/amd64 the autovectorized scalar loop wins at these widths
// (documented in the Phase 4 plan), which is why the dispatch keeps the scalar
// default; the SIMD kernel is benchmarked here so the delta is visible.
func BenchmarkCMulScalar1024(b *testing.B) { benchmarkCMul(b, 1024, CMulScalar) }
func BenchmarkCMulSIMD1024(b *testing.B)   { benchmarkCMul(b, 1024, cmulSIMD) }
func BenchmarkCMulScalar4096(b *testing.B) { benchmarkCMul(b, 4096, CMulScalar) }
func BenchmarkCMulSIMD4096(b *testing.B)   { benchmarkCMul(b, 4096, cmulSIMD) }
