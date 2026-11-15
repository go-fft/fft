package fft

import (
	"fmt"
	"testing"
)

// BenchmarkFusedS times Plan32.FFT with the fused complex64 kernels off and on
// (interleave with -test.count; the gates are forced on, on any AVX-512 host).
func BenchmarkFusedS(b *testing.B) {
	enableFusedS(b)
	isolatePlans(b)
	for _, c := range fusedSCases() {
		for _, on := range []bool{false, true} {
			fusedSEnabled[c.n] = on
			forgetAllPlans()
			p := NewPlan32(c.n)
			x := make([]complex64, c.n)
			for i := range x {
				x[i] = complex(float32(i%13)*0.1, float32(i%11)*0.1)
			}
			d := make([]complex64, c.n)
			b.Run(fmt.Sprintf("%d/fused=%v", c.n, on), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					p.FFT(d, x)
				}
			})
		}
	}
}
